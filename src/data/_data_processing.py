"""
%cd ../
%load_ext autoreload
%autoreload 2
"""

import os
import glob
import json
import cohere
from time import sleep
from time import time
from hashlib import md5
from typing import Dict
from typing import List
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from multiprocessing.pool import ThreadPool
from threading import Lock
from util import config

PROMPT_TEMPLATE = """
You need to extract question or circumstance, and the response to the question in the given legal article. The response should contain clauses, circulars or links support for the answer. You should return in prober multiline JSON format, here is an example of how you should response:
[
    {
    "question": "Đơn vị nhận Lệnh thanh toán trong hệ thống thanh toán điện tử liên ngân hàng "
        "được định nghĩa ra sao? Vấn đề này có được văn bản pháp luật nào nói đến hay không?",
    "answer": "Đơn vị nhận Lệnh thanh toán trong hệ thống thanh toán điện tử liên ngân hàng được "
        "định nghĩa tại Khoản 14 Điều 2 Thông tư 23/2010/TT-NHNN quy định về quản lý, vận "
        "hành và sử dụng hệ thống thanh toán điện tử liên ngân hàng do Ngân hàng Nhà nước "
        "Việt Nam ban hành với nội dung như sau:\n Đơn vị nhận Lệnh thanh toán (viết tắt là "
        "đơn vị nhận lệnh) là thành viên hoặc đơn vị thành viên thay mặt người nhận lệnh nhận "
        "và xử lý Lệnh thanh toán (đến)"
    },
    {
    "question": "Người phạm tội rửa tiền bị truy cứu trách nhiệm hình sự như thế nào?",
    "answer": "Căn cứ theo Điều 324 Bộ luật Hình sự 2015 được sửa đổi bởi khoản 122 Điều 1 "
        "Luật sửa đổi Bộ luật Hình sự 2017 quy định về người phạm tội rửa tiền bị bị truy cứu "
        "hình sự như sau:\n Khung 1: Người nào thực hiện một trong các hành vi sau đây, thì bị phạt "
        "tù từ 01 năm đến 05 năm\n - Tham gia trực tiếp hoặc gián tiếp vào giao dịch tài chính, "
        "ngân hàng hoặc giao dịch khác nhằm che giấu nguồn gốc bất hợp pháp của tiền, "
        "tài sản do mình phạm tội mà có hoặc biết hay có cơ sở để biết là do người khác phạm "
        "tội mà có."
    }
]
Here is the legal article:
{{article}}
"""

class DataProcessing:
    def __init__(self, src_dir: str, tgt_dir: str, api_key: str):
        logger.info(
            f"Initialize Data Processing instance, read data from {src_dir} "
            f"and write to {tgt_dir}, {self}"
        )

        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.api_key = api_key[:10] + "..."
        self.co = cohere.Client(api_key)
        Path(self.tgt_dir).mkdir(exist_ok=True)

    def list_article_paths(self) -> List[str]:
        path = os.path.join(self.src_dir, "*.json")
        return glob.glob(path)
    
    def log_qna(self, raw: Dict[str, str], processed_data: Dict[str, str]):
        data = {
            "url": raw["url"],
            "data": processed_data,
            "extras": raw["extras"]
        }
        name = f"{md5(raw["url"].encode()).hexdigest()}.json"
        path = os.path.join(self.tgt_dir, name)
        json.dump(data, open(path, "w"), ensure_ascii=False)

    def extract_qna(self, raw: Dict[str, object]) -> List[Dict[str, str]]:
        prompt = PROMPT_TEMPLATE.replace("{{article}}", raw["md"])
        resp = self.co.chat(message=prompt, temperature=0)
        return json.loads(resp.text)
    
def extract_load(bar: tqdm, lock: Lock, data_proc: DataProcessing, path: str):
    raw = json.load(open(path))
    start = time()
    try:
        processed_data = data_proc.extract_qna(raw)
        data_proc.log_qna(raw, processed_data)
        wait = max(6 - time() - start, 0)
        sleep(wait)
    except Exception as e:
        logger.warning(f"Failed to extract path={path}, key={data_proc.api_key}, {e}")
    with lock:
        bar.update(1)

def main():
    api_keys = config.COHERE_API_KEY.split(";")
    data_procs = [
        DataProcessing(
            src_dir=config.DATA_ARTICLE_DIR,
            tgt_dir=config.DATA_QNA_DIR,
            api_key=api_key
        )
        for api_key in api_keys
    ]

    paths = data_procs[0].list_article_paths()[1511:]
    bar = tqdm(paths, desc="Processing data...")
    lock = Lock()
    params = [
        (bar, lock, data_procs[i % 2], paths[i])
        for i in range(len(paths))]

    with ThreadPool(processes=len(data_procs)) as pool:
        pool.starmap(extract_load, params)

if __name__ == "__main__":
    main()