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
from hashlib import md5
from typing import Dict
from typing import List
from pathlib import Path
from tqdm import tqdm
from loguru import logger
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
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
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

    # def extract_qna(self, raw: Dict[str, object]):
    #     tools = [cohere.Tool(
    #         name="extract_qna",
    #         description="Extract and log the question and answer and keywords in given legal article.",
    #         parameter_definitions={
    #             "docs": cohere.ToolParameterDefinitionsValue(
    #                 description=(
    #                     "List of dictionaries extracted from the legal article. The dictionary "
    #                     "requires 4 keys: `question`, `answer`. The `question` is the question "
    #                     "or circumstance that people want to be explained, the question should "
    #                     "remove sensitive data such as name, email. The `answer` is the response "
    #                     "for the corresponding question and it should contains detailed "
    #                     "information including terms, definitions, clauses, and circulars."
    #                 ),
    #                 type="List[Dict[str, str]]",
    #                 required=True
    #             ),
    #             # "question": cohere.ToolParameterDefinitionsValue(
    #             #     description=(
    #             #         "The question or circumstance that people want to be explained "
    #             #         "in detailed."
    #             #     ),
    #             #     type="List[str]",
    #             #     required=True
    #             # ),
    #             # "answer": cohere.ToolParameterDefinitionsValue(
    #             #     description=(
    #             #         "The details of answer, keep all the circulars, clauses and "
    #             #         "sanctions, note that summarized answer is not accepted."
    #             #     ),
    #             #     type="str",
    #             #     required=True
    #             # ),
    #             # "conslusions": cohere.ToolParameterDefinitionsValue(
    #             #     description=(
    #             #         "The conclusion of the article, usually directly answer the question."
    #             #     ),
    #             #     type="list",
    #             #     required=True
    #             # ),
    #             # "keywords": cohere.ToolParameterDefinitionsValue(
    #             #     description="The keywords found on the document.",
    #             #     type="str",
    #             #     required=True
    #             # )
    #         }
    #     )]

    #     preamble="""
    #     ## Task Description
    #     You need to extract keywords, questions or circumstances, and answers in the given article.

    #     ## Style Guide
    #     Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling in Vietnamese.
    #     """

    #     document = raw.get("md")
    #     response = self.co.chat(
    #         message=f"## Article\n{document}",
    #         tools=tools,
    #         preamble=preamble
    #     )
    #     params = response.tool_calls[0].parameters
    #     return params
    


def main():
    data_proc = DataProcessing(
        src_dir=config.DATA_ARTICLE_DIR,
        tgt_dir=config.DATA_QNA_DIR,
        api_key=config.COHERE_API_KEY
    )

    for path in tqdm(data_proc.list_article_paths(), desc="Processing data..."):
        raw = json.load(open(path))
        try:
            processed_data = data_proc.extract_qna(raw)
            data_proc.log_qna(raw, processed_data)
            sleep(10)
        except Exception as e:
            logger.warning(f"Failed to extract path={path}, {e}")

if __name__ == "__main__":
    main()