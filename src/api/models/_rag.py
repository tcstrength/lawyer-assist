import os
from typing import Dict, List

import torch
from api.models._base import BaseModel
from data.qdrant_helper import QdrantHelper
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextIteratorStreamer
import string

from dotenv import load_dotenv
load_dotenv()

os.getenv

GEMA_HF_TOKEN = os.getenv('GEMA_HF_TOKEN')

class RAGModel(BaseModel):

    def __init__(self, model_id="tcstrength/gemma-2b-lawyer-assist", token=GEMA_HF_TOKEN):
        self.helper = QdrantHelper()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, )
        self._model = AutoModelForCausalLM.from_pretrained(model_id, token=token, quantization_config=bnb_config)
        self._qa_prompt_tmpl_str = string.Template("""\
Bạn là một luật sư, chuyên giải đáp các câu hỏi liên quan tới văn bản pháp luật.\
Với các văn bản pháp luật dưới đây:
---------------------
${context_str}
---------------------
Trả lời các câu hỏi về pháp luật liên quan tới các văn bản nêu trên.
Vui lòng cung cấp nội dung điều luật trích dẫn trong câu trả lời của bạn. \
Dưới đây là một vài ví dụ để bạn tham khảo.

${few_shot_examples}

Câu hỏi: ${user_input}
Câu trả lời:
""")

    def generate(
        self,
        user_input: str,
        max_tokens: int = 1024,
        history: List[Dict[str, str]] = []
    ):
        documents = [doc.payload['content'] for doc in self.get_relevant_documents(user_input, 10)]
        
        examples = """
Câu hỏi: Nội dung đối với kỳ phiếu được phát hành theo hình thức chứng chỉ hoặc chứng nhận quyền sở hữu theo quy định mới như thế nào?
Câu trả lời: Theo Khoản 3 Điều 11 Thông tư 01/2021/TT-NHNN quy định: Kỳ phiếu, tín phiếu, chứng chỉ tiền gửi phát hành theo hình thức chứng chỉ hoặc chứng nhận quyền sở hữu kỳ phiếu, tín phiếu, chứng chỉ tiền gửi phải bao gồm các nội dung sau: \n- Tên tổ chức phát hành; \n- Tên gọi kỳ phiếu; \n- Ký hiệu, số seri phát hành; \n- Chữ ký của người đại diện hợp pháp của tổ chức tín dụng, chi nhánh ngân hàng nước ngoài phát hành và các chữ ký khác do tổ chức tín dụng, chi nhánh ngân hàng nước ngoài quy định; \n- Mệnh giá, thời hạn, ngày phát hành, ngày đến hạn thanh toán; \n- Lãi suất, phương thức trả lãi, thời điểm trả lãi, địa điểm thanh toán gốc và lãi; \n- Họ tên, số Chứng minh nhân dân hoặc thẻ Căn cước công dân hoặc hộ chiếu còn thời hạn hiệu lực, địa chỉ của người mua (nếu người mua là cá nhân); tên tổ chức mua, số giấy phép thành lập hoặc mã số doanh nghiệp hoặc số giấy chứng nhận đăng ký kinh doanh (trong trường hợp doanh nghiệp chưa có mã số doanh nghiệp), địa chỉ của tổ chức mua (nếu người mua là tổ chức); \n- Đối với kỳ phiếu do công ty tài chính, công ty cho thuê tài chính phát hành, ghi rõ người sở hữu chỉ được chuyển quyền sở hữu cho tổ chức; \n- Các nội dung khác của kỳ phiếu do tổ chức tín dụng, chi nhánh ngân hàng nước ngoài quyết định.
"""

        content = self._qa_prompt_tmpl_str.safe_substitute({'context_str':'\n'.join(documents),
                                                            'few_shot_examples': examples,
                                                            'user_input': user_input})
        
        history = history + [
                {"role": "user", "content": content },
            ]

        input_ids = self._tokenizer.apply_chat_template(
            history, add_generation_prompt=True,
            return_tensors="pt"
        )

        streamer = TextIteratorStreamer(self._tokenizer)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "streamer": streamer,
        }

        import threading
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def get_relevant_documents(self, query: str, k: int = 3):
        return self.helper.search(query, limit=k)
    
if __name__ == "__main__":
    self = RAGModel()
    user_input = "Việc thanh toán giá trị chuyển nhượng cổ phần, phần vốn góp tại doanh nghiệp có vốn đầu tư trực tiếp nước ngoài được quy định như thế nào?"
    # print(self.generate(user_input))
    for text in self.generate(user_input):
        print(text, end="")