"""
%cd ../
%load_ext autoreload
%autoreload 2
"""

from mlx_lm import load, generate
from util import config

def format_prompt(question)-> str:
    return f"<|user|>\n{question}</s>\n<|assistant|>"

BASE_MODEL="tcstrength/tinyllama-lawyer-assist-v0"
model, tokenizer = load(BASE_MODEL)
user_input = "Việc rút tiền từ ATM được quy định như thế nào?"
response = generate(model, tokenizer, prompt=format_prompt(user_input), max_tokens=1024, verbose=True)