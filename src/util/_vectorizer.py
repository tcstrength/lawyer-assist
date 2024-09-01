from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer


MODEL_ID = "dangvantuan/vietnamese-embedding"


class Vectorizer:
    def __init__(self, model_id: str = MODEL_ID):
        self._model_id = model_id
        self._tokenizer = ViTokenizer
        self._model = SentenceTransformer(model_id)

    def vectorize(self, text: str):
        tokens = self._tokenizer.tokenize(text)
        outputs = self._model.encode([tokens])
        return outputs[0]

if __name__ == "__main__":
    self = Vectorizer()
    out = self.vectorize("Tư vấn pháp luật")

