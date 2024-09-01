"""
%cd ../
%load_ext autoreload
%autoreload 2
"""
import os
import glob
import json
from tqdm import tqdm
from typing import List
from util import config
from util import Vectorizer


class MongoDBUploader:
    def __init__(self, data_dir: str, vectorizer: Vectorizer):
        self._data_dir = data_dir
        self._vectorizer = vectorizer
        self._paths = self.list_qna_paths() 
        
    def list_qna_paths(self) -> List[str]:
        path = os.path.join(config.DATA_QNA_DIR, "*.json")
        return glob.glob(path)

    def load_to_vectors(self) -> List[object]:
        documents = []
        results = []
        for path in self.list_qna_paths():
            tmp = json.load(open(path))
            documents += tmp["data"]

        for doc in tqdm(documents, desc="Converting to vectors..."):
            question = doc["question"]
            answer = doc["answer"]
            try:
                question_vector = self._vectorizer.vectorize(question)
                answer_vector = self._vectorizer.vectorize(answer)
                results.append({
                    "question": {
                        "content": question,
                        "vector": question_vector
                    }, 
                    "answer": {
                        "content": answer,
                        "vector": answer_vector
                    }
                })
            except:
                pass

        return results

if __name__ == "__main__":
    self = MongoDBUploader(
        data_dir=config.DATA_QNA_DIR,
        vectorizer=Vectorizer(
            model_id=config.MODEL_VECTORIZER_ID
        )
    )
    results = self.load_to_vectors()

