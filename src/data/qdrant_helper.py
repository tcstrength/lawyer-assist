from itertools import islice
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path
import requests
from qdrant_client import models, QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import Batch
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parents[2]
DATA_DIR = PROJECT_DIR / 'data' / 'raw'
CHUDE_JSON_PATH = DATA_DIR / 'ChuDe.json'
DEMUC_JSON_PATH = DATA_DIR / 'DeMuc.json'
ALL_JSON_PATH = DATA_DIR / 'All.json'
DEMUC_DIR = DATA_DIR / 'demuc'

def load_documents():
    with open(DEMUC_JSON_PATH) as f:
        demuc_json = json.load(f)
    with open(CHUDE_JSON_PATH) as f:
        chude_json = json.load(f)
    
    def find_chude(id):
        for chude in chude_json:
            if chude['Value'] == id:
                return chude['Text']
    
    def parse_file(html_file):
        with open(html_file) as f:
            soup = BeautifulSoup(f, 'html.parser')
        return soup.get_text()

    data = []
    for demuc in tqdm(demuc_json, desc="Load documents from raw data..."):
        item = {}
        item['id'] = demuc['Value']
        item['de_muc'] = demuc['Text']
        item['chu_de'] = find_chude(demuc['ChuDe'])
        item['content'] = parse_file(DEMUC_DIR / f"{demuc['Value']}.html")
        data.append(item)
    return data


class QdrantHelper:

    def __init__(self, host="localhost", 
                 port=6333, 
                 similarity_metric=models.Distance.COSINE,
                 embedding_dimension=384,
                 collection_name="law_collection",
                 batch_size=10,
                 embedding_model_id="sentence-transformers/all-MiniLM-L6-v2"):
        self._client = QdrantClient(host=host, port=port)
        self.embedding_model_id = embedding_model_id
        self.similarity_metric = similarity_metric
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name

        self._embeding_model = SentenceTransformer(embedding_model_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._embeding_model.to(device)

        self.batch_size = batch_size

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=similarity_metric),
            )

    def _embed(self, text):
        return self._embeding_model.encode(text)

    def _batched(self, iterable):
        """Yield successive n-sized chunks from the iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, self.batch_size))
            if not batch:
                break
            yield batch
    
    def upsert_documents(self, documents):
        # Upsert documents in batches with progress tracking
        for batch in tqdm(self._batched(documents), total=len(documents)//self.batch_size + 1, desc="Upserting documents"):
            ids = [doc["id"] for doc in batch]
            vectors = [self._embed(doc["content"]) for doc in batch]
            payloads = batch
            
            # Create a Batch instance
            points_batch = Batch(ids=ids, vectors=vectors, payloads=payloads)
            
            # Upsert the batch to the collection
            self._client.upsert(collection_name=self.collection_name, points=points_batch)

    def search(self, text, limit=1):
        return self._client.search(
            collection_name=self.collection_name,
            query_vector=self._embed(text),
            limit=limit
        )

if __name__ == "__main__":
    # documents = load_documents()
    helper = QdrantHelper()
    # helper.upsert_documents(documents=documents)
    print(helper.search('Các quy định chung về Công tác dân tộc'))
