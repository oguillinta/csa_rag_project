# Basic vector store layer using Chroma
import os
import chromadb
from chromadb import PersistentClient
import chromadb.config
from typing import List, Tuple

def get_collection(path: str, name: str):
    os.makedirs(path, exist_ok=True)
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    client = PersistentClient(path=path, settings=settings)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def clear_collection(path: str, name: str):
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    client = PersistentClient(path=path, settings=settings)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def add_batch(collection, documents: List[str], metadatas: List[dict], ids: List[str], embeddings: List[List[float]]):
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

def query_by_embedding(collection, query_emb: List[float], n_results: int = 4):
    return collection.query(query_embeddings=[query_emb], n_results=n_results, include=["documents","metadatas","distances"])
