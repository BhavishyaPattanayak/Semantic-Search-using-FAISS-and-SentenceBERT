import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index(index, path="semantic.index"):
    faiss.write_index(index, path)

def load_index(path="semantic.index"):
    return faiss.read_index(path)
