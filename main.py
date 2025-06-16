from src.data_loader import load_documents
from src.embedder import get_model, embed_documents
from src.indexer import build_faiss_index, save_index
from src.search import semantic_search

import numpy as np
import pickle

def main():
    print("Loading documents...")
    documents = load_documents()

    print("Loading SBERT model...")
    model = get_model()

    print("Encoding documents...")
    embeddings = embed_documents(model, documents)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving model artifacts...")
    np.save("embeddings.npy", embeddings)
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    save_index(index, "semantic.index")

    print("\nReady! Type a query or 'exit' to quit:")
    while True:
        query = input("\nQuery: ")
        if query.lower() in ["exit", "quit"]:
            break
        results = semantic_search(query, model, index, documents)
        for i, res in enumerate(results):
            print(f"\n🔹 Result:\n{res[:500]}...\n")

if __name__ == "__main__":
    main()
