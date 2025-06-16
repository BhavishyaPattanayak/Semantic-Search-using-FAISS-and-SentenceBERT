import numpy as np

def semantic_search(query, model, index, documents, top_k=5):

    query_vec = model.encode([query])[0].astype('float32')
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)
    return [documents[i] for i in indices[0]]
