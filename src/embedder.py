from sentence_transformers import SentenceTransformer

def get_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_documents(model, documents):
    return model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
