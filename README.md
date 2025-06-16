# Semantic Search using Sentence-BERT and FAISS

## Overview  
This project implements a lightweight semantic search engine using:
- Sentence-BERT (SBERT) to encode text into dense embeddings
- FAISS to perform fast similarity search over large document collections

It allows users to input a query and retrieve the most semantically relevant documents from a pre-indexed dataset.

## Project Structure
- **main.py** – One-time setup script that:
  - Loads and embeds documents
  - Builds and saves the FAISS index
  - Saves processed documents
  - Starts an interactive search session
- **src/data_loader.py** – Loads the 20 Newsgroups dataset
- **src/embedder.py** – Loads SBERT and encodes documents
- **src/indexer.py** – Builds, saves, and loads the FAISS index
- **src/search.py** – Performs semantic search with FAISS

## Files Generated
- **semantic.index** – Saved FAISS index
- **documents.pkl** – Pickled raw document list
- (Optional) **embeddings.npy** – Dense vector representations (not required after index is saved)

## How to Use

1. **Install dependencies**

```bash
pip install sentence-transformers faiss-cpu scikit-learn
```

2. Run for the first time
This encodes documents, builds the FAISS index, and starts the interactive search loop:

```bash
python main.py
```

3. Search
Type your query when prompted. Example:

```bash
 Query: What's a good beginner motorcycle under $1000? 
```
Top semantically similar results will be displayed.

## Dataset Used

20 Newsgroups (via scikit-learn)

**Customizing**

- Replace load_documents() in src/data_loader.py with your own document list or a different dataset.

- SBERT model can be changed in embedder.py by modifying the model name (e.g., all-mpnet-base-v2).

**Why This Works**

- SBERT generates contextual sentence embeddings

- FAISS enables ultra-fast nearest neighbor search

- Together, they replicate a Retrieval-Augmented Generation (RAG-lite) system, entirely offline

