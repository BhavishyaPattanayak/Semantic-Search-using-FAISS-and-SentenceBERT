from sklearn.datasets import fetch_20newsgroups

def load_documents(limit=5000):

    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = data.data[:limit]
    return documents
