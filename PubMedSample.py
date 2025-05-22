from Bio import Entrez
from datetime import datetime
from typing import List
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

Entrez.email = 'anoshandrews@gmail.com'

def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    handle = Entrez.esearch(db = 'pubmed', term = query, retmax = max_results)
    record = Entrez.read(handle)
    return record['IdList']

def fetch_abstracts(ids: List[str]) -> List[Document]:
    handle = Entrez.efetch(db = 'pubmed', id = ','.join(ids), rettype = 'abstract', retmode = 'text')
    raw_text = handle.read()
    abstracts = raw_text.strip().split('\n\n')

    documents = []
    for abstract in abstracts:
        if abstract.strip():
            documents.append(
                Document(
                    page_content = abstract, 
                    metadata = {
                        "source" : 'PubMed',
                        "fetched_at" : str(datetime.utcnow()),
                        "uuid" : str(uuid.uuid4())
                    }
                )
            )
    return documents


def embed_and_store(docs : List[Document], persist_path : str = 'vectorstore'):
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embedding = embeddings)
    vectorstore.save_local(persist_path)
    print(f"Saved {len(docs)} docs to vector DB at: {persist_path}")

if __name__ == "__main__":
    query = 'brest cancer diagnosis AI'
    ids = search_pubmed(query, max_results = 10)
    document = fetch_abstracts(ids)
    embed_and_store(document)