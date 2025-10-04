from RAG import NasaRAG
import os
from pathlib import Path
import pickle
import faiss

directory = 'data'
ROOT = Path(__file__).resolve().parent.parent
print('INIT FOR RAG')
rag = NasaRAG(documents_csv='Nasa_Data_scraped_data_eda.csv',directory='data')
# go to project root

FAISS_PATH = ROOT / directory / 'rag_index.faiss'
print('RAG MADE')
faiss.write_index(rag.index,str(FAISS_PATH))
print('FAISS WRITTEN')

PICKLE_PATH = ROOT / directory / 'rag_docs.pkl'
with open(PICKLE_PATH,'wb') as f:
    pickle.dump(rag.docs, f)

print('SAVED')


