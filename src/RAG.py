import numpy as np
from langchain_core.utils import print_text
from sentence_transformers import SentenceTransformer
import faiss
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import valid_index
from transformers import AutoTokenizer, AutoModelForCausalLM
from textwrap import dedent
import pandas as pd
import torch
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from RAG_DB import RAGDataBase
from pathlib import Path
from google import genai
import pickle



class NasaRAG:

    def __init__(self, model_id="LiquidAI/LFM2-2.6B", documents_csv=None, directory=None):

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()

        # Set defaults
        if documents_csv is None:
            documents_csv = 'full_housing_eda.csv'
        if directory is None:
            directory = 'housing_bank_data'

        ROOT = Path(__file__).resolve().parent.parent
        CSV_PATH = ROOT / directory / documents_csv

        # Filenames for cached data
        FAISS_PATH = ROOT / directory /"rag_index.faiss"
        DOCS_PATH = ROOT / directory /"rag_docs.pkl"
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

        print("🔍 Checking for precomputed data...")
        if FAISS_PATH.exists() and DOCS_PATH.exists():
            print("✅ Loading precomputed FAISS index and docs...")
            # Load FAISS + docs directly
            self.index = faiss.read_index(str(FAISS_PATH))
            with open(DOCS_PATH, "rb") as f:
                self.docs = pickle.load(f)
        else:
            print("⚙️ Precomputed data not found. Building RAG from scratch...")

            # Embedder

            # Chunker
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=['\n\n', '\n', ' ', '']
            )

            # Load CSV
            df = pd.read_csv(CSV_PATH)
            texts = [str(t) for t in df['Content'].to_list()]
            big_text = '\n\n'.join(texts)
            self.docs = self.splitter.split_text(big_text)

            # BM25
            tokenized_docs = [d.split() for d in self.docs]
            self.bm25 = BM25Okapi(tokenized_docs)

            # FAISS embeddings
            doc_embeddings = self.embed_model.encode(self.docs, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(doc_embeddings.shape[1])
            self.index.add(doc_embeddings)

            # Save cache for future runs
            print("💾 Saving FAISS index and docs for next run...")
            faiss.write_index(self.index, str(FAISS_PATH))
            with open(DOCS_PATH, "wb") as f:
                pickle.dump(self.docs, f)

        # Store document mapping
        self.id2doc = {i: self.docs[i] for i in range(len(self.docs))}

        # BM25 (rebuild fast even if cached)
        tokenized_docs = [d.split() for d in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        # LLM model (lazy-load later)
        self.model_id = model_id
        # self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Database
        self.db = RAGDataBase()

        torch.cuda.empty_cache()

    # Search if Question is cashed in Database
    def embed_query(self, query):
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        return q_emb

    def retrieve_images(self,text):
        ids = re.findall(r"<\[(.*?)\]>",text)

        locations = []
        for img_id in ids:
            img_id = int(img_id)
            doc = self.db.images.find_one({'id':img_id})
            if doc and 'location' in doc:
                locations.append(doc['location'])

        text = re.sub(r"<\[(.*?)\]>","",text)
        print(text)
        return text, locations

    # Retrieval - Hybrid
    def retrieve(self, query, top_k=3, alpha=0.5, threshold = 0.2):

        q_emb = self.embed_query(query)

        # FAISS
        D, I = self.index.search(q_emb, top_k)
        faiss_scores = np.zeros(len(self.docs))

        for idx, score in zip(I[0], D[0]):
            faiss_scores[idx] = 1 + (score + 1e-9)

        # BM25
        bm25_scores = self.bm25.get_scores(query.split())

        # normalize
        faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        hybrid_scores = alpha * faiss_scores + (1 - alpha) * bm25_scores

        # collect valid scores (index, score)
        valid_score = [(i, hybrid_scores[i]) for i in range(len(hybrid_scores)) if hybrid_scores[i] >= threshold]
        if not valid_score:
            print("nothing is valid")
            return "", []

        # sort by score descending
        valid_score = sorted(valid_score, key=lambda x: x[1], reverse=True)

        # take top_k document indices (extract the int index, not tuple index!)
        top_indices = [idx for idx, _ in valid_score[:top_k]]

        # now these are plain ints, so this works:
        top_texts = [self.docs[i] for i in top_indices]

        one_text = "\n".join(top_texts)
        text, locations = self.retrieve_images(one_text)
        print(text)
        return text, locations

    # Augmentation
    def augment(self, data_row):
        # Prompt Engineering for answers
        prompt = dedent(f"""
        You are a helpful assistant.
        Only use the information provided below to answer the question,
        If the information does not contain the answer, reply strictly with "I don't Know :-( ", no more, no less, and don't add anything else to the output
        Question: {data_row['question']}

        Information:

        ```
        {data_row['context']}
        ```
        """)
        messages = [
            {"role": "system", "content": "Use only the information to answer the question"},
            {"role": "user", "content": prompt},
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # LLM generation
    def generate(self, prompt, max_new_tokens=512,model: str = "gemini-2.5-flash"):

        api_key = 'AIzaSyC-kQsAv8brM5GF6vOJLTL_1nKb3RbNYYU'

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return resp.text


    # Question Answering - Calls the other methods of RAG
    def search_question(self, query, top_k=3, max_new_tokens=1024, alpha=0.5, use_db=True, threshold= 0.5):

        q_emb = self.embed_query(query)
        question, answer, images = self.db.search_question(q_emb)
        if question is not None and use_db:
            return answer, images

        context, images = self.retrieve(query, top_k, alpha,threshold=threshold)
        data_row = {
            'question': query,
            'context': context
        }

        prompt = self.augment(data_row)

        answer = self.generate(prompt, max_new_tokens)
        if use_db:
            self.db.insert_question(query, answer, q_emb, images)

        return answer, images