import os
import re
import pickle
import tempfile
import requests
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

from dotenv import load_dotenv

# ---------- FASTAPI + SECURITY SECTION ------------
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

#### ---- AUTH SETTINGS (update your API_KEY here if needed) ---- ####
API_KEY = "f5b00b57698f28dbb878319109a149d5cd0a7d430c25cf3c590b30d31cf8b028"

app = FastAPI(
    title="HackRx RAG API",
    description="API to process document and answer insurance queries using RAG pipeline.",
    version="1.0.0"
)

auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

class HackRxRequest(BaseModel):
    documents: HttpUrl  # pdf file URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ------------ RAG PIPELINE SECTION ---------------

@dataclass
class Document:
    content: str
    page_num: int
    start_idx: int
    end_idx: int

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAGPipeline:
    def __init__(self, groq_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.vector_db_path = "faiss_index"  # unused in API version
        self.documents_path = "documents.pkl" # unused in API version

    def read_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        return text

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\'\"\/]", "", text)
        text = re.sub(r"--- Page \d+ ---", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def chunk_text(self, text: str, chunk_size: int = 50, overlap: int = 5) -> List[Document]:
        documents = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) > 0:
                doc = Document(
                    page_num=0,
                    content=chunk_text,
                    start_idx=i,
                    end_idx=min(i + chunk_size, len(words))
                )
                documents.append(doc)
        return documents

    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype("float32"))

    def process_pdf(self, pdf_path: str, chunk_size: int = 50, overlap: int = 5):
        raw_text = self.read_pdf(pdf_path)
        if not raw_text:
            raise Exception("No text could be extracted from the PDF.")
        clean_text = self.clean_text(raw_text)
        self.documents = self.chunk_text(clean_text, chunk_size, overlap)
        self.embeddings = self.create_embeddings(self.documents)
        self.build_faiss_index(self.embeddings)

    def get_context_window(self, text: str, query_words: List[str], context_size: int = 100) -> str:
        words = text.split()
        contexts = []
        for query_word in query_words:
            query_word_lower = query_word.lower()
            for i, word in enumerate(words):
                if query_word_lower in word.lower():
                    start_idx = max(0, i - context_size)
                    end_idx = min(len(words), i + context_size + 1)
                    context = " ".join(words[start_idx:end_idx])
                    contexts.append(context)
                    break
        combined_context = " ... ".join(contexts)
        return combined_context

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.faiss_index is None:
            raise Exception("FAISS index not built.")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding.astype("float32"), top_k)
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                doc = self.documents[idx]
                query_words = query.split()
                context = self.get_context_window(doc.content, query_words, context_size=100)
                results.append({
                    "content": doc.content,
                    "context": context,
                    "score": float(distance),
                    "rank": i + 1,
                    "doc_index": int(idx),
                })
        return results

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"Context {doc['rank']}: {doc['context']}")
        combined_context = "\n\n".join(context_parts)
        prompt = f"""Based on the following context documents, please answer the question. Use the context to provide accurate and relevant information.

Context Documents:
{combined_context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please state that clearly."""
        try:
            response = self.groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    { "role": "system", "content": "Answer these questions the best you can." },
                    { "role": "user", "content": prompt },
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "Sorry, I couldn't generate an answer due to an API error."

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        similar_docs = self.search_similar(question, top_k)
        if not similar_docs:
            return {
                "answer": "No relevant documents found.",
                "context": [],
                "query": question,
            }
        answer = self.generate_answer(question, similar_docs)
        return {"answer": answer, "context": similar_docs, "query": question}


# ---------------- API ENDPOINT -----------------

@app.post('/hackrx/run', response_model=HackRxResponse)
async def run_hackrx(payload: HackRxRequest, token: str = Depends(verify_token)):
    """
    Receives a PDF url and a list of questions.
    For each question, it extracts the document, runs RAG, and produces the answers.
    Returns: {"answers": [ ... ]}
    """
    try:
        # Download the PDF file to a temp file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_url = str(payload.documents)
        response = requests.get(pdf_url)
        if not response.ok:
            raise HTTPException(status_code=400, detail="PDF could not be downloaded from the provided url.")
        temp_pdf.write(response.content)
        temp_pdf.close()
        pdf_path = temp_pdf.name

        # Build the RAG pipeline for this PDF
        rag = RAGPipeline(GROQ_API_KEY)
        rag.process_pdf(pdf_path, chunk_size=40, overlap=5)

        answers = []
        for q in payload.questions:
            result = rag.query(q)
            answers.append(result["answer"])

        # Clean up
        os.remove(pdf_path)

        return HackRxResponse(answers=answers)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


# ---------------- RUN INSTRUCTIONS -------------
# To run locally (from folder with your code): 
# uvicorn main:app --host 0.0.0.0 --port 5001 --reload

# To test in Postman or curl: Supply the endpoint, your Bearer token, the 'documents' field (PDF URL), and a list of 'questions'.
