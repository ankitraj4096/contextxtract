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
from openai import OpenAI

from dotenv import load_dotenv

# ---------- FASTAPI + SECURITY SECTION ------------
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

#### ---- AUTH SETTINGS ---- ####
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
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]


@dataclass
class Document:
    content: str
    page_num: int
    start_idx: int
    end_idx: int

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
    def __init__(self, openai_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.faiss_index = None

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

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.faiss_index is None:
            raise Exception("FAISS index not built.")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding.astype("float32"), top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.documents):
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

    def clean_response(self, response_text: str) -> str:
        """Clean the response by removing <think> tags and formatting properly"""
        # Remove <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Clean up any remaining artifacts
        cleaned = cleaned.strip()
        
        # If response is empty after cleaning, return a default message
        if not cleaned:
            return "The information requested could not be found in the provided context."
        
        return cleaned

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        if not context_docs:
            return "No relevant context found to answer the question."
            
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"Context {doc['rank']}: {doc['context']}")
        combined_context = "\n\n".join(context_parts)
        
        # Improved prompt to get direct, concise answers
        prompt = f"""You are an insurance policy expert. Based on the provided context, answer the question directly and concisely.

Context:
{combined_context}

Question: {query}

Instructions:
- Provide a direct, factual answer
- Be specific with numbers, periods, and conditions when mentioned
- Keep the answer concise but complete
- If specific details aren't in the context, state that clearly
- Do not include reasoning or thought processes in your response"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o-mini for good performance and cost efficiency
                messages=[
                    { 
                        "role": "system", 
                        "content": "You are a helpful insurance policy assistant. Provide direct, concise answers based only on the provided context. Do not show your reasoning process." 
                    },
                    { "role": "user", "content": prompt },
                ],
                temperature=0.0,  # Set to 0 for more consistent outputs
                max_tokens=250,   # Reduced for more concise answers
            )
            
            raw_response = response.choices[0].message.content
            # Clean the response to remove thinking tags
            cleaned_response = self.clean_response(raw_response)
            return cleaned_response
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Sorry, I couldn't generate an answer due to an API error: {str(e)}"

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            similar_docs = self.search_similar(question, top_k)
            if not similar_docs:
                return {
                    "answer": "No relevant documents found for this question.",
                    "context": [],
                    "query": question,
                }
            answer = self.generate_answer(question, similar_docs)
            return {"answer": answer, "context": similar_docs, "query": question}
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "context": [],
                "query": question,
            }


@app.post('/hackrx/run', response_model=HackRxResponse)
async def run_hackrx(payload: HackRxRequest, token: str = Depends(verify_token)):
    """
    Receives a PDF url and a list of questions.
    For each question, it extracts the document, runs RAG, and produces the answers.
    Returns: {"answers": [ ... ]}
    """
    try:
        print(f"Received request with PDF URL: {payload.documents}")
        print(f"Number of questions: {len(payload.questions)}")
        
        # Download the PDF file to a temp file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_url = str(payload.documents)
        
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        if not response.ok:
            raise HTTPException(status_code=400, detail=f"PDF could not be downloaded. Status: {response.status_code}")
        
        temp_pdf.write(response.content)
        temp_pdf.close()
        pdf_path = temp_pdf.name
        
        print(f"PDF downloaded to: {pdf_path}")

        # Build the RAG pipeline for this PDF
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables")
            
        rag = RAGPipeline(OPENAI_API_KEY)
        print("Processing PDF...")
        rag.process_pdf(pdf_path, chunk_size=50, overlap=5)
        print(f"PDF processed. Created {len(rag.documents)} document chunks.")

        answers = []
        for i, q in enumerate(payload.questions):
            print(f"Processing question {i+1}: {q}")
            result = rag.query(q)
            answers.append(result["answer"])
            print(f"Answer {i+1}: {result['answer'][:100]}...")

        # Clean up
        os.remove(pdf_path)
        print("Temporary PDF file cleaned up.")

        return HackRxResponse(answers=answers)
        
    except Exception as ex:
        print(f"Error in run_hackrx: {str(ex)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "HackRx RAG API is running"}
