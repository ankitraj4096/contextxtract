# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_KEY = "f5b00b57698f28dbb878319109a149d5cd0a7d430c25cf3c590b30d31cf8b028"
    API_TITLE = "HackRx RAG API"
    API_DESCRIPTION = "API to process document and answer insurance queries using RAG pipeline."
    API_VERSION = "1.0.0"
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"
    
    # RAG Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 100  # Increased for better context
    CHUNK_OVERLAP = 20  # Increased for better continuity
    TOP_K_RESULTS = 7  # Increased for more context
    CONTEXT_WINDOW_SIZE = 150  # Increased for better context
    
    # Generation Configuration
    TEMPERATURE = 0.0
    MAX_TOKENS = 300  # Increased for more detailed answers
    DOWNLOAD_TIMEOUT = 60  # Increased timeout

settings = Settings()

# ================================================

# models/document.py
from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    content: str
    page_num: int
    start_idx: int
    end_idx: int
    
    def __post_init__(self):
        """Validate document content"""
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")

# ================================================

# models/schemas.py
from pydantic import BaseModel, HttpUrl, validator
from typing import List

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError("Questions list cannot be empty")
        if len(v) > 20:  # Reasonable limit
            raise ValueError("Maximum 20 questions allowed")
        return v

class HackRxResponse(BaseModel):
    answers: List[str]
    
class HealthResponse(BaseModel):
    status: str
    message: str

# ================================================

# utils/pdf_processor.py
import PyPDF2
import re
from typing import List
from models.document import Document

class PDFProcessor:
    def __init__(self):
        pass
    
    def read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} of {total_pages} ---\n"
                            text += page_text
                    except Exception as e:
                        print(f"Error reading page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")
        
        if not text.strip():
            raise Exception("No readable text found in PDF")
            
        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Normalize whitespace
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\'\"\/\%\$]", "", text)
        
        # Remove page markers
        text = re.sub(r"--- Page \d+ of \d+ ---", "", text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return text

    def chunk_text(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[Document]:
        """Split text into overlapping chunks"""
        if not text:
            return []
            
        documents = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_word_count = 0
        start_idx = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            
            # If adding this sentence exceeds chunk size, create a document
            if current_word_count + len(sentence_words) > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():
                    doc = Document(
                        content=chunk_text.strip(),
                        page_num=0,
                        start_idx=start_idx,
                        end_idx=start_idx + current_word_count
                    )
                    documents.append(doc)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
                start_idx += current_word_count - sum(len(s.split()) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_word_count += len(sentence_words)
        
        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():
                doc = Document(
                    content=chunk_text.strip(),
                    page_num=0,
                    start_idx=start_idx,
                    end_idx=start_idx + current_word_count
                )
                documents.append(doc)
        
        return documents

# ================================================

# utils/embeddings.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from models.document import Document

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Create embeddings for documents"""
        if not documents:
            return np.array([])
            
        texts = [doc.content for doc in documents]
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Error creating embeddings: {e}")

    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search"""
        if embeddings.size == 0:
            raise Exception("Cannot build index with empty embeddings")
            
        dimension = embeddings.shape[1]
        # Use IndexFlatIP for cosine similarity with normalized embeddings
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype("float32"))

    def search_similar(self, query: str, documents: List[Document], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.faiss_index is None:
            raise Exception("FAISS index not built")
            
        if not query.strip():
            return []
            
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            scores, indices = self.faiss_index.search(query_embedding.astype("float32"), min(top_k, len(documents)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(documents):
                    doc = documents[idx]
                    results.append({
                        "content": doc.content,
                        "score": float(score),
                        "rank": i + 1,
                        "doc_index": int(idx),
                    })
            
            return results
        except Exception as e:
            raise Exception(f"Error in similarity search: {e}")

# ================================================

# services/llm_service.py
import re
from openai import OpenAI
from typing import List, Dict, Any

class LLMService:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def clean_response(self, response_text: str) -> str:
        """Clean the response by removing unwanted tags and formatting"""
        if not response_text:
            return "The information requested could not be found in the provided context."
            
        # Remove thinking tags and other artifacts
        cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Clean up
        cleaned = cleaned.strip()
        
        if not cleaned:
            return "The information requested could not be found in the provided context."
        
        return cleaned

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]], 
                       temperature: float = 0.0, max_tokens: int = 300) -> str:
        """Generate answer using OpenAI API"""
        if not context_docs:
            return "No relevant context found to answer the question."
            
        # Combine context with better formatting
        context_parts = []
        for doc in context_docs[:5]:  # Limit to top 5 for token efficiency
            context_parts.append(f"[Context {doc['rank']}]: {doc['content']}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Improved prompt for better answers
        prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided context.

CONTEXT:
{combined_context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide a direct, accurate answer based on the context
2. Include specific details like numbers, periods, percentages when available
3. If information is partially available, state what you know and what's unclear
4. If the context doesn't contain relevant information, clearly state this
5. Keep your answer concise but complete
6. Do not make assumptions beyond what's stated in the context

ANSWER:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise insurance policy assistant. Answer only based on provided context. Be direct and factual."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            raw_response = response.choices[0].message.content
            cleaned_response = self.clean_response(raw_response)
            return cleaned_response
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# ================================================

# services/rag_service.py
import tempfile
import requests
import os
from typing import List, Dict, Any
from utils.pdf_processor import PDFProcessor
from utils.embeddings import EmbeddingManager
from services.llm_service import LLMService
from config.settings import settings

class RAGService:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL)
        self.llm_service = LLMService(settings.OPENAI_API_KEY, settings.OPENAI_MODEL)
        self.documents = []
        
    def download_pdf(self, pdf_url: str) -> str:
        """Download PDF from URL to temporary file"""
        try:
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            
            response = requests.get(pdf_url, timeout=settings.DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            temp_pdf.write(response.content)
            temp_pdf.close()
            
            return temp_pdf.name
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download PDF: {e}")
        except Exception as e:
            raise Exception(f"Error downloading PDF: {e}")
    
    def process_document(self, pdf_url: str) -> int:
        """Process PDF document and build search index"""
        pdf_path = None
        try:
            # Download PDF
            pdf_path = self.download_pdf(pdf_url)
            
            # Extract text
            raw_text = self.pdf_processor.read_pdf(pdf_path)
            clean_text = self.pdf_processor.clean_text(raw_text)
            
            # Create document chunks
            self.documents = self.pdf_processor.chunk_text(
                clean_text, 
                settings.CHUNK_SIZE, 
                settings.CHUNK_OVERLAP
            )
            
            if not self.documents:
                raise Exception("No valid document chunks created")
            
            # Create embeddings and build index
            embeddings = self.embedding_manager.create_embeddings(self.documents)
            self.embedding_manager.build_faiss_index(embeddings)
            
            return len(self.documents)
            
        finally:
            # Cleanup temp file
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {pdf_path}: {e}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if not self.documents:
                return {
                    "answer": "No documents have been processed yet.",
                    "query": question,
                    "context_used": 0
                }
            
            # Search for similar documents
            similar_docs = self.embedding_manager.search_similar(
                question, 
                self.documents, 
                settings.TOP_K_RESULTS
            )
            
            if not similar_docs:
                return {
                    "answer": "No relevant documents found for this question.",
                    "query": question,
                    "context_used": 0
                }
            
            # Generate answer
            answer = self.llm_service.generate_answer(
                question, 
                similar_docs,
                settings.TEMPERATURE,
                settings.MAX_TOKENS
            )
            
            return {
                "answer": answer,
                "query": question,
                "context_used": len(similar_docs)
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "query": question,
                "context_used": 0
            }

# ================================================

# auth/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import settings

auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Verify Bearer token"""
    if (not credentials or 
        credentials.scheme != "Bearer" or 
        credentials.credentials != settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# ================================================

# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from models.schemas import HackRxRequest, HackRxResponse, HealthResponse
from services.rag_service import RAGService
from auth.security import verify_token
from config.settings import settings

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

@app.post('/hackrx/run', response_model=HackRxResponse)
async def run_hackrx(payload: HackRxRequest, token: str = Depends(verify_token)):
    """
    Process PDF document and answer questions using RAG pipeline.
    """
    try:
        print(f"Received request with PDF URL: {payload.documents}")
        print(f"Number of questions: {len(payload.questions)}")
        
        # Validate OpenAI API key
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API key not configured"
            )
        
        # Initialize RAG service
        rag_service = RAGService()
        
        # Process document
        print("Processing PDF...")
        num_chunks = rag_service.process_document(str(payload.documents))
        print(f"PDF processed. Created {num_chunks} document chunks.")
        
        # Process questions
        answers = []
        for i, question in enumerate(payload.questions):
            print(f"Processing question {i+1}/{len(payload.questions)}: {question[:50]}...")
            
            result = rag_service.query(question)
            answers.append(result["answer"])
            
            print(f"Answer {i+1}: {result['answer'][:100]}...")
        
        print("All questions processed successfully.")
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in run_hackrx: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        message="HackRx RAG API is running"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
