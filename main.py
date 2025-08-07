import os
import re
import pickle
from typing import List, Dict, Any
from pathlib import Path

import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Document:
    """Document class to store text content and metadata"""

    content: str
    page_num: int
    start_idx: int
    end_idx: int


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RAGPipeline:
    def __init__(self, groq_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG Pipeline

        Args:
            groq_api_key: Your Groq API key
            embedding_model: Sentence transformer model name
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.vector_db_path = "faiss_index"
        self.documents_path = "documents.pkl"

    def read_pdf(self, pdf_path: str) -> str:
        """
        Read PDF file and extract text

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text from PDF
        """
        print(f"Reading PDF: {pdf_path}")
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
        """
        Clean and preprocess text

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        print("Cleaning text...")

        # Remove extra whitespace and normalize line breaks
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\'\"\/]", "", text)

        # Remove page headers/footers (basic pattern)
        text = re.sub(r"--- Page \d+ ---", "", text)

        # Remove excessive spacing
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()
        print(text)

        return text

    def chunk_text(
        self, text: str, chunk_size: int = 50, overlap: int = 5
    ) -> List[Document]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of Document objects
        """
        print(
            f"Chunking text into pieces of {chunk_size} characters with {overlap} overlap..."
        )

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
                    end_idx=min(i + chunk_size, len(words)),
                )
                documents.append(doc)

        return documents

    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for documents

        Args:
            documents: List of Document objects

        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(documents)} documents...")

        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Numpy array of embeddings
        """
        print("Building FAISS index...")

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype("float32"))

        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def save_index_and_documents(self):
        """Save FAISS index and documents to disk"""
        print("Saving FAISS index and documents...")

        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{self.vector_db_path}.index")

        # Save documents
        with open(self.documents_path, "wb") as f:
            pickle.dump(self.documents, f)

        print("Index and documents saved successfully!")

    def load_index_and_documents(self):
        """Load FAISS index and documents from disk"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{self.vector_db_path}.index")

            # Load documents
            with open(self.documents_path, "rb") as f:
                self.documents = pickle.load(f)

            print("Index and documents loaded successfully!")
            return True
        except Exception as e:
            print(f"Could not load existing index: {e}")
            return False

    def process_pdf(self, pdf_path: str, chunk_size: int = 50, overlap: int = 5):
        """
        Complete pipeline to process PDF and build vector database

        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        # Read PDF
        raw_text = self.read_pdf(pdf_path)

        if not raw_text:
            print("No text extracted from PDF")
            return

        # Clean text
        clean_text = self.clean_text(raw_text)

        # Chunk text
        self.documents = self.chunk_text(clean_text, chunk_size, overlap)

        # Create embeddings
        self.embeddings = self.create_embeddings(self.documents)

        # Build FAISS index
        self.build_faiss_index(self.embeddings)

        # Save everything
        self.save_index_and_documents()

        print("PDF processing complete!")

    def get_context_window(
        self, text: str, query_words: List[str], context_size: int = 100
    ) -> str:
        """
        Extract context around matching query words

        Args:
            text: Full text to search in
            query_words: Words to search for
            context_size: Number of words before and after match

        Returns:
            Context window around matches
        """
        words = text.split()
        contexts = []

        for query_word in query_words:
            query_word_lower = query_word.lower()

            for i, word in enumerate(words):
                if query_word_lower in word.lower():
                    # Get context window
                    start_idx = max(0, i - context_size)
                    end_idx = min(len(words), i + context_size + 1)

                    context = " ".join(words[start_idx:end_idx])
                    contexts.append(context)

                    # Only get first match per query word to avoid too much repetition
                    break

        # Combine contexts and remove duplicates
        combined_context = " ... ".join(contexts)
        return combined_context

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of similar documents with scores
        """
        if self.faiss_index is None:
            print("No FAISS index found. Please process a PDF first.")
            return []

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])

        # Search in FAISS
        distances, indices = self.faiss_index.search(
            query_embedding.astype("float32"), top_k
        )

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                doc = self.documents[idx]

                # Get context window around query words
                query_words = query.split()
                context = self.get_context_window(
                    doc.content, query_words, context_size=100
                )

                results.append(
                    {
                        "content": doc.content,
                        "context": context,
                        "score": float(distance),
                        "rank": i + 1,
                        "doc_index": int(idx),
                    }
                )

        return results

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using Groq with DeepSeek model

        Args:
            query: User query
            context_docs: Retrieved context documents

        Returns:
            Generated answer
        """
        # Prepare context from retrieved documents
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"Context {doc['rank']}: {doc['context']}")

        combined_context = "\n\n".join(context_parts)

        # Create prompt
        prompt = f"""Based on the following context documents, please answer the question. Use the context to provide accurate and relevant information.

Context Documents:
{combined_context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please state that clearly."""

        try:
            # Call Groq API with DeepSeek model
            response = self.groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer these questions the best you can.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "Sorry, I couldn't generate an answer due to an API error."

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Complete RAG query pipeline

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer and context
        """
        print(f"Processing query: {question}")

        # Search for similar documents
        similar_docs = self.search_similar(question, top_k)

        if not similar_docs:
            return {
                "answer": "No relevant documents found.",
                "context": [],
                "query": question,
            }
        answer = self.generate_answer(question, similar_docs)

        return {"answer": answer, "context": similar_docs, "query": question}


def main():
    groq_api_key = GROQ_API_KEY
    rag = RAGPipeline(groq_api_key)

    # Try to load existing index
    if not rag.load_index_and_documents():
        # If no existing index, process PDF
        pdf_path = "document1.pdf"
        rag.process_pdf(pdf_path, chunk_size=40, overlap=5)

    # Example queries
    queries = [
        "Does this policy cover knee surgery, and what are the conditions?",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        result = rag.query(query)
        print(f"Query: {result['query']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nContext documents used: {len(result['context'])}")
        for i, ctx in enumerate(result["context"]):
            print(f"  {i + 1}. Score: {ctx['score']:.4f}")


if __name__ == "__main__":
    main()
