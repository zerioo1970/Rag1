"""
Simple RAG (Retrieval-Augmented Generation) implementation.

This module provides a lightweight RAG pipeline that retrieves relevant
document chunks using TF-IDF similarity and returns them as context.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentStore:
    """Stores and indexes text documents for retrieval."""

    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self._matrix = None

    def add_documents(self, docs: list[str]) -> None:
        """Add a list of text documents to the store."""
        if not docs:
            return
        self.documents.extend(docs)
        self._matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve the top_k most relevant documents for the given query."""
        if not self.documents:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class RAGPipeline:
    """A simple RAG pipeline that combines retrieval with a generation step."""

    def __init__(self, document_store: DocumentStore):
        self.store = document_store

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Retrieve relevant context for a question and return a response dict.

        Returns a dict with:
          - 'question': the original question
          - 'context': list of retrieved document chunks
          - 'answer': a synthesized answer string
        """
        context = self.store.retrieve(question, top_k=top_k)
        answer = self._generate(question, context)
        return {
            "question": question,
            "context": context,
            "answer": answer,
        }

    def _generate(self, question: str, context: list[str]) -> str:
        """Generate an answer from the retrieved context."""
        if not context:
            return "No relevant information found."
        combined = " ".join(context)
        return f"Based on the retrieved context: {combined}"
