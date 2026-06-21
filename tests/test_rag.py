"""Tests for the RAG pipeline implementation."""

import pytest
from src.rag import DocumentStore, RAGPipeline


SAMPLE_DOCS = [
    "Python is a high-level programming language known for its readability.",
    "Machine learning is a subset of artificial intelligence.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Vector databases are used to store and search high-dimensional embeddings.",
    "Large language models can generate human-like text.",
]


# ---------------------------------------------------------------------------
# DocumentStore tests
# ---------------------------------------------------------------------------

class TestDocumentStore:
    def test_add_and_retrieve_returns_relevant_doc(self):
        store = DocumentStore()
        store.add_documents(SAMPLE_DOCS)
        results = store.retrieve("What is Python?")
        assert any("Python" in doc for doc in results)

    def test_retrieve_top_k_limits_results(self):
        store = DocumentStore()
        store.add_documents(SAMPLE_DOCS)
        results = store.retrieve("language model generation", top_k=2)
        assert len(results) <= 2

    def test_retrieve_empty_store_returns_empty_list(self):
        store = DocumentStore()
        results = store.retrieve("anything")
        assert results == []

    def test_add_documents_empty_list_is_safe(self):
        store = DocumentStore()
        store.add_documents([])
        assert store.documents == []

    def test_retrieve_no_match_returns_empty_list(self):
        store = DocumentStore()
        store.add_documents(["cats are fluffy animals"])
        results = store.retrieve("quantum physics thermodynamics")
        # Scores of 0 are filtered out
        assert results == []


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def setup_method(self):
        store = DocumentStore()
        store.add_documents(SAMPLE_DOCS)
        self.pipeline = RAGPipeline(store)

    def test_query_returns_expected_keys(self):
        result = self.pipeline.query("What is RAG?")
        assert "question" in result
        assert "context" in result
        assert "answer" in result

    def test_query_preserves_question(self):
        question = "What is machine learning?"
        result = self.pipeline.query(question)
        assert result["question"] == question

    def test_query_context_is_list(self):
        result = self.pipeline.query("Tell me about embeddings")
        assert isinstance(result["context"], list)

    def test_query_answer_references_context(self):
        result = self.pipeline.query("What is RAG?")
        assert result["answer"].startswith("Based on the retrieved context:")

    def test_query_empty_store_returns_no_info_message(self):
        empty_pipeline = RAGPipeline(DocumentStore())
        result = empty_pipeline.query("anything")
        assert result["answer"] == "No relevant information found."

    def test_query_top_k_respected(self):
        result = self.pipeline.query("language model generation text", top_k=1)
        assert len(result["context"]) <= 1
