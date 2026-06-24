# Rag1

A lightweight **Retrieval-Augmented Generation (RAG)** test project built with Python.

## Project structure

```
Rag1/
├── src/
│   └── rag.py          # DocumentStore + RAGPipeline implementation
├── tests/
│   └── test_rag.py     # pytest test suite
├── requirements.txt
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Components

| Class | Description |
|---|---|
| `DocumentStore` | Indexes text documents with TF-IDF and retrieves the top-k most relevant chunks for a query |
| `RAGPipeline` | Wraps a `DocumentStore` and exposes a `query()` method that returns retrieved context and a generated answer |
