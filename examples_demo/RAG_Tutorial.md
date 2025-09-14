# RAG (Retrieval-Augmented Generation) Tutorial

A comprehensive guide to building RAG pipelines with Haystack

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [What is RAG?](#what-is-rag)
3. [Setting Up Your Environment](#setting-up-your-environment)
4. [Basic RAG Pipeline](#basic-rag-pipeline)
5. [Advanced RAG Techniques](#advanced-rag-techniques)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Introduction

This tutorial will guide you through building Retrieval-Augmented Generation (RAG) systems using Haystack. RAG combines the power of information retrieval with generative AI to provide accurate, contextual responses based on your own documents.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances language models by:

1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the prompt with retrieved context
3. **Generating** responses based on both the query and context

### Key Benefits:
- ✅ **Accuracy**: Responses grounded in your documents
- ✅ **Up-to-date**: No need to retrain models
- ✅ **Transparency**: Clear source attribution
- ✅ **Domain-specific**: Works with specialized knowledge

## Setting Up Your Environment

### Prerequisites

```bash
# Install Haystack
pip install haystack-ai

# For offline RAG (using BM25)
pip install haystack-ai

# For semantic search RAG (using embeddings)
pip install sentence-transformers

# For OpenAI-powered RAG
pip install haystack-ai
export OPENAI_API_KEY="your-api-key-here"
```

### Verify Installation

```python
import haystack
print(f"Haystack version: {haystack.__version__}")
```

## Basic RAG Pipeline

### Option 1: Offline RAG (No API Required)

This approach uses keyword-based retrieval (BM25) and works completely offline.

```python
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Initialize document store
document_store = InMemoryDocumentStore()

# 1. Create sample documents
documents = [
    Document(content="Haystack is an end-to-end LLM framework for building AI applications."),
    Document(content="RAG combines information retrieval with text generation for accurate responses."),
    Document(content="Vector databases store high-dimensional vectors for similarity search."),
]

# 2. Store documents
document_store.write_documents(documents)

# 3. Create retrieval pipeline
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(
    template="""
    Context: {% for doc in documents %}{{ doc.content }}{% endfor %}
    Question: {{ question }}
    Answer:
    """
)

# 4. Build pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")

# 5. Run query
result = rag_pipeline.run({
    "retriever": {"query": "What is Haystack?", "top_k": 2},
    "prompt_builder": {"question": "What is Haystack?"}
})

print(result["prompt_builder"]["prompt"])
```

### Option 2: OpenAI-Powered RAG

This approach uses OpenAI embeddings and GPT models for semantic search and generation.

```python
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Initialize document store
document_store = InMemoryDocumentStore()

# 1. Create indexing pipeline
doc_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
doc_writer = DocumentWriter(document_store=document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("doc_embedder", doc_embedder)
indexing_pipeline.add_component("doc_writer", doc_writer)
indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

# 2. Index documents
documents = [
    Document(content="Haystack is an end-to-end LLM framework for building AI applications."),
    Document(content="RAG combines information retrieval with text generation for accurate responses."),
    Document(content="Vector databases store high-dimensional vectors for similarity search."),
]

indexing_pipeline.run({"doc_embedder": {"documents": documents}})

# 3. Create RAG pipeline
text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(
    template="""
    Context:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}

    Question: {{ question }}
    Answer:
    """
)
llm = OpenAIGenerator(model="gpt-4o-mini")

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Connect components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# 4. Ask question
result = rag_pipeline.run({
    "text_embedder": {"text": "What is Haystack?"},
    "retriever": {"top_k": 2},
    "prompt_builder": {"question": "What is Haystack?"}
})

print(result["llm"]["replies"][0])
```

## Advanced RAG Techniques

### 1. Hybrid Search (Combining BM25 + Semantic Search)

```python
from haystack.components.rankers import TransformersSimilarityRanker

# Add both retrievers to pipeline
bm25_retriever = InMemoryBM25Retriever(document_store=document_store)
embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store)

# Use a ranker to combine results
ranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

pipeline = Pipeline()
pipeline.add_component("bm25_retriever", bm25_retriever)
pipeline.add_component("embedding_retriever", embedding_retriever)
pipeline.add_component("ranker", ranker)
# ... connect and configure
```

### 2. Document Preprocessing

```python
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter

# Convert PDFs
pdf_converter = PyPDFToDocument()
documents = pdf_converter.run(sources=["document.pdf"])

# Split long documents
splitter = DocumentSplitter(split_by="passage", split_length=3)
split_docs = splitter.run(documents=documents["documents"])
```

### 3. Query Expansion

```python
# Expand queries for better retrieval
expanded_template = """
Original question: {{ question }}
Expand this question with related terms and synonyms.
Expanded question:
"""

query_expander = OpenAIGenerator(model="gpt-4o-mini")
# Use expanded query for retrieval
```

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple documents at once
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    indexing_pipeline.run({"doc_embedder": {"documents": batch}})
```

### 2. Caching

```python
from haystack.components.caching import CacheChecker

# Add caching to expensive operations
cache_checker = CacheChecker()
pipeline.add_component("cache_checker", cache_checker)
```

### 3. Document Store Optimization

```python
# Use appropriate document stores for production
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.chroma import ChromaDocumentStore
from haystack.document_stores.pinecone import PineconeDocumentStore

# Example with Elasticsearch
document_store = ElasticsearchDocumentStore(
    hosts=["http://localhost:9200"],
    index="haystack_documents"
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. No Results Retrieved

```python
# Check document count
docs = document_store.filter_documents()
print(f"Document count: {len(docs)}")

# Test retrieval directly
retriever = InMemoryBM25Retriever(document_store=document_store)
result = retriever.run(query="your query", top_k=5)
print(f"Retrieved: {len(result['documents'])} documents")
```

#### 2. Poor Retrieval Quality

```python
# Try different search parameters
result = retriever.run(query="your query", top_k=10, scale_score=True)

# Examine document scores
for doc in result["documents"]:
    print(f"Score: {doc.score:.3f} - {doc.content[:100]}...")
```

#### 3. API Rate Limits

```python
# Add rate limiting
import time

def rate_limited_pipeline_run(*args, **kwargs):
    time.sleep(0.1)  # Wait 100ms between requests
    return pipeline.run(*args, **kwargs)
```

## Best Practices

### 1. Document Preparation

- **Chunk documents appropriately**: 200-800 words per chunk
- **Add metadata**: Include source, date, author information
- **Clean text**: Remove formatting artifacts, normalize whitespace
- **Use meaningful IDs**: Make documents traceable

### 2. Retrieval Optimization

- **Tune top_k parameter**: Start with 3-5, adjust based on results
- **Experiment with similarity thresholds**: Filter low-quality matches
- **Consider multiple retrieval strategies**: BM25 for keywords, embeddings for concepts

### 3. Prompt Engineering

```python
template = """
Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{% for document in documents %}
Source: {{ document.meta.source }}
Content: {{ document.content }}
---
{% endfor %}

Question: {{ question }}

Answer:
"""
```

### 4. Evaluation

```python
from haystack.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator

# Evaluate response quality
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = AnswerRelevancyEvaluator()

# Run evaluation
faithfulness_result = faithfulness_evaluator.run(
    questions=[question],
    contexts=[retrieved_docs],
    predicted_answers=[generated_answer]
)
```

### 5. Production Considerations

- **Monitor performance**: Track retrieval quality and response time
- **Use persistent document stores**: Elasticsearch, Chroma, Pinecone
- **Implement proper error handling**: Graceful fallbacks
- **Add logging**: Track queries and results for analysis
- **Security**: Sanitize inputs, manage API keys properly

## Example: Complete RAG Application

Here's a complete example that demonstrates all concepts:

```python
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

class ProductionRAG:
    def __init__(self):
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.rag_pipeline = None
        self._setup_pipelines()
    
    def _setup_pipelines(self):
        # Indexing pipeline
        doc_embedder = OpenAIDocumentEmbedder()
        doc_writer = DocumentWriter(document_store=self.document_store)
        
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("doc_embedder", doc_embedder)
        self.indexing_pipeline.add_component("doc_writer", doc_writer)
        self.indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")
        
        # RAG pipeline
        text_embedder = OpenAITextEmbedder()
        retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        prompt_builder = PromptBuilder(
            template="""
            Answer the question based on the context below. Be concise and accurate.
            
            Context:
            {% for doc in documents %}
            {{ doc.content }}
            {% endfor %}
            
            Question: {{ question }}
            Answer:
            """
        )
        llm = OpenAIGenerator(model="gpt-4o-mini")
        
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("text_embedder", text_embedder)
        self.rag_pipeline.add_component("retriever", retriever)
        self.rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.rag_pipeline.add_component("llm", llm)
        
        # Connect components
        self.rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    
    def add_documents(self, documents):
        """Add documents to the knowledge base"""
        return self.indexing_pipeline.run({"doc_embedder": {"documents": documents}})
    
    def query(self, question, top_k=3):
        """Query the RAG system"""
        result = self.rag_pipeline.run({
            "text_embedder": {"text": question},
            "retriever": {"top_k": top_k},
            "prompt_builder": {"question": question}
        })
        
        return {
            "answer": result["llm"]["replies"][0],
            "sources": result["retriever"]["documents"]
        }

# Usage
rag = ProductionRAG()

# Add documents
documents = [
    Document(content="Python is a popular programming language for AI applications."),
    Document(content="Machine learning models can be deployed using various frameworks."),
]
rag.add_documents(documents)

# Ask questions
response = rag.query("What is Python used for?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])} documents")
```

This tutorial covers the essentials of building RAG systems with Haystack. Start with the basic examples and gradually incorporate advanced techniques as your use case demands.

For more information, visit the [Haystack documentation](https://docs.haystack.deepset.ai/).