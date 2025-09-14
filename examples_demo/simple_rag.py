# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple RAG (Retrieval-Augmented Generation) Pipeline Demo

This demo shows how to create a basic RAG system using Haystack components:
1. Document embedding and storage
2. Question answering with context retrieval
3. LLM-powered response generation
"""

import os
from pathlib import Path
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class SimpleRAGDemo:
    """
    Simple RAG Pipeline demonstration class
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG system

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.rag_pipeline = None

    def setup_indexing_pipeline(self) -> Pipeline:
        """
        Create a pipeline for indexing documents

        Returns:
            Pipeline for document indexing
        """
        # Document embedder - converts documents to vectors
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=self.model_name,
            progress_bar=False
        )

        # Document writer - stores documents in the document store
        doc_writer = DocumentWriter(document_store=self.document_store)

        # Create indexing pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("doc_embedder", doc_embedder)
        indexing_pipeline.add_component("doc_writer", doc_writer)

        # Connect components
        indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

        self.indexing_pipeline = indexing_pipeline
        return indexing_pipeline

    def setup_rag_pipeline(self) -> Pipeline:
        """
        Create a pipeline for RAG question answering

        Returns:
            Pipeline for RAG QA
        """
        # Text embedder - converts questions to vectors
        text_embedder = SentenceTransformersTextEmbedder(
            model=self.model_name,
            progress_bar=False
        )

        # Retriever - finds relevant documents
        retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)

        # Prompt template for RAG
        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """

        prompt_builder = PromptBuilder(template=template)

        # LLM Generator - requires OPENAI_API_KEY environment variable
        try:
            llm = OpenAIGenerator(
                model="gpt-4o-mini",
                system_prompt="You are a helpful assistant. Answer questions based on the provided context."
            )
        except Exception as e:
            print(f"Warning: OpenAI Generator not available: {e}")
            print("Please set OPENAI_API_KEY environment variable to use LLM generation.")
            llm = None

        # Create RAG pipeline
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)

        if llm:
            rag_pipeline.add_component("llm", llm)

        # Connect components
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")

        if llm:
            rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

        self.rag_pipeline = rag_pipeline
        return rag_pipeline

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the knowledge base

        Args:
            documents: List of Document objects to add
        """
        if not self.indexing_pipeline:
            self.setup_indexing_pipeline()

        result = self.indexing_pipeline.run({"doc_embedder": {"documents": documents}})
        print(f"✅ Added {len(documents)} documents to knowledge base")
        return result

    def ask_question(self, question: str, top_k: int = 3) -> dict:
        """
        Ask a question and get an answer using RAG

        Args:
            question: The question to ask
            top_k: Number of relevant documents to retrieve

        Returns:
            Dictionary containing the answer and metadata
        """
        if not self.rag_pipeline:
            self.setup_rag_pipeline()

        # Check if we have an LLM component
        has_llm = "llm" in self.rag_pipeline.graph.nodes

        if has_llm:
            result = self.rag_pipeline.run({
                "text_embedder": {"text": question},
                "retriever": {"top_k": top_k},
                "prompt_builder": {"question": question}
            })

            return {
                "question": question,
                "answer": result["llm"]["replies"][0] if result.get("llm") else "No LLM available",
                "retrieved_documents": result["retriever"]["documents"],
                "prompt": result["prompt_builder"]["prompt"]
            }
        else:
            # Run without LLM - just retrieval
            result = self.rag_pipeline.run({
                "text_embedder": {"text": question},
                "retriever": {"top_k": top_k},
                "prompt_builder": {"question": question}
            })

            return {
                "question": question,
                "retrieved_documents": result["retriever"]["documents"],
                "prompt": result["prompt_builder"]["prompt"],
                "answer": "Please set OPENAI_API_KEY to get LLM-generated answers"
            }


def create_sample_documents() -> List[Document]:
    """
    Create sample documents for testing

    Returns:
        List of sample documents
    """
    documents = [
        Document(content="Haystack is an end-to-end LLM framework for building applications powered by LLMs, Transformer models, and vector search."),
        Document(content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide more accurate and contextual responses."),
        Document(content="Vector databases store high-dimensional vectors that represent text, images, or other data types for similarity search."),
        Document(content="Embedding models convert text into numerical vectors that capture semantic meaning and relationships between words."),
        Document(content="LLMs (Large Language Models) like GPT-4 can understand and generate human-like text based on input prompts."),
        Document(content="Python is a popular programming language for AI and machine learning applications, known for its simplicity and extensive libraries."),
        Document(content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language."),
        Document(content="Machine learning algorithms can learn patterns from data and make predictions or decisions without being explicitly programmed.")
    ]

    return documents


def main():
    """
    Main demonstration function
    """
    print("🚀 Simple RAG Pipeline Demo")
    print("=" * 50)

    # Initialize RAG system
    rag_demo = SimpleRAGDemo()

    # Create and add sample documents
    print("\n📚 Adding sample documents to knowledge base...")
    sample_docs = create_sample_documents()
    rag_demo.add_documents(sample_docs)

    print(f"\n📊 Document store contains {len(rag_demo.document_store.filter_documents())} documents")

    # Setup RAG pipeline
    print("\n🔧 Setting up RAG pipeline...")
    rag_demo.setup_rag_pipeline()
    print("✅ RAG pipeline ready!")

    # Test questions
    questions = [
        "What is Haystack?",
        "How does RAG work?",
        "What are vector databases used for?",
        "Tell me about Python programming language"
    ]

    print("\n❓ Testing RAG pipeline with sample questions:")
    print("-" * 50)

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")

        result = rag_demo.ask_question(question, top_k=2)

        print(f"   Answer: {result['answer']}")
        print(f"   Retrieved documents: {len(result['retrieved_documents'])}")

        # Show retrieved document previews
        for j, doc in enumerate(result['retrieved_documents']):
            preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            print(f"     - Doc {j+1}: {preview}")

    print("\n🎉 Demo completed!")

    # Show pipeline visualization
    print("\n🔍 Pipeline Structure:")
    if rag_demo.rag_pipeline:
        print(f"   Nodes: {list(rag_demo.rag_pipeline.graph.nodes())}")
        print(f"   Edges: {list(rag_demo.rag_pipeline.graph.edges())}")


if __name__ == "__main__":
    main()
