# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple RAG (Retrieval-Augmented Generation) Pipeline Demo - OpenAI Version

This demo shows how to create a basic RAG system using Haystack components with OpenAI:
1. Document embedding and storage using OpenAI embeddings
2. Question answering with context retrieval
3. LLM-powered response generation

Requirements:
- OPENAI_API_KEY environment variable must be set
- pip install haystack-ai
"""

import os
from pathlib import Path
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class SimpleRAGDemo:
    """
    Simple RAG Pipeline demonstration class using OpenAI
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the RAG system

        Args:
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.rag_pipeline = None

        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY environment variable is not set!")
            print("   Please set it to use this demo:")
            print("   export OPENAI_API_KEY='your-api-key-here'")

    def setup_indexing_pipeline(self) -> Pipeline:
        """
        Create a pipeline for indexing documents

        Returns:
            Pipeline for document indexing
        """
        # Document embedder - converts documents to vectors using OpenAI
        doc_embedder = OpenAIDocumentEmbedder(model=self.embedding_model)

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
        # Text embedder - converts questions to vectors using OpenAI
        text_embedder = OpenAITextEmbedder(model=self.embedding_model)

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

        # LLM Generator
        llm = OpenAIGenerator(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant. Answer questions based on the provided context."
        )

        # Create RAG pipeline
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)

        # Connect components
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
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

        try:
            result = self.indexing_pipeline.run({"doc_embedder": {"documents": documents}})
            print(f"✅ Added {len(documents)} documents to knowledge base")
            return result
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            print("   Make sure OPENAI_API_KEY is set correctly.")
            return None

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

        try:
            result = self.rag_pipeline.run({
                "text_embedder": {"text": question},
                "retriever": {"top_k": top_k},
                "prompt_builder": {"question": question}
            })

            return {
                "question": question,
                "answer": result["llm"]["replies"][0] if result.get("llm") else "No answer generated",
                "retrieved_documents": result["retriever"]["documents"],
                "prompt": result["prompt_builder"]["prompt"]
            }
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Error: {e}",
                "retrieved_documents": [],
                "prompt": ""
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
        Document(content="Machine learning algorithms can learn patterns from data and make predictions or decisions without being explicitly programmed."),
        Document(content="OpenAI provides powerful APIs for text generation, embeddings, and other AI capabilities that can be integrated into applications."),
        Document(content="Document stores are databases optimized for storing and retrieving documents, often used in search and RAG applications.")
    ]

    return documents


def print_pipeline_info(pipeline: Pipeline, name: str):
    """
    Print information about a pipeline
    """
    if pipeline:
        print(f"\n🔍 {name} Pipeline Structure:")
        print(f"   Nodes: {list(pipeline.graph.nodes())}")
        print(f"   Edges: {list(pipeline.graph.edges())}")


def main():
    """
    Main demonstration function
    """
    print("🚀 Simple RAG Pipeline Demo - OpenAI Version")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Initialize RAG system
    rag_demo = SimpleRAGDemo()

    # Create and add sample documents
    print("\n📚 Adding sample documents to knowledge base...")
    sample_docs = create_sample_documents()
    result = rag_demo.add_documents(sample_docs)

    if not result:
        print("❌ Failed to add documents. Exiting.")
        return

    print(f"\n📊 Document store contains {len(rag_demo.document_store.filter_documents())} documents")

    # Setup RAG pipeline
    print("\n🔧 Setting up RAG pipeline...")
    rag_demo.setup_rag_pipeline()
    print("✅ RAG pipeline ready!")

    # Show pipeline structures
    print_pipeline_info(rag_demo.indexing_pipeline, "Indexing")
    print_pipeline_info(rag_demo.rag_pipeline, "RAG")

    # Test questions
    questions = [
        "What is Haystack?",
        "How does RAG work?",
        "What are vector databases used for?",
        "Tell me about Python programming language",
        "What can OpenAI APIs do?"
    ]

    print("\n❓ Testing RAG pipeline with sample questions:")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")

        result = rag_demo.ask_question(question, top_k=2)

        print(f"   Answer: {result['answer']}")
        print(f"   Retrieved documents: {len(result['retrieved_documents'])}")

        # Show retrieved document previews
        for j, doc in enumerate(result['retrieved_documents']):
            preview = doc.content[:80] + "..." if len(doc.content) > 80 else doc.content
            score = getattr(doc, 'score', 'N/A')
            print(f"     - Doc {j+1} (score: {score:.3f}): {preview}")

    print("\n🎉 Demo completed successfully!")
    print("\n💡 Tips:")
    print("   - Modify the documents in create_sample_documents() to test with your own content")
    print("   - Adjust top_k parameter in ask_question() to retrieve more/fewer documents")
    print("   - Try different OpenAI models by changing the model parameters")


if __name__ == "__main__":
    main()
