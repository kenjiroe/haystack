# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple RAG (Retrieval-Augmented Generation) Pipeline Demo - Offline Version

This demo shows how to create a basic RAG system using Haystack components without external APIs:
1. Document embedding and storage using simple keyword matching (BM25)
2. Question answering with context retrieval
3. Template-based response generation (no LLM required)

This version works completely offline and doesn't require any API keys.
"""

import os
from pathlib import Path
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class SimpleOfflineRAGDemo:
    """
    Simple RAG Pipeline demonstration class - Offline version
    """

    def __init__(self):
        """
        Initialize the RAG system
        """
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.rag_pipeline = None

    def setup_indexing_pipeline(self) -> Pipeline:
        """
        Create a pipeline for indexing documents

        Returns:
            Pipeline for document indexing
        """
        # Document writer - stores documents in the document store
        doc_writer = DocumentWriter(document_store=self.document_store)

        # Create indexing pipeline (simple version - just store documents)
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("doc_writer", doc_writer)

        self.indexing_pipeline = indexing_pipeline
        return indexing_pipeline

    def setup_rag_pipeline(self) -> Pipeline:
        """
        Create a pipeline for RAG question answering using keyword search

        Returns:
            Pipeline for RAG QA
        """
        # BM25 Retriever - finds relevant documents using keyword matching
        retriever = InMemoryBM25Retriever(document_store=self.document_store)

        # Prompt template for offline RAG
        template = """
Based on the following information, here's what I found relevant to your question:

Context Information:
{% for document in documents %}
• {{ document.content }}
{% endfor %}

Question: {{ question }}

Summary: The above context provides relevant information about your question regarding {{ question|lower }}.
The most relevant documents have been retrieved based on keyword matching.
"""

        prompt_builder = PromptBuilder(template=template)

        # Create RAG pipeline
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)

        # Connect components
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")

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
            # First write documents to store
            result = self.indexing_pipeline.run({"doc_writer": {"documents": documents}})
            print(f"✅ Added {len(documents)} documents to knowledge base")
            return result
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return None

    def ask_question(self, question: str, top_k: int = 3) -> dict:
        """
        Ask a question and get relevant context using keyword search

        Args:
            question: The question to ask
            top_k: Number of relevant documents to retrieve

        Returns:
            Dictionary containing the context and metadata
        """
        if not self.rag_pipeline:
            self.setup_rag_pipeline()

        try:
            # Run the pipeline with correct parameter names
            result = self.rag_pipeline.run({
                "retriever": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question}
            })

            # Extract documents from the prompt content
            retrieved_docs = []
            prompt_text = result.get("prompt_builder", {}).get("prompt", "")

            # Parse documents from prompt text (they appear between "Context Information:" and "Question:")
            if "Context Information:" in prompt_text and "Question:" in prompt_text:
                context_section = prompt_text.split("Context Information:")[1].split("Question:")[0]
                # Count bullet points to estimate number of documents
                bullet_lines = [line.strip() for line in context_section.split('\n') if line.strip().startswith('•')]
                # Create placeholder documents for display
                for i, line in enumerate(bullet_lines):
                    content = line[1:].strip()  # Remove bullet point
                    if content:
                        retrieved_docs.append(Document(content=content))

            return {
                "question": question,
                "context_summary": prompt_text,
                "retrieved_documents": retrieved_docs,
                "retrieval_method": "BM25 (keyword matching)"
            }
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            return {
                "question": question,
                "context_summary": f"Error: {e}",
                "retrieved_documents": [],
                "retrieval_method": "Error"
            }

    def simple_answer_generator(self, question: str, documents: List[Document]) -> str:
        """
        Generate a simple answer based on retrieved documents (rule-based)

        Args:
            question: The question asked
            documents: Retrieved documents

        Returns:
            Simple generated answer
        """
        if not documents:
            return "I couldn't find any relevant information to answer your question."

        question_lower = question.lower()

        # Simple keyword-based answering
        if "what is" in question_lower or "what are" in question_lower:
            return f"Based on the available information: {documents[0].content}"
        elif "how" in question_lower:
            return f"According to the context: {documents[0].content}"
        elif "why" in question_lower:
            return f"The information suggests: {documents[0].content}"
        else:
            return f"Here's relevant information: {documents[0].content}"


def create_sample_documents() -> List[Document]:
    """
    Create sample documents for testing

    Returns:
        List of sample documents
    """
    documents = [
        Document(content="Haystack is an end-to-end LLM framework for building applications powered by LLMs, Transformer models, and vector search. It provides components for document processing, retrieval, and generation."),
        Document(content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide more accurate and contextual responses. It first retrieves relevant documents, then generates answers based on that context."),
        Document(content="Vector databases store high-dimensional vectors that represent text, images, or other data types for similarity search. They enable semantic search by comparing vector representations."),
        Document(content="Embedding models convert text into numerical vectors that capture semantic meaning and relationships between words. These vectors can be used for similarity comparison and search."),
        Document(content="LLMs (Large Language Models) like GPT-4 can understand and generate human-like text based on input prompts. They are trained on vast amounts of text data."),
        Document(content="Python is a popular programming language for AI and machine learning applications, known for its simplicity and extensive libraries like NumPy, pandas, and scikit-learn."),
        Document(content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like text analysis, translation, and generation."),
        Document(content="Machine learning algorithms can learn patterns from data and make predictions or decisions without being explicitly programmed. They improve performance through experience."),
        Document(content="BM25 is a ranking function used for information retrieval that scores documents based on keyword relevance. It's commonly used in search engines for keyword-based matching."),
        Document(content="Document stores are databases optimized for storing and retrieving documents, often used in search and RAG applications. They can support various indexing methods.")
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


def test_document_store_directly(demo: SimpleOfflineRAGDemo):
    """
    Test document store directly without pipeline
    """
    print("\n🧪 Testing direct document store access:")
    docs = demo.document_store.filter_documents()
    print(f"   Total documents in store: {len(docs)}")

    # Test BM25 retriever directly
    retriever = InMemoryBM25Retriever(document_store=demo.document_store)
    try:
        result = retriever.run(query="Haystack framework", top_k=2)
        print(f"   Direct retriever test successful: {len(result['documents'])} documents found")
        for i, doc in enumerate(result['documents'][:2]):
            score = getattr(doc, 'score', 'N/A')
            preview = doc.content[:80] + "..."
            print(f"     - Doc {i+1} (score: {score}): {preview}")
    except Exception as e:
        print(f"   Direct retriever test failed: {e}")


def main():
    """
    Main demonstration function
    """
    print("🚀 Simple RAG Pipeline Demo - Offline Version")
    print("=" * 60)
    print("This demo works completely offline using keyword-based retrieval (BM25)")
    print("No API keys or external services required!")

    # Initialize RAG system
    rag_demo = SimpleOfflineRAGDemo()

    # Create and add sample documents
    print("\n📚 Adding sample documents to knowledge base...")
    sample_docs = create_sample_documents()
    result = rag_demo.add_documents(sample_docs)

    if not result:
        print("❌ Failed to add documents. Exiting.")
        return

    print(f"\n📊 Document store contains {len(rag_demo.document_store.filter_documents())} documents")

    # Test document store directly
    test_document_store_directly(rag_demo)

    # Test simple pipeline run
    print("\n🧪 Testing simple pipeline run:")
    try:
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        retriever = InMemoryBM25Retriever(document_store=rag_demo.document_store)
        test_result = retriever.run(query="What is Haystack", top_k=1)
        print(f"   Simple retriever run successful: {len(test_result['documents'])} docs")
    except Exception as e:
        print(f"   Simple retriever run failed: {e}")

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
        "What is BM25?",
        "How do machine learning algorithms work?"
    ]

    print("\n❓ Testing RAG pipeline with sample questions:")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")

        result = rag_demo.ask_question(question, top_k=2)

        print(f"   Retrieval Method: {result['retrieval_method']}")
        print(f"   Retrieved documents: {len(result['retrieved_documents'])}")

        # Show retrieved document previews
        for j, doc in enumerate(result['retrieved_documents']):
            preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            print(f"     - Doc {j+1}: {preview}")

        # Generate simple answer
        simple_answer = rag_demo.simple_answer_generator(question, result['retrieved_documents'])
        print(f"   Simple Answer: {simple_answer}")

        # Show context summary (optional - comment out if too verbose)
        # print(f"   Full Context:\n{result['context_summary']}")

    print("\n🎉 Demo completed successfully!")
    print("\n💡 About this demo:")
    print("   - Uses BM25 keyword-based retrieval (no embeddings required)")
    print("   - Works completely offline")
    print("   - No external API dependencies")
    print("   - Simple rule-based answer generation")
    print("\n🚀 Next steps:")
    print("   - Try adding your own documents")
    print("   - Experiment with different questions")
    print("   - Upgrade to semantic search with local embedding models")


if __name__ == "__main__":
    main()
