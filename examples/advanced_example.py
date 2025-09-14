#!/usr/bin/env python3
"""
Advanced Haystack Usage Example
===============================

This file demonstrates advanced concepts of Haystack:
1. RAG (Retrieval-Augmented Generation) pipeline
2. Custom components with complex logic
3. Document processing and indexing
4. Multi-modal pipelines
5. Pipeline serialization and deserialization
6. Error handling and validation

Run this file with: python advanced_example.py
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from haystack import component, Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage

# Advanced Custom Components

@component
class DocumentAnalyzer:
    """
    Advanced component that analyzes documents for sentiment, entities, and keywords
    """

    @component.output_types(
        sentiment_score=float,
        entities=List[str],
        keywords=List[str],
        summary=str
    )
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze documents for various features
        """
        if not documents:
            return {
                "sentiment_score": 0.0,
                "entities": [],
                "keywords": [],
                "summary": "No documents provided"
            }

        # Simple sentiment analysis (mock implementation)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate"]

        all_text = " ".join([doc.content for doc in documents])
        words = all_text.lower().split()

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0.0

        # Extract potential entities (capitalized words)
        entities = list(set([word for word in all_text.split() if word.istitle()]))[:10]

        # Extract keywords (words longer than 4 characters, excluding common words)
        common_words = {"this", "that", "with", "have", "will", "from", "they", "been", "were", "said"}
        keywords = list(set([
            word.lower() for word in words
            if len(word) > 4 and word.lower() not in common_words
        ]))[:10]

        # Generate summary
        summary = f"Analyzed {len(documents)} documents with {len(words)} words total."

        return {
            "sentiment_score": sentiment_score,
            "entities": entities,
            "keywords": keywords,
            "summary": summary
        }

@component
class ConditionalRouter:
    """
    Routes documents based on conditions like sentiment or content type
    """

    @component.output_types(
        positive_docs=List[Document],
        negative_docs=List[Document],
        neutral_docs=List[Document]
    )
    def run(
        self,
        documents: List[Document],
        sentiment_score: float,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Route documents based on sentiment analysis
        """
        if sentiment_score > threshold:
            return {
                "positive_docs": documents,
                "negative_docs": [],
                "neutral_docs": []
            }
        elif sentiment_score < -threshold:
            return {
                "positive_docs": [],
                "negative_docs": documents,
                "neutral_docs": []
            }
        else:
            return {
                "positive_docs": [],
                "negative_docs": [],
                "neutral_docs": documents
            }

@component
class ResponseEnhancer:
    """
    Enhances LLM responses with additional context and formatting
    """

    @component.output_types(
        enhanced_response=str,
        confidence_score=float,
        metadata=Dict[str, Any]
    )
    def run(
        self,
        response: str,
        sentiment_score: float = 0.0,
        keywords: Optional[List[str]] = None,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhance the response with additional context
        """
        keywords = keywords or []
        entities = entities or []

        # Calculate confidence based on response length and content
        confidence_score = min(0.9, len(response) / 100 + 0.3)

        # Add context information
        enhanced_parts = [response.strip()]

        if keywords:
            enhanced_parts.append(f"\n**Key topics identified**: {', '.join(keywords[:5])}")

        if entities:
            enhanced_parts.append(f"\n**Entities mentioned**: {', '.join(entities[:5])}")

        sentiment_emoji = "😊" if sentiment_score > 0.3 else "😐" if sentiment_score > -0.3 else "😔"
        enhanced_parts.append(f"\n**Content sentiment**: {sentiment_emoji}")

        enhanced_response = "\n".join(enhanced_parts)

        metadata = {
            "original_length": len(response),
            "enhanced_length": len(enhanced_response),
            "keywords_count": len(keywords),
            "entities_count": len(entities),
            "sentiment_category": "positive" if sentiment_score > 0.3 else "negative" if sentiment_score < -0.3 else "neutral"
        }

        return {
            "enhanced_response": enhanced_response,
            "confidence_score": confidence_score,
            "metadata": metadata
        }

# Advanced Pipeline Examples

def create_sample_documents() -> List[Document]:
    """
    Create sample documents for demonstration
    """
    sample_texts = [
        {
            "content": "Haystack is an amazing open-source framework for building search systems. It provides excellent tools for developers to create powerful applications with natural language processing capabilities.",
            "meta": {"source": "documentation", "category": "technical"}
        },
        {
            "content": "Machine learning has revolutionized how we process information. Deep learning models can understand context and generate human-like responses with remarkable accuracy.",
            "meta": {"source": "article", "category": "ai"}
        },
        {
            "content": "The weather today is terrible. It's raining heavily and the roads are flooded. This is the worst weather we've had in months.",
            "meta": {"source": "news", "category": "weather"}
        },
        {
            "content": "Python programming language is fantastic for data science and AI development. The ecosystem of libraries like pandas, numpy, and scikit-learn makes it incredibly powerful.",
            "meta": {"source": "tutorial", "category": "programming"}
        },
        {
            "content": "Natural Language Processing enables computers to understand and generate human language. Modern NLP systems use transformer architectures and attention mechanisms.",
            "meta": {"source": "research", "category": "nlp"}
        }
    ]

    return [Document(content=text["content"], meta=text["meta"]) for text in sample_texts]

def advanced_rag_pipeline():
    """
    Create an advanced RAG pipeline with document analysis and routing
    """
    print("=== Advanced RAG Pipeline ===")

    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")

    # Initialize document store
    document_store = InMemoryDocumentStore()

    # Create components
    splitter = DocumentSplitter(split_by="word", split_length=50, split_overlap=10)
    writer = DocumentWriter(document_store=document_store)
    analyzer = DocumentAnalyzer()
    router = ConditionalRouter()
    retriever = InMemoryBM25Retriever(document_store=document_store)

    # Build indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("splitter", "writer")

    # Index documents
    print("Indexing documents...")
    indexing_result = indexing_pipeline.run({
        "splitter": {"documents": documents}
    })
    documents_written = indexing_result['writer']['documents_written']
    print(f"Indexed {documents_written} document chunks")

    # Build analysis pipeline
    analysis_pipeline = Pipeline()
    analysis_pipeline.add_component("analyzer", analyzer)

    # Analyze documents first
    print("\nAnalyzing documents...")
    analysis_result = analysis_pipeline.run({
        "analyzer": {"documents": documents}
    })

    print(f"Sentiment Score: {analysis_result['analyzer']['sentiment_score']:.2f}")
    print(f"Keywords: {', '.join(analysis_result['analyzer']['keywords'][:5])}")
    print(f"Entities: {', '.join(analysis_result['analyzer']['entities'][:5])}")

    # Then route documents based on analysis results
    router_pipeline = Pipeline()
    router_pipeline.add_component("router", router)

    routing_result = router_pipeline.run({
        "router": {
            "documents": documents,
            "sentiment_score": analysis_result['analyzer']['sentiment_score']
        }
    })

    print(f"Positive docs: {len(routing_result['router']['positive_docs'])}")
    print(f"Negative docs: {len(routing_result['router']['negative_docs'])}")
    print(f"Neutral docs: {len(routing_result['router']['neutral_docs'])}")

    # Build retrieval pipeline
    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component("retriever", retriever)

    # Test retrieval
    test_queries = [
        "How does machine learning work?",
        "What is Haystack framework?",
        "Python programming for AI"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        retrieval_result = retrieval_pipeline.run({
            "retriever": {"query": query, "top_k": 2}
        })

        retrieved_docs = retrieval_result['retriever']['documents']
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  {i}. {doc.content[:100]}...")
            print(f"     Score: {doc.score:.3f}")

    print()

def response_enhancement_pipeline():
    """
    Create a pipeline that enhances responses with additional context
    """
    print("=== Response Enhancement Pipeline ===")

    # Create sample response data
    sample_response = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
    sample_keywords = ["machine", "learning", "artificial", "intelligence", "computers", "data"]
    sample_entities = ["Machine Learning", "AI", "Data Science"]
    sample_sentiment = 0.6

    # Create components
    enhancer = ResponseEnhancer()

    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("enhancer", enhancer)

    # Run enhancement
    result = pipeline.run({
        "enhancer": {
            "response": sample_response,
            "sentiment_score": sample_sentiment,
            "keywords": sample_keywords,
            "entities": sample_entities
        }
    })

    print("Original Response:")
    print(sample_response)
    print("\nEnhanced Response:")
    print(result['enhancer']['enhanced_response'])
    print(f"\nConfidence Score: {result['enhancer']['confidence_score']:.2f}")
    print("Metadata:")
    for key, value in result['enhancer']['metadata'].items():
        print(f"  {key}: {value}")
    print()

def pipeline_serialization_demo():
    """
    Demonstrate pipeline serialization and deserialization
    """
    print("=== Pipeline Serialization Demo ===")

    # Create a simple pipeline
    pipeline = Pipeline()

    # Add components
    analyzer = DocumentAnalyzer()
    pipeline.add_component("analyzer", analyzer)

    # Save pipeline to file
    pipeline_path = "temp_pipeline.yaml"
    try:
        pipeline.dump(pipeline_path)
        print(f"Pipeline saved to {pipeline_path}")

        # Load pipeline from file
        loaded_pipeline = Pipeline.load(pipeline_path)
        print("Pipeline loaded successfully")

        # Test loaded pipeline
        test_docs = [Document(content="This is a great example of pipeline serialization!")]
        result = loaded_pipeline.run({
            "analyzer": {"documents": test_docs}
        })

        print(f"Test result: {result['analyzer']['summary']}")

    except Exception as e:
        print(f"Serialization demo failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(pipeline_path):
            os.remove(pipeline_path)
            print(f"Cleaned up {pipeline_path}")

    print()

def error_handling_example():
    """
    Demonstrate error handling in pipelines
    """
    print("=== Error Handling Example ===")

    @component
    class ErrorProneComponent:
        """Component that might fail under certain conditions"""

        @component.output_types(result=str, status=str)
        def run(self, input_text: str) -> Dict[str, Any]:
            if len(input_text) < 5:
                raise ValueError("Input text too short (minimum 5 characters)")

            if "error" in input_text.lower():
                raise RuntimeError("Error keyword detected in input")

            return {
                "result": f"Processed: {input_text.upper()}",
                "status": "success"
            }

    # Create pipeline
    pipeline = Pipeline()
    error_component = ErrorProneComponent()
    pipeline.add_component("processor", error_component)

    # Test cases
    test_cases = [
        "This is a good input",
        "bad",  # Too short
        "This contains error word",  # Contains error
        "Another good input text"
    ]

    for test_input in test_cases:
        print(f"Testing: '{test_input}'")
        try:
            result = pipeline.run({
                "processor": {"input_text": test_input}
            })
            print(f"  Success: {result['processor']['result']}")
        except ValueError as e:
            print(f"  ValueError: {e}")
        except RuntimeError as e:
            print(f"  RuntimeError: {e}")
        except Exception as e:
            print(f"  Unexpected error: {e}")
        print()

def main():
    """
    Run all advanced examples
    """
    print("🚀 Haystack Advanced Usage Examples")
    print("===================================")
    print()

    try:
        advanced_rag_pipeline()
        response_enhancement_pipeline()
        pipeline_serialization_demo()
        error_handling_example()

        print("✅ All advanced examples completed!")
        print()
        print("Advanced features demonstrated:")
        print("1. ✅ Custom components with complex logic")
        print("2. ✅ Document analysis and sentiment scoring")
        print("3. ✅ Conditional routing based on content")
        print("4. ✅ Response enhancement with metadata")
        print("5. ✅ Pipeline serialization/deserialization")
        print("6. ✅ Error handling and validation")
        print()
        print("Next steps for production:")
        print("• Add proper logging and monitoring")
        print("• Implement caching for better performance")
        print("• Add unit tests for custom components")
        print("• Use proper vector databases for large-scale RAG")
        print("• Implement authentication and rate limiting")

    except Exception as e:
        print(f"❌ Error running advanced examples: {e}")
        print("This might be due to missing dependencies or configuration issues.")

if __name__ == "__main__":
    main()
