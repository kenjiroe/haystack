#!/usr/bin/env python3
"""
Basic Haystack Usage Example
============================

This file demonstrates the fundamental concepts of Haystack:
1. Creating components
2. Building pipelines
3. Running simple NLP tasks

Run this file with: python basic_example.py
"""

from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage
import os

# Example 1: Simple Text Processing Component
@component
class TextProcessor:
    """
    A simple component that processes text input
    """

    @component.output_types(processed_text=str, word_count=int)
    def run(self, text: str) -> dict:
        """
        Process the input text and return processed text with word count
        """
        processed = text.strip().upper()
        word_count = len(text.split())

        return {
            "processed_text": processed,
            "word_count": word_count
        }

# Example 2: Basic Pipeline without LLM
def basic_text_pipeline():
    """
    Create and run a simple text processing pipeline
    """
    print("=== Basic Text Processing Pipeline ===")

    # Create components
    processor = TextProcessor()

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("processor", processor)

    # Run pipeline
    result = pipeline.run({
        "processor": {
            "text": "hello world from haystack"
        }
    })

    print(f"Original text: hello world from haystack")
    print(f"Processed text: {result['processor']['processed_text']}")
    print(f"Word count: {result['processor']['word_count']}")
    print()

# Example 3: Simple Question Answering (requires OpenAI API key)
def simple_qa_pipeline():
    """
    Create a simple Question Answering pipeline using OpenAI
    Note: Requires OPENAI_API_KEY environment variable
    """
    print("=== Simple Question Answering Pipeline ===")

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Skipping OpenAI example.")
        print("To run this example, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print()
        return

    try:
        # Create components
        prompt_builder = PromptBuilder(
            template="Answer this question: {{question}}"
        )
        generator = OpenAIGenerator(model="gpt-3.5-turbo-instruct")

        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("generator", generator)

        # Connect components
        pipeline.connect("prompt_builder.prompt", "generator.prompt")

        # Run pipeline
        result = pipeline.run({
            "prompt_builder": {
                "question": "What is the capital of Thailand?"
            }
        })

        print("Question: What is the capital of Thailand?")
        print(f"Answer: {result['generator']['replies'][0]}")
        print()

    except Exception as e:
        print(f"Error running OpenAI example: {e}")
        print("Make sure your OPENAI_API_KEY is valid and you have credits.")
        print()

# Example 4: Multi-step Pipeline
@component
class QuestionValidator:
    """
    Component that validates if the input is a proper question
    """

    @component.output_types(is_valid=bool, message=str)
    def run(self, text: str) -> dict:
        """
        Check if the text is a valid question
        """
        is_question = text.strip().endswith('?')

        return {
            "is_valid": is_question,
            "message": "Valid question" if is_question else "Not a question - please add '?'"
        }

def multi_step_pipeline():
    """
    Create a pipeline with multiple processing steps
    """
    print("=== Multi-step Processing Pipeline ===")

    # Test inputs
    test_inputs = [
        "What is machine learning?",
        "Machine learning is great",
        "How does Haystack work?"
    ]

    for test_input in test_inputs:
        print(f"Input: {test_input}")

        # Create fresh components for each iteration
        validator = QuestionValidator()
        processor = TextProcessor()

        # Create separate pipelines for each step to avoid input conflicts
        validation_pipeline = Pipeline()
        validation_pipeline.add_component("validator", validator)

        processing_pipeline = Pipeline()
        processing_pipeline.add_component("processor", processor)

        # Validate first
        validation_result = validation_pipeline.run({
            "validator": {"text": test_input}
        })

        print(f"Validation: {validation_result['validator']['message']}")

        if validation_result['validator']['is_valid']:
            # Process if valid
            processing_result = processing_pipeline.run({
                "processor": {"text": test_input}
            })
            print(f"Processed: {processing_result['processor']['processed_text']}")

        print("---")
    print()

# Example 5: Document-based QA (without external dependencies)
@component
class SimpleDocumentStore:
    """
    A simple in-memory document store
    """

    def __init__(self):
        self.documents = [
            "Haystack is an open-source framework for building search systems.",
            "It supports both traditional search and modern LLM-based applications.",
            "Haystack can work with various LLM providers like OpenAI, Cohere, and local models.",
            "Components in Haystack can be connected to create complex pipelines."
        ]

    @component.output_types(documents=list)
    def run(self, query: str) -> dict:
        """
        Simple keyword-based document retrieval
        """
        query_lower = query.lower()
        matching_docs = []

        for doc in self.documents:
            if any(word in doc.lower() for word in query_lower.split()):
                matching_docs.append(doc)

        return {"documents": matching_docs}

def document_qa_pipeline():
    """
    Create a simple document-based QA system
    """
    print("=== Document-based QA Pipeline ===")

    # Create components
    doc_store = SimpleDocumentStore()

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("doc_store", doc_store)

    # Test queries
    queries = [
        "What is Haystack?",
        "LLM providers",
        "components pipeline"
    ]

    for query in queries:
        result = pipeline.run({
            "doc_store": {"query": query}
        })

        print(f"Query: {query}")
        print("Relevant documents:")
        for i, doc in enumerate(result['doc_store']['documents'], 1):
            print(f"  {i}. {doc}")
        print("---")
    print()

def main():
    """
    Run all examples
    """
    print("🚀 Haystack Basic Usage Examples")
    print("=================================")
    print()

    # Run examples
    basic_text_pipeline()
    simple_qa_pipeline()
    multi_step_pipeline()
    document_qa_pipeline()

    print("✅ All examples completed!")
    print()
    print("Next steps:")
    print("1. Set up OPENAI_API_KEY to try LLM examples")
    print("2. Explore more components in haystack.components")
    print("3. Check out the official documentation at https://docs.haystack.deepset.ai/")
    print("4. Try building your own custom components")

if __name__ == "__main__":
    main()
