# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Haystack Built-in Demo
======================

This demo showcases Haystack's built-in sample components and test data,
demonstrating that Haystack comes with a complete testing ecosystem
ready to use out of the box.

Features:
- Uses built-in sample components from haystack.testing.sample_components
- Demonstrates pipeline creation and execution
- Shows various component types and their interactions
- Uses actual sample data files from e2e/samples
- No external dependencies required (works offline)
"""

import os
from pathlib import Path
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.testing.sample_components import (
    Hello, Greet, Sum, Double, AddFixedValue,
    Concatenate, FString, TextSplitter, Accumulate
)


class HaystackBuiltInDemo:
    """
    Demonstration class using Haystack's built-in components and data
    """

    def __init__(self):
        self.samples_path = Path(__file__).parent.parent / "e2e" / "samples"
        self.document_store = InMemoryDocumentStore()

    def demo_1_basic_components(self):
        """Demo 1: Basic individual components"""
        print("🧪 Demo 1: Basic Sample Components")
        print("=" * 50)

        # Hello component
        hello = Hello()
        result = hello.run(word="Engineer")
        print(f"📝 Hello: {result}")

        # Math components
        double = Double()
        add_val = AddFixedValue(add=10)
        sum_comp = Sum()

        print(f"🔢 Double(5): {double.run(value=5)}")
        print(f"🔢 AddFixedValue(7, +10): {add_val.run(value=7)}")
        print(f"🔢 Sum([1,2,3,4,5]): {sum_comp.run(values=[1,2,3,4,5])}")

        # String components
        concat = Concatenate()
        # Text processing
        splitter = TextSplitter()
        fstring = FString(template="Result: {text} (Score: {score})")

        print(f"📝 Concatenate: {concat.run(first='Hello', second=' World!')}")
        print(f"📝 FString: {fstring.run(text='Great', score=95)}")

        # Text processing
        text_result = splitter.run(sentence="This is a sample text for splitting demonstration")
        print(f"📝 TextSplitter: {text_result}")

    def demo_2_pipeline_with_components(self):
        """Demo 2: Pipeline using sample components"""
        print("\n🔗 Demo 2: Pipeline with Sample Components")
        print("=" * 50)

        # Create a mathematical pipeline
        pipeline = Pipeline()

        # Add components
        pipeline.add_component("doubler", Double())
        pipeline.add_component("adder", AddFixedValue(add=20))
        pipeline.add_component("accumulate", Accumulate())
        pipeline.add_component("greet", Greet(
            message="🎯 Final accumulated values: {value}",
            log_level="INFO"
        ))

        # Connect components
        pipeline.connect("doubler.value", "adder.value")
        pipeline.connect("adder.result", "accumulate.value")
        pipeline.connect("accumulate.value", "greet.value")

        # Show pipeline structure
        print(f"Pipeline nodes: {list(pipeline.graph.nodes())}")
        print(f"Pipeline edges: {list(pipeline.graph.edges())}")

        # Run pipeline multiple times to show accumulation
        print("\nRunning pipeline with different inputs:")
        for i, input_val in enumerate([5, 10, 15], 1):
            print(f"\nRun {i}: Input = {input_val}")
            result = pipeline.run({"doubler": {"value": input_val}})
            calculation = f"{input_val} → double → {input_val*2} → add 20 → {input_val*2+20}"
            print(f"         Calculation: {calculation}")

    def demo_3_document_processing(self):
        """Demo 3: Document processing with built-in data"""
        print("\n📚 Demo 3: Document Processing with Built-in Sample Data")
        print("=" * 50)

        # Check if sample files exist
        sample_files = list(self.samples_path.glob("*.txt"))
        if not sample_files:
            print("⚠️  Sample files not found, creating mock data...")
            # Create sample documents programmatically
            documents = [
                Document(content="Haystack is an end-to-end framework for building LLM applications."),
                Document(content="Components are the building blocks of Haystack pipelines."),
                Document(content="Pipelines orchestrate the execution of multiple components."),
                Document(content="Document stores provide storage and retrieval capabilities."),
                Document(content="Sample components help developers learn and test Haystack features."),
            ]
        else:
            # Use actual sample files
            print(f"📄 Found {len(sample_files)} sample files:")
            for file in sample_files:
                print(f"   - {file.name}")

            # Convert files to documents
            converter = TextFileToDocument()
            documents = []

            for file in sample_files[:2]:  # Use first 2 files to keep demo manageable
                try:
                    result = converter.run(sources=[str(file)])
                    if result.get("documents"):
                        # Truncate long documents for demo
                        for doc in result["documents"]:
                            if len(doc.content) > 500:
                                doc.content = doc.content[:500] + "... [truncated for demo]"
                        documents.extend(result["documents"])
                        print(f"   ✅ Loaded {file.name}: {len(result['documents'])} documents")
                except Exception as e:
                    print(f"   ❌ Failed to load {file.name}: {e}")

        if not documents:
            print("❌ No documents available for processing")
            return

        # Store documents
        self.document_store.write_documents(documents)
        print(f"\n📊 Stored {len(documents)} documents in document store")

        # Create simple search pipeline
        retriever = InMemoryBM25Retriever(document_store=self.document_store)

        # Test retrieval
        queries = ["Haystack framework", "components pipeline", "document processing"]

        print("\n🔍 Testing document retrieval:")
        for query in queries:
            result = retriever.run(query=query, top_k=2)
            print(f"\nQuery: '{query}'")
            print(f"Found {len(result['documents'])} relevant documents:")

            for i, doc in enumerate(result['documents'], 1):
                preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                score = getattr(doc, 'score', 'N/A')
                print(f"  {i}. Score: {score:.3f} - {preview}")

    def demo_4_text_processing_pipeline(self):
        """Demo 4: Advanced text processing pipeline"""
        print("\n📝 Demo 4: Advanced Text Processing Pipeline")
        print("=" * 50)

        # Create text processing pipeline
        pipeline = Pipeline()

        # Add text processing components
        pipeline.add_component("splitter", TextSplitter())
        pipeline.add_component("formatter", FString(template="Chunk {chunk_num}: {text}"))

        # Create a simple pipeline for demonstration
        text_input = "This is a comprehensive demonstration of Haystack's built-in sample components and their capabilities for text processing"

        # Process text manually to show component interaction
        splitter = TextSplitter()
        formatter = FString(template="📄 Chunk {chunk_num}: {text}")

        # Split text
        split_result = splitter.run(sentence=text_input)
        chunks = split_result.get("output", [])

        print(f"Original text: {text_input}")
        print(f"\nSplit into {len(chunks)} chunks:")

        for i, chunk in enumerate(chunks, 1):
            formatted = formatter.run(chunk_num=i, text=chunk)
            print(f"  {formatted['string']}")

    def demo_5_component_showcase(self):
        """Demo 5: Showcase all available sample components"""
        print("\n🎨 Demo 5: Complete Sample Components Showcase")
        print("=" * 50)

        print("Available Haystack sample components:")

        components_demo = {
            "Hello": lambda: Hello().run(word="Haystack"),
            "Double": lambda: Double().run(value=21),
            "AddFixedValue": lambda: AddFixedValue(add=100).run(value=50),
            "Sum": lambda: Sum().run(values=[10, 20, 30, 40]),
            "Concatenate": lambda: Concatenate().run(first=["Hello", " from"], second=[" ", "Haystack"]),
            "FString": lambda: FString(template="🎯 {framework} version {version}").run(framework="Haystack", version="2.18.0"),
            "TextSplitter": lambda: TextSplitter().run(sentence="First sentence. Second sentence. Third sentence. Fourth sentence."),
        }

        for name, demo_func in components_demo.items():
            try:
                result = demo_func()
                print(f"✅ {name:15}: {result}")
            except Exception as e:
                print(f"❌ {name:15}: Error - {e}")

    def run_all_demos(self):
        """Run all demonstration scenarios"""
        print("🚀 Haystack Built-in Components & Data Demo")
        print("=" * 60)
        print("This demo showcases Haystack's ready-to-use testing ecosystem!")
        print("✅ No external APIs required")
        print("✅ No additional dependencies needed")
        print("✅ Everything works out of the box\n")

        try:
            self.demo_1_basic_components()
            self.demo_2_pipeline_with_components()
            self.demo_3_document_processing()
            self.demo_4_text_processing_pipeline()
            self.demo_5_component_showcase()

            print("\n" + "=" * 60)
            print("🎉 All demos completed successfully!")
            print("\n💡 Key Takeaways:")
            print("   ✅ Haystack includes 15+ ready-to-use sample components")
            print("   ✅ Built-in test data and sample documents available")
            print("   ✅ Complete pipeline examples in e2e tests")
            print("   ✅ No setup required - works immediately after installation")
            print("   ✅ Perfect for learning, testing, and prototyping")

            print("\n🔍 Explore more:")
            print("   📁 haystack/testing/sample_components/ - All sample components")
            print("   📁 haystack/e2e/samples/ - Sample data files")
            print("   📁 haystack/e2e/pipelines/ - End-to-end pipeline examples")
            print("   📚 Official docs: https://docs.haystack.deepset.ai/")

        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo"""
    demo = HaystackBuiltInDemo()
    demo.run_all_demos()


if __name__ == "__main__":
    main()
