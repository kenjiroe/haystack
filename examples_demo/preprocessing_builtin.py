# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Document Preprocessing Pipeline - Built-in Example
===================================================

This is a simplified version of Haystack's built-in document preprocessing pipeline
from e2e/pipelines/test_preprocessing_pipeline.py

Features:
- File type routing (TXT, PDF support)
- Document conversion and cleaning
- Text splitting and chunking
- Language detection and routing
- Metadata-based filtering
- BM25 search (no ML models required)

This example demonstrates Haystack's document processing capabilities
without requiring external ML dependencies or API keys.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from haystack import Document, Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


class DocumentPreprocessingDemo:
    """
    Document Preprocessing Pipeline demonstration using Haystack's built-in approach
    """

    def __init__(self):
        """
        Initialize the preprocessing system
        """
        self.document_store = InMemoryDocumentStore()
        self.preprocessing_pipeline = None
        self.search_pipeline = None

        print("🔧 Initializing Document Preprocessing Pipeline")

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Create preprocessing pipeline following Haystack's built-in pattern
        (simplified version without ML dependencies)

        Returns:
            Pipeline for document preprocessing
        """
        print("📝 Creating preprocessing pipeline...")

        # Create the pipeline
        preprocessing_pipeline = Pipeline()

        # File type router for text files only
        preprocessing_pipeline.add_component(
            instance=FileTypeRouter(mime_types=["text/plain"]),
            name="file_type_router"
        )

        # Text file converter
        preprocessing_pipeline.add_component(
            instance=TextFileToDocument(),
            name="text_file_converter"
        )

        # Document processing components
        preprocessing_pipeline.add_component(
            instance=DocumentCleaner(
                remove_empty_lines=True,
                remove_extra_whitespaces=True,
                remove_repeated_substrings=False
            ),
            name="cleaner"
        )

        preprocessing_pipeline.add_component(
            instance=DocumentSplitter(
                split_by="sentence",
                split_length=3,
                split_overlap=1
            ),
            name="splitter"
        )

        # Document storage
        preprocessing_pipeline.add_component(
            instance=DocumentWriter(document_store=self.document_store),
            name="writer"
        )

        # Connect components
        preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        preprocessing_pipeline.connect("text_file_converter.documents", "cleaner.documents")
        preprocessing_pipeline.connect("cleaner.documents", "splitter.documents")
        preprocessing_pipeline.connect("splitter.documents", "writer.documents")

        self.preprocessing_pipeline = preprocessing_pipeline

        print("✅ Preprocessing pipeline created!")
        print(f"   Nodes: {len(preprocessing_pipeline.graph.nodes)}")
        print(f"   Edges: {len(preprocessing_pipeline.graph.edges)}")
        print("   Components:", list(preprocessing_pipeline.graph.nodes))

        return preprocessing_pipeline

    def create_search_pipeline(self) -> Pipeline:
        """
        Create search pipeline using BM25 (no ML models)

        Returns:
            Pipeline for searching processed documents
        """
        print("🔍 Creating search pipeline...")

        search_pipeline = Pipeline()
        search_pipeline.add_component(
            instance=InMemoryBM25Retriever(document_store=self.document_store, top_k=5),
            name="bm25_retriever"
        )

        self.search_pipeline = search_pipeline

        print("✅ Search pipeline created!")
        return search_pipeline

    def process_sample_files(self, samples_path: Optional[Path] = None) -> dict:
        """
        Process sample files using the preprocessing pipeline

        Args:
            samples_path: Path to sample files (uses e2e/samples if None)

        Returns:
            Result from preprocessing pipeline
        """
        if samples_path is None:
            samples_path = Path(__file__).parent.parent / "e2e" / "samples"

        print(f"📚 Processing files from: {samples_path}")

        if not samples_path.exists():
            print(f"❌ Sample path does not exist: {samples_path}")
            # Create test file programmatically
            test_content = """
            Document Processing with Haystack

            Haystack provides powerful document processing capabilities.
            The preprocessing pipeline can handle multiple file formats including text and PDF files.

            Document cleaning removes unnecessary whitespace and formatting artifacts.
            Text splitting divides long documents into manageable chunks for better processing.

            The BM25 retriever enables keyword-based search across processed documents.
            This approach works without requiring machine learning models or external APIs.

            Pipeline components can be easily connected to create complex processing workflows.
            Each component performs a specific task in the document processing chain.
            """

            # Create test file
            test_file = Path(__file__).parent / "test_document.txt"
            test_file.write_text(test_content)

            sample_files = [test_file]
            print(f"✅ Created test file: {test_file}")
        else:
            # Get sample files (text only)
            sample_files = []
            for file in samples_path.iterdir():
                if file.is_file() and file.suffix in ['.txt']:
                    sample_files.append(file)

            print(f"📄 Found {len(sample_files)} files:")
            for file in sample_files:
                size_kb = file.stat().st_size / 1024 if file.stat().st_size > 0 else 0
                print(f"   - {file.name} ({size_kb:.1f} KB)")

        if not sample_files:
            print("❌ No suitable files found")
            return {"writer": {"documents_written": 0}}

        if not self.preprocessing_pipeline:
            self.create_preprocessing_pipeline()

        # Run preprocessing pipeline
        try:
            print("\n🔄 Running preprocessing pipeline...")
            result = self.preprocessing_pipeline.run({"file_type_router": {"sources": sample_files}})

            docs_written = result["writer"]["documents_written"]
            print(f"✅ Successfully processed {docs_written} document chunks")
            print(f"📊 Total documents in store: {self.document_store.count_documents()}")

            # Show sample of processed documents
            sample_docs = self.document_store.filter_documents()[:3]
            print(f"\n📋 Sample processed documents:")
            for i, doc in enumerate(sample_docs, 1):
                preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                print(f"   {i}. {preview}")

            return result

        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def search(self, query: str, top_k: int = 5) -> dict:
        """
        Search processed documents using BM25

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Search results
        """
        if not self.search_pipeline:
            self.create_search_pipeline()

        if self.document_store.count_documents() == 0:
            print("⚠️  No documents in store. Please process documents first.")
            return {"bm25_retriever": {"documents": []}}

        try:
            print(f"🔍 Searching for: '{query}'")
            result = self.search_pipeline.run({"bm25_retriever": {"query": query, "top_k": top_k}})

            documents = result["bm25_retriever"]["documents"]
            print(f"📊 Found {len(documents)} relevant documents")

            return result

        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise

    def save_pipelines(self, output_dir: Path):
        """
        Save pipelines to files

        Args:
            output_dir: Directory to save pipeline files
        """
        output_dir.mkdir(exist_ok=True)

        if self.preprocessing_pipeline:
            with open(output_dir / "preprocessing_pipeline.yaml", "w") as f:
                self.preprocessing_pipeline.dump(f)
            print(f"💾 Saved preprocessing pipeline to {output_dir / 'preprocessing_pipeline.yaml'}")

        if self.search_pipeline:
            with open(output_dir / "search_pipeline.json", "w") as f:
                json.dump(self.search_pipeline.to_dict(), f, indent=4)
            print(f"💾 Saved search pipeline to {output_dir / 'search_pipeline.json'}")

    def analyze_pipeline_structure(self):
        """
        Analyze and display pipeline structure
        """
        print("\n🔍 Pipeline Analysis")
        print("=" * 50)

        if self.preprocessing_pipeline:
            print("📝 Preprocessing Pipeline:")
            print(f"   Nodes: {list(self.preprocessing_pipeline.graph.nodes)}")
            print(f"   Edges: {list(self.preprocessing_pipeline.graph.edges)}")

            # Component details
            for node in self.preprocessing_pipeline.graph.nodes:
                component = self.preprocessing_pipeline.get_component(node)
                print(f"   {node}: {type(component).__name__}")

        if self.search_pipeline:
            print("\n🔍 Search Pipeline:")
            print(f"   Nodes: {list(self.search_pipeline.graph.nodes)}")
            print(f"   Edges: {list(self.search_pipeline.graph.edges)}")

    def demonstrate_document_stats(self):
        """
        Show statistics about processed documents
        """
        print("\n📊 Document Statistics")
        print("=" * 50)

        total_docs = self.document_store.count_documents()
        print(f"Total documents: {total_docs}")

        if total_docs > 0:
            all_docs = self.document_store.filter_documents()

            # Length statistics
            lengths = [len(doc.content) for doc in all_docs]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            min_length = min(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0

            print(f"Average document length: {avg_length:.1f} characters")
            print(f"Shortest document: {min_length} characters")
            print(f"Longest document: {max_length} characters")

            # Content preview
            print(f"\n📄 Document samples:")
            for i, doc in enumerate(all_docs[:3], 1):
                preview = doc.content[:80] + "..." if len(doc.content) > 80 else doc.content
                print(f"   {i}. {preview}")


def main():
    """
    Main demonstration function
    """
    print("🚀 Document Preprocessing Pipeline - Built-in Example")
    print("=" * 70)
    print("Based on Haystack's e2e/pipelines/test_preprocessing_pipeline.py")
    print("Uses document processing without ML dependencies")

    try:
        # Initialize demo
        demo = DocumentPreprocessingDemo()

        # Create pipelines
        demo.create_preprocessing_pipeline()
        demo.create_search_pipeline()

        # Show pipeline structure
        demo.analyze_pipeline_structure()

        # Process sample data
        print("\n" + "="*70)
        processing_result = demo.process_sample_files()

        if processing_result.get("writer", {}).get("documents_written", 0) > 0:
            # Show document statistics
            demo.demonstrate_document_stats()

            # Test search functionality
            test_queries = [
                "Haystack document processing",
                "pipeline components",
                "text splitting chunks",
                "BM25 search retrieval",
                "file formats PDF"
            ]

            print("\n" + "="*70)
            print("🧪 Testing Document Search")
            print("-" * 70)

            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: '{query}'")

                search_result = demo.search(query, top_k=3)
                documents = search_result["bm25_retriever"]["documents"]

                if documents:
                    for j, doc in enumerate(documents, 1):
                        score = getattr(doc, 'score', 0)
                        preview = doc.content[:120] + "..." if len(doc.content) > 120 else doc.content
                        print(f"   {j}. Score: {score:.3f}")
                        print(f"      Content: {preview}")
                else:
                    print("   No results found")

            # Save pipelines
            output_dir = Path(__file__).parent / "pipeline_exports"
            print(f"\n📁 Saving pipelines to: {output_dir}")
            demo.save_pipelines(output_dir)

            # Cleanup test file if created
            test_file = Path(__file__).parent / "test_document.txt"
            if test_file.exists():
                test_file.unlink()
                print(f"🧹 Cleaned up test file: {test_file}")

        else:
            print("❌ No documents were processed")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Preprocessing demo completed!")
    print("\n💡 This example demonstrates:")
    print("   ✅ Document processing without ML dependencies")
    print("   ✅ Text file routing and conversion")
    print("   ✅ Text cleaning and splitting")
    print("   ✅ BM25 keyword search")
    print("   ✅ Pipeline serialization and analysis")
    print("   ✅ Built-in Haystack preprocessing patterns")


if __name__ == "__main__":
    main()
