# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Dense Search Pipeline - Built-in Example
=========================================

This is a simplified version of Haystack's built-in dense document search pipeline
from e2e/pipelines/test_dense_doc_search.py

Features:
- Uses SentenceTransformers for semantic embeddings
- Processes multiple file types (TXT, PDF)
- Document cleaning and splitting
- Semantic search with embedding retrieval
- Works with built-in sample data

This example demonstrates how to use Haystack's production-ready components
for semantic document search without requiring external APIs.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class DenseSearchDemo:
    """
    Dense Search Pipeline demonstration using Haystack's built-in approach
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the dense search system

        Args:
            model_name: SentenceTransformers model for embeddings
        """
        self.model_name = model_name
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.query_pipeline = None

        print(f"🔧 Initializing Dense Search with model: {model_name}")

    def create_indexing_pipeline(self) -> Pipeline:
        """
        Create indexing pipeline following Haystack's built-in pattern

        Returns:
            Pipeline for document indexing
        """
        print("📝 Creating indexing pipeline...")

        # Create the indexing pipeline (from e2e test)
        indexing_pipeline = Pipeline()

        # File type router for different formats
        indexing_pipeline.add_component(
            instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]),
            name="file_type_router"
        )

        # Converters for different file types
        indexing_pipeline.add_component(instance=TextFileToDocument(), name="text_file_converter")
        indexing_pipeline.add_component(instance=PyPDFToDocument(), name="pdf_file_converter")

        # Document processing
        indexing_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
        indexing_pipeline.add_component(instance=DocumentCleaner(), name="cleaner")
        indexing_pipeline.add_component(
            instance=DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1),
            name="splitter"
        )

        # Embeddings and storage
        indexing_pipeline.add_component(
            instance=SentenceTransformersDocumentEmbedder(model=self.model_name, progress_bar=False),
            name="embedder"
        )
        indexing_pipeline.add_component(instance=DocumentWriter(document_store=self.document_store), name="writer")

        # Connect components (following e2e test pattern)
        indexing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        indexing_pipeline.connect("file_type_router.application/pdf", "pdf_file_converter.sources")
        indexing_pipeline.connect("text_file_converter.documents", "joiner.documents")
        indexing_pipeline.connect("pdf_file_converter.documents", "joiner.documents")
        indexing_pipeline.connect("joiner.documents", "cleaner.documents")
        indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        indexing_pipeline.connect("splitter.documents", "embedder.documents")
        indexing_pipeline.connect("embedder.documents", "writer.documents")

        self.indexing_pipeline = indexing_pipeline

        print("✅ Indexing pipeline created!")
        print(f"   Nodes: {len(indexing_pipeline.graph.nodes)}")
        print(f"   Edges: {len(indexing_pipeline.graph.edges)}")

        return indexing_pipeline

    def create_query_pipeline(self) -> Pipeline:
        """
        Create query pipeline following Haystack's built-in pattern

        Returns:
            Pipeline for querying
        """
        print("🔍 Creating query pipeline...")

        # Create the querying pipeline (from e2e test)
        query_pipeline = Pipeline()
        query_pipeline.add_component(
            instance=SentenceTransformersTextEmbedder(model=self.model_name, progress_bar=False),
            name="text_embedder"
        )
        query_pipeline.add_component(
            instance=InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=5),
            name="embedding_retriever"
        )
        query_pipeline.connect("text_embedder", "embedding_retriever")

        self.query_pipeline = query_pipeline

        print("✅ Query pipeline created!")
        print(f"   Nodes: {len(query_pipeline.graph.nodes)}")
        print(f"   Edges: {len(query_pipeline.graph.edges)}")

        return query_pipeline

    def index_sample_files(self, samples_path: Optional[Path] = None) -> dict:
        """
        Index sample files using built-in data

        Args:
            samples_path: Path to sample files (uses e2e/samples if None)

        Returns:
            Result from indexing pipeline
        """
        if samples_path is None:
            samples_path = Path(__file__).parent.parent / "e2e" / "samples"

        print(f"📚 Indexing files from: {samples_path}")

        if not samples_path.exists():
            print(f"❌ Sample path does not exist: {samples_path}")
            # Create some sample documents programmatically
            sample_docs = [
                Document(content="My name is Giorgio and I live in Rome."),
                Document(content="Paris is the capital city of France, known for the Eiffel Tower."),
                Document(content="Berlin is the largest city in Germany and its political and cultural center."),
                Document(content="Tokyo is a bustling metropolis and the capital of Japan."),
                Document(content="London is the capital of England and the United Kingdom.")
            ]

            # Write directly to document store
            self.document_store.write_documents(sample_docs)

            print(f"✅ Created {len(sample_docs)} sample documents")
            return {"writer": {"documents_written": len(sample_docs)}}

        # Get sample files
        sample_files = list(samples_path.iterdir())
        print(f"📄 Found {len(sample_files)} files:")
        for file in sample_files:
            if file.is_file():
                size_kb = file.stat().st_size / 1024 if file.stat().st_size > 0 else 0
                print(f"   - {file.name} ({size_kb:.1f} KB)")

        if not self.indexing_pipeline:
            self.create_indexing_pipeline()

        # Run indexing pipeline
        try:
            result = self.indexing_pipeline.run({"file_type_router": {"sources": sample_files}})
            docs_written = result["writer"]["documents_written"]

            print(f"✅ Successfully indexed {docs_written} documents")
            print(f"📊 Total documents in store: {self.document_store.count_documents()}")

            return result

        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            print("   This might be due to missing dependencies (sentence-transformers)")
            print("   Try: pip install sentence-transformers")
            raise

    def search(self, query: str, top_k: int = 5) -> dict:
        """
        Search documents using semantic similarity

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Search results
        """
        if not self.query_pipeline:
            self.create_query_pipeline()

        if self.document_store.count_documents() == 0:
            print("⚠️  No documents in store. Please index documents first.")
            return {"embedding_retriever": {"documents": []}}

        try:
            print(f"🔍 Searching for: '{query}'")
            result = self.query_pipeline.run({"text_embedder": {"text": query}})

            documents = result["embedding_retriever"]["documents"]
            print(f"📊 Found {len(documents)} relevant documents")

            return result

        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise

    def save_pipelines(self, output_dir: Path):
        """
        Save pipelines to files (following e2e test pattern)

        Args:
            output_dir: Directory to save pipeline files
        """
        output_dir.mkdir(exist_ok=True)

        if self.indexing_pipeline:
            # Save as YAML (like in e2e test)
            with open(output_dir / "indexing_pipeline.yaml", "w") as f:
                self.indexing_pipeline.dump(f)
            print(f"💾 Saved indexing pipeline to {output_dir / 'indexing_pipeline.yaml'}")

        if self.query_pipeline:
            # Save as JSON (like in e2e test)
            with open(output_dir / "query_pipeline.json", "w") as f:
                json.dump(self.query_pipeline.to_dict(), f, indent=4)
            print(f"💾 Saved query pipeline to {output_dir / 'query_pipeline.json'}")

    def load_pipelines(self, pipeline_dir: Path):
        """
        Load pipelines from files

        Args:
            pipeline_dir: Directory containing pipeline files
        """
        indexing_file = pipeline_dir / "indexing_pipeline.yaml"
        query_file = pipeline_dir / "query_pipeline.json"

        if indexing_file.exists():
            with open(indexing_file, "r") as f:
                self.indexing_pipeline = Pipeline.load(f)
            print(f"📂 Loaded indexing pipeline from {indexing_file}")

        if query_file.exists():
            with open(query_file, "r") as f:
                self.query_pipeline = Pipeline.from_dict(json.load(f))
            print(f"📂 Loaded query pipeline from {query_file}")


def main():
    """
    Main demonstration function
    """
    print("🚀 Dense Document Search - Built-in Example")
    print("=" * 60)
    print("Based on Haystack's e2e/pipelines/test_dense_doc_search.py")
    print("Uses SentenceTransformers for semantic search (no API keys needed)")

    # Initialize demo
    demo = DenseSearchDemo()

    # Check if we can create pipelines
    try:
        # Create pipelines
        demo.create_indexing_pipeline()
        demo.create_query_pipeline()

        # Index sample data
        print("\n" + "="*60)
        indexing_result = demo.index_sample_files()

        if indexing_result.get("writer", {}).get("documents_written", 0) > 0:
            # Test queries (from e2e test)
            test_queries = [
                "Who lives in Rome?",
                "Tell me about France",
                "What is the capital of Germany?",
                "Information about Japan",
                "London England"
            ]

            print("\n" + "="*60)
            print("🧪 Testing Semantic Search")
            print("-" * 60)

            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: '{query}'")

                search_result = demo.search(query, top_k=3)
                documents = search_result["embedding_retriever"]["documents"]

                if documents:
                    for j, doc in enumerate(documents, 1):
                        score = getattr(doc, 'score', 0)
                        preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                        print(f"   {j}. Score: {score:.3f} - {preview}")
                else:
                    print("   No results found")

            # Save pipelines for later use
            output_dir = Path(__file__).parent / "pipeline_exports"
            print(f"\n📁 Saving pipelines to: {output_dir}")
            demo.save_pipelines(output_dir)

        else:
            print("❌ No documents were indexed")

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install sentence-transformers")
        print("pip install torch")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Dense search demo completed!")
    print("\n💡 This example shows:")
    print("   ✅ How to use Haystack's built-in pipeline patterns")
    print("   ✅ Semantic search without external APIs")
    print("   ✅ Document processing and splitting")
    print("   ✅ Pipeline serialization and loading")
    print("   ✅ Production-ready component combinations")


if __name__ == "__main__":
    main()
