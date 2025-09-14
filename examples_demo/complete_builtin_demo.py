# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Complete Built-in Pipeline Demo
===============================

This is a comprehensive demonstration of Haystack's built-in capabilities
using actual sample data and production-ready components.

Features:
- Uses real sample data from e2e/samples/test_documents/
- Document processing pipeline (convert, clean, split, store)
- BM25 search pipeline (no ML dependencies)
- Pipeline serialization and loading
- Component analysis and statistics
- Multiple document processing
- Search result ranking and scoring

This demo shows how powerful Haystack is out-of-the-box without
requiring any external APIs or additional ML dependencies.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from haystack import Document, Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


class CompletePipelineDemo:
    """
    Complete pipeline demonstration using Haystack's built-in components
    """

    def __init__(self):
        """Initialize the demo system"""
        self.document_store = InMemoryDocumentStore()
        self.indexing_pipeline = None
        self.search_pipeline = None
        self.processed_files = []
        self.stats = {}

        print("🚀 Initializing Complete Built-in Pipeline Demo")

    def create_indexing_pipeline(self) -> Pipeline:
        """Create document indexing pipeline"""
        print("📝 Creating indexing pipeline...")

        indexing_pipeline = Pipeline()

        # File type routing
        indexing_pipeline.add_component(
            instance=FileTypeRouter(mime_types=["text/plain"]),
            name="file_router"
        )

        # Text file conversion
        indexing_pipeline.add_component(
            instance=TextFileToDocument(),
            name="text_converter"
        )

        # Document cleaning
        indexing_pipeline.add_component(
            instance=DocumentCleaner(
                remove_empty_lines=True,
                remove_extra_whitespaces=True,
                remove_repeated_substrings=False
            ),
            name="cleaner"
        )

        # Document splitting
        indexing_pipeline.add_component(
            instance=DocumentSplitter(
                split_by="sentence",
                split_length=3,
                split_overlap=1
            ),
            name="splitter"
        )

        # Document storage
        indexing_pipeline.add_component(
            instance=DocumentWriter(document_store=self.document_store),
            name="writer"
        )

        # Connect components
        indexing_pipeline.connect("file_router.text/plain", "text_converter.sources")
        indexing_pipeline.connect("text_converter.documents", "cleaner.documents")
        indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        indexing_pipeline.connect("splitter.documents", "writer.documents")

        self.indexing_pipeline = indexing_pipeline

        print("✅ Indexing pipeline created!")
        print(f"   Components: {len(indexing_pipeline.graph.nodes)}")
        print(f"   Connections: {len(indexing_pipeline.graph.edges)}")

        return indexing_pipeline

    def create_search_pipeline(self) -> Pipeline:
        """Create document search pipeline"""
        print("🔍 Creating search pipeline...")

        search_pipeline = Pipeline()
        search_pipeline.add_component(
            instance=InMemoryBM25Retriever(
                document_store=self.document_store,
                top_k=10
            ),
            name="retriever"
        )

        self.search_pipeline = search_pipeline

        print("✅ Search pipeline created!")
        return search_pipeline

    def process_sample_documents(self) -> Dict[str, Any]:
        """Process all sample documents"""
        print("\n📚 Processing Sample Documents")
        print("=" * 50)

        # Find sample files
        samples_path = Path(__file__).parent.parent / "e2e" / "samples" / "test_documents"

        if not samples_path.exists():
            print(f"❌ Sample path not found: {samples_path}")
            return self._create_fallback_documents()

        sample_files = list(samples_path.glob("*.txt"))

        print(f"📄 Found {len(sample_files)} sample files:")
        total_size = 0
        for file in sample_files:
            size_kb = file.stat().st_size / 1024
            total_size += size_kb
            print(f"   - {file.name} ({size_kb:.1f} KB)")

        print(f"   Total size: {total_size:.1f} KB")

        if not sample_files:
            print("❌ No sample files found")
            return self._create_fallback_documents()

        # Create indexing pipeline if not exists
        if not self.indexing_pipeline:
            self.create_indexing_pipeline()

        # Process files
        try:
            print("\n🔄 Running indexing pipeline...")
            result = self.indexing_pipeline.run({"file_router": {"sources": sample_files}})

            docs_written = result["writer"]["documents_written"]
            print(f"✅ Successfully processed {docs_written} document chunks")

            self.processed_files = [f.name for f in sample_files]
            self.stats = {
                "files_processed": len(sample_files),
                "documents_created": docs_written,
                "total_size_kb": total_size,
                "average_file_size": total_size / len(sample_files) if sample_files else 0
            }

            return result

        except Exception as e:
            print(f"❌ Processing failed: {e}")
            print("Falling back to manual document creation...")
            return self._create_fallback_documents()

    def _create_fallback_documents(self) -> Dict[str, Any]:
        """Create fallback documents if sample files not available"""
        print("📝 Creating fallback documents...")

        fallback_docs = [
            Document(
                content="Haystack is an end-to-end framework for building LLM applications. "
                        "It provides components for document processing, retrieval, and generation. "
                        "The framework supports various document stores and retrieval methods."
            ),
            Document(
                content="Document preprocessing is a crucial step in information retrieval systems. "
                        "It involves cleaning text, removing unnecessary whitespace, and splitting "
                        "long documents into manageable chunks for better searchability."
            ),
            Document(
                content="BM25 is a ranking function used in information retrieval systems. "
                        "It scores documents based on keyword relevance and document length. "
                        "BM25 is widely used in search engines and provides good baseline performance."
            ),
            Document(
                content="Pipeline architecture allows for modular and flexible document processing. "
                        "Components can be easily connected to create complex workflows. "
                        "Each component has specific inputs and outputs that define the data flow."
            ),
            Document(
                content="Information retrieval systems help users find relevant documents from large collections. "
                        "They use various techniques including keyword matching, semantic search, "
                        "and machine learning to rank and retrieve the most relevant results."
            )
        ]

        self.document_store.write_documents(fallback_docs)

        self.stats = {
            "files_processed": 0,
            "documents_created": len(fallback_docs),
            "total_size_kb": sum(len(doc.content) for doc in fallback_docs) / 1024,
            "fallback_mode": True
        }

        print(f"✅ Created {len(fallback_docs)} fallback documents")
        return {"writer": {"documents_written": len(fallback_docs)}}

    def demonstrate_search_capabilities(self):
        """Demonstrate search capabilities with various queries"""
        print("\n🔍 Demonstrating Search Capabilities")
        print("=" * 50)

        if not self.search_pipeline:
            self.create_search_pipeline()

        # Test different types of queries
        test_queries = [
            # Factual queries
            ("Haystack framework", "Looking for information about the Haystack framework"),
            ("document processing", "Searching for document processing information"),
            ("BM25 ranking", "Finding content about BM25 ranking algorithm"),

            # Conceptual queries
            ("information retrieval", "Searching for information retrieval concepts"),
            ("pipeline architecture", "Looking for pipeline and architecture details"),

            # Specific terms (if using real sample data)
            ("culture", "Searching for cultural content"),
            ("European", "Looking for European-related content"),
            ("philosophy", "Searching for philosophical content"),
        ]

        print(f"Testing {len(test_queries)} different query types:")

        search_results = []
        for i, (query, description) in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print(f"   Context: {description}")

            try:
                result = self.search_pipeline.run({"retriever": {"query": query, "top_k": 3}})
                documents = result["retriever"]["documents"]

                if documents:
                    print(f"   📊 Found {len(documents)} results:")
                    for j, doc in enumerate(documents, 1):
                        score = getattr(doc, 'score', 0)
                        preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                        print(f"      {j}. Score: {score:.3f}")
                        print(f"         {preview}")

                    # Store best result for analysis
                    search_results.append({
                        "query": query,
                        "results_count": len(documents),
                        "best_score": documents[0].score if documents else 0
                    })
                else:
                    print("   ❌ No results found")
                    search_results.append({
                        "query": query,
                        "results_count": 0,
                        "best_score": 0
                    })

            except Exception as e:
                print(f"   ❌ Search failed: {e}")

        return search_results

    def analyze_system_performance(self, search_results: List[Dict]):
        """Analyze system performance and provide insights"""
        print("\n📊 System Performance Analysis")
        print("=" * 50)

        # Document store statistics
        total_docs = self.document_store.count_documents()
        all_docs = self.document_store.filter_documents()

        doc_lengths = [len(doc.content) for doc in all_docs]
        avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        min_length = min(doc_lengths) if doc_lengths else 0
        max_length = max(doc_lengths) if doc_lengths else 0

        print(f"📚 Document Store Statistics:")
        print(f"   Total documents: {total_docs}")
        print(f"   Average length: {avg_length:.1f} characters")
        print(f"   Length range: {min_length} - {max_length} characters")

        # Search performance
        successful_searches = [r for r in search_results if r["results_count"] > 0]
        success_rate = len(successful_searches) / len(search_results) if search_results else 0
        avg_results = sum(r["results_count"] for r in search_results) / len(search_results) if search_results else 0
        avg_score = sum(r["best_score"] for r in successful_searches) / len(successful_searches) if successful_searches else 0

        print(f"\n🎯 Search Performance:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average results per query: {avg_results:.1f}")
        print(f"   Average best score: {avg_score:.3f}")

        # Processing statistics
        print(f"\n⚙️ Processing Statistics:")
        for key, value in self.stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

        # Top performing queries
        top_queries = sorted(search_results, key=lambda x: x["best_score"], reverse=True)[:3]
        print(f"\n🏆 Top Performing Queries:")
        for i, query_result in enumerate(top_queries, 1):
            print(f"   {i}. '{query_result['query']}' - Score: {query_result['best_score']:.3f}")

    def save_pipeline_configuration(self):
        """Save pipeline configurations for reuse"""
        output_dir = Path(__file__).parent / "pipeline_exports"
        output_dir.mkdir(exist_ok=True)

        print(f"\n💾 Saving Pipeline Configuration")
        print("=" * 50)

        # Save indexing pipeline
        if self.indexing_pipeline:
            indexing_file = output_dir / "complete_indexing_pipeline.yaml"
            with open(indexing_file, "w") as f:
                self.indexing_pipeline.dump(f)
            print(f"📄 Indexing pipeline: {indexing_file}")

        # Save search pipeline
        if self.search_pipeline:
            search_file = output_dir / "complete_search_pipeline.json"
            with open(search_file, "w") as f:
                json.dump(self.search_pipeline.to_dict(), f, indent=2)
            print(f"📄 Search pipeline: {search_file}")

        # Save statistics
        stats_file = output_dir / "demo_statistics.json"
        with open(stats_file, "w") as f:
            json.dump({
                "processed_files": self.processed_files,
                "statistics": self.stats,
                "document_count": self.document_store.count_documents()
            }, f, indent=2)
        print(f"📊 Statistics: {stats_file}")

        print(f"\n✅ All configurations saved to: {output_dir}")

    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🎯 Complete Built-in Pipeline Demo")
        print("=" * 60)
        print("Showcasing Haystack's production-ready capabilities")
        print("✅ No external APIs required")
        print("✅ No ML dependencies needed")
        print("✅ Uses real sample data")
        print("✅ Production-ready components")

        try:
            # Create pipelines
            self.create_indexing_pipeline()
            self.create_search_pipeline()

            # Process documents
            processing_result = self.process_sample_documents()

            if processing_result.get("writer", {}).get("documents_written", 0) > 0:
                # Demonstrate search
                search_results = self.demonstrate_search_capabilities()

                # Analyze performance
                self.analyze_system_performance(search_results)

                # Save configuration
                self.save_pipeline_configuration()

                print("\n🎉 Demo completed successfully!")
                print("\n💡 Key Achievements:")
                print(f"   ✅ Processed {self.stats.get('documents_created', 0)} documents")
                print(f"   ✅ Built complete indexing and search pipelines")
                print(f"   ✅ Demonstrated keyword search capabilities")
                print(f"   ✅ Analyzed system performance")
                print(f"   ✅ Saved reusable pipeline configurations")

            else:
                print("❌ No documents were processed successfully")

        except Exception as e:
            print(f"\n❌ Demo encountered an error: {e}")
            import traceback
            traceback.print_exc()

        print("\n🌟 This demo showcases:")
        print("   📦 Haystack's built-in component ecosystem")
        print("   🔧 Production-ready pipeline patterns")
        print("   📊 Real document processing workflows")
        print("   🔍 Effective search and retrieval")
        print("   💾 Pipeline serialization and reuse")
        print("   📈 Performance analysis and optimization")


def main():
    """Main function to run the complete demo"""
    demo = CompletePipelineDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
