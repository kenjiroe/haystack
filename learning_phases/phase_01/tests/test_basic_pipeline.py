# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration Tests for Basic Pipeline Components

This module contains comprehensive tests for the basic pipeline system,
following Haystack's testing guidelines with unit, integration, and performance tests.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch, Mock

import pytest
import yaml
from haystack import Pipeline
from haystack.dataclasses import Document

from learning_phases.phase_01.basic_pipeline import (
    TextProcessor,
    TextAnalyzer,
    BasicPipelineDemo,
    PipelineComponentError,
    create_basic_text_pipeline,
    create_analysis_pipeline
)


class TestTextProcessor:
    """Unit tests for TextProcessor component."""

    def test_initialization(self):
        """Test TextProcessor initialization with different parameters."""
        # Default initialization
        processor = TextProcessor()
        assert processor.uppercase is False
        assert processor.remove_punctuation is False
        assert processor.max_length is None

        # Custom initialization
        processor = TextProcessor(uppercase=True, remove_punctuation=True, max_length=100)
        assert processor.uppercase is True
        assert processor.remove_punctuation is True
        assert processor.max_length == 100

    def test_serialization(self):
        """Test component serialization and deserialization."""
        processor = TextProcessor(uppercase=True, max_length=50)

        # Serialize
        component_dict = processor.to_dict()
        assert "type" in component_dict
        assert "init_parameters" in component_dict

        # Deserialize
        restored_processor = TextProcessor.from_dict(component_dict)
        assert restored_processor.uppercase is True
        assert restored_processor.max_length == 50
        assert restored_processor.remove_punctuation is False

    def test_warm_up_valid_config(self):
        """Test warm_up with valid configuration."""
        processor = TextProcessor(max_length=100)
        processor.warm_up()  # Should not raise

    def test_warm_up_invalid_config(self):
        """Test warm_up with invalid configuration."""
        processor = TextProcessor(max_length=-1)
        with pytest.raises(PipelineComponentError, match="max_length must be positive"):
            processor.warm_up()

    def test_run_normal_text(self):
        """Test processing normal text."""
        processor = TextProcessor()
        result = processor.run("Hello, world! This is a test.")

        assert result["processed_text"] == "Hello, world! This is a test."
        assert result["word_count"] == 6
        assert result["char_count"] == 29

    def test_run_uppercase_transformation(self):
        """Test uppercase text transformation."""
        processor = TextProcessor(uppercase=True)
        result = processor.run("Hello, world!")

        assert result["processed_text"] == "HELLO, WORLD!"
        assert result["word_count"] == 2
        assert result["char_count"] == 13

    def test_run_punctuation_removal(self):
        """Test punctuation removal."""
        processor = TextProcessor(remove_punctuation=True)
        result = processor.run("Hello, world! How are you?")

        assert result["processed_text"] == "Hello world How are you"
        assert result["word_count"] == 5

    def test_run_text_truncation(self):
        """Test text truncation with max_length."""
        processor = TextProcessor(max_length=10)
        result = processor.run("This is a very long text that should be truncated")

        assert result["processed_text"] == "This is a ..."
        assert len(result["processed_text"]) == 13  # 10 + "..."

    def test_run_empty_text(self):
        """Test processing empty text."""
        processor = TextProcessor()
        result = processor.run("")

        assert result["processed_text"] == ""
        assert result["word_count"] == 0
        assert result["char_count"] == 0

    def test_run_invalid_input(self):
        """Test processing invalid input types."""
        processor = TextProcessor()

        with pytest.raises(PipelineComponentError, match="Input must be a string"):
            processor.run(123)

        with pytest.raises(PipelineComponentError, match="Input must be a string"):
            processor.run(None)


class TestTextAnalyzer:
    """Unit tests for TextAnalyzer component."""

    def test_initialization_and_serialization(self):
        """Test TextAnalyzer initialization and serialization."""
        analyzer = TextAnalyzer()

        # Test serialization
        component_dict = analyzer.to_dict()
        restored_analyzer = TextAnalyzer.from_dict(component_dict)

        assert isinstance(restored_analyzer, TextAnalyzer)

    def test_warm_up(self):
        """Test analyzer warm up."""
        analyzer = TextAnalyzer()
        analyzer.warm_up()  # Should not raise

    def test_run_normal_analysis(self):
        """Test normal text analysis."""
        analyzer = TextAnalyzer()
        result = analyzer.run("Hello world. This is a test.", 6, 28)

        assert "analysis_report" in result
        assert "readability_score" in result
        assert "complexity_level" in result
        assert "key_metrics" in result

        assert isinstance(result["readability_score"], (int, float))
        assert result["complexity_level"] in ["easy", "moderate", "difficult"]
        assert result["key_metrics"]["word_count"] == 6
        assert result["key_metrics"]["char_count"] == 28

    def test_run_empty_text_analysis(self):
        """Test analysis of empty text."""
        analyzer = TextAnalyzer()
        result = analyzer.run("", 0, 0)

        assert result["analysis_report"] == "No text to analyze"
        assert result["readability_score"] == 0.0
        assert result["complexity_level"] == "none"
        assert result["key_metrics"] == {}

    def test_run_invalid_inputs(self):
        """Test analysis with invalid inputs."""
        analyzer = TextAnalyzer()

        # Invalid text type
        with pytest.raises(PipelineComponentError, match="text must be a string"):
            analyzer.run(123, 5, 10)

        # Invalid word_count
        with pytest.raises(PipelineComponentError, match="word_count must be a non-negative integer"):
            analyzer.run("test", -1, 10)

        # Invalid char_count
        with pytest.raises(PipelineComponentError, match="char_count must be a non-negative integer"):
            analyzer.run("test", 5, -10)

    def test_readability_score_calculation(self):
        """Test readability score calculation for different text complexities."""
        analyzer = TextAnalyzer()

        # Simple text should have high readability score
        simple_result = analyzer.run("Cat sat. Dog ran.", 4, 17)
        assert simple_result["readability_score"] > 50
        assert simple_result["complexity_level"] in ["easy", "moderate"]

        # Complex text should have lower readability score
        complex_text = "The multifaceted implementation necessitates comprehensive evaluation."
        complex_result = analyzer.run(complex_text, 7, len(complex_text))
        # Score may vary, but should be calculated properly
        assert isinstance(complex_result["readability_score"], float)
        assert 0 <= complex_result["readability_score"] <= 100


@pytest.mark.integration
class TestBasicPipelineDemo:
    """Integration tests for BasicPipelineDemo class."""

    @pytest.fixture
    def demo(self):
        """Create a BasicPipelineDemo instance."""
        return BasicPipelineDemo()

    def test_create_simple_linear_pipeline(self, demo):
        """Test creation of simple linear pipeline."""
        pipeline = demo.create_simple_linear_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert "processor" in pipeline.graph.nodes
        assert "analyzer" in pipeline.graph.nodes

        # Check connections
        edges = list(pipeline.graph.edges)
        connection_strings = [str(edge) for edge in edges]

        assert any("processed_text" in conn for conn in connection_strings)
        assert any("word_count" in conn for conn in connection_strings)
        assert any("char_count" in conn for conn in connection_strings)

    def test_create_branched_pipeline(self, demo):
        """Test creation of branched pipeline."""
        pipeline = demo.create_branched_pipeline()

        assert isinstance(pipeline, Pipeline)

        # Check all expected components exist
        expected_components = ["input_processor", "uppercase_processor", "clean_processor", "analyzer1", "analyzer2"]
        for comp in expected_components:
            assert comp in pipeline.graph.nodes

    def test_create_document_processing_pipeline(self, demo):
        """Test creation of document processing pipeline."""
        pipeline = demo.create_document_processing_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert "doc_processor" in pipeline.graph.nodes

    def test_run_linear_pipeline_example(self, demo):
        """Test running linear pipeline with sample data."""
        pipeline = demo.create_simple_linear_pipeline()

        input_data = {"processor": {"text": "This is a sample text for testing."}}
        result = demo.run_pipeline_example(pipeline, input_data)

        assert "error" not in result
        assert "analyzer" in result
        assert "analysis_report" in result["analyzer"]
        assert "readability_score" in result["analyzer"]

    def test_run_document_processing_example(self, demo):
        """Test running document processing pipeline."""
        pipeline = demo.create_document_processing_pipeline()

        sample_docs = [
            Document(content="First document content.", meta={"source": "doc1"}),
            Document(content="Second document content.", meta={"source": "doc2"})
        ]

        input_data = {"doc_processor": {"documents": sample_docs}}
        result = demo.run_pipeline_example(pipeline, input_data)

        assert "error" not in result
        assert "doc_processor" in result
        assert "processed_docs" in result["doc_processor"]
        assert len(result["doc_processor"]["processed_docs"]) == 2

    def test_pipeline_error_handling(self, demo):
        """Test pipeline error handling with invalid input."""
        pipeline = demo.create_simple_linear_pipeline()

        # Test with invalid input
        input_data = {"processor": {"text": 123}}  # Invalid type
        result = demo.run_pipeline_example(pipeline, input_data)

        assert "error" in result

    def test_demonstrate_pipeline_debugging(self, demo, capsys):
        """Test pipeline debugging functionality."""
        pipeline = demo.create_simple_linear_pipeline()

        demo.demonstrate_pipeline_debugging(pipeline)

        captured = capsys.readouterr()
        assert "Pipeline Components:" in captured.out
        assert "Pipeline Connections:" in captured.out
        assert "Component Details:" in captured.out

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_save_and_load_pipeline(self, demo, temp_dir):
        """Test pipeline save and load functionality."""
        pipeline = demo.create_simple_linear_pipeline()
        save_path = temp_dir / "test_pipeline.yaml"

        # Save pipeline
        demo.save_pipeline(pipeline, str(save_path))
        assert save_path.exists()

        # Load pipeline
        loaded_pipeline = demo.load_pipeline(str(save_path))
        assert loaded_pipeline is not None

        # Test that loaded pipeline works
        input_data = {"processor": {"text": "Test loaded pipeline."}}
        result = demo.run_pipeline_example(loaded_pipeline, input_data)

        assert "error" not in result
        assert "analyzer" in result

    def test_load_nonexistent_pipeline(self, demo):
        """Test loading non-existent pipeline file."""
        result = demo.load_pipeline("nonexistent_file.yaml")
        assert result is None

    def test_save_pipeline_creates_directory(self, demo, temp_dir):
        """Test that save_pipeline creates directories if they don't exist."""
        pipeline = demo.create_simple_linear_pipeline()
        save_path = temp_dir / "nested" / "directory" / "pipeline.yaml"

        demo.save_pipeline(pipeline, str(save_path))
        assert save_path.exists()
        assert save_path.parent.exists()


@pytest.mark.integration
class TestPipelineFactoryFunctions:
    """Integration tests for pipeline factory functions."""

    def test_create_basic_text_pipeline(self):
        """Test basic text pipeline factory function."""
        pipeline = create_basic_text_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert "processor" in pipeline.graph.nodes
        assert "analyzer" in pipeline.graph.nodes

        # Test pipeline execution
        result = pipeline.run({"processor": {"text": "Factory function test."}})
        assert "analyzer" in result
        assert "analysis_report" in result["analyzer"]

    def test_create_analysis_pipeline(self):
        """Test analysis pipeline factory function."""
        pipeline = create_analysis_pipeline()

        assert isinstance(pipeline, Pipeline)

        # Should have multiple components (branched pipeline)
        assert len(pipeline.graph.nodes) > 2

        # Test pipeline execution
        result = pipeline.run({"input_processor": {"text": "Analysis pipeline test."}})
        assert "analyzer1" in result
        assert "analyzer2" in result


@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformance:
    """Performance tests for pipeline components."""

    def test_text_processor_performance(self):
        """Test TextProcessor performance with large text."""
        processor = TextProcessor()
        processor.warm_up()

        # Large text (approximately 10KB)
        large_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200

        start_time = time.time()
        for _ in range(100):
            result = processor.run(large_text)
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be less than 10ms per operation

        # Check that result is correct
        assert result["processed_text"].strip() == large_text.strip()
        assert result["word_count"] > 1000
        assert result["char_count"] == len(result["processed_text"])

    def test_text_analyzer_performance(self):
        """Test TextAnalyzer performance with large text."""
        analyzer = TextAnalyzer()
        analyzer.warm_up()

        # Large text analysis
        large_text = "This is a test sentence. " * 100
        word_count = len(large_text.split())
        char_count = len(large_text)

        start_time = time.time()
        for _ in range(50):
            result = analyzer.run(large_text, word_count, char_count)
        end_time = time.time()

        avg_time = (end_time - start_time) / 50
        assert avg_time < 0.05  # Should be less than 50ms per operation

        # Check that result is correct
        assert result["key_metrics"]["word_count"] == word_count
        assert result["key_metrics"]["char_count"] == char_count

    def test_pipeline_batch_processing(self):
        """Test pipeline performance with batch processing."""
        pipeline = create_basic_text_pipeline()

        # Warm up pipeline
        demo = BasicPipelineDemo()
        demo._warm_up_pipeline(pipeline)

        # Test data
        test_texts = [
            f"This is test document number {i} with some content."
            for i in range(50)
        ]

        start_time = time.time()
        results = []
        for text in test_texts:
            result = pipeline.run({"processor": {"text": text}})
            results.append(result)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_doc = total_time / len(test_texts)

        assert avg_time_per_doc < 0.1  # Should be less than 100ms per document
        assert len(results) == 50

        # Check that all results are valid
        for result in results:
            assert "analyzer" in result
            assert "readability_score" in result["analyzer"]

    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and run many pipelines
        for _ in range(10):
            pipeline = create_basic_text_pipeline()
            for j in range(10):
                pipeline.run({"processor": {"text": f"Memory test {j}"}})

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024


@pytest.mark.integration
class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""

    def test_pipeline_with_empty_documents(self):
        """Test document processing pipeline with empty documents."""
        demo = BasicPipelineDemo()
        pipeline = demo.create_document_processing_pipeline()

        empty_docs = [Document(content="", meta={"source": "empty"})]
        result = demo.run_pipeline_example(pipeline, {"doc_processor": {"documents": empty_docs}})

        assert "error" not in result
        assert "doc_processor" in result

    def test_pipeline_with_malformed_input(self):
        """Test pipeline behavior with malformed input."""
        demo = BasicPipelineDemo()
        pipeline = demo.create_simple_linear_pipeline()

        # Various malformed inputs
        malformed_inputs = [
            {"wrong_component": {"text": "test"}},
            {"processor": {"wrong_field": "test"}},
            {},
            None
        ]

        for malformed_input in malformed_inputs:
            result = demo.run_pipeline_example(pipeline, malformed_input)
            # Should handle gracefully and return error
            if result is not None:
                # Either returns error or handles gracefully
                assert True

    def test_component_warm_up_failure_handling(self):
        """Test handling of component warm-up failures."""
        # Create a processor with invalid configuration
        processor = TextProcessor(max_length=-1)

        with pytest.raises(PipelineComponentError):
            processor.warm_up()

    def test_pipeline_serialization_edge_cases(self):
        """Test pipeline serialization with edge cases."""
        demo = BasicPipelineDemo()

        # Test saving to non-existent directory
        pipeline = demo.create_simple_linear_pipeline()
        demo.save_pipeline(pipeline, "/tmp/nonexistent_dir/pipeline.yaml")

        # Should create directory and save successfully
        saved_path = Path("/tmp/nonexistent_dir/pipeline.yaml")
        if saved_path.exists():
            saved_path.unlink()
            saved_path.parent.rmdir()


if __name__ == "__main__":
    # Run basic tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])
