# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Basic Pipeline Creation for Haystack Learning Project

This module demonstrates the creation and usage of basic Haystack pipelines,
following the learning spec for Phase 1, Task 2.2.

Key learning objectives:
- Understanding pipeline construction
- Component connections
- Data flow between components
- Error handling in pipelines
- Pipeline execution and results
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

from haystack import Pipeline, component, logging, default_to_dict, default_from_dict
from haystack.core.errors import ComponentError
from haystack.dataclasses import Document

from learning_phases.phase_01.learning_tracker import LearningTracker

logger = logging.getLogger(__name__)


class PipelineComponentError(ComponentError):
    """Exception raised when pipeline components encounter errors."""
    pass


@component
class TextProcessor:
    """
    A basic text processing component for learning pipeline fundamentals.

    This component demonstrates:
    - Component input/output definition
    - Text transformation logic
    - Proper error handling
    - Logging best practices
    - Haystack serialization support
    """

    def __init__(self,
                 uppercase: bool = False,
                 remove_punctuation: bool = False,
                 max_length: Optional[int] = None):
        """
        Initialize the text processor.

        :param uppercase: Convert text to uppercase
        :param remove_punctuation: Remove punctuation from text
        :param max_length: Maximum text length (truncate if longer)
        """
        self.uppercase = uppercase
        self.remove_punctuation = remove_punctuation
        self.max_length = max_length

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with the component's configuration.
        """
        return default_to_dict(
            self,
            uppercase=self.uppercase,
            remove_punctuation=self.remove_punctuation,
            max_length=self.max_length,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextProcessor":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize the component from.
        :returns: Deserialized component.
        """
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Warm up the component by validating configuration.
        """
        if self.max_length is not None and self.max_length <= 0:
            raise PipelineComponentError("max_length must be positive if specified")
        logger.info("TextProcessor warmed up successfully")

    @component.output_types(processed_text=str, word_count=int, char_count=int)
    def run(self, text: str) -> Dict[str, Any]:
        """
        Process the input text according to configuration.

        :param text: Input text to process
        :returns: Dictionary with processed text and statistics
        :raises PipelineComponentError: If processing fails
        """
        if not isinstance(text, str):
            raise PipelineComponentError("Input must be a string")

        if not text:
            logger.warning("Empty text input received")
            return {
                "processed_text": "",
                "word_count": 0,
                "char_count": 0
            }

        try:
            processed = text.strip()

            # Apply transformations
            if self.remove_punctuation:
                import string
                processed = processed.translate(str.maketrans('', '', string.punctuation))

            if self.uppercase:
                processed = processed.upper()

            if self.max_length and len(processed) > self.max_length:
                processed = processed[:self.max_length] + "..."
                logger.info(f"Text truncated to {self.max_length} characters")

            # Calculate statistics
            word_count = len(processed.split()) if processed else 0
            char_count = len(processed)

            logger.debug(f"Processed text: {word_count} words, {char_count} characters")

            return {
                "processed_text": processed,
                "word_count": word_count,
                "char_count": char_count
            }

        except Exception as e:
            raise PipelineComponentError(f"Text processing failed: {e}") from e


@component
class TextAnalyzer:
    """
    A component that analyzes text characteristics.

    Demonstrates:
    - Multiple output types
    - Statistical calculations
    - Component chaining preparation
    - Proper serialization support
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with the component's configuration.
        """
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextAnalyzer":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize the component from.
        :returns: Deserialized component.
        """
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Warm up the component.
        """
        logger.info("TextAnalyzer warmed up successfully")

    @component.output_types(
        analysis_report=str,
        readability_score=float,
        complexity_level=str,
        key_metrics=Dict[str, Any]
    )
    def run(self, text: str, word_count: int, char_count: int) -> Dict[str, Any]:
        """
        Analyze text characteristics and generate report.

        :param text: Input text to analyze
        :param word_count: Number of words (from previous component)
        :param char_count: Number of characters (from previous component)
        :returns: Analysis results and metrics
        :raises PipelineComponentError: If analysis fails
        """
        # Validate inputs
        if not isinstance(text, str):
            raise PipelineComponentError("text must be a string")
        if not isinstance(word_count, int) or word_count < 0:
            raise PipelineComponentError("word_count must be a non-negative integer")
        if not isinstance(char_count, int) or char_count < 0:
            raise PipelineComponentError("char_count must be a non-negative integer")

        if not text:
            return {
                "analysis_report": "No text to analyze",
                "readability_score": 0.0,
                "complexity_level": "none",
                "key_metrics": {}
            }

        try:
            # Calculate basic metrics
            sentences = text.count('.') + text.count('!') + text.count('?')
            sentences = max(sentences, 1)  # Avoid division by zero

            avg_words_per_sentence = word_count / sentences
            avg_chars_per_word = char_count / max(word_count, 1)

            # Simple readability score (Flesch-like approximation)
            readability_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (avg_chars_per_word / 4.7))
            readability_score = max(0, min(100, readability_score))  # Clamp to 0-100

            # Determine complexity level
            if readability_score >= 60:
                complexity_level = "easy"
            elif readability_score >= 30:
                complexity_level = "moderate"
            else:
                complexity_level = "difficult"

            # Generate analysis report
            analysis_report = f"""
📊 Text Analysis Report
======================

📝 Basic Metrics:
   • Words: {word_count}
   • Characters: {char_count}
   • Sentences: {sentences}

📈 Advanced Metrics:
   • Avg words/sentence: {avg_words_per_sentence:.1f}
   • Avg chars/word: {avg_chars_per_word:.1f}
   • Readability score: {readability_score:.1f}/100

🎯 Assessment:
   • Complexity: {complexity_level.title()}
   • Reading level: {'Easy to read' if readability_score >= 60 else 'Moderate difficulty' if readability_score >= 30 else 'Difficult to read'}
""".strip()

            key_metrics = {
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentences,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_chars_per_word": avg_chars_per_word,
                "readability_score": readability_score,
                "complexity_level": complexity_level
            }

            logger.info(f"Text analysis completed: {complexity_level} complexity, {readability_score:.1f} readability")

            return {
                "analysis_report": analysis_report,
                "readability_score": readability_score,
                "complexity_level": complexity_level,
                "key_metrics": key_metrics
            }

        except Exception as e:
            raise PipelineComponentError(f"Text analysis failed: {e}") from e


class BasicPipelineDemo:
    """
    Demonstration class for basic pipeline concepts and patterns.

    This class shows how to:
    - Create different types of pipelines
    - Connect components properly
    - Handle pipeline execution
    - Debug pipeline issues
    - Save and load pipelines
    """

    def __init__(self):
        """Initialize the demo with a learning tracker."""
        self.tracker = LearningTracker(
            progress_file="basic_pipeline_progress.json",
            auto_save=True
        )

    def create_simple_linear_pipeline(self) -> Pipeline:
        """
        Create a simple linear pipeline: TextProcessor -> TextAnalyzer.

        This demonstrates:
        - Linear pipeline structure
        - Component-to-component connections
        - Data flow through pipeline

        :returns: Configured pipeline
        """
        logger.info("Creating simple linear pipeline")

        # Create pipeline
        pipeline = Pipeline()

        # Add components
        pipeline.add_component("processor", TextProcessor(uppercase=False, remove_punctuation=False))
        pipeline.add_component("analyzer", TextAnalyzer())

        # Connect components
        pipeline.connect("processor.processed_text", "analyzer.text")
        pipeline.connect("processor.word_count", "analyzer.word_count")
        pipeline.connect("processor.char_count", "analyzer.char_count")

        logger.info("Linear pipeline created successfully")
        return pipeline

    def create_branched_pipeline(self) -> Pipeline:
        """
        Create a branched pipeline with parallel processing paths.

        This demonstrates:
        - Branched pipeline structure
        - Multiple outputs from single component
        - Parallel processing concepts

        :returns: Configured branched pipeline
        """
        logger.info("Creating branched pipeline")

        pipeline = Pipeline()

        # Add components
        pipeline.add_component("input_processor", TextProcessor())
        pipeline.add_component("uppercase_processor", TextProcessor(uppercase=True))
        pipeline.add_component("clean_processor", TextProcessor(remove_punctuation=True))
        pipeline.add_component("analyzer1", TextAnalyzer())
        pipeline.add_component("analyzer2", TextAnalyzer())

        # Connect input to both processors
        pipeline.connect("input_processor.processed_text", "uppercase_processor.text")
        pipeline.connect("input_processor.processed_text", "clean_processor.text")

        # Connect processors to analyzers
        pipeline.connect("uppercase_processor.processed_text", "analyzer1.text")
        pipeline.connect("uppercase_processor.word_count", "analyzer1.word_count")
        pipeline.connect("uppercase_processor.char_count", "analyzer1.char_count")

        pipeline.connect("clean_processor.processed_text", "analyzer2.text")
        pipeline.connect("clean_processor.word_count", "analyzer2.word_count")
        pipeline.connect("clean_processor.char_count", "analyzer2.char_count")

        logger.info("Branched pipeline created successfully")
        return pipeline

    def create_document_processing_pipeline(self) -> Pipeline:
        """
        Create a pipeline that processes Document objects.

        This demonstrates:
        - Working with Haystack Document objects
        - Document metadata handling
        - More complex data structures

        :returns: Document processing pipeline
        """
        logger.info("Creating document processing pipeline")

        @component
        class DocumentProcessor:
            def to_dict(self) -> Dict[str, Any]:
                return default_to_dict(self)

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "DocumentProcessor":
                return default_from_dict(cls, data)

            def warm_up(self) -> None:
                logger.info("DocumentProcessor warmed up successfully")

            @component.output_types(processed_docs=List[Document])
            def run(self, documents: List[Document]) -> Dict[str, Any]:
                if not isinstance(documents, list):
                    raise PipelineComponentError("documents must be a list")

                processed_docs = []
                try:
                    for doc in documents:
                        if not isinstance(doc, Document):
                            raise PipelineComponentError("All items in documents must be Document objects")

                        # Process document content
                        content = doc.content or ""
                        processed_content = content.strip().replace('\n', ' ')

                        # Create new document with processed content
                        processed_doc = Document(
                            content=processed_content,
                            meta={
                                **doc.meta,
                                "processed": True,
                                "original_length": len(content),
                                "processed_length": len(processed_content)
                            }
                        )
                        processed_docs.append(processed_doc)

                    return {"processed_docs": processed_docs}

                except Exception as e:
                    raise PipelineComponentError(f"Document processing failed: {e}") from e

        pipeline = Pipeline()
        pipeline.add_component("doc_processor", DocumentProcessor())

        logger.info("Document processing pipeline created successfully")
        return pipeline

    def run_pipeline_example(self, pipeline: Pipeline, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a pipeline with error handling and logging.

        :param pipeline: Pipeline to execute
        :param input_data: Input data for pipeline
        :returns: Pipeline execution results
        """
        try:
            logger.info(f"Executing pipeline with input: {list(input_data.keys())}")

            # Warm up pipeline components before execution
            self._warm_up_pipeline(pipeline)

            result = pipeline.run(input_data)
            logger.info("Pipeline executed successfully")
            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {"error": str(e)}

    def _warm_up_pipeline(self, pipeline: Pipeline) -> None:
        """
        Warm up all components in the pipeline.

        :param pipeline: Pipeline to warm up
        """
        try:
            for component_name, component_instance in pipeline.graph.nodes.items():
                if hasattr(component_instance, 'warm_up'):
                    component_instance.warm_up()
                    logger.debug(f"Warmed up component: {component_name}")
        except Exception as e:
            logger.warning(f"Failed to warm up some components: {e}")

    def demonstrate_pipeline_debugging(self, pipeline: Pipeline):
        """
        Demonstrate pipeline debugging techniques.

        :param pipeline: Pipeline to debug
        """
        logger.info("=== Pipeline Debugging Demo ===")

        # Show pipeline graph
        print("Pipeline Components:")
        for name, component in pipeline.graph.nodes.items():
            print(f"  • {name}: {type(component).__name__}")

        print("\nPipeline Connections:")
        for connection in pipeline.graph.edges:
            print(f"  • {connection}")

        # Show component inputs/outputs
        print("\nComponent Details:")
        for name, component in pipeline.graph.nodes.items():
            inputs = getattr(component, '__haystack_input__', None)
            outputs = getattr(component, '__haystack_output__', None)

            if inputs:
                input_sockets = list(inputs._sockets_dict.keys())
                print(f"  • {name} inputs: {input_sockets}")

            if outputs:
                output_sockets = list(outputs._sockets_dict.keys())
                print(f"  • {name} outputs: {output_sockets}")

    def save_pipeline(self, pipeline: Pipeline, filename: str):
        """
        Save pipeline to YAML file.

        :param pipeline: Pipeline to save
        :param filename: Output filename
        """
        try:
            pipeline_dict = pipeline.to_dict()

            # Ensure directory exists
            file_path = Path(filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to YAML file
            with open(file_path, 'w') as f:
                yaml.dump(pipeline_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Pipeline saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")

    def load_pipeline(self, filename: str) -> Optional[Pipeline]:
        """
        Load pipeline from YAML file.

        :param filename: Input filename
        :returns: Loaded pipeline or None if failed
        """
        try:
            file_path = Path(filename)
            if not file_path.exists():
                logger.error(f"Pipeline file does not exist: {filename}")
                return None

            with open(file_path, 'r') as f:
                pipeline_dict = yaml.safe_load(f)

            pipeline = Pipeline.from_dict(pipeline_dict)
            logger.info(f"Pipeline loaded from {filename}")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return None

    def run_learning_exercises(self):
        """
        Run through all basic pipeline learning exercises.
        """
        print("🚀 Starting Basic Pipeline Learning Exercises")
        print("=" * 50)

        # Exercise 1: Simple Linear Pipeline
        print("\n📝 Exercise 1: Simple Linear Pipeline")
        linear_pipeline = self.create_simple_linear_pipeline()

        sample_text = "The quick brown fox jumps over the lazy dog. This is a sample text for testing our pipeline components."

        result = self.run_pipeline_example(linear_pipeline, {"processor": {"text": sample_text}})

        if "analyzer" in result:
            print("✅ Linear pipeline executed successfully!")
            print(result["analyzer"]["analysis_report"])

        # Track progress
        self.tracker.run(
            exercise_name="Simple Linear Pipeline",
            phase="Phase 1",
            total_exercises=4,
            difficulty_level="beginner",
            estimated_time=30
        )

        # Exercise 2: Branched Pipeline
        print("\n📝 Exercise 2: Branched Pipeline")
        branched_pipeline = self.create_branched_pipeline()

        result = self.run_pipeline_example(branched_pipeline, {"input_processor": {"text": sample_text}})

        if "analyzer1" in result and "analyzer2" in result:
            print("✅ Branched pipeline executed successfully!")
            print("Uppercase Analysis:")
            print(result["analyzer1"]["analysis_report"])
            print("\nCleaned Analysis:")
            print(result["analyzer2"]["analysis_report"])

        # Track progress
        self.tracker.run(
            exercise_name="Branched Pipeline",
            phase="Phase 1",
            total_exercises=4,
            difficulty_level="intermediate",
            estimated_time=45
        )

        # Exercise 3: Document Processing
        print("\n📝 Exercise 3: Document Processing Pipeline")
        doc_pipeline = self.create_document_processing_pipeline()

        sample_docs = [
            Document(content="First document content\nwith multiple lines.", meta={"source": "doc1.txt"}),
            Document(content="Second document with different content.", meta={"source": "doc2.txt"})
        ]

        result = self.run_pipeline_example(doc_pipeline, {"doc_processor": {"documents": sample_docs}})

        if "doc_processor" in result:
            print("✅ Document processing pipeline executed successfully!")
            for doc in result["doc_processor"]["processed_docs"]:
                print(f"Document: {doc.meta['source']}")
                print(f"  Original length: {doc.meta['original_length']}")
                print(f"  Processed length: {doc.meta['processed_length']}")

        # Track progress
        self.tracker.run(
            exercise_name="Document Processing Pipeline",
            phase="Phase 1",
            total_exercises=4,
            difficulty_level="intermediate",
            estimated_time=60
        )

        # Exercise 4: Pipeline Debugging
        print("\n📝 Exercise 4: Pipeline Debugging")
        self.demonstrate_pipeline_debugging(linear_pipeline)

        # Track progress
        progress_result = self.tracker.run(
            exercise_name="Pipeline Debugging",
            phase="Phase 1",
            total_exercises=4,
            difficulty_level="advanced",
            estimated_time=30
        )

        print("\n🎯 Learning Progress:")
        print(progress_result["progress_report"])

        if progress_result["completion_rate"] >= 1.0:
            print("\n🎉 Congratulations! You've completed Phase 1 basic pipeline exercises!")
            print("You're ready to move on to Phase 2: Document Processing")


# Utility functions for standalone usage
def create_basic_text_pipeline() -> Pipeline:
    """
    Factory function to create a basic text processing pipeline.

    :returns: Ready-to-use text processing pipeline
    """
    demo = BasicPipelineDemo()
    return demo.create_simple_linear_pipeline()


def create_analysis_pipeline() -> Pipeline:
    """
    Factory function to create a text analysis pipeline.

    :returns: Ready-to-use text analysis pipeline
    """
    demo = BasicPipelineDemo()
    return demo.create_branched_pipeline()


if __name__ == "__main__":
    # Run the complete learning exercise when executed directly
    demo = BasicPipelineDemo()
    demo.run_learning_exercises()

    # Test pipeline serialization
    print("\n🔧 Testing Pipeline Serialization:")
    pipeline = create_basic_text_pipeline()
    demo.save_pipeline(pipeline, "test_pipeline.yaml")
    loaded_pipeline = demo.load_pipeline("test_pipeline.yaml")

    if loaded_pipeline:
        print("✅ Pipeline serialization successful!")

        # Test loaded pipeline
        test_result = demo.run_pipeline_example(
            loaded_pipeline,
            {"processor": {"text": "Test serialization functionality."}}
        )
        if "analyzer" in test_result:
            print("✅ Loaded pipeline execution successful!")
        else:
            print("❌ Loaded pipeline execution failed!")

        # Cleanup
        Path("test_pipeline.yaml").unlink(missing_ok=True)
    else:
        print("❌ Pipeline serialization failed!")
