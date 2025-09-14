#!/usr/bin/env python3
"""
My First Haystack Component - Phase 1 Foundation
===============================================

This is your very first custom Haystack component!
It demonstrates the basic structure and patterns you'll use
throughout your Haystack learning journey.

Author: [Your Name]
Date: [Current Date]
Phase: 1 - Foundation
"""

from haystack import component, logging
from typing import Dict, Any, Optional
from datetime import datetime
import re

# Set up logging for our component
logger = logging.getLogger(__name__)

@component
class HelloHaystack:
    """
    Your very first Haystack component!

    This component demonstrates:
    - Basic component structure with @component decorator
    - Proper output type definitions
    - Input validation and error handling
    - Logging for debugging and monitoring
    - Documentation best practices
    """

    def __init__(self, greeting_style: str = "friendly"):
        """
        Initialize the component with configuration options.

        Args:
            greeting_style: Style of greeting ('friendly', 'formal', 'casual')
        """
        self.greeting_style = greeting_style
        logger.info(f"HelloHaystack component initialized with style: {greeting_style}")

    @component.output_types(
        greeting=str,
        timestamp=str,
        metadata=dict,
        word_count=int
    )
    def run(self, name: str = "World", message: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a personalized greeting with metadata.

        Args:
            name: Name of the person to greet
            message: Optional custom message to include

        Returns:
            Dictionary containing:
            - greeting: The formatted greeting message
            - timestamp: ISO format timestamp
            - metadata: Additional information about the greeting
            - word_count: Number of words in the greeting

        Raises:
            ValueError: If name contains invalid characters
        """
        try:
            # Input validation
            if not name or not isinstance(name, str):
                raise ValueError("Name must be a non-empty string")

            # Clean the name (remove numbers and special chars for this example)
            clean_name = re.sub(r'[^a-zA-Z\s-]', '', name).strip()
            if not clean_name:
                raise ValueError("Name must contain at least one letter")

            # Generate greeting based on style
            greeting_templates = {
                "friendly": f"Hello there, {clean_name}! Welcome to your Haystack learning journey! 🚀",
                "formal": f"Good day, {clean_name}. Welcome to the Haystack framework.",
                "casual": f"Hey {clean_name}! Ready to build some awesome AI stuff?"
            }

            base_greeting = greeting_templates.get(
                self.greeting_style,
                greeting_templates["friendly"]
            )

            # Add custom message if provided
            if message:
                clean_message = message.strip()
                if clean_message:
                    base_greeting += f" {clean_message}"

            # Generate timestamp
            timestamp = datetime.now().isoformat()

            # Calculate word count
            word_count = len(base_greeting.split())

            # Create metadata
            metadata = {
                "component_name": self.__class__.__name__,
                "greeting_style": self.greeting_style,
                "processed_name": clean_name,
                "original_name": name,
                "has_custom_message": message is not None,
                "generation_time": timestamp,
                "component_version": "1.0.0"
            }

            logger.info(f"Generated greeting for '{clean_name}' with {word_count} words")

            return {
                "greeting": base_greeting,
                "timestamp": timestamp,
                "metadata": metadata,
                "word_count": word_count
            }

        except Exception as e:
            logger.error(f"Error generating greeting: {str(e)}")
            # Re-raise with more context
            raise ValueError(f"Failed to generate greeting: {str(e)}")


@component
class TextAnalyzer:
    """
    A more advanced component that analyzes text properties.

    This demonstrates:
    - Multiple input parameters
    - Complex processing logic
    - Comprehensive output data
    - Performance monitoring
    """

    @component.output_types(
        analysis_results=dict,
        processing_time_ms=float,
        text_stats=dict
    )
    def run(self, text: str, analyze_sentiment: bool = False) -> Dict[str, Any]:
        """
        Analyze text for various properties.

        Args:
            text: Text to analyze
            analyze_sentiment: Whether to perform sentiment analysis

        Returns:
            Comprehensive analysis results
        """
        start_time = datetime.now()

        try:
            # Basic text statistics
            words = text.split()
            sentences = text.split('.')
            paragraphs = text.split('\n\n')

            # Character analysis
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))

            # Word analysis
            word_count = len(words)
            unique_words = len(set(word.lower().strip('.,!?";') for word in words))
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

            # Sentence analysis
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

            # Simple readability score (simplified Flesch formula)
            if sentence_count > 0 and word_count > 0:
                readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (char_count_no_spaces / word_count))
                readability_score = max(0, min(100, readability_score))  # Clamp to 0-100
            else:
                readability_score = 0

            # Language detection (very basic - just check for common English words)
            english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
            english_word_count = sum(1 for word in words if word.lower() in english_indicators)
            likely_language = "English" if english_word_count > len(words) * 0.1 else "Unknown"

            # Simple sentiment analysis if requested
            sentiment_score = None
            if analyze_sentiment:
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']

                positive_count = sum(1 for word in words if word.lower() in positive_words)
                negative_count = sum(1 for word in words if word.lower() in negative_words)

                if positive_count + negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    sentiment_score = 0.0

            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            analysis_results = {
                "readability_score": readability_score,
                "likely_language": likely_language,
                "sentiment_score": sentiment_score,
                "vocabulary_richness": unique_words / word_count if word_count > 0 else 0,
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2)
            }

            text_stats = {
                "character_count": char_count,
                "character_count_no_spaces": char_count_no_spaces,
                "word_count": word_count,
                "unique_word_count": unique_words,
                "sentence_count": sentence_count,
                "paragraph_count": len(paragraphs)
            }

            logger.info(f"Analyzed text with {word_count} words in {processing_time_ms:.2f}ms")

            return {
                "analysis_results": analysis_results,
                "processing_time_ms": processing_time_ms,
                "text_stats": text_stats
            }

        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise ValueError(f"Text analysis failed: {str(e)}")


def demo_components():
    """
    Demonstrate how to use the components.
    This is a great pattern for testing and showcasing your work!
    """
    print("🚀 Welcome to your first Haystack components demo!")
    print("=" * 50)

    # Demo 1: Basic HelloHaystack component
    print("\n1. Testing HelloHaystack component:")
    hello_component = HelloHaystack(greeting_style="friendly")

    result = hello_component.run(name="Haystack Learner", message="Let's build something amazing!")
    print(f"Greeting: {result['greeting']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Generated at: {result['timestamp']}")

    # Demo 2: Different greeting styles
    print("\n2. Testing different greeting styles:")
    styles = ["friendly", "formal", "casual"]
    for style in styles:
        component = HelloHaystack(greeting_style=style)
        result = component.run(name="Developer")
        print(f"{style.capitalize()}: {result['greeting']}")

    # Demo 3: TextAnalyzer component
    print("\n3. Testing TextAnalyzer component:")
    analyzer = TextAnalyzer()

    sample_text = """
    Haystack is an amazing framework for building AI applications.
    It makes creating intelligent search systems and chatbots incredibly easy.
    I'm excited to learn more about its capabilities!
    """

    analysis_result = analyzer.run(text=sample_text, analyze_sentiment=True)

    print("Text Analysis Results:")
    print(f"  - Word Count: {analysis_result['text_stats']['word_count']}")
    print(f"  - Readability Score: {analysis_result['analysis_results']['readability_score']:.1f}")
    print(f"  - Sentiment Score: {analysis_result['analysis_results']['sentiment_score']:.2f}")
    print(f"  - Processing Time: {analysis_result['processing_time_ms']:.2f}ms")
    print(f"  - Language: {analysis_result['analysis_results']['likely_language']}")

    print("\n✅ All components working perfectly!")
    print("🎉 Congratulations on creating your first Haystack components!")


def test_error_handling():
    """
    Demonstrate error handling in components.
    Good error handling is crucial for production systems!
    """
    print("\n🔧 Testing Error Handling:")
    print("-" * 30)

    hello_component = HelloHaystack()

    # Test with invalid input
    try:
        result = hello_component.run(name="")  # Empty name should raise error
        print("❌ This should have raised an error!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    try:
        result = hello_component.run(name="123")  # Numbers only should raise error
        print("❌ This should have raised an error!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test with valid input after errors
    try:
        result = hello_component.run(name="Valid User")
        print(f"✅ Recovered successfully: {result['greeting'][:50]}...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    """
    Run the demonstration when this file is executed directly.
    This is a great way to test your components during development!
    """
    try:
        demo_components()
        test_error_handling()

        print("\n" + "=" * 50)
        print("🎓 Learning Tips:")
        print("1. Always use the @component decorator")
        print("2. Define clear output types with @component.output_types")
        print("3. Add proper error handling and logging")
        print("4. Write comprehensive docstrings")
        print("5. Test your components thoroughly")
        print("6. Keep components focused and reusable")

        print("\n🚀 Next Steps:")
        print("1. Try modifying the greeting styles")
        print("2. Add more analysis features to TextAnalyzer")
        print("3. Create your own component from scratch")
        print("4. Combine components into a pipeline")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo execution failed: {str(e)}")
