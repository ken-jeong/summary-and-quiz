"""Tests for generator module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.generator import SummaryQuizGenerator, GenerationMode
from src.config import Config


class TestGenerationMode:
    """Test cases for GenerationMode enum."""

    def test_summary_mode(self):
        """Test summary mode value."""
        assert GenerationMode.SUMMARY.value == "summary"

    def test_quiz_mode(self):
        """Test quiz mode value."""
        assert GenerationMode.QUIZ.value == "quiz"


class TestSummaryQuizGenerator:
    """Test cases for SummaryQuizGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = SummaryQuizGenerator()

        assert generator.config is not None
        assert generator.model is None
        assert generator.processor is None
        assert generator._is_loaded is False

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = Config(max_new_tokens=512)
        generator = SummaryQuizGenerator(config)

        assert generator.config.max_new_tokens == 512

    def test_init_invalid_config(self):
        """Test initialization fails with invalid config."""
        config = Config(max_new_tokens=-1)

        with pytest.raises(ValueError):
            SummaryQuizGenerator(config)

    def test_generate_empty_text(self):
        """Test generate raises error for empty text."""
        generator = SummaryQuizGenerator()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            generator.generate("summary", "")

    def test_generate_whitespace_text(self):
        """Test generate raises error for whitespace-only text."""
        generator = SummaryQuizGenerator()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            generator.generate("summary", "   \n\t  ")

    def test_generate_invalid_mode(self):
        """Test generate raises error for invalid mode."""
        generator = SummaryQuizGenerator()

        with pytest.raises(ValueError, match="Invalid mode"):
            generator.generate("invalid", "test text")

    def test_extract_response_success(self):
        """Test response extraction with valid output."""
        generator = SummaryQuizGenerator()
        output = "<|start_header_id|>assistant<|end_header_id|>extracted text<|eot_id|>"

        result = generator._extract_response(output)

        assert result == "extracted text"

    def test_extract_response_with_whitespace(self):
        """Test response extraction strips whitespace."""
        generator = SummaryQuizGenerator()
        output = "<|start_header_id|>assistant<|end_header_id|>  text with spaces  <|eot_id|>"

        result = generator._extract_response(output)

        assert result == "text with spaces"

    def test_extract_response_fallback(self):
        """Test fallback pattern when primary fails."""
        generator = SummaryQuizGenerator()
        output = "assistant<|end_header_id|>\nfallback text"

        result = generator._extract_response(output)

        assert result == "fallback text"

    def test_extract_response_failure(self):
        """Test response extraction fails with invalid output."""
        generator = SummaryQuizGenerator()
        output = "completely invalid output format"

        with pytest.raises(RuntimeError, match="Could not extract response"):
            generator._extract_response(output)

    @patch.object(SummaryQuizGenerator, 'generate')
    def test_summarize_calls_generate(self, mock_generate):
        """Test summarize is a convenience wrapper."""
        mock_generate.return_value = "summary result"
        generator = SummaryQuizGenerator()

        result = generator.summarize("test text")

        mock_generate.assert_called_once_with("summary", "test text")
        assert result == "summary result"

    @patch.object(SummaryQuizGenerator, 'generate')
    def test_create_quiz_calls_generate(self, mock_generate):
        """Test create_quiz is a convenience wrapper."""
        mock_generate.return_value = "quiz result"
        generator = SummaryQuizGenerator()

        result = generator.create_quiz("test text")

        mock_generate.assert_called_once_with("quiz", "test text")
        assert result == "quiz result"

    @patch.object(SummaryQuizGenerator, 'summarize')
    @patch.object(SummaryQuizGenerator, 'create_quiz')
    def test_summarize_and_quiz(self, mock_quiz, mock_summarize):
        """Test combined summarize and quiz generation."""
        mock_summarize.return_value = "summary"
        mock_quiz.return_value = "quiz"
        generator = SummaryQuizGenerator()

        result = generator.summarize_and_quiz("test text")

        assert result["summary"] == "summary"
        assert result["quiz"] == "quiz"
        mock_summarize.assert_called_once_with("test text")
        mock_quiz.assert_called_once_with("summary")

    @patch.object(SummaryQuizGenerator, 'load_model')
    @patch.object(SummaryQuizGenerator, 'unload_model')
    def test_context_manager(self, mock_unload, mock_load):
        """Test context manager calls load and unload."""
        with SummaryQuizGenerator() as generator:
            mock_load.assert_called_once()

        mock_unload.assert_called_once()
