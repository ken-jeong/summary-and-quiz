"""Main generator module for text summarization and quiz generation."""

import re
import logging
from typing import Optional, Literal
from enum import Enum

import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor

from .config import Config, DEFAULT_CONFIG


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GenerationMode(str, Enum):
    """Available generation modes."""
    SUMMARY = "summary"
    QUIZ = "quiz"


class SummaryQuizGenerator:
    """Generator for text summarization and quiz creation.

    This class handles loading the LLM model and generating summaries
    or quiz questions from Korean text input.

    Example:
        >>> generator = SummaryQuizGenerator()
        >>> summary = generator.generate("summary", "긴 텍스트...")
        >>> quiz = generator.generate("quiz", summary)
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the generator with configuration.

        Args:
            config: Configuration object. Uses DEFAULT_CONFIG if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        self.config.validate()

        self.model: Optional[MllamaForConditionalGeneration] = None
        self.processor: Optional[MllamaProcessor] = None
        self._is_loaded = False

        logger.info(f"Initialized generator with model: {self.config.model_name}")

    def load_model(self) -> None:
        """Load the model and processor into memory.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return

        logger.info(f"Loading model: {self.config.model_name}")

        try:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map
            )

            self.processor = MllamaProcessor.from_pretrained(
                self.config.model_name
            )

            self._is_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def generate(
        self,
        mode: Literal["summary", "quiz"],
        text: str
    ) -> str:
        """Generate summary or quiz from input text.

        Args:
            mode: Generation mode - "summary" or "quiz"
            text: Input text to process

        Returns:
            Generated text (summary or quiz Q&A)

        Raises:
            ValueError: If mode is invalid or text is empty
            RuntimeError: If model is not loaded or generation fails
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if mode not in [m.value for m in GenerationMode]:
            raise ValueError(f"Invalid mode: {mode}. Use 'summary' or 'quiz'")

        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        # Select prompt based on mode
        if mode == GenerationMode.SUMMARY.value:
            prompt = self.config.summary_prompt
        else:
            prompt = self.config.quiz_prompt

        logger.info(f"Generating {mode} for text of length {len(text)}")

        # Build chat messages
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt + text}]
        }]

        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize inputs
        inputs = self.processor(
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate output
        try:
            eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                self.config.eos_token
            )

            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                eos_token_id=eos_token_id,
                use_cache=self.config.use_cache
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}") from e

        # Decode and extract response
        output_text = self.processor.decode(output[0])
        result = self._extract_response(output_text)

        logger.info(f"Generated {mode} of length {len(result)}")
        return result

    def _extract_response(self, output_text: str) -> str:
        """Extract assistant response from model output.

        Args:
            output_text: Raw model output text

        Returns:
            Extracted response text

        Raises:
            RuntimeError: If response pattern not found
        """
        match = re.search(
            self.config.response_pattern,
            output_text,
            re.DOTALL
        )

        if match:
            return match.group(1).strip()

        # Fallback: try to find any text after assistant header
        fallback_pattern = r'assistant<\|end_header_id\|>\s*(.*?)(?:<\||$)'
        fallback_match = re.search(fallback_pattern, output_text, re.DOTALL)

        if fallback_match:
            logger.warning("Used fallback pattern for response extraction")
            return fallback_match.group(1).strip()

        logger.error("Failed to extract response from model output")
        raise RuntimeError(
            "Could not extract response from model output. "
            "The model output format may have changed."
        )

    def summarize(self, text: str) -> str:
        """Convenience method for generating summary.

        Args:
            text: Text to summarize

        Returns:
            Summary text
        """
        return self.generate("summary", text)

    def create_quiz(self, text: str) -> str:
        """Convenience method for generating quiz.

        Args:
            text: Text to create quiz from (usually a summary)

        Returns:
            Quiz question and answer
        """
        return self.generate("quiz", text)

    def summarize_and_quiz(self, text: str) -> dict:
        """Generate both summary and quiz from text.

        Args:
            text: Original text to process

        Returns:
            Dictionary with 'summary' and 'quiz' keys
        """
        summary = self.summarize(text)
        quiz = self.create_quiz(summary)

        return {
            "summary": summary,
            "quiz": quiz
        }

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self._is_loaded:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._is_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded and memory cleared")

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
        return False
