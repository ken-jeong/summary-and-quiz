"""Configuration management for Summary and Quiz Generator."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Config:
    """Configuration class for model and generation settings.

    Attributes:
        model_name: HuggingFace model identifier
        torch_dtype: PyTorch data type for model weights
        device_map: Device mapping strategy ('auto', 'cpu', 'cuda')
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        use_cache: Whether to use KV cache during generation
    """

    # Model settings
    model_name: str = "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"

    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.1
    use_cache: bool = False

    # Prompt templates
    summary_prompt: str = "다음 텍스트를 요약해줘: "
    quiz_prompt: str = "다음 텍스트로 문제와 정답을 1개만 만들어줘: "

    # Token patterns for response extraction
    response_pattern: str = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'
    eos_token: str = "<|eot_id|>"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")


# Default configuration instance
DEFAULT_CONFIG = Config()
