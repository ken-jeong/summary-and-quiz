"""Tests for configuration module."""

import pytest
import torch

from src.config import Config, DEFAULT_CONFIG


class TestConfig:
    """Test cases for Config class."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = Config()

        assert config.model_name == "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"
        assert config.torch_dtype == torch.bfloat16
        assert config.device_map == "auto"
        assert config.max_new_tokens == 256
        assert config.temperature == 0.1

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = Config(
            model_name="custom/model",
            max_new_tokens=512,
            temperature=0.7
        )

        assert config.model_name == "custom/model"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = Config()
        config.validate()  # Should not raise

    def test_validate_invalid_max_tokens(self):
        """Test validation fails for invalid max_new_tokens."""
        config = Config(max_new_tokens=0)

        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            config.validate()

    def test_validate_invalid_temperature_low(self):
        """Test validation fails for temperature below 0."""
        config = Config(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between"):
            config.validate()

    def test_validate_invalid_temperature_high(self):
        """Test validation fails for temperature above 2."""
        config = Config(temperature=2.5)

        with pytest.raises(ValueError, match="temperature must be between"):
            config.validate()

    def test_validate_empty_model_name(self):
        """Test validation fails for empty model name."""
        config = Config(model_name="")

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            config.validate()

    def test_default_config_instance(self):
        """Test DEFAULT_CONFIG is a valid Config instance."""
        assert isinstance(DEFAULT_CONFIG, Config)
        DEFAULT_CONFIG.validate()  # Should not raise
