"""
Unit tests for src/utils/config_loader.py

Tests configuration loading, validation, and environment variable handling.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from src.utils.config_loader import (
    load_config,
    validate_provider_model,
    Config,
    DatasetConfig,
    AgentConfig,
    SplitConfig,
    SystemConfig
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config_yaml):
        """Test loading a valid config.yaml file."""
        config = load_config(sample_config_yaml)

        assert isinstance(config, Config)
        assert config.dataset.provider == 'google'
        assert config.dataset.model == 'gemini-2.0-flash-exp'
        assert config.agent.mode == 'create'
        assert config.split.enabled is True

    def test_load_config_with_defaults(self, tmp_path):
        """Test loading config with missing optional fields uses defaults."""
        minimal_config = """
dataset:
  provider: google
  model: gemini-2.0-flash-exp

agent:
  mode: create
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(minimal_config)

        config = load_config(str(config_file))

        # Check defaults are applied
        assert config.agent.variety == 'medium'  # default
        assert config.agent.length == 'medium'  # default
        assert config.split.enabled is False  # default

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_invalid_yaml_syntax_raises_error(self, tmp_path):
        """Test that invalid YAML syntax raises appropriate error."""
        invalid_config = """
dataset:
  provider: google
  model: [unclosed bracket
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(invalid_config)

        with pytest.raises(yaml.YAMLError):
            load_config(str(config_file))

    def test_missing_required_fields_raises_error(self, tmp_path):
        """Test that missing required fields raises ValueError."""
        incomplete_config = """
dataset:
  provider: google
  # missing model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(incomplete_config)

        with pytest.raises((KeyError, ValueError, TypeError)):
            load_config(str(config_file))

    def test_load_env_variables(self, mock_env_vars):
        """Test that environment variables are loaded correctly."""
        # Environment variables set by mock_env_vars fixture
        import os

        assert os.getenv('GOOGLE_API_KEY') == 'test-google-api-key-12345'
        assert os.getenv('AGENT_PROVIDER') == 'google'
        assert os.getenv('AGENT_MODEL') == 'gemini-2.0-flash-exp'

    def test_config_with_all_providers(self, tmp_path):
        """Test config loading with different providers."""
        providers = ['google', 'openai', 'anthropic', 'mistral', 'llama']

        for provider in providers:
            config_content = f"""
dataset:
  provider: {provider}
  model: test-model

agent:
  mode: create
"""
            config_file = tmp_path / f"config_{provider}.yaml"
            config_file.write_text(config_content)

            config = load_config(str(config_file))
            assert config.dataset.provider == provider


class TestValidateProviderModel:
    """Tests for validate_provider_model function."""

    def test_validate_google_models(self):
        """Test validation of Google Gemini models."""
        valid_models = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash-exp'
        ]

        for model in valid_models:
            assert validate_provider_model('google', model) is True

    def test_validate_openai_models(self):
        """Test validation of OpenAI models."""
        valid_models = [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4.1',
            'gpt-4.1-mini',
            'gpt-3.5-turbo'
        ]

        for model in valid_models:
            assert validate_provider_model('openai', model) is True

    def test_validate_anthropic_models(self):
        """Test validation of Anthropic models."""
        valid_models = ['claude-3-haiku-20240307']

        for model in valid_models:
            assert validate_provider_model('anthropic', model) is True

    def test_validate_mistral_models(self):
        """Test validation of Mistral models."""
        valid_models = [
            'mistral-7b',
            'mistral-large-v2',
            'mistral-nemo',
            'mixtral-8x7b'
        ]

        for model in valid_models:
            assert validate_provider_model('mistral', model) is True

    def test_validate_llama_models(self):
        """Test validation of Meta Llama models."""
        valid_models = [
            'llama-3.1-8b',
            'llama-3.1-70b',
            'llama-3.1-405b'
        ]

        for model in valid_models:
            assert validate_provider_model('llama', model) is True

    def test_invalid_provider_model_combination(self):
        """Test that invalid provider-model combinations return False."""
        assert validate_provider_model('google', 'gpt-4o') is False
        assert validate_provider_model('openai', 'gemini-2.5-pro') is False
        assert validate_provider_model('anthropic', 'llama-3.1-8b') is False

    def test_invalid_provider_returns_false(self):
        """Test that invalid provider returns False."""
        assert validate_provider_model('invalid-provider', 'some-model') is False

    def test_invalid_model_returns_false(self):
        """Test that invalid model for valid provider returns False."""
        assert validate_provider_model('google', 'nonexistent-model') is False


class TestDataclassValidation:
    """Tests for dataclass type validation."""

    def test_dataset_config_dataclass(self):
        """Test DatasetConfig dataclass creation."""
        dataset = DatasetConfig(
            name='test',
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=100,
            output_format='jsonl'
        )

        assert dataset.name == 'test'
        assert dataset.provider == 'google'
        assert dataset.model == 'gemini-2.0-flash-exp'
        assert dataset.target_samples == 100

    def test_agent_config_dataclass(self):
        """Test AgentConfig dataclass creation."""
        agent = AgentConfig(
            mode='create',
            variety='high',
            length='long',
            min_conversation_turns=3,
            confidence_threshold=0.8,
            generation_ratio=0.6,
            deduplicate=True
        )

        assert agent.mode == 'create'
        assert agent.variety == 'high'
        assert agent.length == 'long'
        assert agent.generation_ratio == 0.6

    def test_split_config_dataclass(self):
        """Test SplitConfig dataclass creation."""
        split = SplitConfig(
            enabled=True,
            ratio=0.8,
            strategy='weighted'
        )

        assert split.enabled is True
        assert split.ratio == 0.8
        assert split.strategy == 'weighted'

    def test_system_config_dataclass(self):
        """Test SystemConfig dataclass creation."""
        system = SystemConfig(
            agentic_llm={'provider': 'google', 'model': 'gemini-2.0-flash-exp'},
            max_retries=5,
            rate_limit_delay=2.0,
            debug=True
        )

        assert system.max_retries == 5
        assert system.rate_limit_delay == 2.0
        assert system.debug is True

    def test_config_dataclass_complete(self):
        """Test complete Config dataclass creation."""
        dataset = DatasetConfig(
            name='test',
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=100,
            output_format='jsonl'
        )

        agent = AgentConfig(
            mode='create',
            variety='medium',
            length='medium',
            min_conversation_turns=2,
            confidence_threshold=0.7,
            generation_ratio=0.5,
            deduplicate=True
        )

        split = SplitConfig(
            enabled=True,
            ratio=0.8,
            strategy='random'
        )

        system = SystemConfig(
            agentic_llm={'provider': 'google', 'model': 'gemini-2.0-flash-exp'},
            max_retries=3,
            rate_limit_delay=1.0,
            debug=False
        )

        config = Config(
            dataset=dataset,
            agent=agent,
            split=split,
            system=system
        )

        assert config.dataset.provider == 'google'
        assert config.agent.mode == 'create'
        assert config.split.ratio == 0.8
        assert config.system.max_retries == 3


class TestConfigEdgeCases:
    """Tests for edge cases in configuration."""

    def test_empty_config_file(self, tmp_path):
        """Test handling of empty config file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises((ValueError, TypeError, AttributeError)):
            load_config(str(config_file))

    def test_config_with_extra_fields(self, tmp_path):
        """Test that extra unknown fields are handled gracefully."""
        config_with_extras = """
dataset:
  provider: google
  model: gemini-2.0-flash-exp
  extra_field: this_should_be_ignored

agent:
  mode: create
  unknown_param: also_ignored
"""
        config_file = tmp_path / "config_extras.yaml"
        config_file.write_text(config_with_extras)

        # Should load successfully, ignoring extra fields
        config = load_config(str(config_file))
        assert config.dataset.provider == 'google'

    def test_config_with_wrong_types(self, tmp_path):
        """Test that wrong types in config raise appropriate errors."""
        wrong_types_config = """
dataset:
  provider: google
  model: gemini-2.0-flash-exp
  target_samples: "not_a_number"  # should be int

agent:
  mode: create
"""
        config_file = tmp_path / "wrong_types.yaml"
        config_file.write_text(wrong_types_config)

        with pytest.raises((ValueError, TypeError)):
            load_config(str(config_file))

    def test_special_characters_in_dataset_name(self, tmp_path):
        """Test handling of special characters in dataset name."""
        special_char_config = """
dataset:
  name: "test-dataset_2024.v1"
  provider: google
  model: gemini-2.0-flash-exp

agent:
  mode: create
"""
        config_file = tmp_path / "special_chars.yaml"
        config_file.write_text(special_char_config)

        config = load_config(str(config_file))
        assert config.dataset.name == "test-dataset_2024.v1"

    def test_boundary_split_ratios(self, tmp_path):
        """Test boundary values for split ratios."""
        # Test minimum valid ratio
        min_ratio_config = """
dataset:
  provider: google
  model: gemini-2.0-flash-exp

agent:
  mode: create

split:
  enabled: true
  ratio: 0.1
  strategy: random
"""
        config_file = tmp_path / "min_ratio.yaml"
        config_file.write_text(min_ratio_config)

        config = load_config(str(config_file))
        assert config.split.ratio == 0.1

        # Test maximum valid ratio
        max_ratio_config = """
dataset:
  provider: google
  model: gemini-2.0-flash-exp

agent:
  mode: create

split:
  enabled: true
  ratio: 0.99
  strategy: random
"""
        config_file = tmp_path / "max_ratio.yaml"
        config_file.write_text(max_ratio_config)

        config = load_config(str(config_file))
        assert config.split.ratio == 0.99
