"""
Unit tests for all provider implementations.

Tests all 5 providers: OpenAI, Anthropic, Google, Mistral, Llama.
"""

import pytest
from src.providers.base_provider import BaseProvider, ProviderSpec, ValidationResult
from src.providers.openai_provider import OpenAIProvider
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.google_provider import GoogleProvider
from src.providers.mistral_provider import MistralProvider
from src.providers.llama_provider import LlamaProvider
from src.providers import get_provider


class TestProviderFactory:
    """Tests for provider factory function."""

    def test_get_provider_openai(self):
        """Test factory returns OpenAI provider."""
        provider = get_provider('openai')
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_anthropic(self):
        """Test factory returns Anthropic provider."""
        provider = get_provider('anthropic')
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_google(self):
        """Test factory returns Google provider."""
        provider = get_provider('google')
        assert isinstance(provider, GoogleProvider)

    def test_get_provider_mistral(self):
        """Test factory returns Mistral provider."""
        provider = get_provider('mistral')
        assert isinstance(provider, MistralProvider)

    def test_get_provider_llama(self):
        """Test factory returns Llama provider."""
        provider = get_provider('llama')
        assert isinstance(provider, LlamaProvider)

    def test_get_provider_invalid(self):
        """Test factory raises error for invalid provider."""
        with pytest.raises((ValueError, KeyError)):
            get_provider('invalid_provider')

    def test_get_provider_case_insensitive(self):
        """Test factory handles case-insensitive provider names."""
        provider_upper = get_provider('GOOGLE')
        provider_lower = get_provider('google')

        assert isinstance(provider_upper, GoogleProvider)
        assert isinstance(provider_lower, GoogleProvider)


class TestAllProvidersImplementInterface:
    """Tests that all providers implement required interface."""

    @pytest.fixture
    def all_providers(self):
        """Fixture providing all provider instances."""
        return [
            OpenAIProvider(),
            AnthropicProvider(),
            GoogleProvider(),
            MistralProvider(),
            LlamaProvider()
        ]

    def test_all_have_get_models(self, all_providers):
        """Test all providers implement get_models()."""
        for provider in all_providers:
            models = provider.get_models()
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(m, str) for m in models)

    def test_all_have_get_token_limit(self, all_providers):
        """Test all providers implement get_token_limit()."""
        for provider in all_providers:
            models = provider.get_models()
            for model in models[:2]:  # Test first 2 models
                limit = provider.get_token_limit(model)
                assert isinstance(limit, (int, type(None)))
                if limit is not None:
                    assert limit > 0

    def test_all_have_format_example(self, all_providers):
        """Test all providers implement format_example()."""
        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test'},
                {'role': 'assistant', 'content': 'Response'}
            ]
        }

        for provider in all_providers:
            formatted = provider.format_example(sample)
            assert isinstance(formatted, dict)

    def test_all_have_validate_example(self, all_providers):
        """Test all providers implement validate_example()."""
        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test'},
                {'role': 'assistant', 'content': 'Response'}
            ]
        }

        for provider in all_providers:
            result = provider.validate_example(sample)
            assert isinstance(result, (bool, ValidationResult))

    def test_all_have_get_spec(self, all_providers):
        """Test all providers implement get_spec()."""
        for provider in all_providers:
            spec = provider.get_spec()
            assert isinstance(spec, ProviderSpec)
            assert hasattr(spec, 'name')
            assert hasattr(spec, 'models')


class TestOpenAIProvider:
    """Tests specific to OpenAI provider."""

    def test_get_models(self):
        """Test OpenAI model list."""
        provider = OpenAIProvider()
        models = provider.get_models()

        expected_models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-3.5-turbo']

        for expected in expected_models:
            assert expected in models

    def test_get_token_limit_gpt4o(self):
        """Test token limit for GPT-4o."""
        provider = OpenAIProvider()
        limit = provider.get_token_limit('gpt-4o')

        assert limit == 65536

    def test_get_token_limit_gpt35_turbo(self):
        """Test token limit for GPT-3.5-turbo."""
        provider = OpenAIProvider()
        limit = provider.get_token_limit('gpt-3.5-turbo')

        # GPT-3.5-turbo-0125 has 16,385 limit
        assert limit in [16385, 16000]  # Allow some variation

    def test_format_example_with_system(self):
        """Test OpenAI format with system message."""
        provider = OpenAIProvider()

        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'messages' in formatted
        assert formatted['messages'][0]['role'] == 'system'

    def test_format_example_without_system(self):
        """Test OpenAI format without system message."""
        provider = OpenAIProvider()

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'messages' in formatted
        assert all(m['role'] in ['user', 'assistant'] for m in formatted['messages'])


class TestAnthropicProvider:
    """Tests specific to Anthropic provider."""

    def test_get_models(self):
        """Test Anthropic model list."""
        provider = AnthropicProvider()
        models = provider.get_models()

        assert 'claude-3-haiku-20240307' in models

    def test_get_token_limit(self):
        """Test token limit for Claude."""
        provider = AnthropicProvider()
        limit = provider.get_token_limit('claude-3-haiku-20240307')

        assert limit == 32000

    def test_format_example_extracts_system(self):
        """Test Anthropic format extracts system to separate field."""
        provider = AnthropicProvider()

        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'system' in formatted
        assert formatted['system'] == 'You are helpful.'
        assert all(m['role'] in ['user', 'assistant'] for m in formatted['messages'])

    def test_format_example_without_system(self):
        """Test Anthropic format without system message."""
        provider = AnthropicProvider()

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'messages' in formatted


class TestGoogleProvider:
    """Tests specific to Google provider."""

    def test_get_models(self):
        """Test Google model list."""
        provider = GoogleProvider()
        models = provider.get_models()

        expected_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash-exp']

        for expected in expected_models:
            assert expected in models

    def test_get_token_limit(self):
        """Test token limit for Gemini models."""
        provider = GoogleProvider()

        # Gemini models have varying limits
        limit = provider.get_token_limit('gemini-2.0-flash-exp')
        assert isinstance(limit, (int, type(None)))

    def test_minimum_samples_requirement(self):
        """Test that Google enforces 100+ minimum samples."""
        provider = GoogleProvider()
        spec = provider.get_spec()

        # Check if spec has min_samples attribute
        if hasattr(spec, 'min_samples'):
            assert spec.min_samples >= 100

    def test_format_example(self):
        """Test Google format."""
        provider = GoogleProvider()

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test'},
                {'role': 'assistant', 'content': 'Response'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'messages' in formatted


class TestMistralProvider:
    """Tests specific to Mistral provider."""

    def test_get_models(self):
        """Test Mistral model list."""
        provider = MistralProvider()
        models = provider.get_models()

        expected_models = ['mistral-7b', 'mistral-large-v2', 'mistral-nemo', 'mixtral-8x7b']

        for expected in expected_models:
            assert expected in models

    def test_format_example(self):
        """Test Mistral format (similar to OpenAI)."""
        provider = MistralProvider()

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test'},
                {'role': 'assistant', 'content': 'Response'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'messages' in formatted


class TestLlamaProvider:
    """Tests specific to Llama provider."""

    def test_get_models(self):
        """Test Llama model list."""
        provider = LlamaProvider()
        models = provider.get_models()

        expected_models = ['llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b']

        for expected in expected_models:
            assert expected in models

    def test_flexible_limits(self):
        """Test that Llama has flexible limits (open source)."""
        provider = LlamaProvider()

        # Llama limits may be None or high values
        limit = provider.get_token_limit('llama-3.1-8b')
        assert isinstance(limit, (int, type(None)))

    def test_format_example(self):
        """Test Llama format."""
        provider = LlamaProvider()

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test'},
                {'role': 'assistant', 'content': 'Response'}
            ]
        }

        formatted = provider.format_example(sample)

        assert 'messages' in formatted


class TestFormatDifferences:
    """Tests comparing format differences between providers."""

    def test_openai_vs_anthropic_system_handling(self):
        """Test different system message handling between OpenAI and Anthropic."""
        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        openai = OpenAIProvider()
        anthropic = AnthropicProvider()

        openai_formatted = openai.format_example(sample)
        anthropic_formatted = anthropic.format_example(sample)

        # OpenAI keeps system in messages array
        assert openai_formatted['messages'][0]['role'] == 'system'

        # Anthropic extracts system to separate field
        assert 'system' in anthropic_formatted
        assert all(m['role'] != 'system' for m in anthropic_formatted['messages'])

    def test_all_providers_handle_basic_conversation(self):
        """Test that all providers can format basic user/assistant conversation."""
        sample = {
            'messages': [
                {'role': 'user', 'content': 'What is Python?'},
                {'role': 'assistant', 'content': 'Python is a programming language.'}
            ]
        }

        providers = [
            OpenAIProvider(),
            AnthropicProvider(),
            GoogleProvider(),
            MistralProvider(),
            LlamaProvider()
        ]

        for provider in providers:
            formatted = provider.format_example(sample)
            assert 'messages' in formatted
            assert len(formatted['messages']) == 2


class TestProviderSpecDataclass:
    """Tests for ProviderSpec dataclass."""

    def test_provider_spec_creation(self):
        """Test ProviderSpec dataclass creation."""
        spec = ProviderSpec(
            name='test_provider',
            models=['model1', 'model2'],
            token_limits={'model1': 1000, 'model2': 2000},
            max_training_tokens=50000000,
            min_samples=100
        )

        assert spec.name == 'test_provider'
        assert len(spec.models) == 2
        assert spec.token_limits['model1'] == 1000

    def test_all_providers_return_valid_spec(self):
        """Test that all providers return valid ProviderSpec."""
        providers = [
            OpenAIProvider(),
            AnthropicProvider(),
            GoogleProvider(),
            MistralProvider(),
            LlamaProvider()
        ]

        for provider in providers:
            spec = provider.get_spec()
            assert isinstance(spec, ProviderSpec)
            assert spec.name is not None
            assert len(spec.models) > 0


class TestValidationResultDataclass:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=['Minor warning']
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=['Error 1', 'Error 2'],
            warnings=[]
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
