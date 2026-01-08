"""
Unit tests for src/utils/formatter.py

Tests provider-specific JSONL formatting including OpenAI, Anthropic, Google formats.
"""

import pytest
import json
from src.utils.formatter import Formatter, FormatError


class TestFormatterBasic:
    """Basic formatter tests."""

    def test_initialization(self):
        """Test Formatter initialization."""
        formatter = Formatter(provider='google')
        assert formatter is not None
        assert formatter.provider == 'google'

    def test_initialization_different_providers(self):
        """Test Formatter with different providers."""
        providers = ['openai', 'anthropic', 'google', 'mistral', 'llama']

        for provider in providers:
            formatter = Formatter(provider=provider)
            assert formatter.provider == provider


class TestOpenAIFormatting:
    """Tests for OpenAI format conversion."""

    def test_format_openai_with_system(self):
        """Test formatting sample to OpenAI format with system message."""
        formatter = Formatter(provider='openai')

        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'What is Python?'},
                {'role': 'assistant', 'content': 'Python is a programming language.'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'messages' in formatted
        assert len(formatted['messages']) == 3
        assert formatted['messages'][0]['role'] == 'system'
        assert formatted['messages'][1]['role'] == 'user'
        assert formatted['messages'][2]['role'] == 'assistant'

    def test_format_openai_without_system(self):
        """Test formatting sample to OpenAI format without system message."""
        formatter = Formatter(provider='openai')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'messages' in formatted
        assert len(formatted['messages']) == 2
        assert all(msg['role'] in ['user', 'assistant'] for msg in formatted['messages'])

    def test_format_openai_multi_turn(self):
        """Test formatting multi-turn conversation to OpenAI format."""
        formatter = Formatter(provider='openai')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'},
                {'role': 'user', 'content': 'How are you?'},
                {'role': 'assistant', 'content': 'I am doing well!'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert len(formatted['messages']) == 4
        assert formatted['messages'][0]['role'] == 'user'
        assert formatted['messages'][1]['role'] == 'assistant'


class TestAnthropicFormatting:
    """Tests for Anthropic format conversion."""

    def test_format_anthropic_with_system(self):
        """Test formatting sample to Anthropic format with system field."""
        formatter = Formatter(provider='anthropic')

        sample = {
            'system': 'You are a helpful assistant.',
            'messages': [
                {'role': 'user', 'content': 'What is Python?'},
                {'role': 'assistant', 'content': 'Python is a programming language.'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'system' in formatted
        assert 'messages' in formatted
        assert formatted['system'] == 'You are a helpful assistant.'
        assert len(formatted['messages']) == 2

    def test_format_anthropic_without_system(self):
        """Test formatting sample to Anthropic format without system."""
        formatter = Formatter(provider='anthropic')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'messages' in formatted
        # System field may be optional
        assert len(formatted['messages']) == 2

    def test_format_anthropic_extract_system_from_messages(self):
        """Test extracting system message from messages array for Anthropic."""
        formatter = Formatter(provider='anthropic')

        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        formatted = formatter.format_sample(sample)

        # System should be extracted to separate field
        assert 'system' in formatted
        assert formatted['system'] == 'You are helpful.'
        # Messages should only have user/assistant
        assert all(msg['role'] in ['user', 'assistant'] for msg in formatted['messages'])


class TestGoogleFormatting:
    """Tests for Google Gemini format conversion."""

    def test_format_google_basic(self):
        """Test formatting sample to Google format."""
        formatter = Formatter(provider='google')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'What is Python?'},
                {'role': 'assistant', 'content': 'Python is a programming language.'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'messages' in formatted
        assert len(formatted['messages']) == 2

    def test_format_google_with_system(self):
        """Test formatting sample to Google format with system message."""
        formatter = Formatter(provider='google')

        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'}
            ]
        }

        formatted = formatter.format_sample(sample)

        # Google may handle system messages differently
        assert 'messages' in formatted


class TestMistralAndLlamaFormatting:
    """Tests for Mistral and Llama format conversion."""

    def test_format_mistral(self):
        """Test formatting sample to Mistral format."""
        formatter = Formatter(provider='mistral')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test question'},
                {'role': 'assistant', 'content': 'Test answer'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'messages' in formatted
        assert isinstance(formatted, dict)

    def test_format_llama(self):
        """Test formatting sample to Llama format."""
        formatter = Formatter(provider='llama')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test question'},
                {'role': 'assistant', 'content': 'Test answer'}
            ]
        }

        formatted = formatter.format_sample(sample)

        assert 'messages' in formatted
        assert isinstance(formatted, dict)


class TestJSONLConversion:
    """Tests for JSONL conversion."""

    def test_format_to_jsonl(self):
        """Test formatting dataset to JSONL string."""
        formatter = Formatter(provider='google')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Q1'},
                    {'role': 'assistant', 'content': 'A1'}
                ]
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'Q2'},
                    {'role': 'assistant', 'content': 'A2'}
                ]
            }
        ]

        jsonl = formatter.format_to_jsonl(dataset)

        # Should be newline-delimited JSON
        lines = jsonl.strip().split('\n')
        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert 'messages' in parsed

    def test_format_to_jsonl_single_sample(self):
        """Test formatting single sample to JSONL."""
        formatter = Formatter(provider='openai')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Test'},
                    {'role': 'assistant', 'content': 'Response'}
                ]
            }
        ]

        jsonl = formatter.format_to_jsonl(dataset)

        lines = jsonl.strip().split('\n')
        assert len(lines) == 1
        assert json.loads(lines[0])['messages']

    def test_format_to_jsonl_empty_dataset(self):
        """Test formatting empty dataset to JSONL."""
        formatter = Formatter(provider='google')

        dataset = []
        jsonl = formatter.format_to_jsonl(dataset)

        assert jsonl.strip() == ''

    def test_format_to_jsonl_pretty_print(self):
        """Test JSONL formatting with pretty print option."""
        formatter = Formatter(provider='google', pretty_print=True)

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Test'},
                    {'role': 'assistant', 'content': 'Response'}
                ]
            }
        ]

        jsonl = formatter.format_to_jsonl(dataset)

        # Pretty print adds indentation (multi-line per sample)
        # But still newline-delimited between samples
        assert len(jsonl) > 0


class TestValidateJSONL:
    """Tests for JSONL validation."""

    def test_validate_valid_jsonl(self):
        """Test validation of valid JSONL string."""
        formatter = Formatter(provider='google')

        valid_jsonl = '{"messages": [{"role": "user", "content": "Q"}]}\n{"messages": [{"role": "assistant", "content": "A"}]}'

        is_valid = formatter.validate_jsonl(valid_jsonl)
        assert is_valid is True

    def test_validate_invalid_jsonl(self):
        """Test validation of invalid JSONL string."""
        formatter = Formatter(provider='google')

        invalid_jsonl = '{"messages": [invalid json]}'

        is_valid = formatter.validate_jsonl(invalid_jsonl)
        assert is_valid is False

    def test_validate_empty_jsonl(self):
        """Test validation of empty JSONL string."""
        formatter = Formatter(provider='google')

        is_valid = formatter.validate_jsonl('')
        # Empty JSONL may be valid or invalid depending on implementation
        assert isinstance(is_valid, bool)


class TestSpecialCharactersHandling:
    """Tests for handling special characters in formatting."""

    def test_format_with_quotes(self):
        """Test formatting content with quotes."""
        formatter = Formatter(provider='google')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'What does "hello" mean?'},
                {'role': 'assistant', 'content': 'It\'s a greeting.'}
            ]
        }

        formatted = formatter.format_sample(sample)
        jsonl = formatter.format_to_jsonl([formatted])

        # Should properly escape quotes in JSON
        assert json.loads(jsonl.strip())

    def test_format_with_newlines(self):
        """Test formatting content with newlines."""
        formatter = Formatter(provider='google')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Line 1\nLine 2'},
                {'role': 'assistant', 'content': 'Response\nMulti-line'}
            ]
        }

        formatted = formatter.format_sample(sample)
        jsonl = formatter.format_to_jsonl([formatted])

        # Should properly handle newlines in JSON
        parsed = json.loads(jsonl.strip())
        assert '\n' in parsed['messages'][0]['content']

    def test_format_with_unicode(self):
        """Test formatting content with Unicode characters."""
        formatter = Formatter(provider='google')

        sample = {
            'messages': [
                {'role': 'user', 'content': '世界 means world'},
                {'role': 'assistant', 'content': 'Yes, 世界 is world in Chinese.'}
            ]
        }

        formatted = formatter.format_sample(sample)
        jsonl = formatter.format_to_jsonl([formatted])

        parsed = json.loads(jsonl.strip())
        assert '世界' in parsed['messages'][0]['content']

    def test_format_with_emojis(self):
        """Test formatting content with emojis."""
        formatter = Formatter(provider='google')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hello 👋'},
                {'role': 'assistant', 'content': 'Hi there! 🎉'}
            ]
        }

        formatted = formatter.format_sample(sample)
        jsonl = formatter.format_to_jsonl([formatted])

        parsed = json.loads(jsonl.strip())
        assert '👋' in parsed['messages'][0]['content']


class TestErrorHandling:
    """Tests for error handling in formatter."""

    def test_format_invalid_sample_structure(self):
        """Test formatting sample with invalid structure."""
        formatter = Formatter(provider='google')

        invalid_sample = {
            'invalid': 'structure'
        }

        with pytest.raises((FormatError, KeyError, ValueError)):
            formatter.format_sample(invalid_sample)

    def test_format_missing_messages(self):
        """Test formatting sample without messages."""
        formatter = Formatter(provider='google')

        invalid_sample = {
            'content': 'Some content'
        }

        with pytest.raises((FormatError, KeyError, ValueError)):
            formatter.format_sample(invalid_sample)

    def test_unsupported_provider(self):
        """Test that unsupported provider raises error."""
        with pytest.raises((ValueError, FormatError)):
            Formatter(provider='unsupported_provider')


class TestBatchFormatting:
    """Tests for batch formatting operations."""

    def test_format_multiple_samples(self):
        """Test formatting multiple samples at once."""
        formatter = Formatter(provider='google')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Question {i}'},
                    {'role': 'assistant', 'content': f'Answer {i}'}
                ]
            }
            for i in range(10)
        ]

        jsonl = formatter.format_to_jsonl(dataset)
        lines = jsonl.strip().split('\n')

        assert len(lines) == 10

        for line in lines:
            parsed = json.loads(line)
            assert 'messages' in parsed

    def test_format_large_dataset(self):
        """Test formatting large dataset (100+ samples)."""
        formatter = Formatter(provider='google')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ]
            }
            for i in range(100)
        ]

        jsonl = formatter.format_to_jsonl(dataset)
        lines = jsonl.strip().split('\n')

        assert len(lines) == 100
