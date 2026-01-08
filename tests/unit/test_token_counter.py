"""
Unit tests for src/utils/token_counter.py

Tests provider-specific token counting including tiktoken, approximations, and caching.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.utils.token_counter import TokenCounter


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_initialization(self):
        """Test TokenCounter initialization."""
        counter = TokenCounter()
        assert counter is not None
        assert hasattr(counter, 'cache')

    def test_openai_token_counting(self):
        """Test OpenAI token counting using tiktoken."""
        counter = TokenCounter()
        text = "This is a test sentence for token counting."

        # Test with gpt-4o model
        tokens = counter.count_tokens(text, 'openai', 'gpt-4o')
        assert isinstance(tokens, int)
        assert tokens > 0

        # Should be approximately 9-11 tokens for this sentence
        assert 5 < tokens < 15

    def test_openai_different_models(self):
        """Test OpenAI tokenization with different models."""
        counter = TokenCounter()
        text = "Hello, world!"

        models = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']

        for model in models:
            tokens = counter.count_tokens(text, 'openai', model)
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_anthropic_approximation(self):
        """Test Anthropic token count approximation (~4 chars/token)."""
        counter = TokenCounter()

        # Test with known character count
        text = "1234"  # 4 characters = ~1 token
        tokens = counter.count_tokens(text, 'anthropic', 'claude-3-haiku-20240307')

        assert tokens == 1

        # Test with longer text
        text = "A" * 100  # 100 characters = ~25 tokens
        tokens = counter.count_tokens(text, 'anthropic', 'claude-3-haiku-20240307')

        assert tokens == 25

    def test_google_token_counting_mocked(self):
        """Test Google token counting with mocked API call."""
        counter = TokenCounter()
        text = "Test text for Google tokenization."

        with patch.object(counter, '_count_google', return_value=10) as mock_google:
            tokens = counter.count_tokens(text, 'google', 'gemini-2.0-flash-exp')

            assert tokens == 10
            mock_google.assert_called_once()

    def test_mistral_token_counting(self):
        """Test Mistral token counting (uses tiktoken as proxy)."""
        counter = TokenCounter()
        text = "This is a test for Mistral tokenization."

        tokens = counter.count_tokens(text, 'mistral', 'mistral-7b')
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_llama_token_counting(self):
        """Test Llama token counting (uses tiktoken as proxy)."""
        counter = TokenCounter()
        text = "This is a test for Llama tokenization."

        tokens = counter.count_tokens(text, 'llama', 'llama-3.1-8b')
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_empty_string_returns_zero(self):
        """Test that empty string returns 0 tokens."""
        counter = TokenCounter()

        providers_models = [
            ('openai', 'gpt-4o'),
            ('anthropic', 'claude-3-haiku-20240307'),
            ('mistral', 'mistral-7b'),
            ('llama', 'llama-3.1-8b')
        ]

        for provider, model in providers_models:
            tokens = counter.count_tokens("", provider, model)
            assert tokens == 0

    def test_very_long_text(self):
        """Test tokenization of very long text."""
        counter = TokenCounter()

        # Create a very long text (10,000 words)
        long_text = " ".join(["word"] * 10000)

        tokens = counter.count_tokens(long_text, 'openai', 'gpt-4o')
        assert isinstance(tokens, int)
        assert tokens > 10000  # Should be roughly equal to word count
        assert tokens < 15000  # Reasonable upper bound

    def test_caching_works(self):
        """Test that caching mechanism works correctly."""
        counter = TokenCounter()
        text = "Test caching mechanism."

        # First call - should compute
        tokens1 = counter.count_tokens(text, 'openai', 'gpt-4o')

        # Second call with same inputs - should use cache
        tokens2 = counter.count_tokens(text, 'openai', 'gpt-4o')

        assert tokens1 == tokens2

        # Verify cache was used (both calls return same result)
        assert len(counter.cache) > 0

    def test_cache_different_providers(self):
        """Test that cache differentiates between providers."""
        counter = TokenCounter()
        text = "Test text"

        # Count with different providers
        tokens_openai = counter.count_tokens(text, 'openai', 'gpt-4o')
        tokens_anthropic = counter.count_tokens(text, 'anthropic', 'claude-3-haiku-20240307')

        # Results should be different (different tokenization methods)
        assert tokens_openai != tokens_anthropic

    def test_special_characters_handling(self):
        """Test tokenization of text with special characters."""
        counter = TokenCounter()

        special_texts = [
            "Hello! How are you?",
            "Test with 数字 and 한글",
            "Emoji test 😀🎉",
            "Code: print('hello')",
            "Math: 2 + 2 = 4",
        ]

        for text in special_texts:
            tokens = counter.count_tokens(text, 'openai', 'gpt-4o')
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_multiline_text(self):
        """Test tokenization of multiline text."""
        counter = TokenCounter()

        multiline = """Line 1
Line 2
Line 3"""

        tokens = counter.count_tokens(multiline, 'openai', 'gpt-4o')
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises appropriate error."""
        counter = TokenCounter()

        with pytest.raises((ValueError, KeyError, AttributeError)):
            counter.count_tokens("test", "unsupported_provider", "model")

    def test_token_count_consistency(self):
        """Test that token count is consistent across multiple calls."""
        counter = TokenCounter()
        text = "Consistency test text."

        # Call multiple times
        results = [counter.count_tokens(text, 'openai', 'gpt-4o') for _ in range(5)]

        # All results should be identical
        assert len(set(results)) == 1

    def test_whitespace_handling(self):
        """Test handling of various whitespace patterns."""
        counter = TokenCounter()

        whitespace_texts = [
            "  leading spaces",
            "trailing spaces  ",
            "multiple    spaces",
            "\ttabs\t",
            "\nnewlines\n"
        ]

        for text in whitespace_texts:
            tokens = counter.count_tokens(text, 'openai', 'gpt-4o')
            assert isinstance(tokens, int)
            assert tokens >= 0


class TestTokenCounterEdgeCases:
    """Tests for edge cases in token counting."""

    def test_single_character(self):
        """Test tokenization of single character."""
        counter = TokenCounter()

        tokens = counter.count_tokens("a", 'openai', 'gpt-4o')
        assert tokens == 1

    def test_single_word(self):
        """Test tokenization of single word."""
        counter = TokenCounter()

        tokens = counter.count_tokens("hello", 'openai', 'gpt-4o')
        assert tokens >= 1

    def test_numbers_only(self):
        """Test tokenization of numbers."""
        counter = TokenCounter()

        tokens = counter.count_tokens("123456789", 'openai', 'gpt-4o')
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_punctuation_only(self):
        """Test tokenization of punctuation."""
        counter = TokenCounter()

        tokens = counter.count_tokens("!!!", 'openai', 'gpt-4o')
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_mixed_languages(self):
        """Test tokenization of mixed language text."""
        counter = TokenCounter()

        mixed = "Hello world 你好世界 こんにちは"
        tokens = counter.count_tokens(mixed, 'openai', 'gpt-4o')

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_code_snippets(self):
        """Test tokenization of code snippets."""
        counter = TokenCounter()

        code = """
def hello():
    print("Hello, world!")
"""
        tokens = counter.count_tokens(code, 'openai', 'gpt-4o')

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_json_content(self):
        """Test tokenization of JSON content."""
        counter = TokenCounter()

        json_text = '{"key": "value", "number": 123}'
        tokens = counter.count_tokens(json_text, 'openai', 'gpt-4o')

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_repeated_tokens(self):
        """Test tokenization of repeated tokens."""
        counter = TokenCounter()

        repeated = "test " * 100
        tokens = counter.count_tokens(repeated, 'openai', 'gpt-4o')

        assert isinstance(tokens, int)
        # Should be roughly 100 tokens (one per "test")
        assert 90 < tokens < 110
