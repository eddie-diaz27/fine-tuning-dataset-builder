"""
Unit tests for src/agents/base_agent.py

Tests base agent initialization, retry logic, conversation parsing, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.base_agent import BaseAgent
import time


class TestBaseAgentInitialization:
    """Tests for BaseAgent initialization."""

    def test_initialization_google(self, mock_env_vars):
        """Test BaseAgent initialization with Google provider."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

                assert agent.provider == 'google'
                assert agent.model == 'gemini-2.0-flash-exp'

    def test_initialization_openai(self, mock_env_vars):
        """Test BaseAgent initialization with OpenAI provider."""
        with patch('openai.OpenAI') as mock_openai:
            agent = BaseAgent(provider='openai', model='gpt-4o')

            assert agent.provider == 'openai'
            assert agent.model == 'gpt-4o'

    def test_initialization_anthropic(self, mock_env_vars):
        """Test BaseAgent initialization with Anthropic provider."""
        with patch('anthropic.Anthropic') as mock_anthropic:
            agent = BaseAgent(provider='anthropic', model='claude-3-haiku-20240307')

            assert agent.provider == 'anthropic'
            assert agent.model == 'claude-3-haiku-20240307'

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises error."""
        with pytest.raises(ValueError) as exc_info:
            BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        assert 'API key' in str(exc_info.value) or 'GOOGLE_API_KEY' in str(exc_info.value)

    def test_initialization_invalid_provider(self, mock_env_vars):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            BaseAgent(provider='invalid_provider', model='model')

        assert 'Unsupported' in str(exc_info.value) or 'provider' in str(exc_info.value).lower()


class TestTemperatureSettings:
    """Tests for temperature and variety settings."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_low(self, mock_model, mock_configure, mock_env_vars):
        """Test that 'low' variety sets low temperature."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', variety='low')

        # Low variety should have low temperature (e.g., 0.3)
        assert agent.temperature <= 0.4

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_medium(self, mock_model, mock_configure, mock_env_vars):
        """Test that 'medium' variety sets medium temperature."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', variety='medium')

        # Medium variety should have medium temperature (e.g., 0.7)
        assert 0.5 <= agent.temperature <= 0.8

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_high(self, mock_model, mock_configure, mock_env_vars):
        """Test that 'high' variety sets high temperature."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', variety='high')

        # High variety should have high temperature (e.g., 0.9)
        assert agent.temperature >= 0.8


class TestConversationParsing:
    """Tests for conversation parsing."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_parse_conversation_basic(self, mock_model, mock_configure, mock_env_vars):
        """Test parsing basic conversation format."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        conversation_text = """SYSTEM: You are a helpful assistant.
USER: What is Python?
ASSISTANT: Python is a programming language."""

        parsed = agent.parse_conversation(conversation_text)

        assert isinstance(parsed, dict)
        assert 'messages' in parsed
        assert len(parsed['messages']) == 3
        assert parsed['messages'][0]['role'] == 'system'
        assert parsed['messages'][1]['role'] == 'user'
        assert parsed['messages'][2]['role'] == 'assistant'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_parse_conversation_no_system(self, mock_model, mock_configure, mock_env_vars):
        """Test parsing conversation without system message."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        conversation_text = """USER: Hello
ASSISTANT: Hi there!"""

        parsed = agent.parse_conversation(conversation_text)

        assert 'messages' in parsed
        assert len(parsed['messages']) == 2
        assert parsed['messages'][0]['role'] == 'user'
        assert parsed['messages'][1]['role'] == 'assistant'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_parse_conversation_multiline(self, mock_model, mock_configure, mock_env_vars):
        """Test parsing conversation with multiline content."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        conversation_text = """USER: Can you explain?
Line 1
Line 2
ASSISTANT: Sure!
Response line 1
Response line 2"""

        parsed = agent.parse_conversation(conversation_text)

        assert 'messages' in parsed
        # Should handle multiline content correctly
        user_content = parsed['messages'][0]['content']
        assert 'Line 1' in user_content or 'Line 2' in user_content


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_keywords_mocked(self, mock_model, mock_configure, mock_env_vars):
        """Test keyword extraction with mocked LLM response."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock the LLM response
        mock_response = Mock()
        mock_response.text = "python, programming, coding, language"
        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        conversation_text = "Python is a programming language used for coding."

        keywords = agent.extract_keywords(conversation_text)

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert any('python' in kw.lower() for kw in keywords)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_keywords_empty_text(self, mock_model, mock_configure, mock_env_vars):
        """Test keyword extraction with empty text."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        keywords = agent.extract_keywords("")

        assert isinstance(keywords, list)
        assert len(keywords) == 0


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_retry_on_failure(self, mock_model, mock_configure, mock_env_vars):
        """Test retry logic when LLM call fails."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', max_retries=3)

        # Mock LLM client to fail twice then succeed
        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(
            side_effect=[
                Exception("API Error"),
                Exception("API Error"),
                Mock(text="Success response")
            ]
        )

        # Should succeed on 3rd attempt
        response = agent.call_llm_with_retry("Test prompt")

        assert response.text == "Success response"
        assert agent.llm_client.generate_content.call_count == 3

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_retry_exponential_backoff(self, mock_model, mock_configure, mock_env_vars):
        """Test exponential backoff between retries."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', max_retries=3)

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(
            side_effect=[
                Exception("API Error"),
                Mock(text="Success")
            ]
        )

        start_time = time.time()
        agent.call_llm_with_retry("Test prompt")
        elapsed = time.time() - start_time

        # Should have some delay from backoff (at least 1 second for first retry)
        assert elapsed >= 0.5  # Allow for timing variance

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_retry_max_attempts_exceeded(self, mock_model, mock_configure, mock_env_vars):
        """Test that max retries raises error after exhausting attempts."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', max_retries=2)

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(side_effect=Exception("API Error"))

        with pytest.raises(Exception) as exc_info:
            agent.call_llm_with_retry("Test prompt")

        assert "API Error" in str(exc_info.value)
        assert agent.llm_client.generate_content.call_count == 2


class TestErrorHandling:
    """Tests for error handling."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_auth_error(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of authentication errors."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(
            side_effect=Exception("Invalid API key")
        )

        with pytest.raises(Exception) as exc_info:
            agent.call_llm_with_retry("Test prompt", max_retries=1)

        assert 'API key' in str(exc_info.value) or 'Invalid' in str(exc_info.value)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_rate_limit_error(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of rate limit errors."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp', max_retries=2)

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(
            side_effect=[
                Exception("Rate limit exceeded"),
                Mock(text="Success after rate limit")
            ]
        )

        # Should retry and succeed
        response = agent.call_llm_with_retry("Test prompt")
        assert response.text == "Success after rate limit"


class TestPromptLoading:
    """Tests for loading system prompts."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_system_prompt(self, mock_model, mock_configure, mock_env_vars, tmp_path):
        """Test loading system prompt from file."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        # Create temporary prompt file
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("You are a helpful assistant for testing.")

        loaded_prompt = agent.load_system_prompt(str(prompt_file))

        assert loaded_prompt == "You are a helpful assistant for testing."

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_system_prompt_missing_file(self, mock_model, mock_configure, mock_env_vars):
        """Test loading prompt from non-existent file."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        with pytest.raises(FileNotFoundError):
            agent.load_system_prompt("nonexistent_prompt.txt")


class TestTokenTracking:
    """Tests for token usage tracking."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_track_tokens(self, mock_model, mock_configure, mock_env_vars):
        """Test that agent tracks token usage."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        # Initialize token counter if not already done
        if hasattr(agent, 'total_tokens'):
            initial_tokens = agent.total_tokens
        else:
            initial_tokens = 0

        # Simulate processing
        test_text = "This is a test message for token counting."
        agent.track_tokens(test_text)

        # Should have updated token count
        if hasattr(agent, 'total_tokens'):
            assert agent.total_tokens > initial_tokens

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_reset_token_count(self, mock_model, mock_configure, mock_env_vars):
        """Test resetting token count."""
        agent = BaseAgent(provider='google', model='gemini-2.0-flash-exp')

        # Track some tokens
        agent.track_tokens("Test message")

        # Reset
        if hasattr(agent, 'reset_tokens'):
            agent.reset_tokens()
            assert agent.total_tokens == 0
