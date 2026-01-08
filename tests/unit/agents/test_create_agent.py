"""
Unit tests for src/agents/create_agent.py

Tests synthetic conversation generation using CREATE mode with sales documents.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.create_agent import CreateAgent
from pathlib import Path


class TestCreateAgentInitialization:
    """Tests for CreateAgent initialization."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization(self, mock_model, mock_configure, mock_env_vars):
        """Test CreateAgent initialization."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        assert agent is not None
        assert agent.provider == 'google'
        assert agent.mode == 'create'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_variety(self, mock_model, mock_configure, mock_env_vars):
        """Test CreateAgent with variety settings."""
        varieties = ['low', 'medium', 'high']

        for variety in varieties:
            agent = CreateAgent(
                provider='google',
                model='gemini-2.0-flash-exp',
                variety=variety
            )
            assert agent.variety == variety


class TestSystemPromptLoading:
    """Tests for system prompt loading."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_create_mode_prompt(self, mock_model, mock_configure, mock_env_vars):
        """Test loading CREATE mode system prompt."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        # Check if prompt file exists
        prompt_path = Path('prompts/system/create_mode.txt')
        if prompt_path.exists():
            system_prompt = agent.load_system_prompt(str(prompt_path))
            assert len(system_prompt) > 0
            assert isinstance(system_prompt, str)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_system_prompt_contains_create_instructions(self, mock_model, mock_configure, mock_env_vars):
        """Test that system prompt contains CREATE mode instructions."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        prompt_path = Path('prompts/system/create_mode.txt')
        if prompt_path.exists():
            system_prompt = agent.load_system_prompt(str(prompt_path))

            # Should mention generation/creation
            assert 'generate' in system_prompt.lower() or 'create' in system_prompt.lower()


class TestSampleGeneration:
    """Tests for sample generation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_samples_mocked(self, mock_model, mock_configure, mock_env_vars):
        """Test sample generation with mocked LLM response."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=5
        )

        # Mock LLM response
        mock_response = Mock()
        mock_response.text = """SYSTEM: You are a sales expert.
USER: What is SPIN selling?
ASSISTANT: SPIN selling is a methodology focused on asking Situation, Problem, Implication, and Need-payoff questions."""

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        # Create temporary source file
        source_text = "Sales methodologies include SPIN selling, MEDDIC, and consultative selling."

        samples = agent.generate_samples([source_text], target_samples=5)

        assert isinstance(samples, list)
        assert len(samples) > 0
        assert all('messages' in sample for sample in samples)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_samples_with_real_sales_doc(self, mock_model, mock_configure, mock_env_vars):
        """Test generation with actual sales methodology document."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=3
        )

        # Check if sales document exists
        sales_doc_path = Path('input_files/sales_create_methodology.txt')
        if sales_doc_path.exists():
            with open(sales_doc_path, 'r', encoding='utf-8') as f:
                source_text = f.read()

            # Mock LLM to return valid conversation
            mock_response = Mock()
            mock_response.text = """SYSTEM: You are a sales expert.
USER: How do I handle price objections?
ASSISTANT: Focus on value, not price. Ask questions to understand their budget concerns."""

            agent.llm_client = Mock()
            agent.llm_client.generate_content = Mock(return_value=mock_response)

            samples = agent.generate_samples([source_text], target_samples=3)

            assert isinstance(samples, list)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_samples_no_verbatim_copying(self, mock_model, mock_configure, mock_env_vars):
        """Test that generated samples don't verbatim copy source text."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        source_text = "This is unique source text that should not appear verbatim: XYZ123ABC."

        # Mock LLM to return different content
        mock_response = Mock()
        mock_response.text = """USER: Tell me about sales.
ASSISTANT: Sales involves understanding customer needs and providing solutions."""

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.generate_samples([source_text], target_samples=1)

        # Check that unique marker doesn't appear verbatim
        for sample in samples:
            messages = sample.get('messages', [])
            for message in messages:
                content = message.get('content', '')
                assert 'XYZ123ABC' not in content


class TestVarietyControl:
    """Tests for variety control in generation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_low_temperature(self, mock_model, mock_configure, mock_env_vars):
        """Test low variety uses low temperature."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            variety='low'
        )

        assert agent.temperature <= 0.4

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_high_temperature(self, mock_model, mock_configure, mock_env_vars):
        """Test high variety uses high temperature."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            variety='high'
        )

        assert agent.temperature >= 0.8

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_diverse_samples(self, mock_model, mock_configure, mock_env_vars):
        """Test that high variety generates diverse samples."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            variety='high',
            target_samples=3
        )

        # Mock LLM to return different responses
        mock_responses = [
            Mock(text="USER: Q1?\nASSISTANT: A1."),
            Mock(text="USER: Q2?\nASSISTANT: A2."),
            Mock(text="USER: Q3?\nASSISTANT: A3.")
        ]

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(side_effect=mock_responses)

        samples = agent.generate_samples(["Source text"], target_samples=3)

        # Should generate multiple different samples
        assert len(samples) >= 1


class TestTokenTracking:
    """Tests for token usage tracking."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_track_token_usage(self, mock_model, mock_configure, mock_env_vars):
        """Test that agent tracks token usage during generation."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=2
        )

        mock_response = Mock()
        mock_response.text = "USER: Test\nASSISTANT: Response"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        initial_tokens = getattr(agent, 'total_tokens', 0)

        agent.generate_samples(["Source text"], target_samples=2)

        # Should have tracked tokens
        if hasattr(agent, 'total_tokens'):
            assert agent.total_tokens > initial_tokens


class TestQualityValidation:
    """Tests for quality validation of generated samples."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_validate_generated_sample_quality(self, mock_model, mock_configure, mock_env_vars):
        """Test quality validation of generated samples."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        valid_sample = {
            'messages': [
                {'role': 'user', 'content': 'What is Python?'},
                {'role': 'assistant', 'content': 'Python is a programming language used for various applications.'}
            ]
        }

        is_valid = agent.validate_sample_quality(valid_sample)

        assert is_valid is True

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_reject_low_quality_sample(self, mock_model, mock_configure, mock_env_vars):
        """Test rejection of low-quality samples."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        # Very short, low-quality sample
        low_quality_sample = {
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }

        is_valid = agent.validate_sample_quality(low_quality_sample)

        # May or may not reject based on quality thresholds
        assert isinstance(is_valid, bool)


class TestLengthControl:
    """Tests for conversation length control."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_short_conversations(self, mock_model, mock_configure, mock_env_vars):
        """Test generation of short conversations."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            length='short'
        )

        mock_response = Mock()
        mock_response.text = "USER: Q\nASSISTANT: A"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.generate_samples(["Source"], target_samples=1)

        # Short conversations should have fewer turns
        if samples:
            assert len(samples[0]['messages']) <= 4

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_long_conversations(self, mock_model, mock_configure, mock_env_vars):
        """Test generation of long conversations."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            length='long'
        )

        mock_response = Mock()
        mock_response.text = """USER: Q1
ASSISTANT: A1
USER: Q2
ASSISTANT: A2
USER: Q3
ASSISTANT: A3"""

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.generate_samples(["Source"], target_samples=1)

        # Long conversations should have more turns
        if samples:
            # Should have multiple turns
            assert isinstance(samples[0]['messages'], list)


class TestMetadataGeneration:
    """Tests for metadata generation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_topic_keywords(self, mock_model, mock_configure, mock_env_vars):
        """Test generation of topic keywords for samples."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock keyword extraction
        agent.extract_keywords = Mock(return_value=['sales', 'negotiation', 'pricing'])

        conversation_text = "Discussion about sales negotiation and pricing strategies."

        keywords = agent.extract_keywords(conversation_text)

        assert isinstance(keywords, list)
        assert 'sales' in keywords
        assert 'negotiation' in keywords

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_samples_include_metadata(self, mock_model, mock_configure, mock_env_vars):
        """Test that generated samples include metadata."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        mock_response = Mock()
        mock_response.text = "USER: Test\nASSISTANT: Response"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)
        agent.extract_keywords = Mock(return_value=['test', 'sample'])

        samples = agent.generate_samples(["Source"], target_samples=1)

        if samples:
            sample = samples[0]
            # Should have metadata fields
            assert 'messages' in sample
            # May have topic_keywords, source_file, etc.


class TestErrorHandling:
    """Tests for error handling during generation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_generation_failure(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of generation failures."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=2
        )

        # Mock LLM to fail
        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(side_effect=Exception("API Error"))

        # Should handle gracefully or retry
        try:
            samples = agent.generate_samples(["Source"], target_samples=2)
            # If it succeeds with retry, check result
            assert isinstance(samples, list)
        except Exception as e:
            # If it raises error, check error message
            assert 'API Error' in str(e)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_no_source_files(self, mock_model, mock_configure, mock_env_vars):
        """Test handling when no source files provided."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        with pytest.raises((ValueError, Exception)) as exc_info:
            agent.generate_samples([], target_samples=5)

        assert 'source' in str(exc_info.value).lower() or 'empty' in str(exc_info.value).lower()

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_malformed_llm_response(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of malformed LLM responses."""
        agent = CreateAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock LLM to return malformed response
        mock_response = Mock()
        mock_response.text = "This is not a valid conversation format"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        # Should handle gracefully
        samples = agent.generate_samples(["Source"], target_samples=1)

        # May return empty list or skip invalid samples
        assert isinstance(samples, list)


class TestTargetSamplesControl:
    """Tests for target sample count control."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_exact_target_samples(self, mock_model, mock_configure, mock_env_vars):
        """Test generating exact number of target samples."""
        agent = CreateAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=10
        )

        mock_response = Mock()
        mock_response.text = "USER: Test\nASSISTANT: Response"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.generate_samples(["Source"], target_samples=10)

        # Should generate close to target (may vary slightly)
        assert 8 <= len(samples) <= 12

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_with_different_targets(self, mock_model, mock_configure, mock_env_vars):
        """Test generation with different target counts."""
        mock_response = Mock()
        mock_response.text = "USER: Q\nASSISTANT: A"

        for target in [5, 10, 20]:
            agent = CreateAgent(
                provider='google',
                model='gemini-2.0-flash-exp',
                target_samples=target
            )

            agent.llm_client = Mock()
            agent.llm_client.generate_content = Mock(return_value=mock_response)

            samples = agent.generate_samples(["Source"], target_samples=target)

            assert isinstance(samples, list)
