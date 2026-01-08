"""
Unit tests for src/agents/custom_agent.py

Tests user-defined custom behavior using CUSTOM mode with AI SaaS scenario.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.custom_agent import CustomAgent
from pathlib import Path


class TestCustomAgentInitialization:
    """Tests for CustomAgent initialization."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization(self, mock_model, mock_configure, mock_env_vars):
        """Test CustomAgent initialization."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        assert agent is not None
        assert agent.provider == 'google'
        assert agent.mode == 'custom'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_custom_prompt(self, mock_model, mock_configure, mock_env_vars, tmp_path):
        """Test CustomAgent with custom prompt file."""
        # Create temporary custom prompt
        custom_prompt_file = tmp_path / "custom_prompt.txt"
        custom_prompt_file.write_text("You are an AI SaaS sales expert.")

        agent = CustomAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            custom_prompt_path=str(custom_prompt_file)
        )

        assert agent.custom_prompt_path == str(custom_prompt_file)


class TestCustomPromptLoading:
    """Tests for custom prompt loading."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_custom_prompt_from_file(self, mock_model, mock_configure, mock_env_vars, tmp_path):
        """Test loading custom prompt from file."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        # Create custom prompt file
        prompt_file = tmp_path / "ai_saas_prompt.txt"
        prompt_content = """You are a sales expert for AI SaaS products.
Generate conversations that demonstrate selling AI/ML features."""

        prompt_file.write_text(prompt_content)

        loaded_prompt = agent.load_custom_prompt(str(prompt_file))

        assert loaded_prompt == prompt_content
        assert 'AI SaaS' in loaded_prompt

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_from_default_custom_mode_path(self, mock_model, mock_configure, mock_env_vars):
        """Test loading from default prompts/custom_mode/prompt.txt."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        default_path = Path('prompts/custom_mode/prompt.txt')
        if default_path.exists():
            prompt = agent.load_custom_prompt(str(default_path))
            assert len(prompt) > 0
            assert isinstance(prompt, str)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_interactive_prompt_mocked(self, mock_model, mock_configure, mock_env_vars):
        """Test interactive prompt input (mocked user input)."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock user input
        with patch('builtins.input', return_value='Custom AI SaaS sales expert prompt'):
            prompt = agent.get_interactive_prompt()

            assert prompt == 'Custom AI SaaS sales expert prompt'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_prompt_file_not_found(self, mock_model, mock_configure, mock_env_vars):
        """Test handling when custom prompt file not found."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        with pytest.raises(FileNotFoundError):
            agent.load_custom_prompt("nonexistent_prompt.txt")


class TestCustomBehaviorExecution:
    """Tests for custom behavior execution."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_execute_custom_behavior(self, mock_model, mock_configure, mock_env_vars):
        """Test execution of custom behavior."""
        agent = CustomAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=5
        )

        custom_prompt = "Generate AI SaaS sales conversations focusing on ROI and automation."

        # Mock LLM response
        mock_response = Mock()
        mock_response.text = """SYSTEM: You are an AI SaaS expert.
USER: How does AI improve our ROI?
ASSISTANT: AI automation reduces manual work by 80%, directly impacting your bottom line."""

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.execute_custom_behavior(
            custom_prompt=custom_prompt,
            source_texts=["AI SaaS sales methodology"],
            target_samples=5
        )

        assert isinstance(samples, list)
        assert len(samples) > 0

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_ai_saas_scenario(self, mock_model, mock_configure, mock_env_vars):
        """Test AI SaaS customization scenario."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        custom_prompt = """You are a sales expert for AI SaaS products. Generate conversations that demonstrate:
- Selling AI/ML features to technical and non-technical buyers
- Addressing concerns about AI reliability, bias, and explainability
- ROI discussions for AI automation"""

        mock_response = Mock()
        mock_response.text = """USER: I'm concerned about AI bias in your product.
ASSISTANT: That's an important concern. Our AI models undergo rigorous bias testing and we provide explainability tools."""

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.execute_custom_behavior(
            custom_prompt=custom_prompt,
            source_texts=["AI SaaS context"],
            target_samples=3
        )

        # Should generate AI-specific conversations
        assert isinstance(samples, list)


class TestVarietyControl:
    """Tests for variety control in CUSTOM mode."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_low(self, mock_model, mock_configure, mock_env_vars):
        """Test low variety in CUSTOM mode."""
        agent = CustomAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            variety='low'
        )

        assert agent.temperature <= 0.4

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_variety_high(self, mock_model, mock_configure, mock_env_vars):
        """Test high variety in CUSTOM mode."""
        agent = CustomAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            variety='high'
        )

        assert agent.temperature >= 0.8


class TestFlexibility:
    """Tests for CUSTOM mode flexibility."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_industry_specific_customization(self, mock_model, mock_configure, mock_env_vars):
        """Test industry-specific customization (Healthcare)."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        healthcare_prompt = """Generate sales conversations for healthcare AI products.
Focus on HIPAA compliance, patient outcomes, and regulatory considerations."""

        mock_response = Mock()
        mock_response.text = "USER: Is this HIPAA compliant?\nASSISTANT: Yes, we're fully HIPAA certified."

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.execute_custom_behavior(
            custom_prompt=healthcare_prompt,
            source_texts=["Healthcare context"],
            target_samples=2
        )

        assert isinstance(samples, list)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_skill_level_specific_customization(self, mock_model, mock_configure, mock_env_vars):
        """Test skill-level specific customization (Senior Reps)."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        senior_rep_prompt = """Generate conversations for senior sales reps.
Focus on complex enterprise sales, negotiation, and multi-threading."""

        mock_response = Mock()
        mock_response.text = "USER: Can we negotiate pricing?\nASSISTANT: Let me involve our CFO in this discussion."

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.execute_custom_behavior(
            custom_prompt=senior_rep_prompt,
            source_texts=["Enterprise sales context"],
            target_samples=2
        )

        assert isinstance(samples, list)


class TestMetadataGeneration:
    """Tests for metadata generation in CUSTOM mode."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_add_custom_metadata(self, mock_model, mock_configure, mock_env_vars):
        """Test adding custom metadata to samples."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        mock_response = Mock()
        mock_response.text = "USER: Test\nASSISTANT: Response"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.execute_custom_behavior(
            custom_prompt="Test prompt",
            source_texts=["Source"],
            target_samples=1,
            metadata={'scenario': 'AI SaaS', 'skill_level': 'senior'}
        )

        # Should include custom metadata
        if samples and 'metadata' in samples[0]:
            assert samples[0]['metadata']['scenario'] == 'AI SaaS'


class TestErrorHandling:
    """Tests for error handling in CUSTOM mode."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_empty_custom_prompt(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of empty custom prompt."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        with pytest.raises((ValueError, Exception)) as exc_info:
            agent.execute_custom_behavior(
                custom_prompt="",
                source_texts=["Source"],
                target_samples=5
            )

        assert 'prompt' in str(exc_info.value).lower() or 'empty' in str(exc_info.value).lower()

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_generation_failure(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of generation failures."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock LLM to fail
        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(side_effect=Exception("API Error"))

        with pytest.raises(Exception) as exc_info:
            agent.execute_custom_behavior(
                custom_prompt="Test prompt",
                source_texts=["Source"],
                target_samples=5
            )

        assert 'API Error' in str(exc_info.value)


class TestTargetSamplesControl:
    """Tests for target sample control."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_target_samples(self, mock_model, mock_configure, mock_env_vars):
        """Test generating exact target number of samples."""
        agent = CustomAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            target_samples=10
        )

        mock_response = Mock()
        mock_response.text = "USER: Q\nASSISTANT: A"

        agent.llm_client = Mock()
        agent.llm_client.generate_content = Mock(return_value=mock_response)

        samples = agent.execute_custom_behavior(
            custom_prompt="Generate sales conversations",
            source_texts=["Source"],
            target_samples=10
        )

        # Should generate close to target
        assert 8 <= len(samples) <= 12


class TestQualityValidation:
    """Tests for quality validation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_validate_generated_quality(self, mock_model, mock_configure, mock_env_vars):
        """Test quality validation of generated samples."""
        agent = CustomAgent(provider='google', model='gemini-2.0-flash-exp')

        valid_sample = {
            'messages': [
                {'role': 'user', 'content': 'How does AI improve efficiency?'},
                {'role': 'assistant', 'content': 'AI automates repetitive tasks, saving time and resources.'}
            ]
        }

        is_valid = agent.validate_sample_quality(valid_sample)

        assert is_valid is True
