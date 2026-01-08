"""
Integration tests for CREATE mode end-to-end workflow.
Tests the complete flow with real API calls (mocked).
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from utils.config_loader import Config
from agents.create_agent import CreateAgent
from utils.validator import Validator
from utils.splitter import Splitter
from persistence.sqlite_client import SQLiteClient


class TestCreateModeWorkflow:
    """Test CREATE mode end-to-end workflow"""

    @pytest.fixture
    def workflow_config(self, sample_config):
        """Config specifically for CREATE mode"""
        config = sample_config.copy()
        config['agent']['mode'] = 'create'
        config['dataset']['target_samples'] = 10
        config['split']['enabled'] = True
        config['split']['ratio'] = 0.8
        return config

    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for CREATE mode"""
        return [
            """SYSTEM: You are a helpful sales assistant.
USER: What is SPIN selling?
ASSISTANT: SPIN selling is a methodology that uses four types of questions: Situation, Problem, Implication, and Need-payoff questions to guide sales conversations.""",
            """SYSTEM: You are a sales expert.
USER: How do I handle price objections?
ASSISTANT: Address price objections by focusing on value delivered, ROI, and comparing total cost of ownership rather than just upfront price."""
        ] * 5  # 10 responses total

    @patch('agents.create_agent.genai.GenerativeModel')
    def test_create_mode_full_workflow(
        self,
        mock_genai_model,
        workflow_config,
        mock_llm_responses,
        temp_input_files,
        mock_env_vars,
        tmp_path
    ):
        """Test complete CREATE mode workflow from config to export"""
        # Setup mocks
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Create response mocks
        mock_responses = []
        for response_text in mock_llm_responses:
            mock_response = Mock()
            mock_response.text = response_text
            mock_responses.append(mock_response)

        mock_model_instance.generate_content.side_effect = mock_responses

        # Setup database
        db_path = tmp_path / "test_workflow.db"
        client = SQLiteClient(str(db_path))

        # Step 1: Load configuration
        config = Config(**workflow_config)
        assert config.agent.mode == 'create'
        assert config.dataset.target_samples == 10

        # Step 2: Initialize agent
        agent = CreateAgent(config)
        assert agent is not None

        # Step 3: Generate samples
        samples = agent.generate(
            input_files=list(temp_input_files.glob('*.txt')),
            target_count=config.dataset.target_samples
        )

        # Verify samples generated
        assert len(samples) == 10
        assert all(s.mode == 'create' for s in samples)
        assert all(s.content['user'] for s in samples)
        assert all(s.content['assistant'] for s in samples)

        # Step 4: Validate samples
        from utils.token_counter import TokenCounter
        from providers import get_provider

        provider = get_provider(config.dataset.provider)
        token_counter = TokenCounter()
        validator = Validator(provider, config.dataset.model, token_counter)

        validation_result = validator.validate_dataset(samples)
        assert validation_result.valid
        assert validation_result.total_samples == 10
        assert validation_result.total_tokens > 0

        # Step 5: Split dataset
        splitter = Splitter()
        train_samples, val_samples = splitter.split(
            samples,
            ratio=config.split.ratio,
            strategy=config.split.strategy
        )

        assert len(train_samples) == 8
        assert len(val_samples) == 2
        assert all(s.split == 'train' for s in train_samples)
        assert all(s.split == 'validation' for s in val_samples)

        # Step 6: Save to database
        dataset_id = client.save_dataset(
            name="test_create_workflow",
            provider=config.dataset.provider,
            model=config.dataset.model,
            mode=config.agent.mode,
            samples=samples,
            config=workflow_config
        )

        assert dataset_id is not None

        # Step 7: Export JSONL
        from utils.formatter import Formatter

        formatter = Formatter(provider)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train_path = output_dir / "train.jsonl"
        formatter.write_jsonl(train_samples, str(train_path))

        # Verify JSONL file
        assert train_path.exists()
        with open(train_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 8

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert 'messages' in data or 'system' in data

        # Step 8: Verify database persistence
        retrieved_samples = client.get_samples(dataset_id)
        assert len(retrieved_samples) == 10

    @patch('agents.create_agent.genai.GenerativeModel')
    def test_create_mode_with_variety_settings(
        self,
        mock_genai_model,
        workflow_config,
        mock_llm_responses,
        temp_input_files,
        mock_env_vars
    ):
        """Test CREATE mode with different variety settings"""
        # Test with high variety
        workflow_config['agent']['variety'] = 'high'
        config = Config(**workflow_config)

        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        mock_response = Mock()
        mock_response.text = mock_llm_responses[0]
        mock_model_instance.generate_content.return_value = mock_response

        agent = CreateAgent(config)

        # Verify temperature is set high
        assert agent.temperature == 0.9

        # Test with low variety
        workflow_config['agent']['variety'] = 'low'
        config = Config(**workflow_config)
        agent = CreateAgent(config)
        assert agent.temperature == 0.3

    @patch('agents.create_agent.genai.GenerativeModel')
    def test_create_mode_respects_token_limits(
        self,
        mock_genai_model,
        workflow_config,
        temp_input_files,
        mock_env_vars
    ):
        """Test CREATE mode respects provider token limits"""
        # Setup mock to return very long response
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        long_response = Mock()
        long_response.text = f"""SYSTEM: You are a sales expert.
USER: Tell me about sales.
ASSISTANT: {'Very long response ' * 10000}"""  # Intentionally very long

        mock_model_instance.generate_content.return_value = long_response

        config = Config(**workflow_config)
        agent = CreateAgent(config)

        # Generate one sample
        samples = agent.generate(
            input_files=list(temp_input_files.glob('*.txt')),
            target_count=1
        )

        # Verify token limit enforcement
        from utils.token_counter import TokenCounter
        from providers import get_provider

        provider = get_provider(config.dataset.provider)
        token_counter = TokenCounter()

        sample_tokens = token_counter.count_tokens(
            json.dumps(samples[0].content),
            config.dataset.provider,
            config.dataset.model
        )

        max_tokens = provider.get_token_limit(config.dataset.model)
        assert sample_tokens <= max_tokens

    @patch('agents.create_agent.genai.GenerativeModel')
    def test_create_mode_no_verbatim_copying(
        self,
        mock_genai_model,
        workflow_config,
        temp_input_files,
        mock_env_vars
    ):
        """Test CREATE mode doesn't copy verbatim from source files"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = """SYSTEM: Sales training assistant.
USER: What is consultative selling?
ASSISTANT: Consultative selling is an approach where you focus on understanding customer needs through questions and providing tailored solutions."""

        mock_model_instance.generate_content.return_value = mock_response

        # Read source file content
        source_content = (temp_input_files / "sample.txt").read_text()

        config = Config(**workflow_config)
        agent = CreateAgent(config)

        samples = agent.generate(
            input_files=list(temp_input_files.glob('*.txt')),
            target_count=1
        )

        # Verify generated content is different from source
        generated_text = samples[0].content['assistant']
        assert generated_text != source_content

        # Allow some overlap but not full verbatim copying
        # (Checking that it's not just a substring)
        assert generated_text not in source_content

    @patch('agents.create_agent.genai.GenerativeModel')
    def test_create_mode_error_handling(
        self,
        mock_genai_model,
        workflow_config,
        temp_input_files,
        mock_env_vars
    ):
        """Test CREATE mode handles API errors gracefully"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Simulate API error
        mock_model_instance.generate_content.side_effect = Exception("API rate limit exceeded")

        config = Config(**workflow_config)
        agent = CreateAgent(config)

        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            agent.generate(
                input_files=list(temp_input_files.glob('*.txt')),
                target_count=1
            )

        assert "API rate limit exceeded" in str(exc_info.value)

    @patch('agents.create_agent.genai.GenerativeModel')
    def test_create_mode_metadata_tracking(
        self,
        mock_genai_model,
        workflow_config,
        mock_llm_responses,
        temp_input_files,
        mock_env_vars
    ):
        """Test CREATE mode tracks metadata correctly"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = mock_llm_responses[0]
        mock_model_instance.generate_content.return_value = mock_response

        config = Config(**workflow_config)
        agent = CreateAgent(config)

        samples = agent.generate(
            input_files=list(temp_input_files.glob('*.txt')),
            target_count=1
        )

        sample = samples[0]

        # Verify metadata
        assert sample.id is not None
        assert sample.mode == 'create'
        assert sample.token_length > 0
        assert len(sample.topic_keywords) > 0
        assert sample.source_file is not None
        assert 'system' in sample.content or 'user' in sample.content
        assert 'assistant' in sample.content
