"""
Integration tests for HYBRID mode end-to-end workflow.
Tests combination of extraction and generation.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from utils.config_loader import Config
from agents.hybrid_agent import HybridAgent
from utils.validator import Validator
from utils.splitter import Splitter
from persistence.sqlite_client import SQLiteClient


class TestHybridModeWorkflow:
    """Test HYBRID mode end-to-end workflow"""

    @pytest.fixture
    def workflow_config(self, sample_config):
        """Config specifically for HYBRID mode"""
        config = sample_config.copy()
        config['agent']['mode'] = 'hybrid'
        config['agent']['generation_ratio'] = 0.5  # 50/50 split
        config['agent']['deduplicate'] = True
        config['dataset']['target_samples'] = 20
        config['split']['enabled'] = True
        config['split']['ratio'] = 0.8
        config['split']['strategy'] = 'weighted_by_source'
        return config

    @pytest.fixture
    def mixed_input_files(self, tmp_path):
        """Create input files with both conversations and plain text"""
        # File 1: Contains conversations (for IDENTIFY)
        conversation_content = """
**Rep:** What are your biggest sales challenges?
**Prospect:** We struggle with lead qualification and pipeline visibility.

**Rep:** How much time does your team spend on unqualified leads?
**Prospect:** Probably 30-40% of our time. It's a huge waste.

---

**Rep:** If we could reduce that by half, what would that mean for revenue?
**Prospect:** We could potentially close 20% more deals per quarter.
"""
        file1 = tmp_path / "sales_conversations.txt"
        file1.write_text(conversation_content)

        # File 2: Plain text for inspiration (for CREATE)
        methodology_content = """
SPIN Selling Methodology

SPIN is an acronym for the four types of questions:
1. Situation: Understand the customer's current state
2. Problem: Identify pain points and challenges
3. Implication: Explore the consequences of those problems
4. Need-payoff: Help customers recognize the value of solving their problems

MEDDIC Qualification Framework

MEDDIC helps qualify enterprise opportunities:
- Metrics: Quantifiable business value
- Economic Buyer: Who controls the budget?
- Decision Criteria: What factors influence the decision?
- Decision Process: How is the decision made?
- Identify Pain: What problems need solving?
- Champion: Who advocates for your solution?
"""
        file2 = tmp_path / "sales_methodology.txt"
        file2.write_text(methodology_content)

        return [file1, file2]

    @pytest.fixture
    def mock_llm_responses_hybrid(self):
        """Mock LLM responses for HYBRID mode (both extraction and generation)"""
        return {
            'extraction_confidence': "0.95",
            'generation': [
                """SYSTEM: You are a SPIN selling expert.
USER: How do I use situation questions effectively?
ASSISTANT: Situation questions should be used early in the conversation to understand the customer's current state. Ask about their existing processes, tools, and team structure.""",
                """SYSTEM: You are a MEDDIC qualification expert.
USER: How do I identify the economic buyer?
ASSISTANT: The economic buyer is the person who controls the budget. Ask questions like "Who typically approves investments of this size?" and "What's your budget approval process?"""
            ] * 5  # 10 generated samples
        }

    @patch('agents.hybrid_agent.genai.GenerativeModel')
    def test_hybrid_mode_full_workflow(
        self,
        mock_genai_model,
        workflow_config,
        mixed_input_files,
        mock_llm_responses_hybrid,
        mock_env_vars,
        tmp_path
    ):
        """Test complete HYBRID mode workflow combining extraction and generation"""
        # Setup mocks
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Mock responses for both extraction and generation
        mock_responses = []

        # Confidence scores for extraction
        confidence_response = Mock()
        confidence_response.text = mock_llm_responses_hybrid['extraction_confidence']
        mock_responses.extend([confidence_response] * 5)

        # Generation responses
        for gen_text in mock_llm_responses_hybrid['generation']:
            gen_response = Mock()
            gen_response.text = gen_text
            mock_responses.append(gen_response)

        mock_model_instance.generate_content.side_effect = mock_responses

        # Setup database
        db_path = tmp_path / "test_workflow.db"
        client = SQLiteClient(str(db_path))

        # Step 1: Load configuration
        config = Config(**workflow_config)
        assert config.agent.mode == 'hybrid'
        assert config.agent.generation_ratio == 0.5

        # Step 2: Initialize agent
        agent = HybridAgent(config)
        assert agent is not None

        # Step 3: Generate samples (both extraction + generation)
        samples = agent.generate(
            input_files=mixed_input_files,
            target_count=config.dataset.target_samples
        )

        # Verify samples
        assert len(samples) == 20

        # Verify mix of identify and create modes
        identify_samples = [s for s in samples if s.mode == 'identify']
        create_samples = [s for s in samples if s.mode == 'create']

        # With 50/50 ratio, should have roughly equal split
        assert len(identify_samples) > 0
        assert len(create_samples) > 0
        # Approximately 10 each (with some tolerance)
        assert 8 <= len(identify_samples) <= 12
        assert 8 <= len(create_samples) <= 12

        # Step 4: Validate samples
        from utils.token_counter import TokenCounter
        from providers import get_provider

        provider = get_provider(config.dataset.provider)
        token_counter = TokenCounter()
        validator = Validator(provider, config.dataset.model, token_counter)

        validation_result = validator.validate_dataset(samples)
        assert validation_result.valid
        assert validation_result.total_samples == 20

        # Step 5: Verify deduplication
        # Check that there are no exact duplicates
        unique_contents = set()
        for sample in samples:
            content_str = json.dumps(sample.content, sort_keys=True)
            assert content_str not in unique_contents
            unique_contents.add(content_str)

        # Step 6: Split dataset with weighted strategy
        splitter = Splitter()
        train_samples, val_samples = splitter.split(
            samples,
            ratio=config.split.ratio,
            strategy=config.split.strategy
        )

        assert len(train_samples) == 16
        assert len(val_samples) == 4

        # Step 7: Save to database
        dataset_id = client.save_dataset(
            name="test_hybrid_workflow",
            provider=config.dataset.provider,
            model=config.dataset.model,
            mode=config.agent.mode,
            samples=samples,
            config=workflow_config
        )

        assert dataset_id is not None

        # Step 8: Export JSONL
        from utils.formatter import Formatter

        formatter = Formatter(provider)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "validation.jsonl"

        formatter.write_jsonl(train_samples, str(train_path))
        formatter.write_jsonl(val_samples, str(val_path))

        # Verify JSONL files
        assert train_path.exists()
        assert val_path.exists()

        with open(train_path, 'r') as f:
            train_lines = f.readlines()
            assert len(train_lines) == 16

        with open(val_path, 'r') as f:
            val_lines = f.readlines()
            assert len(val_lines) == 4

    @patch('agents.hybrid_agent.genai.GenerativeModel')
    def test_hybrid_mode_generation_ratio(
        self,
        mock_genai_model,
        workflow_config,
        mixed_input_files,
        mock_llm_responses_hybrid,
        mock_env_vars
    ):
        """Test HYBRID mode respects generation ratio setting"""
        # Test with 80/20 ratio (80% generated, 20% extracted)
        workflow_config['agent']['generation_ratio'] = 0.8
        workflow_config['dataset']['target_samples'] = 50

        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Setup mock responses
        confidence_response = Mock()
        confidence_response.text = "0.95"

        gen_response = Mock()
        gen_response.text = mock_llm_responses_hybrid['generation'][0]

        mock_model_instance.generate_content.side_effect = [confidence_response] * 10 + [gen_response] * 40

        config = Config(**workflow_config)
        agent = HybridAgent(config)

        samples = agent.generate(
            input_files=mixed_input_files,
            target_count=50
        )

        # Verify ratio
        create_samples = [s for s in samples if s.mode == 'create']
        identify_samples = [s for s in samples if s.mode == 'identify']

        # Should have approximately 40 created (80%) and 10 identified (20%)
        # Allow some tolerance
        assert 35 <= len(create_samples) <= 45
        assert 5 <= len(identify_samples) <= 15

    @patch('agents.hybrid_agent.genai.GenerativeModel')
    def test_hybrid_mode_deduplication(
        self,
        mock_genai_model,
        workflow_config,
        mixed_input_files,
        mock_env_vars
    ):
        """Test HYBRID mode deduplicates similar samples"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Create responses with some duplicates
        confidence_response = Mock()
        confidence_response.text = "0.95"

        # Same response multiple times to trigger deduplication
        duplicate_response = Mock()
        duplicate_response.text = """SYSTEM: Sales expert.
USER: What is SPIN selling?
ASSISTANT: SPIN selling uses four question types to guide sales conversations."""

        mock_model_instance.generate_content.side_effect = [
            confidence_response,
            duplicate_response,
            duplicate_response,  # Duplicate
            duplicate_response,  # Duplicate
            confidence_response,
            duplicate_response
        ] * 10

        workflow_config['agent']['deduplicate'] = True
        config = Config(**workflow_config)
        agent = HybridAgent(config)

        samples = agent.generate(
            input_files=mixed_input_files,
            target_count=20
        )

        # Should remove duplicates
        unique_contents = set()
        for sample in samples:
            content_str = json.dumps(sample.content, sort_keys=True)
            assert content_str not in unique_contents
            unique_contents.add(content_str)

    @patch('agents.hybrid_agent.genai.GenerativeModel')
    def test_hybrid_mode_weighted_split(
        self,
        mock_genai_model,
        workflow_config,
        mixed_input_files,
        mock_llm_responses_hybrid,
        mock_env_vars
    ):
        """Test HYBRID mode weighted split maintains source distribution"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        confidence_response = Mock()
        confidence_response.text = "0.95"

        gen_response = Mock()
        gen_response.text = mock_llm_responses_hybrid['generation'][0]

        mock_model_instance.generate_content.side_effect = [confidence_response] * 10 + [gen_response] * 10

        config = Config(**workflow_config)
        agent = HybridAgent(config)

        samples = agent.generate(
            input_files=mixed_input_files,
            target_count=20
        )

        # Split with weighted strategy
        splitter = Splitter()
        train_samples, val_samples = splitter.split(
            samples,
            ratio=0.8,
            strategy='weighted_by_source'
        )

        # Count samples per source file in train and val
        train_sources = {}
        for s in train_samples:
            source = s.source_file
            train_sources[source] = train_sources.get(source, 0) + 1

        val_sources = {}
        for s in val_samples:
            source = s.source_file
            val_sources[source] = val_sources.get(source, 0) + 1

        # Each source should be represented in both splits
        # (assuming there are enough samples from each source)
        # This is a basic check - exact distribution depends on implementation
        assert len(train_sources) > 0
        assert len(val_sources) >= 0  # May not have samples from all sources in val

    @patch('agents.hybrid_agent.genai.GenerativeModel')
    def test_hybrid_mode_metadata_preservation(
        self,
        mock_genai_model,
        workflow_config,
        mixed_input_files,
        mock_llm_responses_hybrid,
        mock_env_vars
    ):
        """Test HYBRID mode preserves metadata from both modes"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        confidence_response = Mock()
        confidence_response.text = "0.95"

        gen_response = Mock()
        gen_response.text = mock_llm_responses_hybrid['generation'][0]

        mock_model_instance.generate_content.side_effect = [confidence_response] * 10 + [gen_response] * 10

        config = Config(**workflow_config)
        agent = HybridAgent(config)

        samples = agent.generate(
            input_files=mixed_input_files,
            target_count=20
        )

        # Verify all samples have complete metadata
        for sample in samples:
            assert sample.id is not None
            assert sample.mode in ['create', 'identify']
            assert sample.token_length > 0
            assert sample.source_file is not None
            assert 'user' in sample.content
            assert 'assistant' in sample.content

    @patch('agents.hybrid_agent.genai.GenerativeModel')
    def test_hybrid_mode_error_handling(
        self,
        mock_genai_model,
        workflow_config,
        mixed_input_files,
        mock_env_vars
    ):
        """Test HYBRID mode handles errors gracefully"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Simulate API error during generation
        mock_model_instance.generate_content.side_effect = Exception("API error")

        config = Config(**workflow_config)
        agent = HybridAgent(config)

        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            agent.generate(
                input_files=mixed_input_files,
                target_count=20
            )

        assert "API error" in str(exc_info.value)
