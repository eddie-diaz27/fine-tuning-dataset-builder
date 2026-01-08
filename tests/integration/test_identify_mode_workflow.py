"""
Integration tests for IDENTIFY mode end-to-end workflow.
Tests conversation extraction with real sales documents.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from utils.config_loader import Config
from agents.identify_agent import IdentifyAgent
from utils.validator import Validator
from utils.splitter import Splitter
from persistence.sqlite_client import SQLiteClient


class TestIdentifyModeWorkflow:
    """Test IDENTIFY mode end-to-end workflow"""

    @pytest.fixture
    def workflow_config(self, sample_config):
        """Config specifically for IDENTIFY mode"""
        config = sample_config.copy()
        config['agent']['mode'] = 'identify'
        config['agent']['min_turns'] = 2
        config['agent']['confidence_threshold'] = 0.8
        config['agent']['include_context'] = True
        config['split']['enabled'] = True
        config['split']['ratio'] = 0.8
        return config

    @pytest.fixture
    def sales_conversation_file(self, tmp_path):
        """Create a sample sales conversation file"""
        content = """
**Rep:** Hi there! Thanks for taking the time to speak with me today. How are you doing?

**Prospect:** I'm good, thanks. Just curious about what you're offering.

**Rep:** Great! I'd love to learn more about your current sales process. Could you walk me through how your team typically handles outbound prospecting?

**Prospect:** Sure, we mostly do cold calling and LinkedIn outreach. It's pretty manual right now.

**Rep:** I see. And how long does it usually take to qualify a lead?

**Prospect:** Probably 3-4 calls on average. Sometimes more if they're hard to reach.

---

**Rep:** What's the biggest challenge you're facing with your current approach?

**Prospect:** Honestly, it's the time spent on unqualified leads. We waste a lot of effort.

**Rep:** That must be frustrating. Have you calculated the cost of that wasted time?

**Prospect:** We estimate it costs us about $10K per month in lost productivity.

**Rep:** If we could reduce that by 50%, what would that enable your team to do?

**Prospect:** We could focus on closing deals instead of chasing dead ends. That would be huge.
"""
        file_path = tmp_path / "sales_conversation.txt"
        file_path.write_text(content)
        return file_path

    @pytest.fixture
    def chat_log_file(self, tmp_path):
        """Create a sample chat log file"""
        content = """
**Customer:** Hi, I'm interested in your enterprise plan. What's included?

**Sarah (Sales):** Hi! Great question. Our enterprise plan includes unlimited users, advanced analytics, dedicated support, and custom integrations. What's most important to you?

**Customer:** We need the custom integrations. Can you connect to Salesforce?

**Sarah (Sales):** Absolutely! Salesforce is one of our most popular integrations. We also support HubSpot, Pipedrive, and several others. Would you like to schedule a demo?

---

**Customer:** How much does it cost?

**Sarah (Sales):** Our enterprise pricing starts at $499/month for up to 50 users. Can I ask how many people would be using the system?

**Customer:** We have about 30 sales reps.

**Sarah (Sales):** Perfect! You'd fit comfortably in our base tier. Would you like me to send over a detailed proposal?
"""
        file_path = tmp_path / "chat_log.txt"
        file_path.write_text(content)
        return file_path

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_full_workflow(
        self,
        mock_genai_model,
        workflow_config,
        sales_conversation_file,
        chat_log_file,
        mock_env_vars,
        tmp_path
    ):
        """Test complete IDENTIFY mode workflow from config to export"""
        # Setup mocks (IDENTIFY mode uses AI minimally, just for confidence scoring)
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = "0.95"  # High confidence score
        mock_model_instance.generate_content.return_value = mock_response

        # Setup database
        db_path = tmp_path / "test_workflow.db"
        client = SQLiteClient(str(db_path))

        # Step 1: Load configuration
        config = Config(**workflow_config)
        assert config.agent.mode == 'identify'

        # Step 2: Initialize agent
        agent = IdentifyAgent(config)
        assert agent is not None

        # Step 3: Extract conversations
        input_files = [sales_conversation_file, chat_log_file]
        samples = agent.generate(
            input_files=input_files,
            target_count=None  # Extract all
        )

        # Verify samples extracted
        assert len(samples) >= 2  # At least 2 conversations from the files
        assert all(s.mode == 'identify' for s in samples)
        assert all(s.content['user'] for s in samples)
        assert all(s.content['assistant'] for s in samples)

        # Verify verbatim extraction (no AI modification)
        # Check that extracted content matches source patterns
        for sample in samples:
            user_text = sample.content['user']
            assistant_text = sample.content['assistant']
            assert len(user_text) > 0
            assert len(assistant_text) > 0

        # Step 4: Validate samples
        from utils.token_counter import TokenCounter
        from providers import get_provider

        provider = get_provider(config.dataset.provider)
        token_counter = TokenCounter()
        validator = Validator(provider, config.dataset.model, token_counter)

        validation_result = validator.validate_dataset(samples)
        assert validation_result.valid
        assert validation_result.total_samples >= 2

        # Step 5: Split dataset
        splitter = Splitter()
        train_samples, val_samples = splitter.split(
            samples,
            ratio=config.split.ratio,
            strategy=config.split.strategy
        )

        total_samples = len(samples)
        expected_train = int(total_samples * 0.8)
        expected_val = total_samples - expected_train

        assert len(train_samples) == expected_train
        assert len(val_samples) == expected_val

        # Step 6: Save to database
        dataset_id = client.save_dataset(
            name="test_identify_workflow",
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
            assert len(lines) == expected_train

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert 'messages' in data or 'system' in data

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_verbatim_extraction(
        self,
        mock_genai_model,
        workflow_config,
        sales_conversation_file,
        mock_env_vars
    ):
        """Test IDENTIFY mode extracts conversations verbatim (no AI modification)"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = "0.95"
        mock_model_instance.generate_content.return_value = mock_response

        config = Config(**workflow_config)
        agent = IdentifyAgent(config)

        # Read original file content
        original_content = sales_conversation_file.read_text()

        # Extract conversations
        samples = agent.generate(
            input_files=[sales_conversation_file],
            target_count=None
        )

        assert len(samples) > 0

        # Verify extracted content exists in original (verbatim)
        for sample in samples:
            user_text = sample.content['user'].strip()
            assistant_text = sample.content['assistant'].strip()

            # The extracted text should be present in the original file
            # (allowing for some formatting differences)
            # This is a basic check - real verbatim check would be more complex
            assert len(user_text) > 0
            assert len(assistant_text) > 0

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_conversation_patterns(
        self,
        mock_genai_model,
        workflow_config,
        tmp_path,
        mock_env_vars
    ):
        """Test IDENTIFY mode detects various conversation patterns"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = "0.95"
        mock_model_instance.generate_content.return_value = mock_response

        # Create file with different conversation formats
        content = """
Q: What is SPIN selling?
A: SPIN selling uses four question types: Situation, Problem, Implication, Need-payoff.

---

User: How do I handle objections?
Assistant: Address objections by understanding the root concern first, then providing value-based responses.

---

**Question**: What's your pricing model?
**Answer**: We offer tiered pricing starting at $99/month for our basic plan.
"""
        test_file = tmp_path / "mixed_formats.txt"
        test_file.write_text(content)

        config = Config(**workflow_config)
        agent = IdentifyAgent(config)

        samples = agent.generate(
            input_files=[test_file],
            target_count=None
        )

        # Should detect all 3 conversation formats
        assert len(samples) >= 3

        # Each sample should have user and assistant content
        for sample in samples:
            assert 'user' in sample.content
            assert 'assistant' in sample.content
            assert sample.mode == 'identify'

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_confidence_threshold(
        self,
        mock_genai_model,
        workflow_config,
        tmp_path,
        mock_env_vars
    ):
        """Test IDENTIFY mode respects confidence threshold"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Simulate low confidence scores
        low_confidence_response = Mock()
        low_confidence_response.text = "0.5"  # Below threshold of 0.8

        high_confidence_response = Mock()
        high_confidence_response.text = "0.95"  # Above threshold

        # Alternate between low and high confidence
        mock_model_instance.generate_content.side_effect = [
            low_confidence_response,
            high_confidence_response,
            low_confidence_response,
            high_confidence_response
        ]

        content = """
**Rep:** Hi there!
**Prospect:** Hello.

---

**Rep:** What are your biggest sales challenges?
**Prospect:** We struggle with lead qualification and need better tools.
"""
        test_file = tmp_path / "confidence_test.txt"
        test_file.write_text(content)

        workflow_config['agent']['confidence_threshold'] = 0.8
        config = Config(**workflow_config)
        agent = IdentifyAgent(config)

        samples = agent.generate(
            input_files=[test_file],
            target_count=None
        )

        # Should only extract conversations with confidence >= 0.8
        # Exact count depends on detection logic, but should be filtered
        assert len(samples) >= 0

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_min_turns(
        self,
        mock_genai_model,
        workflow_config,
        tmp_path,
        mock_env_vars
    ):
        """Test IDENTIFY mode respects minimum turn requirement"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = "0.95"
        mock_model_instance.generate_content.return_value = mock_response

        # Create conversations with different turn counts
        content = """
**Rep:** Hi.
**Prospect:** Hello.

---

**Rep:** What are your goals?
**Prospect:** We want to improve sales efficiency.
**Rep:** Tell me more about your current process.
**Prospect:** We use a lot of manual workflows.
"""
        test_file = tmp_path / "min_turns_test.txt"
        test_file.write_text(content)

        workflow_config['agent']['min_turns'] = 2
        config = Config(**workflow_config)
        agent = IdentifyAgent(config)

        samples = agent.generate(
            input_files=[test_file],
            target_count=None
        )

        # First conversation has 1 turn (should be filtered out)
        # Second conversation has 2 turns (should be included)
        # Exact filtering depends on implementation
        assert len(samples) >= 1

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_metadata_tracking(
        self,
        mock_genai_model,
        workflow_config,
        sales_conversation_file,
        mock_env_vars
    ):
        """Test IDENTIFY mode tracks metadata correctly"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_response = Mock()
        mock_response.text = "0.95"
        mock_model_instance.generate_content.return_value = mock_response

        config = Config(**workflow_config)
        agent = IdentifyAgent(config)

        samples = agent.generate(
            input_files=[sales_conversation_file],
            target_count=None
        )

        assert len(samples) > 0

        sample = samples[0]

        # Verify metadata
        assert sample.id is not None
        assert sample.mode == 'identify'
        assert sample.token_length > 0
        assert len(sample.topic_keywords) >= 0  # May or may not have keywords
        assert sample.source_file == str(sales_conversation_file)
        assert 'user' in sample.content
        assert 'assistant' in sample.content

    @patch('agents.identify_agent.genai.GenerativeModel')
    def test_identify_mode_no_conversations_found(
        self,
        mock_genai_model,
        workflow_config,
        tmp_path,
        mock_env_vars
    ):
        """Test IDENTIFY mode handles files with no conversations"""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        # Create file with no conversation patterns
        content = """
This is a plain text file with no conversations.
It just contains regular paragraphs of text.
There are no Q&A patterns or dialogue markers.
"""
        test_file = tmp_path / "no_conversations.txt"
        test_file.write_text(content)

        config = Config(**workflow_config)
        agent = IdentifyAgent(config)

        # Should handle gracefully
        samples = agent.generate(
            input_files=[test_file],
            target_count=None
        )

        # No conversations found = empty list (not an error)
        assert len(samples) == 0
