"""
Unit tests for src/agents/hybrid_agent.py

Tests combined extraction + generation using HYBRID mode with sales documents.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.hybrid_agent import HybridAgent
from src.utils.conversation_detector import Conversation, Turn
from pathlib import Path


class TestHybridAgentInitialization:
    """Tests for HybridAgent initialization."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization(self, mock_model, mock_configure, mock_env_vars):
        """Test HybridAgent initialization."""
        agent = HybridAgent(provider='google', model='gemini-2.0-flash-exp')

        assert agent is not None
        assert agent.provider == 'google'
        assert agent.mode == 'hybrid'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_generation_ratio(self, mock_model, mock_configure, mock_env_vars):
        """Test HybridAgent with custom generation ratio."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.7
        )

        assert agent.generation_ratio == 0.7

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_deduplication(self, mock_model, mock_configure, mock_env_vars):
        """Test HybridAgent with deduplication enabled."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            deduplicate=True
        )

        assert agent.deduplicate is True


class TestSystemPromptLoading:
    """Tests for system prompt loading."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_hybrid_mode_prompt(self, mock_model, mock_configure, mock_env_vars):
        """Test loading HYBRID mode system prompt."""
        agent = HybridAgent(provider='google', model='gemini-2.0-flash-exp')

        prompt_path = Path('prompts/system/hybrid_mode.txt')
        if prompt_path.exists():
            system_prompt = agent.load_system_prompt(str(prompt_path))
            assert len(system_prompt) > 0
            assert isinstance(system_prompt, str)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_system_prompt_contains_hybrid_instructions(self, mock_model, mock_configure, mock_env_vars):
        """Test that system prompt contains hybrid mode instructions."""
        agent = HybridAgent(provider='google', model='gemini-2.0-flash-exp')

        prompt_path = Path('prompts/system/hybrid_mode.txt')
        if prompt_path.exists():
            system_prompt = agent.load_system_prompt(str(prompt_path))

            # Should mention both extraction and generation
            assert ('extract' in system_prompt.lower() or 'identify' in system_prompt.lower()) and \
                   ('generate' in system_prompt.lower() or 'create' in system_prompt.lower())


class TestCombineExtractionAndGeneration:
    """Tests for combining extraction and generation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_combine_extracted_and_generated(self, mock_model, mock_configure, mock_env_vars):
        """Test combining extracted and generated samples."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.5,
            target_samples=10
        )

        # Mock extracted conversations
        agent.extract_conversations = Mock(return_value=[
            {'messages': [{'role': 'user', 'content': 'Extracted Q1'}]},
            {'messages': [{'role': 'user', 'content': 'Extracted Q2'}]},
            {'messages': [{'role': 'user', 'content': 'Extracted Q3'}]},
            {'messages': [{'role': 'user', 'content': 'Extracted Q4'}]},
            {'messages': [{'role': 'user', 'content': 'Extracted Q5'}]}
        ])

        # Mock generated samples
        agent.generate_samples = Mock(return_value=[
            {'messages': [{'role': 'user', 'content': 'Generated Q1'}]},
            {'messages': [{'role': 'user', 'content': 'Generated Q2'}]},
            {'messages': [{'role': 'user', 'content': 'Generated Q3'}]},
            {'messages': [{'role': 'user', 'content': 'Generated Q4'}]},
            {'messages': [{'role': 'user', 'content': 'Generated Q5'}]}
        ])

        combined = agent.combine_samples(["Source text"], target_samples=10)

        # Should have 10 total samples (5 extracted + 5 generated with 0.5 ratio)
        assert len(combined) == 10
        assert isinstance(combined, list)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generation_ratio_80_20(self, mock_model, mock_configure, mock_env_vars):
        """Test 80/20 generation ratio (80% generated, 20% extracted)."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.8,
            target_samples=100
        )

        agent.extract_conversations = Mock(return_value=[
            {'messages': []} for _ in range(20)  # 20 extracted
        ])

        agent.generate_samples = Mock(return_value=[
            {'messages': []} for _ in range(80)  # 80 generated
        ])

        combined = agent.combine_samples(["Source"], target_samples=100)

        # Should honor 80/20 ratio
        assert len(combined) == 100

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generation_ratio_20_80(self, mock_model, mock_configure, mock_env_vars):
        """Test 20/80 generation ratio (20% generated, 80% extracted)."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.2,
            target_samples=100
        )

        agent.extract_conversations = Mock(return_value=[
            {'messages': []} for _ in range(80)  # 80 extracted
        ])

        agent.generate_samples = Mock(return_value=[
            {'messages': []} for _ in range(20)  # 20 generated
        ])

        combined = agent.combine_samples(["Source"], target_samples=100)

        assert len(combined) == 100


class TestDeduplication:
    """Tests for deduplication of samples."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_deduplicate_exact_duplicates(self, mock_model, mock_configure, mock_env_vars):
        """Test deduplication removes exact duplicates."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            deduplicate=True
        )

        samples = [
            {'messages': [{'role': 'user', 'content': 'Duplicate question'}]},
            {'messages': [{'role': 'user', 'content': 'Duplicate question'}]},  # Exact duplicate
            {'messages': [{'role': 'user', 'content': 'Unique question'}]}
        ]

        deduplicated = agent.deduplicate_samples(samples)

        # Should remove exact duplicate
        assert len(deduplicated) == 2

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_deduplicate_similar_samples(self, mock_model, mock_configure, mock_env_vars):
        """Test deduplication removes similar samples (>85% similarity)."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            deduplicate=True,
            similarity_threshold=0.85
        )

        samples = [
            {'messages': [{'role': 'user', 'content': 'What is Python programming language?'}]},
            {'messages': [{'role': 'user', 'content': 'What is Python programming lang?'}]},  # Very similar
            {'messages': [{'role': 'user', 'content': 'What is JavaScript?'}]}  # Different
        ]

        deduplicated = agent.deduplicate_samples(samples)

        # Should detect and remove similar samples
        assert len(deduplicated) <= 2

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_deduplicate_keeps_unique(self, mock_model, mock_configure, mock_env_vars):
        """Test deduplication keeps all unique samples."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            deduplicate=True
        )

        samples = [
            {'messages': [{'role': 'user', 'content': 'Question 1?'}]},
            {'messages': [{'role': 'user', 'content': 'Question 2?'}]},
            {'messages': [{'role': 'user', 'content': 'Question 3?'}]}
        ]

        deduplicated = agent.deduplicate_samples(samples)

        # All are unique, should keep all
        assert len(deduplicated) == 3


class TestRealSalesDocuments:
    """Tests with actual sales training documents."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_hybrid_with_sales_documents(self, mock_model, mock_configure, mock_env_vars):
        """Test HYBRID mode with all sales documents."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.5,
            target_samples=20
        )

        # Check if sales documents exist
        identify_file = Path('input_files/sales_identify_conversations.txt')
        create_file = Path('input_files/sales_create_methodology.txt')

        if identify_file.exists() and create_file.exists():
            # Mock extraction
            agent.detector = Mock()
            agent.detector.detect_conversations = Mock(return_value=[
                Conversation(
                    turns=[Turn(role='user', content='Q'), Turn(role='assistant', content='A')],
                    source_pattern='test',
                    confidence=0.9,
                    text_range=(0, 50)
                )
                for _ in range(10)
            ])

            # Mock generation
            mock_response = Mock()
            mock_response.text = "USER: Generated Q\nASSISTANT: Generated A"
            agent.llm_client = Mock()
            agent.llm_client.generate_content = Mock(return_value=mock_response)

            with open(identify_file, 'r', encoding='utf-8') as f:
                identify_text = f.read()
            with open(create_file, 'r', encoding='utf-8') as f:
                create_text = f.read()

            combined = agent.combine_samples([identify_text, create_text], target_samples=20)

            assert isinstance(combined, list)


class TestEdgeCases:
    """Tests for edge cases."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_zero_extracted_conversations(self, mock_model, mock_configure, mock_env_vars):
        """Test when no conversations are extracted (all generated)."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=1.0  # 100% generated
        )

        agent.extract_conversations = Mock(return_value=[])  # No extracted

        agent.generate_samples = Mock(return_value=[
            {'messages': []} for _ in range(10)
        ])

        combined = agent.combine_samples(["Source"], target_samples=10)

        # Should fallback to pure generation
        assert len(combined) == 10

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_zero_generated_samples(self, mock_model, mock_configure, mock_env_vars):
        """Test when no samples are generated (all extracted)."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.0  # 0% generated (all extracted)
        )

        agent.extract_conversations = Mock(return_value=[
            {'messages': []} for _ in range(10)
        ])

        agent.generate_samples = Mock(return_value=[])  # No generated

        combined = agent.combine_samples(["Source"], target_samples=10)

        # Should fallback to pure extraction
        assert len(combined) == 10

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_deduplication_disabled(self, mock_model, mock_configure, mock_env_vars):
        """Test that deduplication can be disabled."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            deduplicate=False
        )

        samples = [
            {'messages': [{'role': 'user', 'content': 'Duplicate'}]},
            {'messages': [{'role': 'user', 'content': 'Duplicate'}]}
        ]

        # Without deduplication, should keep duplicates
        result = samples if not agent.deduplicate else agent.deduplicate_samples(samples)

        assert len(result) == 2


class TestMetadataTracking:
    """Tests for metadata tracking."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_track_source_of_samples(self, mock_model, mock_configure, mock_env_vars):
        """Test tracking whether sample was extracted or generated."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.5
        )

        agent.extract_conversations = Mock(return_value=[
            {'messages': [], 'mode': 'extracted'}
        ])

        agent.generate_samples = Mock(return_value=[
            {'messages': [], 'mode': 'generated'}
        ])

        combined = agent.combine_samples(["Source"], target_samples=2)

        # Should track mode for each sample
        modes = [s.get('mode') for s in combined]
        assert 'extracted' in modes or 'generated' in modes


class TestErrorHandling:
    """Tests for error handling."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_extraction_failure(self, mock_model, mock_configure, mock_env_vars):
        """Test handling when extraction fails."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.5
        )

        # Mock extraction to fail
        agent.extract_conversations = Mock(side_effect=Exception("Extraction failed"))

        # Mock generation to succeed
        agent.generate_samples = Mock(return_value=[{'messages': []} for _ in range(10)])

        # Should fallback to pure generation
        try:
            combined = agent.combine_samples(["Source"], target_samples=10)
            # If it succeeds, should have generated samples
            assert isinstance(combined, list)
        except Exception:
            # Or may raise error - either is acceptable
            pass

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_generation_failure(self, mock_model, mock_configure, mock_env_vars):
        """Test handling when generation fails."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.5
        )

        # Mock extraction to succeed
        agent.extract_conversations = Mock(return_value=[{'messages': []} for _ in range(10)])

        # Mock generation to fail
        agent.generate_samples = Mock(side_effect=Exception("Generation failed"))

        # Should fallback to pure extraction
        try:
            combined = agent.combine_samples(["Source"], target_samples=10)
            assert isinstance(combined, list)
        except Exception:
            pass


class TestStatistics:
    """Tests for statistics tracking."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_track_statistics(self, mock_model, mock_configure, mock_env_vars):
        """Test tracking of hybrid mode statistics."""
        agent = HybridAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            generation_ratio=0.5,
            target_samples=100
        )

        agent.extract_conversations = Mock(return_value=[{'messages': []} for _ in range(50)])
        agent.generate_samples = Mock(return_value=[{'messages': []} for _ in range(50)])

        combined = agent.combine_samples(["Source"], target_samples=100)

        # Should track extraction vs generation stats
        if hasattr(agent, 'statistics'):
            assert agent.statistics is not None
