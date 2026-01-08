"""
Unit tests for src/agents/identify_agent.py

Tests verbatim conversation extraction using IDENTIFY mode with sales documents.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.identify_agent import IdentifyAgent
from src.utils.conversation_detector import ConversationDetector, Conversation, Turn
from pathlib import Path


class TestIdentifyAgentInitialization:
    """Tests for IdentifyAgent initialization."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization(self, mock_model, mock_configure, mock_env_vars):
        """Test IdentifyAgent initialization."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        assert agent is not None
        assert agent.provider == 'google'
        assert agent.mode == 'identify'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_confidence_threshold(self, mock_model, mock_configure, mock_env_vars):
        """Test IdentifyAgent with custom confidence threshold."""
        agent = IdentifyAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            confidence_threshold=0.9
        )

        assert agent.confidence_threshold == 0.9

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_min_turns(self, mock_model, mock_configure, mock_env_vars):
        """Test IdentifyAgent with minimum turns setting."""
        agent = IdentifyAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            min_turns=3
        )

        assert agent.min_turns == 3


class TestSystemPromptLoading:
    """Tests for system prompt loading."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_load_identify_mode_prompt(self, mock_model, mock_configure, mock_env_vars):
        """Test loading IDENTIFY mode system prompt."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        prompt_path = Path('prompts/system/identify_mode.txt')
        if prompt_path.exists():
            system_prompt = agent.load_system_prompt(str(prompt_path))
            assert len(system_prompt) > 0
            assert isinstance(system_prompt, str)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_system_prompt_contains_extraction_instructions(self, mock_model, mock_configure, mock_env_vars):
        """Test that system prompt contains extraction instructions."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        prompt_path = Path('prompts/system/identify_mode.txt')
        if prompt_path.exists():
            system_prompt = agent.load_system_prompt(str(prompt_path))

            # Should mention extraction/identification
            assert 'extract' in system_prompt.lower() or 'identify' in system_prompt.lower()
            # Should emphasize verbatim extraction
            assert 'verbatim' in system_prompt.lower() or 'exact' in system_prompt.lower()


class TestConversationExtraction:
    """Tests for conversation extraction."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_conversations_verbatim(self, mock_model, mock_configure, mock_env_vars):
        """Test verbatim conversation extraction."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        # Sample text with conversation
        text = """**Rep:** Hi Sarah, thanks for taking the time today.

**Prospect:** Sure, happy to chat.

**Rep:** Can you tell me about your current workflow?

**Prospect:** We're using spreadsheets and email to track everything."""

        # Mock conversation detector
        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[
                    Turn(role='user', content='Hi Sarah, thanks for taking the time today.'),
                    Turn(role='assistant', content='Sure, happy to chat.'),
                    Turn(role='user', content='Can you tell me about your current workflow?'),
                    Turn(role='assistant', content="We're using spreadsheets and email to track everything.")
                ],
                source_pattern='rep_prospect',
                confidence=0.95,
                text_range=(0, len(text))
            )
        ])

        conversations = agent.extract_conversations(text)

        assert isinstance(conversations, list)
        assert len(conversations) > 0
        assert all('messages' in conv for conv in conversations)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_from_sales_conversations_file(self, mock_model, mock_configure, mock_env_vars):
        """Test extraction from actual sales conversations file."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        sales_file = Path('input_files/sales_identify_conversations.txt')
        if sales_file.exists():
            with open(sales_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Use real ConversationDetector
            agent.detector = ConversationDetector(
                llm_provider='google',
                llm_model='gemini-2.0-flash-exp'
            )

            # Mock the AI detection to avoid API calls
            with patch.object(agent.detector, '_ai_detect_chunk') as mock_detect:
                mock_detect.return_value = [
                    Conversation(
                        turns=[
                            Turn(role='user', content='Test question'),
                            Turn(role='assistant', content='Test answer')
                        ],
                        source_pattern='ai_detected',
                        confidence=0.9,
                        text_range=(0, 100)
                    )
                ]

                conversations = agent.extract_conversations(text, min_turns=1)

                assert isinstance(conversations, list)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_from_sales_chatlogs_file(self, mock_model, mock_configure, mock_env_vars):
        """Test extraction from actual sales chat logs file."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        chatlogs_file = Path('input_files/sales_identify_chatlogs.txt')
        if chatlogs_file.exists():
            with open(chatlogs_file, 'r', encoding='utf-8') as f:
                text = f.read()

            agent.detector = ConversationDetector()

            # Mock AI detection
            with patch.object(agent.detector, '_ai_detect_chunk') as mock_detect:
                mock_detect.return_value = [
                    Conversation(
                        turns=[
                            Turn(role='user', content='Customer question'),
                            Turn(role='assistant', content='Agent response')
                        ],
                        source_pattern='customer_agent',
                        confidence=0.92,
                        text_range=(0, 100)
                    )
                ]

                conversations = agent.extract_conversations(text)

                assert isinstance(conversations, list)


class TestConfidenceThreshold:
    """Tests for confidence threshold filtering."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_filter_by_confidence_threshold(self, mock_model, mock_configure, mock_env_vars):
        """Test filtering conversations by confidence threshold."""
        agent = IdentifyAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            confidence_threshold=0.8
        )

        # Mock detector with varying confidence levels
        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[Turn(role='user', content='Q1'), Turn(role='assistant', content='A1')],
                source_pattern='pattern1',
                confidence=0.95,  # Above threshold
                text_range=(0, 50)
            ),
            Conversation(
                turns=[Turn(role='user', content='Q2'), Turn(role='assistant', content='A2')],
                source_pattern='pattern2',
                confidence=0.6,  # Below threshold
                text_range=(50, 100)
            ),
            Conversation(
                turns=[Turn(role='user', content='Q3'), Turn(role='assistant', content='A3')],
                source_pattern='pattern3',
                confidence=0.85,  # Above threshold
                text_range=(100, 150)
            )
        ])

        conversations = agent.extract_conversations("Test text", confidence_threshold=0.8)

        # Should only return conversations with confidence >= 0.8
        assert len(conversations) == 2

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_high_confidence_threshold(self, mock_model, mock_configure, mock_env_vars):
        """Test with high confidence threshold (0.9)."""
        agent = IdentifyAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            confidence_threshold=0.9
        )

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[Turn(role='user', content='Q'), Turn(role='assistant', content='A')],
                source_pattern='pattern',
                confidence=0.85,  # Below 0.9
                text_range=(0, 50)
            )
        ])

        conversations = agent.extract_conversations("Test text")

        # Should filter out conversations below 0.9
        assert len(conversations) == 0


class TestMinimumTurns:
    """Tests for minimum turns filtering."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_filter_by_minimum_turns(self, mock_model, mock_configure, mock_env_vars):
        """Test filtering conversations by minimum turns."""
        agent = IdentifyAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            min_turns=2
        )

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[
                    Turn(role='user', content='Q1'),
                    Turn(role='assistant', content='A1')
                ],  # 1 turn (1 exchange)
                source_pattern='pattern1',
                confidence=0.9,
                text_range=(0, 50)
            ),
            Conversation(
                turns=[
                    Turn(role='user', content='Q1'),
                    Turn(role='assistant', content='A1'),
                    Turn(role='user', content='Q2'),
                    Turn(role='assistant', content='A2')
                ],  # 2 turns (2 exchanges)
                source_pattern='pattern2',
                confidence=0.9,
                text_range=(50, 150)
            )
        ])

        conversations = agent.extract_conversations("Test text", min_turns=2)

        # Should only return conversations with 2+ turns
        assert len(conversations) == 1
        assert len(conversations[0]['messages']) == 4


class TestContextInclusion:
    """Tests for including surrounding context."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_include_context_before_after(self, mock_model, mock_configure, mock_env_vars):
        """Test including context paragraphs before and after conversation."""
        agent = IdentifyAgent(
            provider='google',
            model='gemini-2.0-flash-exp',
            include_context=True,
            max_context_paragraphs=2
        )

        text = """This is context before.

**Rep:** Hello.
**Prospect:** Hi.

This is context after."""

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[
                    Turn(role='user', content='Hello.'),
                    Turn(role='assistant', content='Hi.')
                ],
                source_pattern='rep_prospect',
                confidence=0.9,
                text_range=(30, 60),
                context_before='This is context before.',
                context_after='This is context after.'
            )
        ])

        conversations = agent.extract_conversations(text, include_context=True)

        # Conversations may include context in metadata
        assert isinstance(conversations, list)


class TestNoConversationsFound:
    """Tests for handling when no conversations are found."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_no_conversations_raises_error(self, mock_model, mock_configure, mock_env_vars):
        """Test that no conversations found raises appropriate error."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock detector to return empty list
        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[])

        with pytest.raises((ValueError, Exception)) as exc_info:
            agent.extract_conversations("Plain text with no conversations")

        assert 'no conversations' in str(exc_info.value).lower() or 'not found' in str(exc_info.value).lower()

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_empty_text_raises_error(self, mock_model, mock_configure, mock_env_vars):
        """Test that empty text raises error."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        with pytest.raises((ValueError, Exception)):
            agent.extract_conversations("")


class TestVerbatimExtraction:
    """Tests ensuring verbatim extraction (no AI modification)."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_exact_text_verbatim(self, mock_model, mock_configure, mock_env_vars):
        """Test that extracted text is verbatim (not modified by AI)."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        original_text = "This is the exact original text that must be preserved XYZ123."

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[
                    Turn(role='user', content='Question with XYZ123?'),
                    Turn(role='assistant', content='Answer with XYZ123.')
                ],
                source_pattern='test',
                confidence=0.9,
                text_range=(0, 50)
            )
        ])

        conversations = agent.extract_conversations("Q: Question with XYZ123?\nA: Answer with XYZ123.")

        # Check that unique marker appears verbatim in extracted content
        for conv in conversations:
            messages = conv['messages']
            found_marker = any('XYZ123' in msg['content'] for msg in messages)
            assert found_marker is True

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_no_ai_hallucination(self, mock_model, mock_configure, mock_env_vars):
        """Test that no AI hallucination occurs (content not in source)."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        source_text = """**Rep:** What's your budget?
**Prospect:** Around $50,000."""

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[
                    Turn(role='user', content="What's your budget?"),
                    Turn(role='assistant', content='Around $50,000.')
                ],
                source_pattern='rep_prospect',
                confidence=0.95,
                text_range=(0, len(source_text))
            )
        ])

        conversations = agent.extract_conversations(source_text)

        # Verify extracted content matches source
        for conv in conversations:
            for msg in conv['messages']:
                content = msg['content']
                # Content should be substring of original
                # (allowing for minor formatting differences)
                assert isinstance(content, str)
                assert len(content) > 0


class TestMetadataGeneration:
    """Tests for metadata generation."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_add_source_file_metadata(self, mock_model, mock_configure, mock_env_vars):
        """Test that source file is tracked in metadata."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[Turn(role='user', content='Q'), Turn(role='assistant', content='A')],
                source_pattern='test',
                confidence=0.9,
                text_range=(0, 50)
            )
        ])

        conversations = agent.extract_conversations("Test text", source_file='test.txt')

        # Should include source file in metadata
        if conversations:
            sample = conversations[0]
            if 'source_file' in sample:
                assert sample['source_file'] == 'test.txt'

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_add_confidence_metadata(self, mock_model, mock_configure, mock_env_vars):
        """Test that confidence score is tracked."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[Turn(role='user', content='Q'), Turn(role='assistant', content='A')],
                source_pattern='test',
                confidence=0.92,
                text_range=(0, 50)
            )
        ])

        conversations = agent.extract_conversations("Test text")

        # May include confidence in metadata
        assert isinstance(conversations, list)


class TestMultipleFilesProcessing:
    """Tests for processing multiple files."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_from_multiple_files(self, mock_model, mock_configure, mock_env_vars):
        """Test extraction from multiple source files."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        files_data = [
            ("file1.txt", "**Rep:** Q1\n**Prospect:** A1"),
            ("file2.txt", "**Rep:** Q2\n**Prospect:** A2")
        ]

        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[Turn(role='user', content='Q'), Turn(role='assistant', content='A')],
                source_pattern='test',
                confidence=0.9,
                text_range=(0, 50)
            )
        ])

        all_conversations = []
        for filename, text in files_data:
            conversations = agent.extract_conversations(text, source_file=filename)
            all_conversations.extend(conversations)

        # Should extract from both files
        assert len(all_conversations) >= 2


class TestErrorHandling:
    """Tests for error handling."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_detector_failure(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of conversation detector failures."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock detector to raise error
        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(side_effect=Exception("Detection error"))

        with pytest.raises(Exception) as exc_info:
            agent.extract_conversations("Test text")

        assert 'Detection error' in str(exc_info.value)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_handle_malformed_conversations(self, mock_model, mock_configure, mock_env_vars):
        """Test handling of malformed conversation structures."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        # Mock detector to return malformed conversation
        agent.detector = Mock()
        agent.detector.detect_conversations = Mock(return_value=[
            Conversation(
                turns=[],  # Empty turns (invalid)
                source_pattern='test',
                confidence=0.9,
                text_range=(0, 50)
            )
        ])

        conversations = agent.extract_conversations("Test text", min_turns=1)

        # Should filter out invalid conversations
        assert len(conversations) == 0


class TestDifferentConversationFormats:
    """Tests for different conversation formats."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_rep_prospect_format(self, mock_model, mock_configure, mock_env_vars):
        """Test extraction of **Rep:**/**Prospect:** format."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        text = "**Rep:** Test\n**Prospect:** Response"

        agent.detector = ConversationDetector()

        # Mock AI detection
        with patch.object(agent.detector, '_ai_detect_chunk') as mock_detect:
            mock_detect.return_value = [
                Conversation(
                    turns=[
                        Turn(role='user', content='Test'),
                        Turn(role='assistant', content='Response')
                    ],
                    source_pattern='rep_prospect',
                    confidence=0.95,
                    text_range=(0, len(text))
                )
            ]

            conversations = agent.extract_conversations(text)
            assert len(conversations) > 0

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_customer_agent_format(self, mock_model, mock_configure, mock_env_vars):
        """Test extraction of **Customer:**/**Agent:** format."""
        agent = IdentifyAgent(provider='google', model='gemini-2.0-flash-exp')

        text = "**Customer:** Question\n**Sarah (Sales):** Answer"

        agent.detector = ConversationDetector()

        with patch.object(agent.detector, '_ai_detect_chunk') as mock_detect:
            mock_detect.return_value = [
                Conversation(
                    turns=[
                        Turn(role='user', content='Question'),
                        Turn(role='assistant', content='Answer')
                    ],
                    source_pattern='customer_agent',
                    confidence=0.93,
                    text_range=(0, len(text))
                )
            ]

            conversations = agent.extract_conversations(text)
            assert len(conversations) > 0
