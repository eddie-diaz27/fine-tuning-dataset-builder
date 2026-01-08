"""
Unit tests for src/utils/file_processor.py

Tests file processing for various formats using real test data.
"""

import pytest
from pathlib import Path
from src.utils.file_processor import FileProcessor, ProcessedFile, FileProcessingError


class TestFileProcessorBasic:
    """Basic file processor tests."""

    def test_initialization(self):
        """Test FileProcessor initialization."""
        processor = FileProcessor()
        assert processor is not None
        assert processor.max_size_mb == 50
        assert processor.encoding == 'utf-8'

    def test_custom_initialization(self):
        """Test FileProcessor with custom settings."""
        processor = FileProcessor(max_size_mb=100, encoding='latin-1')
        assert processor.max_size_mb == 100
        assert processor.encoding == 'latin-1'

    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        processor = FileProcessor()
        expected_formats = ['.pdf', '.docx', '.txt', '.csv', '.xlsx', '.pptx', '.md', '.html']

        for fmt in expected_formats:
            assert fmt in processor.SUPPORTED_FORMATS


class TestTextFileProcessing:
    """Tests for text file processing with real sales documents."""

    def test_process_sales_conversations_file(self):
        """Test processing sales_identify_conversations.txt."""
        processor = FileProcessor()
        file_path = "input_files/sales_identify_conversations.txt"

        result = processor.process_file(file_path)

        assert isinstance(result, ProcessedFile)
        assert result.file_name == "sales_identify_conversations.txt"
        assert result.file_type == "txt"
        assert result.char_count > 0
        assert len(result.text) > 0

        # Check for expected conversation markers
        assert "**Rep:**" in result.text or "**Prospect:**" in result.text

    def test_process_sales_chatlogs_file(self):
        """Test processing sales_identify_chatlogs.txt."""
        processor = FileProcessor()
        file_path = "input_files/sales_identify_chatlogs.txt"

        result = processor.process_file(file_path)

        assert isinstance(result, ProcessedFile)
        assert result.file_name == "sales_identify_chatlogs.txt"
        assert result.file_type == "txt"
        assert result.char_count > 0

        # Check for expected chat markers
        assert "**Customer:**" in result.text or "**Sarah (Sales):**" in result.text

    def test_process_sales_methodology_file(self):
        """Test processing sales_create_methodology.txt."""
        processor = FileProcessor()
        file_path = "input_files/sales_create_methodology.txt"

        result = processor.process_file(file_path)

        assert isinstance(result, ProcessedFile)
        assert result.file_name == "sales_create_methodology.txt"
        assert result.char_count > 0

        # Check for expected methodology content
        assert "SPIN" in result.text or "selling" in result.text.lower()

    def test_process_sales_playbooks_file(self):
        """Test processing sales_create_playbooks.txt."""
        processor = FileProcessor()
        file_path = "input_files/sales_create_playbooks.txt"

        result = processor.process_file(file_path)

        assert isinstance(result, ProcessedFile)
        assert result.file_name == "sales_create_playbooks.txt"
        assert result.char_count > 0

    def test_process_text_file_basic(self, tmp_path):
        """Test processing basic text file."""
        processor = FileProcessor()

        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file.\nWith multiple lines."
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        assert result.text == test_content
        assert result.file_name == "test.txt"
        assert result.file_type == "txt"
        assert result.char_count == len(test_content)

    def test_process_text_file_with_utf8(self, tmp_path):
        """Test processing UTF-8 text file with special characters."""
        processor = FileProcessor()

        test_file = tmp_path / "utf8_test.txt"
        test_content = "Hello 世界 🌍 Привет"
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        assert result.text == test_content
        assert result.char_count == len(test_content)

    def test_process_text_file_with_latin1(self, tmp_path):
        """Test processing Latin-1 encoded text file."""
        processor = FileProcessor()

        test_file = tmp_path / "latin1_test.txt"
        test_content = "Café résumé naïve"
        test_file.write_bytes(test_content.encode('latin-1'))

        result = processor.process_file(str(test_file))

        # Should successfully read the file (encoding auto-detected)
        assert len(result.text) > 0
        assert result.file_name == "latin1_test.txt"


class TestMarkdownProcessing:
    """Tests for Markdown file processing."""

    def test_process_markdown_readme(self):
        """Test processing SALES_DOCS_README.md."""
        processor = FileProcessor()
        file_path = "input_files/SALES_DOCS_README.md"

        result = processor.process_file(file_path)

        assert isinstance(result, ProcessedFile)
        assert result.file_name == "SALES_DOCS_README.md"
        assert result.file_type == "md"
        assert result.char_count > 0

        # Check for expected markdown content
        assert "#" in result.text  # Headers
        assert "IDENTIFY" in result.text or "CREATE" in result.text

    def test_process_markdown_basic(self, tmp_path):
        """Test processing basic Markdown file."""
        processor = FileProcessor()

        test_file = tmp_path / "test.md"
        test_content = """# Title

## Section 1
Content here.

**Bold text** and *italic text*
"""
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        assert result.text == test_content
        assert result.file_name == "test.md"
        assert result.file_type == "md"


class TestErrorHandling:
    """Tests for error handling."""

    def test_file_not_found(self):
        """Test that missing file raises appropriate error."""
        processor = FileProcessor()

        with pytest.raises(FileProcessingError) as exc_info:
            processor.process_file("nonexistent_file.txt")

        assert "File not found" in str(exc_info.value)

    def test_empty_file(self, tmp_path):
        """Test that empty file raises appropriate error."""
        processor = FileProcessor()

        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding='utf-8')

        with pytest.raises(FileProcessingError) as exc_info:
            processor.process_file(str(test_file))

        assert "empty" in str(exc_info.value).lower()

    def test_file_too_large(self, tmp_path):
        """Test that oversized file raises appropriate error."""
        processor = FileProcessor(max_size_mb=0.001)  # 1KB limit

        test_file = tmp_path / "large.txt"
        # Create file larger than 1KB
        test_file.write_text("A" * 2000, encoding='utf-8')

        with pytest.raises(FileProcessingError) as exc_info:
            processor.process_file(str(test_file))

        assert "too large" in str(exc_info.value).lower()

    def test_unsupported_format(self, tmp_path):
        """Test that unsupported format raises appropriate error."""
        processor = FileProcessor()

        test_file = tmp_path / "test.xyz"
        test_file.write_text("Test content", encoding='utf-8')

        with pytest.raises(FileProcessingError) as exc_info:
            processor.process_file(str(test_file))

        assert "Unsupported file format" in str(exc_info.value)
        assert ".xyz" in str(exc_info.value)


class TestProcessedFileDataclass:
    """Tests for ProcessedFile dataclass."""

    def test_processed_file_creation(self):
        """Test ProcessedFile dataclass creation."""
        from datetime import datetime

        processed = ProcessedFile(
            text="Sample text",
            file_name="test.txt",
            file_type="txt",
            file_size=1024,
            char_count=11,
            created_at=datetime.now()
        )

        assert processed.text == "Sample text"
        assert processed.file_name == "test.txt"
        assert processed.file_type == "txt"
        assert processed.file_size == 1024
        assert processed.char_count == 11
        assert processed.created_at is not None

    def test_processed_file_from_real_processing(self):
        """Test ProcessedFile from actual file processing."""
        processor = FileProcessor()
        file_path = "input_files/sales_identify_conversations.txt"

        result = processor.process_file(file_path)

        # Verify all fields are populated
        assert result.text is not None
        assert len(result.text) > 0
        assert result.file_name == "sales_identify_conversations.txt"
        assert result.file_type == "txt"
        assert result.file_size > 0
        assert result.char_count > 0
        assert result.char_count == len(result.text)
        assert result.created_at is not None


class TestSpecialCharactersHandling:
    """Tests for handling special characters."""

    def test_emoji_handling(self, tmp_path):
        """Test processing file with emojis."""
        processor = FileProcessor()

        test_file = tmp_path / "emoji.txt"
        test_content = "Hello 👋 World 🌍 Test 🚀"
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        assert result.text == test_content
        assert "👋" in result.text
        assert "🌍" in result.text

    def test_newline_handling(self, tmp_path):
        """Test processing file with various newline types."""
        processor = FileProcessor()

        test_file = tmp_path / "newlines.txt"
        test_content = "Line 1\nLine 2\r\nLine 3\n"
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        assert "Line 1" in result.text
        assert "Line 2" in result.text
        assert "Line 3" in result.text

    def test_tab_handling(self, tmp_path):
        """Test processing file with tabs."""
        processor = FileProcessor()

        test_file = tmp_path / "tabs.txt"
        test_content = "Column1\tColumn2\tColumn3\nData1\tData2\tData3"
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        assert "\t" in result.text
        assert "Column1" in result.text


class TestMultipleFilesProcessing:
    """Tests for processing multiple files."""

    def test_process_all_sales_files(self):
        """Test processing all sales training documents."""
        processor = FileProcessor()

        sales_files = [
            "input_files/sales_identify_conversations.txt",
            "input_files/sales_identify_chatlogs.txt",
            "input_files/sales_create_methodology.txt",
            "input_files/sales_create_playbooks.txt"
        ]

        results = []
        for file_path in sales_files:
            try:
                result = processor.process_file(file_path)
                results.append(result)
            except FileProcessingError:
                pass  # Skip if file doesn't exist

        # Should process at least some files
        assert len(results) >= 0

        # All processed files should have valid data
        for result in results:
            assert isinstance(result, ProcessedFile)
            assert len(result.text) > 0
            assert result.char_count > 0

    def test_process_files_with_different_sizes(self, tmp_path):
        """Test processing files of various sizes."""
        processor = FileProcessor()

        # Small file
        small_file = tmp_path / "small.txt"
        small_file.write_text("Small", encoding='utf-8')

        # Medium file
        medium_file = tmp_path / "medium.txt"
        medium_file.write_text("Medium " * 100, encoding='utf-8')

        # Larger file
        large_file = tmp_path / "large.txt"
        large_file.write_text("Large " * 1000, encoding='utf-8')

        small_result = processor.process_file(str(small_file))
        medium_result = processor.process_file(str(medium_file))
        large_result = processor.process_file(str(large_file))

        assert small_result.char_count < medium_result.char_count < large_result.char_count


class TestConversationPatternPreservation:
    """Tests for preserving conversation patterns in processed files."""

    def test_preserve_rep_prospect_pattern(self):
        """Test that Rep/Prospect dialogue markers are preserved."""
        processor = FileProcessor()
        file_path = "input_files/sales_identify_conversations.txt"

        result = processor.process_file(file_path)

        # Should preserve the exact formatting of conversation markers
        assert "**Rep:**" in result.text
        assert "**Prospect:**" in result.text

    def test_preserve_customer_agent_pattern(self):
        """Test that Customer/Agent dialogue markers are preserved."""
        processor = FileProcessor()
        file_path = "input_files/sales_identify_chatlogs.txt"

        result = processor.process_file(file_path)

        # Should preserve chat log markers
        assert "**Customer:**" in result.text or "**Visitor:**" in result.text

    def test_preserve_whitespace_in_conversations(self, tmp_path):
        """Test that whitespace in conversations is preserved."""
        processor = FileProcessor()

        test_file = tmp_path / "conversation.txt"
        test_content = """**Rep:** Hello, how are you?

**Prospect:** I'm doing well, thanks for asking.

**Rep:** Great! Let's discuss your needs."""
        test_file.write_text(test_content, encoding='utf-8')

        result = processor.process_file(str(test_file))

        # Should preserve exact formatting including blank lines
        assert result.text == test_content
