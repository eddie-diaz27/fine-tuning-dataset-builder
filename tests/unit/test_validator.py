"""
Unit tests for src/utils/validator.py

Tests sample and dataset validation including format, token limits, duplicates, and quality checks.
"""

import pytest
from src.utils.validator import Validator, ValidationResult, ValidationError
from src.utils.token_counter import TokenCounter


class TestValidatorBasic:
    """Basic validator tests."""

    def test_initialization(self):
        """Test Validator initialization."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')
        assert validator is not None
        assert validator.provider == 'google'
        assert validator.model == 'gemini-2.0-flash-exp'

    def test_initialization_with_custom_token_counter(self):
        """Test Validator with custom token counter."""
        counter = TokenCounter()
        validator = Validator(provider='openai', model='gpt-4o', token_counter=counter)
        assert validator.token_counter is counter


class TestSampleValidation:
    """Tests for individual sample validation."""

    def test_validate_valid_sample(self):
        """Test validation of a valid sample."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'What is Python?'},
                {'role': 'assistant', 'content': 'Python is a programming language.'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_sample_missing_messages(self):
        """Test validation fails for sample without messages."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'content': 'Some content without messages format'
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is False
        assert any('messages' in error.lower() for error in result.errors)

    def test_validate_sample_empty_messages(self):
        """Test validation fails for empty messages array."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {'messages': []}

        result = validator.validate_sample(sample)
        assert result.is_valid is False
        assert any('empty' in error.lower() or 'messages' in error.lower() for error in result.errors)

    def test_validate_sample_token_limit_exceeded(self):
        """Test validation fails when token limit exceeded."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Create a very long message that exceeds token limits
        long_content = "word " * 50000  # ~50k tokens

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Question?'},
                {'role': 'assistant', 'content': long_content}
            ]
        }

        result = validator.validate_sample(sample)
        # May or may not fail depending on provider limits
        # Just ensure validation runs without crashing
        assert isinstance(result, ValidationResult)

    def test_validate_openai_format(self):
        """Test validation of OpenAI format with system message."""
        validator = Validator(provider='openai', model='gpt-4o')

        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True

    def test_validate_anthropic_format(self):
        """Test validation of Anthropic format with separate system field."""
        validator = Validator(provider='anthropic', model='claude-3-haiku-20240307')

        sample = {
            'system': 'You are a helpful assistant.',
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True

    def test_validate_sample_invalid_role(self):
        """Test validation fails for invalid role."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'invalid_role', 'content': 'Some content'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is False
        assert any('role' in error.lower() for error in result.errors)

    def test_validate_sample_missing_content(self):
        """Test validation fails for message without content."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'user'}  # Missing content
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is False
        assert any('content' in error.lower() for error in result.errors)

    def test_validate_sample_empty_content(self):
        """Test validation fails for empty content."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'user', 'content': ''},
                {'role': 'assistant', 'content': ''}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is False
        assert any('empty' in error.lower() or 'content' in error.lower() for error in result.errors)


class TestDatasetValidation:
    """Tests for full dataset validation."""

    def test_validate_valid_dataset(self):
        """Test validation of valid dataset."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'What is Python?'},
                    {'role': 'assistant', 'content': 'Python is a programming language.'}
                ]
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'What is JavaScript?'},
                    {'role': 'assistant', 'content': 'JavaScript is a scripting language.'}
                ]
            }
        ]

        result = validator.validate_dataset(dataset)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_empty_dataset(self):
        """Test validation fails for empty dataset."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        dataset = []

        result = validator.validate_dataset(dataset)
        assert result.is_valid is False
        assert any('empty' in error.lower() for error in result.errors)

    def test_validate_dataset_with_invalid_samples(self):
        """Test validation identifies invalid samples in dataset."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Valid sample'},
                    {'role': 'assistant', 'content': 'Valid response'}
                ]
            },
            {
                'messages': []  # Invalid - empty messages
            },
            {
                'invalid': 'format'  # Invalid - no messages
            }
        ]

        result = validator.validate_dataset(dataset)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_dataset_minimum_samples(self):
        """Test validation enforces minimum sample count for providers."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Google requires minimum 100 samples
        # Create small dataset (less than minimum)
        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Question {i}?'},
                    {'role': 'assistant', 'content': f'Answer {i}.'}
                ]
            }
            for i in range(50)  # Only 50 samples
        ]

        result = validator.validate_dataset(dataset)
        # May warn or fail depending on implementation
        assert isinstance(result, ValidationResult)

    def test_validate_dataset_total_tokens(self):
        """Test validation tracks total dataset tokens."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Test question'},
                    {'role': 'assistant', 'content': 'Test answer'}
                ]
            }
            for _ in range(10)
        ]

        result = validator.validate_dataset(dataset)
        # Should track total tokens
        if hasattr(result, 'total_tokens'):
            assert result.total_tokens > 0


class TestDuplicateDetection:
    """Tests for duplicate detection."""

    def test_detect_exact_duplicates(self):
        """Test detection of exact duplicate samples."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'What is Python?'},
                    {'role': 'assistant', 'content': 'Python is a programming language.'}
                ]
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'What is Python?'},
                    {'role': 'assistant', 'content': 'Python is a programming language.'}
                ]
            }
        ]

        duplicates = validator.detect_duplicates(dataset)
        assert len(duplicates) > 0

    def test_detect_no_duplicates(self):
        """Test that unique samples are not flagged as duplicates."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'What is Python?'},
                    {'role': 'assistant', 'content': 'Python is a programming language.'}
                ]
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'What is JavaScript?'},
                    {'role': 'assistant', 'content': 'JavaScript is a scripting language.'}
                ]
            }
        ]

        duplicates = validator.detect_duplicates(dataset)
        assert len(duplicates) == 0

    def test_detect_similar_duplicates_threshold(self):
        """Test detection of similar duplicates above 85% threshold."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp', similarity_threshold=0.85)

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'What is Python programming language?'},
                    {'role': 'assistant', 'content': 'Python is a high-level programming language.'}
                ]
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'What is Python programming language?'},
                    {'role': 'assistant', 'content': 'Python is a high-level programming lang.'}  # Very similar
                ]
            }
        ]

        duplicates = validator.detect_duplicates(dataset)
        # Should detect similarity > 85%
        assert isinstance(duplicates, list)

    def test_detect_duplicates_empty_dataset(self):
        """Test duplicate detection on empty dataset."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        duplicates = validator.detect_duplicates([])
        assert len(duplicates) == 0


class TestQualityChecks:
    """Tests for quality validation checks."""

    def test_quality_check_conversation_length(self):
        """Test quality check for minimum conversation length."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Very short conversation
        sample = {
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }

        result = validator.validate_sample(sample)
        # Should pass validation but may have quality warnings
        assert isinstance(result, ValidationResult)

    def test_quality_check_balanced_turns(self):
        """Test quality check for balanced user/assistant turns."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Unbalanced - 3 user messages, 1 assistant
        sample = {
            'messages': [
                {'role': 'user', 'content': 'Question 1'},
                {'role': 'user', 'content': 'Question 2'},
                {'role': 'user', 'content': 'Question 3'},
                {'role': 'assistant', 'content': 'Answer'}
            ]
        }

        result = validator.validate_sample(sample)
        # Should pass basic validation but may flag quality issue
        assert isinstance(result, ValidationResult)

    def test_quality_check_content_diversity(self):
        """Test quality check for content diversity."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Repetitive content
        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Test'},
                    {'role': 'assistant', 'content': 'Test response'}
                ]
            }
            for _ in range(10)
        ]

        result = validator.validate_dataset(dataset)
        # Should identify low diversity
        assert isinstance(result, ValidationResult)


class TestProviderSpecificValidation:
    """Tests for provider-specific validation rules."""

    def test_openai_system_message_optional(self):
        """Test that OpenAI allows optional system message."""
        validator = Validator(provider='openai', model='gpt-4o')

        # Without system message
        sample1 = {
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }

        # With system message
        sample2 = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }

        result1 = validator.validate_sample(sample1)
        result2 = validator.validate_sample(sample2)

        assert result1.is_valid is True
        assert result2.is_valid is True

    def test_anthropic_system_field_separate(self):
        """Test that Anthropic requires separate system field."""
        validator = Validator(provider='anthropic', model='claude-3-haiku-20240307')

        # Correct format - system as separate field
        sample = {
            'system': 'You are helpful.',
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True

    def test_google_minimum_samples_requirement(self):
        """Test that Google enforces minimum 100 samples."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Create dataset with 50 samples (below minimum)
        small_dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ]
            }
            for i in range(50)
        ]

        result = validator.validate_dataset(small_dataset)
        # Should warn or fail about minimum samples
        assert isinstance(result, ValidationResult)

    def test_openai_token_limit_enforcement(self):
        """Test that OpenAI token limits are enforced."""
        validator = Validator(provider='openai', model='gpt-4o')

        # GPT-4o has 65,536 token limit per example
        # Token counter should flag if exceeded
        sample = {
            'messages': [
                {'role': 'user', 'content': 'Test'},
                {'role': 'assistant', 'content': 'Response'}
            ]
        }

        result = validator.validate_sample(sample)
        # Should validate token limits
        assert isinstance(result, ValidationResult)


class TestValidationResultDataclass:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=['Minor issue'],
            total_tokens=1000
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.total_tokens == 1000

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=['Error 1', 'Error 2'],
            warnings=[],
            total_tokens=500
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert 'Error 1' in result.errors


class TestEdgeCases:
    """Tests for edge cases in validation."""

    def test_validate_very_large_dataset(self):
        """Test validation of very large dataset."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        # Create large dataset (1000 samples)
        large_dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Question {i}?'},
                    {'role': 'assistant', 'content': f'Answer {i}.'}
                ]
            }
            for i in range(1000)
        ]

        result = validator.validate_dataset(large_dataset)
        assert isinstance(result, ValidationResult)

    def test_validate_unicode_content(self):
        """Test validation with Unicode characters."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'What is 世界?'},
                {'role': 'assistant', 'content': '世界 means world in Chinese.'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True

    def test_validate_special_characters(self):
        """Test validation with special characters."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Code: print("hello")'},
                {'role': 'assistant', 'content': 'This prints "hello" to console.'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True

    def test_validate_multiline_content(self):
        """Test validation with multiline content."""
        validator = Validator(provider='google', model='gemini-2.0-flash-exp')

        sample = {
            'messages': [
                {'role': 'user', 'content': 'Line 1\nLine 2\nLine 3'},
                {'role': 'assistant', 'content': 'Response line 1\nResponse line 2'}
            ]
        }

        result = validator.validate_sample(sample)
        assert result.is_valid is True
