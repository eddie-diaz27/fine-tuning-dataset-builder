"""
Pytest configuration and fixtures for FineDatasets tests.
"""

import pytest
from pathlib import Path
import os
import tempfile
import json
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_config():
    """Fixture for valid configuration dictionary."""
    return {
        'dataset': {
            'name': 'test_dataset',
            'provider': 'google',
            'model': 'gemini-2.0-flash-exp',
            'target_samples': 10,
            'output_format': 'jsonl'
        },
        'agent': {
            'mode': 'create',
            'variety': 'medium',
            'length': 'medium',
            'min_conversation_turns': 2,
            'confidence_threshold': 0.7,
            'generation_ratio': 0.5,
            'deduplicate': True
        },
        'split': {
            'enabled': True,
            'ratio': 0.8,
            'strategy': 'random'
        },
        'system': {
            'agentic_llm': {
                'provider': 'google',
                'model': 'gemini-2.0-flash-exp'
            },
            'max_retries': 3,
            'rate_limit_delay': 1.0,
            'debug': False
        },
        'file_processing': {
            'max_file_size_mb': 50,
            'supported_formats': ['.txt', '.pdf', '.docx', '.csv', '.xlsx', '.pptx', '.md', '.html']
        }
    }


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Fixture for a temporary config.yaml file."""
    config_content = """
dataset:
  name: test_dataset
  provider: google
  model: gemini-2.0-flash-exp
  target_samples: 10
  output_format: jsonl

agent:
  mode: create
  variety: medium
  length: medium

split:
  enabled: true
  ratio: 0.8
  strategy: random

system:
  agentic_llm:
    provider: google
    model: gemini-2.0-flash-exp
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


# ============================================================================
# Environment Variable Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture for environment variables."""
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-google-api-key-12345')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-api-key-67890')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-api-key-abcde')
    monkeypatch.setenv('MISTRAL_API_KEY', 'test-mistral-api-key-fghij')
    monkeypatch.setenv('AGENT_PROVIDER', 'google')
    monkeypatch.setenv('AGENT_MODEL', 'gemini-2.0-flash-exp')


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_input_files(tmp_path):
    """Fixture for temporary input files directory with sample files."""
    input_dir = tmp_path / "input_files"
    input_dir.mkdir()

    # Create sample.txt
    (input_dir / "sample.txt").write_text("Sample text content for testing.", encoding='utf-8')

    # Create sample with more content
    (input_dir / "sample_long.txt").write_text(
        "This is a longer sample file.\n" * 100,
        encoding='utf-8'
    )

    return input_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture for temporary output directory."""
    output_dir = tmp_path / "output_datasets"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Fixture for temporary memory directory."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_conversation_text():
    """Fixture for conversation text in Q&A format."""
    return """Q: What is Python?
A: Python is a high-level, interpreted programming language known for its simplicity and readability.

Q: What are the main features of Python?
A: Python features include dynamic typing, automatic memory management, a large standard library, and support for multiple programming paradigms.

Q: Why is Python popular?
A: Python is popular because of its easy-to-learn syntax, versatility, and strong community support."""


@pytest.fixture
def sample_chat_text():
    """Fixture for conversation text in User/Assistant format."""
    return """User: Tell me about machine learning.
Assistant: Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data.

User: What are the types of machine learning?
Assistant: The main types are supervised learning, unsupervised learning, and reinforcement learning."""


@pytest.fixture
def sample_markdown_qa():
    """Fixture for conversation text in Markdown Q&A format."""
    return """**Question:** What is a neural network?
**Answer:** A neural network is a computational model inspired by biological neural networks in animal brains.

**Question:** How do neural networks learn?
**Answer:** Neural networks learn by adjusting weights based on training data through a process called backpropagation."""


# ============================================================================
# Mock LLM Response Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Fixture for mocked LLM response in expected format."""
    return """SYSTEM: You are a helpful AI assistant specialized in Python programming.
USER: What is list comprehension in Python?
ASSISTANT: List comprehension is a concise way to create lists in Python. It allows you to generate a new list by applying an expression to each item in an existing iterable."""


@pytest.fixture
def mock_llm_response_multiline():
    """Fixture for multi-turn mocked LLM response."""
    return """SYSTEM: You are a helpful AI assistant.
USER: What is Python?
ASSISTANT: Python is a high-level programming language.
USER: What makes it popular?
ASSISTANT: Its simplicity, readability, and extensive libraries make it popular."""


@pytest.fixture
def mock_google_response():
    """Fixture for mocked Google Gemini API response."""
    class MockResponse:
        def __init__(self, text):
            self.text = text

    return MockResponse(
        text="""SYSTEM: You are a helpful assistant.
USER: Explain recursion.
ASSISTANT: Recursion is when a function calls itself to solve a problem."""
    )


@pytest.fixture
def mock_openai_response():
    """Fixture for mocked OpenAI API response."""
    class MockChoice:
        def __init__(self, message):
            self.message = message

    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(MockMessage(content))]

    return MockResponse(
        content="""SYSTEM: You are a helpful assistant.
USER: What is a decorator?
ASSISTANT: A decorator is a function that modifies the behavior of another function."""
    )


@pytest.fixture
def mock_anthropic_response():
    """Fixture for mocked Anthropic API response."""
    class MockContent:
        def __init__(self, text):
            self.text = text

    class MockResponse:
        def __init__(self, text):
            self.content = [MockContent(text)]

    return MockResponse(
        text="""SYSTEM: You are a helpful assistant.
USER: What is a closure?
ASSISTANT: A closure is a function that has access to variables from its outer scope."""
    )


# ============================================================================
# Sample Object Fixtures
# ============================================================================

@pytest.fixture
def sample_processed_file():
    """Fixture for a ProcessedFile object."""
    from src.utils.file_processor import ProcessedFile

    return ProcessedFile(
        file_path="/test/sample.txt",
        file_name="sample.txt",
        file_type=".txt",
        text="Sample text for testing purposes.",
        char_count=35,
        encoding="utf-8"
    )


@pytest.fixture
def sample_conversation():
    """Fixture for a Conversation object."""
    from src.utils.conversation_detector import Conversation

    return Conversation(
        turns=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ],
        pattern_type="qa_colon",
        confidence=0.95,
        source_start=0,
        source_end=100,
        keywords=["Python", "programming"],
        metadata={}
    )


@pytest.fixture
def sample_sample():
    """Fixture for a Sample object."""
    from src.utils.conversation_detector import Sample

    return Sample(
        id="test-sample-123",
        content={
            'system': 'You are a helpful assistant.',
            'user': 'What is Python?',
            'assistant': 'Python is a programming language.'
        },
        token_length=50,
        topic_keywords=["Python", "programming"],
        source_file="sample.txt",
        split="train"
    )


@pytest.fixture
def sample_samples_list():
    """Fixture for a list of Sample objects."""
    from src.utils.conversation_detector import Sample

    return [
        Sample(
            id=f"sample-{i}",
            content={
                'system': 'You are a helpful assistant.',
                'user': f'Question {i}',
                'assistant': f'Answer {i}'
            },
            token_length=30 + i * 5,
            topic_keywords=["test", "sample"],
            source_file=f"file_{i % 3}.txt",
            split=None
        )
        for i in range(10)
    ]


# ============================================================================
# Provider Fixtures
# ============================================================================

@pytest.fixture
def mock_google_provider():
    """Fixture for mocked Google provider."""
    from src.providers.google_provider import GoogleProvider

    provider = GoogleProvider()
    return provider


@pytest.fixture
def mock_openai_provider():
    """Fixture for mocked OpenAI provider."""
    from src.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    return provider


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def in_memory_db():
    """Fixture for in-memory SQLite database."""
    from src.persistence.sqlite_client import SQLiteClient

    # Use :memory: for fast, isolated tests
    client = SQLiteClient(':memory:')
    yield client
    # Cleanup happens automatically with in-memory DB


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture for temporary SQLite database file."""
    db_path = tmp_path / "test_database.db"
    return str(db_path)


# ============================================================================
# Token Counter Fixtures
# ============================================================================

@pytest.fixture
def mock_token_counter():
    """Fixture for mocked TokenCounter."""
    from src.utils.token_counter import TokenCounter

    counter = TokenCounter()
    return counter


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def sample_jsonl_content():
    """Fixture for sample JSONL content."""
    samples = [
        {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
        {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        {"messages": [{"role": "user", "content": "Q3"}, {"role": "assistant", "content": "A3"}]}
    ]
    return [json.dumps(s) for s in samples]


@pytest.fixture
def sample_csv_content():
    """Fixture for sample CSV content."""
    return """sample_id,content_preview,token_length,topic_keywords,source_file,split
sample-1,"User: Q1...",50,"test,sample",file1.txt,train
sample-2,"User: Q2...",45,"test,sample",file2.txt,validation"""


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def create_temp_file():
    """Fixture that returns a function to create temporary files."""
    def _create_file(directory, filename, content, encoding='utf-8'):
        file_path = directory / filename
        file_path.write_text(content, encoding=encoding)
        return file_path

    return _create_file


@pytest.fixture
def create_mock_response():
    """Fixture that returns a function to create mock API responses."""
    def _create_response(provider, text):
        if provider == 'google':
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            return MockResponse(text)

        elif provider == 'openai':
            class MockChoice:
                def __init__(self, message):
                    self.message = message

            class MockMessage:
                def __init__(self, content):
                    self.content = content

            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(MockMessage(content))]

            return MockResponse(text)

        elif provider == 'anthropic':
            class MockContent:
                def __init__(self, text):
                    self.text = text

            class MockResponse:
                def __init__(self, text):
                    self.content = [MockContent(text)]

            return MockResponse(text)

    return _create_response


# ============================================================================
# Parametrization Fixtures
# ============================================================================

@pytest.fixture(params=['google', 'openai', 'anthropic', 'mistral', 'llama'])
def provider_name(request):
    """Parametrized fixture for provider names."""
    return request.param


@pytest.fixture(params=['create', 'identify', 'hybrid', 'custom'])
def agent_mode(request):
    """Parametrized fixture for agent modes."""
    return request.param


@pytest.fixture(params=[0.7, 0.8, 0.9])
def split_ratio(request):
    """Parametrized fixture for split ratios."""
    return request.param


@pytest.fixture(params=['random', 'weighted'])
def split_strategy(request):
    """Parametrized fixture for split strategies."""
    return request.param
