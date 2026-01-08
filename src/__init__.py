"""
Fine-Tuning Dataset Builder

An intelligent agentic system for generating high-quality fine-tuning datasets
for multiple LLM providers (OpenAI, Anthropic, Google, Mistral, Meta Llama).

Supports 4 generation modes:
- CREATE: Generate synthetic conversations from source files
- IDENTIFY: Extract existing conversations verbatim
- HYBRID: Combine extraction and generation
- CUSTOM: User-defined custom behavior
"""

__version__ = "0.1.0"
__author__ = "FineDatasets Team"

from .main import DatasetBuilder, main

__all__ = ['DatasetBuilder', 'main']
