"""
Base provider abstract class for all LLM providers.
Defines interface for provider implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProviderSpec:
    """Provider specifications"""
    name: str
    models: List[str]
    token_limits: Dict[str, int]  # model -> limit
    max_training_tokens: Dict[str, Optional[int]]  # model -> max
    format_type: str  # "openai", "anthropic", "google", etc.
    min_examples: int


@dataclass
class ValidationResult:
    """Result of dataset validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]  # total_tokens, avg_tokens, etc.


class BaseProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def get_spec(self) -> ProviderSpec:
        """
        Return provider specifications.

        Returns:
            ProviderSpec with models, limits, etc.
        """
        pass

    @abstractmethod
    def format_example(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> Dict[str, Any]:
        """
        Format a single-turn conversation into provider-specific JSONL.

        Args:
            system: System message
            user: User message
            assistant: Assistant response

        Returns:
            Dict that will be serialized to JSON
        """
        pass

    @abstractmethod
    def format_multi_turn(
        self,
        system: str,
        turns: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Format multi-turn conversation.

        Args:
            system: System message
            turns: List of dicts with 'role' and 'content'

        Returns:
            Dict that will be serialized to JSON
        """
        pass

    @abstractmethod
    def validate_example(self, example: Dict) -> bool:
        """
        Validate single example against provider rules.

        Args:
            example: Example dict to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If example is invalid (with details)
        """
        pass

    def validate_dataset(
        self,
        examples: List[Dict],
        model: str
    ) -> ValidationResult:
        """
        Validate entire dataset.
        Checks: token limits, format, total tokens.

        Args:
            examples: List of example dicts
            model: Model name

        Returns:
            ValidationResult with warnings/errors
        """
        errors = []
        warnings = []
        stats = {}

        spec = self.get_spec()

        # Check if model is valid for this provider
        if model not in spec.models:
            errors.append(
                f"Model '{model}' not available for {spec.name}. "
                f"Available: {', '.join(spec.models)}"
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                stats=stats
            )

        # Get limits
        token_limit = self.get_model_token_limit(model)
        max_training_tokens = self.get_max_training_tokens(model)

        # Validate each example
        total_tokens = 0

        for i, example in enumerate(examples):
            try:
                self.validate_example(example)
            except Exception as e:
                errors.append(f"Example {i}: {str(e)}")

        # Check minimum examples
        if len(examples) < spec.min_examples:
            warnings.append(
                f"Dataset has {len(examples)} examples, "
                f"but {spec.min_examples}+ recommended for {spec.name}"
            )

        # Compute stats
        if not errors:
            stats = {
                "total_examples": len(examples),
                "model": model,
                "provider": spec.name
            }

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def get_model_token_limit(self, model: str) -> int:
        """
        Get token limit for specific model.

        Args:
            model: Model name

        Returns:
            Token limit per example
        """
        spec = self.get_spec()
        return spec.token_limits.get(model, 4096)  # Default fallback

    def get_max_training_tokens(self, model: str) -> Optional[int]:
        """
        Get maximum training tokens for model.

        Args:
            model: Model name

        Returns:
            Max training tokens (None if unlimited)
        """
        spec = self.get_spec()
        return spec.max_training_tokens.get(model)

    def get_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List of model names
        """
        return self.get_spec().models


class ValidationError(Exception):
    """Raised when example validation fails"""
    pass
