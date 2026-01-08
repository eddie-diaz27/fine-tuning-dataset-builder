"""
Dataset validator for checking against provider limits.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

from providers.base_provider import BaseProvider, ValidationResult


@dataclass
class Sample:
    """Dataset sample with metadata"""
    id: str  # Unique sample identifier
    content: Dict[str, str]  # {system, user, assistant}
    token_length: int
    topic_keywords: List[str]
    source_file: str
    mode: str
    metadata: Dict[str, Any] = None


class Validator:
    """Validate datasets against provider requirements"""

    def __init__(self, provider: BaseProvider, model: str, token_counter):
        """
        Initialize validator.

        Args:
            provider: Provider instance
            model: Model name
            token_counter: TokenCounter instance
        """
        self.provider = provider
        self.model = model
        self.token_counter = token_counter
        self.spec = provider.get_spec()

    def validate_dataset(
        self,
        samples: List[Sample]
    ) -> ValidationResult:
        """
        Comprehensive dataset validation.

        Checks:
        1. Individual sample token limits
        2. Total training token limits
        3. Format compliance
        4. Minimum example count
        5. Quality issues (empty content, duplicates)

        Args:
            samples: List of Sample objects

        Returns:
            ValidationResult with errors, warnings, and stats
        """
        errors = []
        warnings = []

        # Get limits
        token_limit = self.provider.get_model_token_limit(self.model)
        max_training_tokens = self.provider.get_max_training_tokens(self.model)
        min_examples = self.spec.min_examples

        # Check minimum examples
        if len(samples) < min_examples:
            warnings.append(
                f"Dataset has {len(samples)} examples, "
                f"but {min_examples}+ recommended for {self.spec.name}"
            )

        # Check if dataset is empty
        if len(samples) == 0:
            errors.append("Dataset is empty. No samples to validate.")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                stats={}
            )

        # Validate individual samples
        total_tokens = 0
        token_lengths = []

        for i, sample in enumerate(samples):
            # Token limit check
            if sample.token_length > token_limit:
                errors.append(
                    f"Sample {i} ({sample.source_file}): "
                    f"exceeds token limit ({sample.token_length} > {token_limit})"
                )

            total_tokens += sample.token_length
            token_lengths.append(sample.token_length)

            # Format check
            try:
                system = sample.content.get('system', '')
                user = sample.content.get('user', '')
                assistant = sample.content.get('assistant', '')

                formatted = self.provider.format_example(system, user, assistant)
                self.provider.validate_example(formatted)

            except Exception as e:
                errors.append(f"Sample {i} format error: {e}")

            # Empty content check
            if not sample.content.get('user') or not sample.content.get('assistant'):
                errors.append(
                    f"Sample {i}: has empty user or assistant content"
                )

            # Check for very short content (likely low quality)
            user_len = len(sample.content.get('user', ''))
            assistant_len = len(sample.content.get('assistant', ''))

            if user_len < 10:
                warnings.append(
                    f"Sample {i}: user message very short ({user_len} chars)"
                )
            if assistant_len < 10:
                warnings.append(
                    f"Sample {i}: assistant message very short ({assistant_len} chars)"
                )

        # Check total tokens
        if max_training_tokens and total_tokens > max_training_tokens:
            errors.append(
                f"Total tokens ({total_tokens:,}) exceeds provider limit "
                f"({max_training_tokens:,}). Reduce dataset size to "
                f"{int(len(samples) * max_training_tokens / total_tokens)} samples."
            )

        # Check for duplicates
        duplicate_pairs = self._find_duplicates(samples)
        if duplicate_pairs:
            warnings.append(
                f"Found {len(duplicate_pairs)} potential duplicate samples. "
                f"Consider removing duplicates for better training."
            )

        # Calculate statistics
        stats = {
            "total_samples": len(samples),
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / len(samples) if samples else 0,
            "max_sample_tokens": max(token_lengths) if token_lengths else 0,
            "min_sample_tokens": min(token_lengths) if token_lengths else 0,
            "median_tokens": sorted(token_lengths)[len(token_lengths)//2] if token_lengths else 0,
            "provider": self.spec.name,
            "model": self.model,
            "token_limit_per_sample": token_limit,
            "max_training_tokens": max_training_tokens,
            "duplicate_count": len(duplicate_pairs)
        }

        # Add source file distribution
        source_distribution = {}
        for sample in samples:
            source = sample.source_file
            source_distribution[source] = source_distribution.get(source, 0) + 1
        stats["source_distribution"] = source_distribution

        # Add mode distribution
        mode_distribution = {}
        for sample in samples:
            mode = sample.mode
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        stats["mode_distribution"] = mode_distribution

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def _find_duplicates(
        self,
        samples: List[Sample],
        threshold: float = 0.85
    ) -> List[tuple]:
        """
        Find duplicate or near-duplicate samples.

        Args:
            samples: List of samples
            threshold: Similarity threshold (0.85 = 85% similar)

        Returns:
            List of (index1, index2) duplicate pairs
        """
        duplicates = []

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                # Compare user + assistant content
                text1 = (samples[i].content.get('user', '') +
                        samples[i].content.get('assistant', ''))
                text2 = (samples[j].content.get('user', '') +
                        samples[j].content.get('assistant', ''))

                # Calculate similarity
                similarity = SequenceMatcher(None, text1, text2).ratio()

                if similarity >= threshold:
                    duplicates.append((i, j))

        return duplicates

    def validate_sample(self, sample: Sample) -> tuple:
        """
        Validate a single sample.

        Args:
            sample: Sample to validate

        Returns:
            (is_valid: bool, errors: List[str])
        """
        errors = []

        # Token limit
        token_limit = self.provider.get_model_token_limit(self.model)
        if sample.token_length > token_limit:
            errors.append(
                f"Exceeds token limit ({sample.token_length} > {token_limit})"
            )

        # Format
        try:
            system = sample.content.get('system', '')
            user = sample.content.get('user', '')
            assistant = sample.content.get('assistant', '')

            formatted = self.provider.format_example(system, user, assistant)
            self.provider.validate_example(formatted)
        except Exception as e:
            errors.append(f"Format error: {e}")

        # Empty content
        if not sample.content.get('user') or not sample.content.get('assistant'):
            errors.append("Empty user or assistant content")

        return (len(errors) == 0, errors)

    def suggest_dataset_size(
        self,
        desired_size: int,
        avg_tokens_per_sample: int
    ) -> int:
        """
        Suggest optimal dataset size based on provider limits.

        Args:
            desired_size: Desired number of samples
            avg_tokens_per_sample: Average tokens per sample

        Returns:
            Suggested dataset size
        """
        max_training_tokens = self.provider.get_max_training_tokens(self.model)

        if not max_training_tokens:
            # No limit, use desired size
            return desired_size

        # Calculate max samples that fit
        max_samples = max_training_tokens // avg_tokens_per_sample

        if desired_size > max_samples:
            return max_samples

        return desired_size
