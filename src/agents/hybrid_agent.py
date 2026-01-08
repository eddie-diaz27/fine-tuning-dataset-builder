"""
HYBRID mode agent combining CREATE and IDENTIFY approaches.
Extracts real conversations + generates synthetic ones.
"""

from typing import List, Dict, Any
from pathlib import Path

from agents.base_agent import BaseAgent, AgentError
from agents.create_agent import CreateAgent
from agents.identify_agent import IdentifyAgent
from utils.validator import Sample


class HybridAgent(BaseAgent):
    """Combine extraction and generation for hybrid datasets"""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        config: Dict[str, Any]
    ):
        """
        Initialize HYBRID agent.

        Args:
            provider: LLM provider name
            model: Model name
            api_key: API key
            config: Agent configuration dict
        """
        super().__init__(provider, model, api_key, config)

        # Initialize both sub-agents
        self.create_agent = CreateAgent(provider, model, api_key, config)
        self.identify_agent = IdentifyAgent(provider, model, api_key, config)

    def _load_system_prompt(self) -> str:
        """
        Load HYBRID mode system prompt.

        Returns:
            System prompt string
        """
        prompt_path = Path("prompts/system/hybrid_mode.txt")

        if not prompt_path.exists():
            raise AgentError(
                f"HYBRID mode system prompt not found at {prompt_path}.\n"
                f"Please ensure prompts/system/hybrid_mode.txt exists."
            )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise AgentError(f"Failed to load HYBRID mode system prompt: {e}")

    def generate_samples(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter
    ) -> List:
        """
        Generate hybrid dataset combining extraction and generation.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Total number of samples to generate
            target_provider: Provider for final dataset
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects
        """
        if not source_files:
            raise AgentError("No source files provided for HYBRID mode")

        print(f"\n{'='*60}")
        print(f"HYBRID Mode: Generating {target_count} samples")
        print(f"{'='*60}\n")

        all_samples = []
        extracted_samples = []
        synthetic_samples = []

        # Phase 1: Extract ALL available conversations first
        print(f"{'='*60}")
        print("Phase 1: Extracting ALL Available Conversations")
        print(f"{'='*60}\n")

        try:
            extracted_samples = self.identify_agent.generate_samples(
                source_files,
                target_count=None,  # Extract all, no limit
                target_provider=target_provider,
                token_counter=token_counter
            )

            extracted_count = len(extracted_samples)
            all_samples.extend(extracted_samples)
            print(f"✓ Extracted {extracted_count} conversations\n")

        except AgentError as e:
            # If extraction fails completely, warn and proceed with all synthetic
            print(f"⚠️  Extraction failed: {e}")
            print(f"⚠️  Will generate all {target_count} samples synthetically\n")
            extracted_count = 0

        # Phase 2: Calculate how many synthetic samples needed to reach target
        remaining = max(0, target_count - extracted_count)

        print(f"{'='*60}")
        print("Phase 2: Generating Synthetic to Reach Target")
        print(f"{'='*60}")
        print(f"Target: {target_count} total samples")
        print(f"Extracted: {extracted_count} samples")
        print(f"Synthetic needed: {remaining} samples\n")

        if remaining > 0:
            synthetic_samples = self.create_agent.generate_samples(
                source_files,
                remaining,
                target_provider,
                token_counter
            )

            all_samples.extend(synthetic_samples)
            print(f"✓ Generated {len(synthetic_samples)} synthetic samples\n")

        # Phase 3: Deduplicate if configured
        if self.config.get('deduplicate', True):
            print(f"{'='*60}")
            print("Phase 3: Deduplication")
            print(f"{'='*60}\n")

            original_count = len(all_samples)
            all_samples = self._deduplicate_samples(all_samples)
            removed = original_count - len(all_samples)

            if removed > 0:
                print(f"✓ Removed {removed} duplicate samples\n")
            else:
                print(f"✓ No duplicates found\n")

        # Final statistics
        print(f"{'='*60}")
        print(f"HYBRID Mode Complete: {len(all_samples)} total samples")
        print(f"{'='*60}")
        print(f"Final Distribution:")
        print(f"  - Extracted: {len(extracted_samples)} samples ({len(extracted_samples)/len(all_samples)*100:.1f}%)")
        print(f"  - Synthetic: {len(synthetic_samples)} samples ({len(synthetic_samples)/len(all_samples)*100:.1f}%)")
        print(f"{'='*60}\n")

        return all_samples

    def _deduplicate_samples(
        self,
        samples: List[Sample],
        similarity_threshold: float = 0.85
    ) -> List[Sample]:
        """
        Remove duplicate or highly similar samples.

        Args:
            samples: List of Sample objects
            similarity_threshold: Similarity threshold (0-1)

        Returns:
            Deduplicated list
        """
        from difflib import SequenceMatcher

        if not samples:
            return samples

        unique_samples = []
        seen_content = []

        for sample in samples:
            # Get sample text
            user = sample.content.get('user', '')
            assistant = sample.content.get('assistant', '')
            sample_text = user + ' ' + assistant

            # Check against all seen content
            is_duplicate = False

            for seen in seen_content:
                similarity = SequenceMatcher(None, sample_text, seen).ratio()

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_samples.append(sample)
                seen_content.append(sample_text)

        return unique_samples

    def get_hybrid_statistics(
        self,
        source_files: List
    ) -> Dict[str, Any]:
        """
        Get statistics about potential hybrid generation.

        Args:
            source_files: List of ProcessedFile objects

        Returns:
            Dict with statistics
        """
        # Get extraction stats
        extract_stats = self.identify_agent.get_extraction_statistics(source_files)

        # Get basic file stats
        total_chars = sum(len(f.text) for f in source_files)
        avg_chars = total_chars / len(source_files) if source_files else 0

        stats = {
            'total_files': len(source_files),
            'total_characters': total_chars,
            'avg_characters_per_file': avg_chars,
            'extractable_conversations': extract_stats['total_conversations'],
            'files_with_conversations': extract_stats['files_with_conversations'],
            'recommended_ratio': self._recommend_generation_ratio(extract_stats)
        }

        return stats

    def _recommend_generation_ratio(
        self,
        extract_stats: Dict[str, Any]
    ) -> float:
        """
        Recommend generation ratio based on extraction potential.

        Args:
            extract_stats: Extraction statistics

        Returns:
            Recommended generation ratio (0-1)
        """
        extractable = extract_stats['total_conversations']

        if extractable == 0:
            # No extractable conversations, use 100% synthetic
            return 1.0
        elif extractable < 10:
            # Few conversations, mostly synthetic
            return 0.8
        elif extractable < 50:
            # Moderate conversations, balanced
            return 0.5
        else:
            # Many conversations, mostly extract
            return 0.3

    def preview_hybrid_distribution(
        self,
        source_files: List,
        target_count: int
    ) -> Dict[str, Any]:
        """
        Preview hybrid distribution without generating.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Target sample count

        Returns:
            Dict with preview information
        """
        stats = self.get_hybrid_statistics(source_files)
        generation_ratio = self.config.get('generation_ratio', 0.5)

        synthetic_count = int(target_count * generation_ratio)
        extracted_count = target_count - synthetic_count

        preview = {
            'target_total': target_count,
            'target_extracted': extracted_count,
            'target_synthetic': synthetic_count,
            'generation_ratio': generation_ratio,
            'available_conversations': stats['extractable_conversations'],
            'extraction_feasible': stats['extractable_conversations'] >= extracted_count,
            'recommended_ratio': stats['recommended_ratio']
        }

        return preview

    def adjust_ratio_dynamically(
        self,
        source_files: List,
        target_count: int
    ) -> float:
        """
        Dynamically adjust generation ratio based on available conversations.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Target sample count

        Returns:
            Adjusted generation ratio
        """
        stats = self.get_hybrid_statistics(source_files)
        extractable = stats['extractable_conversations']

        if extractable == 0:
            # No conversations to extract
            return 1.0

        # Calculate what ratio would give us all extractable conversations
        # extracted_count = target_count * (1 - ratio)
        # ratio = 1 - (extractable / target_count)

        if extractable >= target_count:
            # We have enough to extract everything
            return 0.0
        else:
            # Adjust to use all extractable + fill rest with synthetic
            return 1.0 - (extractable / target_count)
