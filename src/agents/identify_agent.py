"""
IDENTIFY mode agent for conversation extraction.
Extracts existing conversations verbatim from source files.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import uuid

from agents.base_agent import BaseAgent, AgentError
from utils.conversation_detector import ConversationDetector, Conversation
from utils.validator import Sample


class IdentifyAgent(BaseAgent):
    """Extract existing conversations from source files"""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        config: Dict[str, Any]
    ):
        """
        Initialize IDENTIFY agent.

        Args:
            provider: LLM provider name
            model: Model name
            api_key: API key
            config: Agent configuration dict
        """
        super().__init__(provider, model, api_key, config)
        self.detector = ConversationDetector(
            llm_provider=provider,
            llm_model=model
        )

    def _load_system_prompt(self) -> str:
        """
        Load IDENTIFY mode system prompt.

        Returns:
            System prompt string
        """
        prompt_path = Path("prompts/system/identify_mode.txt")

        if not prompt_path.exists():
            raise AgentError(
                f"IDENTIFY mode system prompt not found at {prompt_path}.\n"
                f"Please ensure prompts/system/identify_mode.txt exists."
            )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise AgentError(f"Failed to load IDENTIFY mode system prompt: {e}")

    def generate_samples(
        self,
        source_files: List,
        target_count: int = None,
        target_provider = None,
        token_counter = None
    ) -> List:
        """
        Extract conversation samples from source files.

        CRITICAL: Extracts conversations VERBATIM without modification.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Advisory count for warning purposes (None = no limit)
                         IDENTIFY mode always extracts ALL available conversations
            target_provider: Provider for final dataset
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects (all conversations found)
        """
        if not source_files:
            raise AgentError("No source files provided for IDENTIFY mode")

        print(f"\n{'='*60}")
        print(f"IDENTIFY Mode: Extracting conversations from {len(source_files)} files")
        print(f"{'='*60}\n")

        all_samples = []

        # Extract from each file
        for file in source_files:
            print(f"Processing: {file.file_name}")

            # Detect conversations
            conversations = self.detector.detect_conversations(
                file.text,
                min_turns=self.config.get('min_conversation_turns', 2),
                confidence_threshold=self.config.get('confidence_threshold', 0.7)
            )

            if not conversations:
                print(f"  ⚠️  No conversations found in {file.file_name}")
                continue

            print(f"  ✓ Found {len(conversations)} conversations")

            # Convert to samples
            for conv in conversations:
                try:
                    sample = self._conversation_to_sample(
                        conv,
                        file,
                        target_provider,
                        token_counter
                    )

                    all_samples.append(sample)

                except Exception as e:
                    print(f"  ⚠️  Failed to convert conversation: {e}")
                    continue

        if not all_samples:
            raise AgentError(
                "No conversations found in source files.\n"
                "IDENTIFY mode requires files with existing Q&A or chat formats.\n"
                "Suggestions:\n"
                "  - Verify files contain conversation patterns (Q&A, User/Assistant, etc.)\n"
                "  - Try using CREATE or HYBRID mode instead."
            )

        # Check against target and issue warnings
        if target_count is not None:
            if len(all_samples) < target_count:
                print(f"\n⚠️  WARNING: Found fewer conversations than target")
                print(f"   Target: {target_count}")
                print(f"   Found:  {len(all_samples)}")
                print(f"   Consider using CREATE or HYBRID mode to reach target\n")
            elif len(all_samples) > target_count:
                print(f"\n📊 INFO: Found more conversations than target")
                print(f"   Target: {target_count}")
                print(f"   Found:  {len(all_samples)}")
                print(f"   All {len(all_samples)} conversations will be extracted\n")

        print(f"\n{'='*60}")
        print(f"IDENTIFY Mode Complete: {len(all_samples)} conversations extracted")
        if target_count:
            print(f"Target was {target_count}, extraction is complete regardless of target")
        print(f"{'='*60}\n")

        return all_samples

    def _conversation_to_sample(
        self,
        conversation: Conversation,
        file,
        target_provider,
        token_counter
    ) -> Sample:
        """
        Convert Conversation to Sample object.

        CRITICAL: Preserves original text VERBATIM, no modification.

        Args:
            conversation: Conversation object
            file: ProcessedFile object
            target_provider: Provider instance
            token_counter: TokenCounter instance

        Returns:
            Sample object
        """
        sample_id = str(uuid.uuid4())

        # Extract turns
        turns = conversation.turns

        if len(turns) < 2:
            raise ValueError("Conversation must have at least 2 turns")

        # Build content
        # Assume first turn is user, second is assistant
        # For multi-turn, concatenate alternating turns
        user_parts = []
        assistant_parts = []

        for i, turn in enumerate(turns):
            if i % 2 == 0:
                # User turn
                user_parts.append(turn.content)
            else:
                # Assistant turn
                assistant_parts.append(turn.content)

        # Join with newlines to preserve multi-turn structure
        user_content = '\n\n'.join(user_parts)
        assistant_content = '\n\n'.join(assistant_parts)

        # Create system message (use default or extract from conversation)
        system_content = self._infer_system_message(conversation)

        content = {
            'system': system_content,
            'user': user_content,
            'assistant': assistant_content
        }

        # Count tokens
        content_str = json.dumps(content)
        token_length = token_counter.count_tokens(
            content_str,
            target_provider.get_spec().name.lower(),
            self.config.get('model', '')
        )

        # Extract keywords from conversation content
        sample_keywords = self._extract_keywords(
            user_content + ' ' + assistant_content,
            count=3
        )

        return Sample(
            id=sample_id,
            content=content,
            token_length=token_length,
            topic_keywords=sample_keywords,
            source_file=file.file_name,
            mode='identify'
        )

    def _infer_system_message(self, conversation: Conversation) -> str:
        """
        Infer system message from conversation context.

        Args:
            conversation: Conversation object

        Returns:
            System message string
        """
        # Use a generic system message based on source pattern
        # conversation.source_pattern indicates how it was detected
        source_pattern = conversation.source_pattern.lower() if conversation.source_pattern else ''

        if 'qa' in source_pattern or 'question' in source_pattern:
            return "You are a helpful assistant that provides clear, accurate answers to questions."
        elif 'user' in source_pattern or 'assistant' in source_pattern:
            return "You are a helpful AI assistant."
        elif 'sales' in source_pattern or 'rep' in source_pattern:
            return "You are a professional sales assistant helping customers make informed decisions."
        else:
            # Generic fallback
            return "You are a helpful, harmless, and honest AI assistant."

    def extract_with_context(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter,
        include_context: bool = True,
        context_lines: int = 2
    ) -> List:
        """
        Extract conversations with surrounding context.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Target number of samples
            target_provider: Provider instance
            token_counter: TokenCounter instance
            include_context: Whether to include context paragraphs
            context_lines: Number of context lines before/after

        Returns:
            List of Sample objects
        """
        # For now, delegate to main method
        # Future enhancement: Add context extraction logic
        return self.generate_samples(
            source_files,
            target_count,
            target_provider,
            token_counter
        )

    def validate_extraction_quality(
        self,
        samples: List[Sample],
        min_confidence: float = 0.7
    ) -> List[Sample]:
        """
        Validate extracted samples meet quality standards.

        Args:
            samples: List of Sample objects
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of high-quality samples
        """
        validated = []

        for sample in samples:
            user = sample.content.get('user', '')
            assistant = sample.content.get('assistant', '')

            # Check minimum lengths
            if len(user) < 10 or len(assistant) < 20:
                continue

            # Check for artifacts (page numbers, headers, etc.)
            artifacts = ['page ', 'chapter ', '©', 'copyright', 'all rights reserved']
            has_artifacts = any(
                artifact in user.lower() or artifact in assistant.lower()
                for artifact in artifacts
            )

            if has_artifacts:
                continue

            # Passed validation
            validated.append(sample)

        return validated

    def get_extraction_statistics(self, source_files: List) -> Dict[str, Any]:
        """
        Get statistics about extractable conversations without generating samples.

        Args:
            source_files: List of ProcessedFile objects

        Returns:
            Dict with extraction statistics
        """
        stats = {
            'total_files': len(source_files),
            'files_with_conversations': 0,
            'total_conversations': 0,
            'conversations_by_pattern': {},
            'avg_confidence': 0.0
        }

        all_confidences = []

        for file in source_files:
            conversations = self.detector.detect_conversations(
                file.text,
                min_turns=self.config.get('min_conversation_turns', 2),
                confidence_threshold=self.config.get('confidence_threshold', 0.7)
            )

            if conversations:
                stats['files_with_conversations'] += 1
                stats['total_conversations'] += len(conversations)

                for conv in conversations:
                    pattern = conv.pattern_type
                    stats['conversations_by_pattern'][pattern] = \
                        stats['conversations_by_pattern'].get(pattern, 0) + 1

                    all_confidences.append(conv.confidence)

        if all_confidences:
            stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)

        return stats
