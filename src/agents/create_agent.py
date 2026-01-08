"""
CREATE mode agent for synthetic dataset generation.
Generates conversations inspired by source files.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import uuid

from agents.base_agent import BaseAgent, AgentError
from utils.validator import Sample


class CreateAgent(BaseAgent):
    """Generate synthetic conversations from source files"""

    def _load_system_prompt(self) -> str:
        """
        Load CREATE mode system prompt.

        Returns:
            System prompt string
        """
        prompt_path = Path("prompts/system/create_mode.txt")

        if not prompt_path.exists():
            raise AgentError(
                f"CREATE mode system prompt not found at {prompt_path}.\n"
                f"Please ensure prompts/system/create_mode.txt exists."
            )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise AgentError(f"Failed to load CREATE mode system prompt: {e}")

    def generate_samples(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter
    ) -> List:
        """
        Generate synthetic conversation samples.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Number of samples to generate
            target_provider: Provider for final dataset
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects
        """
        if not source_files:
            raise AgentError("No source files provided for CREATE mode")

        print(f"\n{'='*60}")
        print(f"CREATE Mode: Generating {target_count} synthetic samples")
        print(f"{'='*60}\n")

        samples = []
        variety = self.config.get('variety', 'medium')
        temperature = self._get_temperature_for_variety(variety)
        length_preference = self._get_length_preference()

        # Generate topic keywords from all source files
        print("Analyzing source files...")
        all_topics = []
        for file in source_files:
            topics = self._extract_keywords(file.text, count=3)
            all_topics.extend(topics)

        # Deduplicate topics
        unique_topics = list(set(all_topics))
        print(f"Identified {len(unique_topics)} unique topics\n")

        # Generate samples
        generated = 0
        attempts = 0
        max_attempts = target_count * 3  # Allow retries

        while generated < target_count and attempts < max_attempts:
            attempts += 1

            # Select random source file
            import random
            file = random.choice(source_files)

            # Create generation prompt
            file_summary = self._summarize_files([file], max_length=1500)

            generation_prompt = f"""{self.system_prompt}

**Source Material:**
{file_summary}

**Generation Parameters:**
- Conversation Length: {length_preference} exchanges
- Variety: {variety}
- Topics: Focus on {', '.join(random.sample(unique_topics, min(3, len(unique_topics))))}

**Instructions:**
Generate a {length_preference} conversation inspired by the source material above.
The conversation should be natural and educational.

Output format:
SYSTEM: <system message describing the assistant's role>
USER: <user's question or request>
ASSISTANT: <assistant's helpful response>
USER: <follow-up question>
ASSISTANT: <follow-up response>
... (continue for {length_preference} turns)

Generate now:"""

            try:
                # Call LLM
                response = self._call_llm(generation_prompt, temperature=temperature)

                # Parse conversation
                parsed = self._parse_conversation(response)

                # Validate parsed output
                if not parsed.get('user') or not parsed.get('assistant'):
                    print(f"  ⚠️  Attempt {attempts}: Invalid format, retrying...")
                    continue

                # Create sample
                sample_id = str(uuid.uuid4())
                content = {
                    'system': parsed.get('system', ''),
                    'user': parsed.get('user', ''),
                    'assistant': parsed.get('assistant', '')
                }

                # Count tokens
                content_str = json.dumps(content)
                token_length = token_counter.count_tokens(
                    content_str,
                    target_provider.get_spec().name.lower(),
                    self.config.get('model', '')
                )

                # Extract keywords for this sample
                sample_keywords = self._extract_keywords(
                    parsed.get('user', '') + ' ' + parsed.get('assistant', ''),
                    count=3
                )

                # Create Sample object
                sample = Sample(
                    id=sample_id,
                    content=content,
                    token_length=token_length,
                    topic_keywords=sample_keywords,
                    source_file=file.file_name,
                    mode='create'
                )

                samples.append(sample)
                generated += 1

                print(f"  ✓ Generated {generated}/{target_count} samples "
                      f"({token_length} tokens)")

            except AgentError:
                # Propagate agent errors (auth, rate limit)
                raise
            except Exception as e:
                print(f"  ⚠️  Attempt {attempts}: Generation failed - {e}")
                continue

        if generated < target_count:
            print(f"\n⚠️  Warning: Only generated {generated}/{target_count} samples "
                  f"after {max_attempts} attempts")

        print(f"\n{'='*60}")
        print(f"CREATE Mode Complete: {len(samples)} samples generated")
        print(f"{'='*60}\n")

        return samples

    def _generate_with_constraints(
        self,
        file,
        unique_topics: List[str],
        temperature: float,
        length_preference: str,
        variety: str
    ) -> str:
        """
        Generate a conversation with specific constraints.

        Args:
            file: ProcessedFile object
            unique_topics: List of topics to focus on
            temperature: Generation temperature
            length_preference: Conversation length preference
            variety: Variety setting

        Returns:
            Generated text
        """
        import random

        file_summary = self._summarize_files([file], max_length=1500)

        # Select 2-3 random topics
        selected_topics = random.sample(
            unique_topics,
            min(3, len(unique_topics))
        )

        prompt = f"""{self.system_prompt}

**Source Material:**
{file_summary}

**Generation Parameters:**
- Conversation Length: {length_preference} exchanges
- Variety: {variety}
- Topics: Focus on {', '.join(selected_topics)}

**Instructions:**
Generate a {length_preference} conversation inspired by the source material above.
The conversation should be natural, educational, and vary in style.

Output format:
SYSTEM: <system message>
USER: <user message>
ASSISTANT: <assistant message>
USER: <follow-up>
ASSISTANT: <response>
... (continue for {length_preference} turns)

Generate now:"""

        return self._call_llm(prompt, temperature=temperature)

    def _validate_sample_quality(
        self,
        sample: Dict[str, str],
        min_user_length: int = 10,
        min_assistant_length: int = 20
    ) -> bool:
        """
        Validate sample meets quality standards.

        Args:
            sample: Parsed sample dict
            min_user_length: Minimum user message length
            min_assistant_length: Minimum assistant message length

        Returns:
            True if valid
        """
        user = sample.get('user', '')
        assistant = sample.get('assistant', '')

        # Check minimum lengths
        if len(user) < min_user_length:
            return False

        if len(assistant) < min_assistant_length:
            return False

        # Check for placeholder text
        placeholders = ['...', '[insert', 'TODO', 'example here']
        for placeholder in placeholders:
            if placeholder in user.lower() or placeholder in assistant.lower():
                return False

        # Check for reasonable structure
        # User should be a question or request
        if not any(c in user for c in ['?', 'please', 'could', 'would', 'how', 'what', 'why']):
            # Allow statements too, but ensure some variety
            pass

        return True

    def _create_sample_object(
        self,
        parsed: Dict[str, str],
        file,
        target_provider,
        token_counter,
        sample_keywords: List[str]
    ) -> Sample:
        """
        Create a Sample object from parsed conversation.

        Args:
            parsed: Parsed conversation dict
            file: Source ProcessedFile
            target_provider: Provider instance
            token_counter: TokenCounter instance
            sample_keywords: List of keywords

        Returns:
            Sample object
        """
        sample_id = str(uuid.uuid4())

        content = {
            'system': parsed.get('system', ''),
            'user': parsed.get('user', ''),
            'assistant': parsed.get('assistant', '')
        }

        # Count tokens
        content_str = json.dumps(content)
        token_length = token_counter.count_tokens(
            content_str,
            target_provider.get_spec().name.lower(),
            self.config.get('model', '')
        )

        return Sample(
            id=sample_id,
            content=content,
            token_length=token_length,
            topic_keywords=sample_keywords,
            source_file=file.file_name,
            mode='create'
        )

    def generate_batch(
        self,
        source_files: List,
        batch_size: int,
        target_provider,
        token_counter
    ) -> List:
        """
        Generate a batch of samples (used by hybrid mode).

        Args:
            source_files: List of ProcessedFile objects
            batch_size: Number of samples to generate
            target_provider: Provider instance
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects
        """
        # Reuse main generate_samples method
        return self.generate_samples(
            source_files,
            batch_size,
            target_provider,
            token_counter
        )
