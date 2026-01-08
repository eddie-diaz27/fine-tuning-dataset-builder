"""
CUSTOM mode agent for user-defined custom behavior.
Allows maximum flexibility for experimental workflows.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import uuid

from agents.base_agent import BaseAgent, AgentError
from utils.validator import Sample


class CustomAgent(BaseAgent):
    """User-defined custom dataset generation"""

    def _load_system_prompt(self) -> str:
        """
        Load CUSTOM mode system prompt.

        Checks for user-defined prompt at prompts/custom_mode/prompt.txt.
        If not found, prompts user interactively.

        Returns:
            System prompt string
        """
        prompt_path = Path("prompts/custom_mode/prompt.txt")

        if prompt_path.exists():
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    custom_prompt = f.read().strip()

                if custom_prompt:
                    print(f"\n✓ Loaded custom prompt from {prompt_path}\n")
                    return custom_prompt
            except Exception as e:
                print(f"⚠️  Failed to load custom prompt: {e}")
                print("⚠️  Falling back to interactive mode\n")

        # Prompt user interactively
        print(f"\n{'='*60}")
        print("CUSTOM Mode: Custom Instructions")
        print(f"{'='*60}\n")
        print("Please provide custom instructions for dataset generation.")
        print("These instructions will guide the AI in creating your dataset.\n")
        print("Example instructions:")
        print("  - 'Generate conversations about Python programming'")
        print("  - 'Create Q&A pairs for customer support scenarios'")
        print("  - 'Extract technical documentation into conversational format'\n")

        # Get user input
        custom_prompt = input("Enter your instructions (or press Enter for default): ").strip()

        if not custom_prompt:
            # Default fallback
            custom_prompt = (
                "You are an AI assistant that generates high-quality "
                "conversational training data. Create natural, helpful, "
                "and educational conversations based on the provided source material."
            )
            print(f"\n✓ Using default instructions\n")
        else:
            print(f"\n✓ Custom instructions received\n")

            # Optionally save for future use
            save_prompt = input("Save these instructions for future use? (y/n): ").strip().lower()
            if save_prompt == 'y':
                try:
                    prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(prompt_path, 'w', encoding='utf-8') as f:
                        f.write(custom_prompt)
                    print(f"✓ Saved to {prompt_path}\n")
                except Exception as e:
                    print(f"⚠️  Failed to save: {e}\n")

        return custom_prompt

    def generate_samples(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter
    ) -> List:
        """
        Generate samples using user-defined behavior.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Number of samples to generate
            target_provider: Provider for final dataset
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects
        """
        if not source_files:
            raise AgentError("No source files provided for CUSTOM mode")

        print(f"\n{'='*60}")
        print(f"CUSTOM Mode: Generating {target_count} samples")
        print(f"{'='*60}\n")

        # Get custom generation parameters
        temperature = self.config.get('temperature', 0.7)
        max_retries = self.config.get('max_retries', 3)

        samples = []
        generated = 0
        attempts = 0
        max_attempts = target_count * 3

        # Prepare source material summary
        source_summary = self._summarize_files(source_files, max_length=2000)

        print(f"Custom Instructions:\n{self.system_prompt}\n")
        print(f"{'='*60}\n")

        while generated < target_count and attempts < max_attempts:
            attempts += 1

            # Select random source file
            import random
            file = random.choice(source_files)

            # Create custom prompt
            generation_prompt = f"""{self.system_prompt}

**Source Material:**
{self._summarize_files([file], max_length=1500)}

**Task:**
Generate a single high-quality conversation based on the source material above.

**Output Format:**
SYSTEM: <system message describing the assistant's role>
USER: <user's question or request>
ASSISTANT: <assistant's helpful response>

You may include multiple turns if appropriate:
USER: <follow-up question>
ASSISTANT: <follow-up response>

Generate now:"""

            try:
                # Call LLM with custom prompt
                response = self._call_llm(generation_prompt, temperature=temperature)

                # Parse conversation
                parsed = self._parse_conversation(response)

                # Validate
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

                # Extract keywords
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
                    mode='custom'
                )

                samples.append(sample)
                generated += 1

                print(f"  ✓ Generated {generated}/{target_count} samples "
                      f"({token_length} tokens)")

            except AgentError:
                # Propagate agent errors
                raise
            except Exception as e:
                print(f"  ⚠️  Attempt {attempts}: Generation failed - {e}")
                continue

        if generated < target_count:
            print(f"\n⚠️  Warning: Only generated {generated}/{target_count} samples "
                  f"after {max_attempts} attempts")

        print(f"\n{'='*60}")
        print(f"CUSTOM Mode Complete: {len(samples)} samples generated")
        print(f"{'='*60}\n")

        return samples

    def generate_with_custom_format(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter,
        custom_format: str
    ) -> List:
        """
        Generate samples with a custom output format.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Number of samples to generate
            target_provider: Provider instance
            token_counter: TokenCounter instance
            custom_format: Custom format specification

        Returns:
            List of Sample objects
        """
        # Store original system prompt
        original_prompt = self.system_prompt

        # Append custom format instructions
        self.system_prompt = f"{original_prompt}\n\n**Custom Format:**\n{custom_format}"

        try:
            # Generate with modified prompt
            samples = self.generate_samples(
                source_files,
                target_count,
                target_provider,
                token_counter
            )

            return samples

        finally:
            # Restore original prompt
            self.system_prompt = original_prompt

    def batch_generate_with_variations(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter,
        temperature_range: tuple = (0.5, 0.9)
    ) -> List:
        """
        Generate samples with varying temperatures for diversity.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Number of samples to generate
            target_provider: Provider instance
            token_counter: TokenCounter instance
            temperature_range: (min_temp, max_temp) tuple

        Returns:
            List of Sample objects
        """
        import random

        # Temporarily override config temperature
        original_temp = self.config.get('temperature', 0.7)

        samples = []
        generated = 0
        attempts = 0
        max_attempts = target_count * 3

        while generated < target_count and attempts < max_attempts:
            attempts += 1

            # Vary temperature
            temperature = random.uniform(*temperature_range)
            self.config['temperature'] = temperature

            file = random.choice(source_files)

            generation_prompt = f"""{self.system_prompt}

**Source Material:**
{self._summarize_files([file], max_length=1500)}

Generate a conversation:
SYSTEM: <system message>
USER: <user message>
ASSISTANT: <assistant message>

Generate now:"""

            try:
                response = self._call_llm(generation_prompt, temperature=temperature)
                parsed = self._parse_conversation(response)

                if not parsed.get('user') or not parsed.get('assistant'):
                    continue

                sample_id = str(uuid.uuid4())
                content = {
                    'system': parsed.get('system', ''),
                    'user': parsed.get('user', ''),
                    'assistant': parsed.get('assistant', '')
                }

                content_str = json.dumps(content)
                token_length = token_counter.count_tokens(
                    content_str,
                    target_provider.get_spec().name.lower(),
                    self.config.get('model', '')
                )

                sample_keywords = self._extract_keywords(
                    parsed.get('user', '') + ' ' + parsed.get('assistant', ''),
                    count=3
                )

                sample = Sample(
                    id=sample_id,
                    content=content,
                    token_length=token_length,
                    topic_keywords=sample_keywords,
                    source_file=file.file_name,
                    mode='custom'
                )

                samples.append(sample)
                generated += 1

                print(f"  ✓ Generated {generated}/{target_count} samples "
                      f"(temp={temperature:.2f}, {token_length} tokens)")

            except AgentError:
                raise
            except Exception as e:
                print(f"  ⚠️  Attempt {attempts}: Failed - {e}")
                continue

        # Restore original temperature
        self.config['temperature'] = original_temp

        return samples

    def interactive_refinement(
        self,
        source_files: List,
        target_provider,
        token_counter
    ) -> List:
        """
        Interactive mode for iterative refinement.

        Args:
            source_files: List of ProcessedFile objects
            target_provider: Provider instance
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects
        """
        print(f"\n{'='*60}")
        print("CUSTOM Mode: Interactive Refinement")
        print(f"{'='*60}\n")

        samples = []

        while True:
            print("\nOptions:")
            print("  1. Generate a single sample")
            print("  2. Generate a batch (specify count)")
            print("  3. Modify instructions")
            print("  4. Review samples")
            print("  5. Finish")

            choice = input("\nSelect option (1-5): ").strip()

            if choice == '1':
                batch = self.generate_samples(
                    source_files, 1, target_provider, token_counter
                )
                samples.extend(batch)

            elif choice == '2':
                count = int(input("Enter batch size: ").strip())
                batch = self.generate_samples(
                    source_files, count, target_provider, token_counter
                )
                samples.extend(batch)

            elif choice == '3':
                new_prompt = input("Enter new instructions: ").strip()
                if new_prompt:
                    self.system_prompt = new_prompt
                    print("✓ Instructions updated")

            elif choice == '4':
                print(f"\nGenerated {len(samples)} samples so far")
                if samples:
                    preview = input("Preview last sample? (y/n): ").strip().lower()
                    if preview == 'y':
                        last = samples[-1]
                        print(f"\nSample ID: {last.id}")
                        print(f"User: {last.content.get('user', '')[:100]}...")
                        print(f"Assistant: {last.content.get('assistant', '')[:100]}...")

            elif choice == '5':
                print(f"\n✓ Finished with {len(samples)} samples")
                break

            else:
                print("Invalid option")

        return samples
