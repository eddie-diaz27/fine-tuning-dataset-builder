"""
Base agent class for dataset generation.
Provides common functionality for all agent modes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import time
import random


class AgentError(Exception):
    """Raised when agent operations fail"""
    pass


class BaseAgent(ABC):
    """Abstract base agent for dataset generation"""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        config: Dict[str, Any]
    ):
        """
        Initialize base agent.

        Args:
            provider: LLM provider name (google, openai, etc.)
            model: Model name
            api_key: API key
            config: Agent configuration dict
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.config = config
        self.client = self._init_client(provider, api_key)
        self.system_prompt = self._load_system_prompt()

    @abstractmethod
    def _load_system_prompt(self) -> str:
        """
        Load mode-specific system prompt.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def generate_samples(
        self,
        source_files: List,
        target_count: int,
        target_provider,
        token_counter
    ) -> List:
        """
        Generate dataset samples.

        Args:
            source_files: List of ProcessedFile objects
            target_count: Number of samples to generate
            target_provider: Provider for final dataset
            token_counter: TokenCounter instance

        Returns:
            List of Sample objects
        """
        pass

    def _init_client(self, provider: str, api_key: str):
        """
        Initialize LLM client based on provider.

        Args:
            provider: Provider name
            api_key: API key

        Returns:
            Client instance
        """
        if provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                return genai.GenerativeModel(self.model)
            except ImportError:
                raise AgentError(
                    "google-generativeai is required for Google provider.\n"
                    "Install with: pip install google-generativeai"
                )

        elif provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=api_key)
            except ImportError:
                raise AgentError(
                    "openai is required for OpenAI provider.\n"
                    "Install with: pip install openai"
                )

        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=api_key)
            except ImportError:
                raise AgentError(
                    "anthropic is required for Anthropic provider.\n"
                    "Install with: pip install anthropic"
                )

        else:
            raise AgentError(f"Provider '{provider}' not yet supported for agents")

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """
        Call LLM with retry logic.

        Args:
            prompt: Prompt to send
            temperature: Generation temperature
            max_retries: Maximum retry attempts

        Returns:
            Generated text

        Raises:
            AgentError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "google":
                    response = self.client.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    return response.text

                elif self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    return response.choices[0].message.content

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    return response.content[0].text

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limiting
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.random()
                        print(f"Rate limit hit. Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue

                # Check for authentication errors
                if "auth" in error_str or "api key" in error_str or "401" in error_str:
                    raise AgentError(
                        f"Authentication failed for {self.provider}.\n"
                        f"Please check your API key in .env file."
                    )

                # Last attempt
                if attempt == max_retries - 1:
                    raise AgentError(f"LLM call failed after {max_retries} attempts: {e}")

                # Retry for other errors
                time.sleep(1)

        raise AgentError("LLM call failed")

    def _extract_keywords(self, text: str, count: int = 5) -> List[str]:
        """
        Extract topic keywords from text using LLM.

        Args:
            text: Text to analyze
            count: Number of keywords to extract

        Returns:
            List of keywords
        """
        # Truncate very long text
        text_sample = text[:1000] if len(text) > 1000 else text

        prompt = f"""Extract {count} topic keywords from this text.
Return only the keywords, comma-separated, no explanations.

Text: {text_sample}

Keywords:"""

        try:
            response = self._call_llm(prompt, temperature=0.3)
            # Parse keywords
            keywords = [k.strip() for k in response.split(',')]
            # Clean and limit
            keywords = [k for k in keywords if k and len(k) > 2][:count]
            return keywords if keywords else ["general"]
        except Exception:
            return ["general"]

    def _get_temperature_for_variety(self, variety: str) -> float:
        """
        Get temperature based on variety setting.

        Args:
            variety: "low", "medium", or "high"

        Returns:
            Temperature value
        """
        variety_map = {
            "low": 0.3,
            "medium": 0.7,
            "high": 0.9
        }
        return variety_map.get(variety, 0.7)

    def _get_length_preference(self) -> str:
        """
        Get conversation length preference from config.

        Returns:
            Length descriptor
        """
        length = self.config.get('length', 'medium')
        length_map = {
            "short": "2-3 turn",
            "medium": "4-6 turn",
            "long": "7+ turn"
        }
        return length_map.get(length, "4-6 turn")

    def _parse_conversation(self, text: str) -> Dict[str, str]:
        """
        Parse LLM output into structured conversation.

        Expected format:
        SYSTEM: <system message>
        USER: <user message>
        ASSISTANT: <assistant message>

        Args:
            text: LLM output

        Returns:
            Dict with system, user, assistant keys
        """
        lines = text.strip().split('\n')

        system = ""
        user = ""
        assistant = ""

        current_role = None
        current_content = []

        for line in lines:
            line_upper = line.upper().strip()

            if line_upper.startswith("SYSTEM:"):
                if current_role and current_content:
                    # Save previous role
                    if current_role == "system":
                        system = '\n'.join(current_content)
                    elif current_role == "user":
                        user = '\n'.join(current_content)
                    elif current_role == "assistant":
                        assistant = '\n'.join(current_content)

                current_role = "system"
                current_content = [line[7:].strip()]  # Remove "SYSTEM:" prefix

            elif line_upper.startswith("USER:"):
                if current_role and current_content:
                    if current_role == "system":
                        system = '\n'.join(current_content)
                    elif current_role == "user":
                        user = '\n'.join(current_content)
                    elif current_role == "assistant":
                        assistant = '\n'.join(current_content)

                current_role = "user"
                current_content = [line[5:].strip()]

            elif line_upper.startswith("ASSISTANT:"):
                if current_role and current_content:
                    if current_role == "system":
                        system = '\n'.join(current_content)
                    elif current_role == "user":
                        user = '\n'.join(current_content)
                    elif current_role == "assistant":
                        assistant = '\n'.join(current_content)

                current_role = "assistant"
                current_content = [line[10:].strip()]

            else:
                # Continue current role
                if current_role and line.strip():
                    current_content.append(line.strip())

        # Save last role
        if current_role and current_content:
            if current_role == "system":
                system = '\n'.join(current_content)
            elif current_role == "user":
                user = '\n'.join(current_content)
            elif current_role == "assistant":
                assistant = '\n'.join(current_content)

        return {
            "system": system,
            "user": user,
            "assistant": assistant
        }

    def _summarize_files(self, files: List, max_length: int = 2000) -> str:
        """
        Create summary of source files for prompting.

        Args:
            files: List of ProcessedFile objects
            max_length: Max chars per file

        Returns:
            Summarized text
        """
        summaries = []

        for file in files[:5]:  # Limit to 5 files to avoid huge prompts
            # Truncate text
            text = file.text[:max_length]
            if len(file.text) > max_length:
                text += "..."

            summaries.append(f"=== {file.file_name} ===\n{text}")

        return "\n\n".join(summaries)
