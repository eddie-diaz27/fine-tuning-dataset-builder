"""
OpenAI provider implementation.
Supports: GPT-4o, GPT-4o mini, GPT-4.1, GPT-4.1 mini, GPT-3.5-turbo
"""

from typing import Dict, Any, List
from providers.base_provider import BaseProvider, ProviderSpec, ValidationError


class OpenAIProvider(BaseProvider):
    """OpenAI GPT fine-tuning provider"""

    def get_spec(self) -> ProviderSpec:
        """Return OpenAI provider specifications"""
        return ProviderSpec(
            name="OpenAI",
            models=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-3.5-turbo"
            ],
            token_limits={
                "gpt-4o": 65536,
                "gpt-4o-mini": 65536,
                "gpt-4.1": 65536,
                "gpt-4.1-mini": 65536,
                "gpt-3.5-turbo": 16385
            },
            max_training_tokens={
                "gpt-4o": 50_000_000,
                "gpt-4o-mini": 50_000_000,
                "gpt-4.1": 50_000_000,
                "gpt-4.1-mini": 50_000_000,
                "gpt-3.5-turbo": 50_000_000
            },
            format_type="openai",
            min_examples=50
        )

    def format_example(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> Dict[str, Any]:
        """
        Format example in OpenAI JSONL format.

        OpenAI format:
        {
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

        Args:
            system: System message
            user: User message
            assistant: Assistant response

        Returns:
            Formatted example dict
        """
        messages = []

        # Add system message if provided
        if system and system.strip():
            messages.append({"role": "system", "content": system})

        # Add user and assistant messages
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

        return {"messages": messages}

    def format_multi_turn(
        self,
        system: str,
        turns: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Format multi-turn conversation.

        Args:
            system: System message
            turns: List of {"role": ..., "content": ...}

        Returns:
            Formatted example dict
        """
        messages = []

        # Add system message if provided
        if system and system.strip():
            messages.append({"role": "system", "content": system})

        # Add turns
        for turn in turns:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

        return {"messages": messages}

    def validate_example(self, example: Dict) -> bool:
        """
        Validate OpenAI format example.

        Rules:
        - Must have "messages" key
        - Messages must be a list
        - Each message must have "role" and "content"
        - Last message must be from assistant
        - Roles must be valid (system, user, assistant)

        Args:
            example: Example to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If invalid
        """
        # Check for messages key
        if "messages" not in example:
            raise ValidationError("Missing 'messages' key")

        messages = example["messages"]

        # Check if messages is a list
        if not isinstance(messages, list):
            raise ValidationError("'messages' must be a list")

        # Check if messages is not empty
        if not messages:
            raise ValidationError("'messages' cannot be empty")

        # Validate each message
        valid_roles = {"system", "user", "assistant"}

        for i, msg in enumerate(messages):
            # Check for role and content
            if "role" not in msg:
                raise ValidationError(f"Message {i}: missing 'role'")
            if "content" not in msg:
                raise ValidationError(f"Message {i}: missing 'content'")

            # Check role is valid
            if msg["role"] not in valid_roles:
                raise ValidationError(
                    f"Message {i}: invalid role '{msg['role']}'. "
                    f"Must be one of: {', '.join(valid_roles)}"
                )

            # Check content is not empty
            if not msg["content"] or not msg["content"].strip():
                raise ValidationError(f"Message {i}: empty content")

        # Check last message is from assistant
        if messages[-1]["role"] != "assistant":
            raise ValidationError(
                "Last message must be from assistant, "
                f"got: {messages[-1]['role']}"
            )

        return True
