"""
Mistral AI provider implementation.
Supports: Mistral 7B, Large v2, Nemo, Mixtral 8x7B
"""

from typing import Dict, Any, List
from providers.base_provider import BaseProvider, ProviderSpec, ValidationError


class MistralProvider(BaseProvider):
    """Mistral AI fine-tuning provider"""

    def get_spec(self) -> ProviderSpec:
        """Return Mistral provider specifications"""
        return ProviderSpec(
            name="Mistral",
            models=[
                "mistral-7b",
                "mistral-large-v2",
                "mistral-nemo",
                "mixtral-8x7b"
            ],
            token_limits={
                "mistral-7b": 32000,
                "mistral-large-v2": 8192,  # Recommended for memory
                "mistral-nemo": 32000,
                "mixtral-8x7b": 32000
            },
            max_training_tokens={
                "mistral-7b": None,  # Flexible
                "mistral-large-v2": None,
                "mistral-nemo": None,
                "mixtral-8x7b": None
            },
            format_type="mistral",
            min_examples=50
        )

    def format_example(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> Dict[str, Any]:
        """
        Format example in Mistral JSONL format.

        Mistral format (similar to OpenAI):
        {
          "messages": [
            {"role": "system", "content": "..."} (optional),
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
        Validate Mistral format example.

        Rules (similar to OpenAI):
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

        # Check messages is a list
        if not isinstance(messages, list):
            raise ValidationError("'messages' must be a list")

        # Check messages is not empty
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
