"""
Anthropic Claude provider implementation.
Supports: Claude 3 Haiku (via Amazon Bedrock only)
"""

from typing import Dict, Any, List
from providers.base_provider import BaseProvider, ProviderSpec, ValidationError


class AnthropicProvider(BaseProvider):
    """Anthropic Claude fine-tuning provider (Bedrock only)"""

    def get_spec(self) -> ProviderSpec:
        """Return Anthropic provider specifications"""
        return ProviderSpec(
            name="Anthropic",
            models=["claude-3-haiku"],
            token_limits={"claude-3-haiku": 32000},
            max_training_tokens={"claude-3-haiku": None},  # Not specified
            format_type="anthropic",
            min_examples=50
        )

    def format_example(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> Dict[str, Any]:
        """
        Format example in Anthropic JSONL format.

        Anthropic format:
        {
          "system": "...",
          "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

        Note: System message is separate, not in messages array

        Args:
            system: System message
            user: User message
            assistant: Assistant response

        Returns:
            Formatted example dict
        """
        example = {
            "system": system if system else "",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ]
        }

        return example

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
        # Filter out system role from turns (it goes in separate field)
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in turns
            if turn["role"] != "system"
        ]

        return {
            "system": system if system else "",
            "messages": messages
        }

    def validate_example(self, example: Dict) -> bool:
        """
        Validate Anthropic format example.

        Rules:
        - Must have "system" and "messages" keys
        - System must be a string
        - Messages must be a list
        - Each message must have "role" and "content"
        - Roles must alternate between user and assistant
        - Last message must be from assistant
        - No system role in messages array

        Args:
            example: Example to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If invalid
        """
        # Check for required keys
        if "system" not in example:
            raise ValidationError("Missing 'system' key")
        if "messages" not in example:
            raise ValidationError("Missing 'messages' key")

        # Check system is a string
        if not isinstance(example["system"], str):
            raise ValidationError("'system' must be a string")

        messages = example["messages"]

        # Check messages is a list
        if not isinstance(messages, list):
            raise ValidationError("'messages' must be a list")

        # Check messages is not empty
        if not messages:
            raise ValidationError("'messages' cannot be empty")

        # Validate each message
        valid_roles = {"user", "assistant"}
        expected_role = "user"  # First message should be from user

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
                    f"Must be 'user' or 'assistant' (system goes in separate field)"
                )

            # Check alternating roles
            if msg["role"] != expected_role:
                raise ValidationError(
                    f"Message {i}: expected role '{expected_role}', "
                    f"got '{msg['role']}'. Roles must alternate."
                )

            # Toggle expected role
            expected_role = "assistant" if expected_role == "user" else "user"

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
