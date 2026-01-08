"""
Meta Llama provider implementation.
Supports: Llama 3.1 (8B, 70B, 405B)
"""

from typing import Dict, Any, List
from providers.base_provider import BaseProvider, ProviderSpec, ValidationError


class LlamaProvider(BaseProvider):
    """Meta Llama fine-tuning provider (open source)"""

    def get_spec(self) -> ProviderSpec:
        """Return Llama provider specifications"""
        return ProviderSpec(
            name="Llama",
            models=[
                "llama-3.1-8b",
                "llama-3.1-70b",
                "llama-3.1-405b"
            ],
            token_limits={
                "llama-3.1-8b": 128000,  # Flexible for open source
                "llama-3.1-70b": 128000,
                "llama-3.1-405b": 128000
            },
            max_training_tokens={
                "llama-3.1-8b": None,  # Flexible
                "llama-3.1-70b": None,
                "llama-3.1-405b": None
            },
            format_type="llama",
            min_examples=50
        )

    def format_example(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> Dict[str, Any]:
        """
        Format example in Llama conversational JSONL format.

        Llama format (conversational):
        {
          "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

        Note: System message typically included in user message or separate

        Args:
            system: System message
            user: User message
            assistant: Assistant response

        Returns:
            Formatted example dict
        """
        messages = []

        # Prepend system to user message if provided
        user_content = user
        if system and system.strip():
            user_content = f"[INST] {system}\n\n{user} [/INST]"
        else:
            user_content = f"[INST] {user} [/INST]"

        messages.append({"role": "user", "content": user_content})
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
        first_user_turn = True

        for turn in turns:
            role = turn["role"]
            content = turn["content"]

            # Skip system role
            if role == "system":
                continue

            # Add [INST] tags to first user message with system
            if role == "user" and first_user_turn:
                if system and system.strip():
                    content = f"[INST] {system}\n\n{content} [/INST]"
                else:
                    content = f"[INST] {content} [/INST]"
                first_user_turn = False
            elif role == "user":
                content = f"[INST] {content} [/INST]"

            messages.append({"role": role, "content": content})

        return {"messages": messages}

    def validate_example(self, example: Dict) -> bool:
        """
        Validate Llama format example.

        Rules:
        - Must have "messages" key
        - Messages must be a list
        - Each message must have "role" and "content"
        - Roles must alternate between user and assistant
        - Last message must be from assistant

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
                    f"Must be 'user' or 'assistant'"
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
