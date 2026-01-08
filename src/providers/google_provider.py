"""
Google Gemini provider implementation.
Supports: Gemini 2.5 Pro, Flash, Flash-lite, 2.0 Flash Exp
"""

from typing import Dict, Any, List
from providers.base_provider import BaseProvider, ProviderSpec, ValidationError


class GoogleProvider(BaseProvider):
    """Google Gemini fine-tuning provider (Vertex AI)"""

    def get_spec(self) -> ProviderSpec:
        """Return Google provider specifications"""
        return ProviderSpec(
            name="Google",
            models=[
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash-exp"
            ],
            token_limits={
                "gemini-2.5-pro": 100000,  # Approximate
                "gemini-2.5-flash": 100000,
                "gemini-2.5-flash-lite": 100000,
                "gemini-2.0-flash-exp": 100000
            },
            max_training_tokens={
                "gemini-2.5-pro": None,  # Not publicly specified
                "gemini-2.5-flash": None,
                "gemini-2.5-flash-lite": None,
                "gemini-2.0-flash-exp": None
            },
            format_type="google",
            min_examples=100
        )

    def format_example(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> Dict[str, Any]:
        """
        Format example in Google Gemini JSONL format.

        Google format (simplified):
        {
          "messages": [
            {"role": "user", "content": "..."},
            {"role": "model", "content": "..."}
          ]
        }

        Note: System message typically prepended to first user message

        Args:
            system: System message
            user: User message
            assistant: Assistant response

        Returns:
            Formatted example dict
        """
        # Prepend system message to user message if provided
        user_content = user
        if system and system.strip():
            user_content = f"{system}\n\n{user}"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "model", "content": assistant}
            ]
        }

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
        first_user_message = True

        for turn in turns:
            role = turn["role"]
            content = turn["content"]

            # Convert "assistant" role to "model" for Google
            if role == "assistant":
                role = "model"

            # Skip system messages (they get prepended)
            if role == "system":
                continue

            # Prepend system message to first user message
            if role == "user" and first_user_message and system:
                content = f"{system}\n\n{content}"
                first_user_message = False

            messages.append({"role": role, "content": content})

        return {"messages": messages}

    def validate_example(self, example: Dict) -> bool:
        """
        Validate Google format example.

        Rules:
        - Must have "messages" key
        - Messages must be a list
        - Each message must have "role" and "content"
        - Roles must alternate between user and model
        - Last message must be from model
        - No system role in messages

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
        valid_roles = {"user", "model"}
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
                    f"Must be 'user' or 'model'"
                )

            # Check alternating roles
            if msg["role"] != expected_role:
                raise ValidationError(
                    f"Message {i}: expected role '{expected_role}', "
                    f"got '{msg['role']}'. Roles must alternate."
                )

            # Toggle expected role
            expected_role = "model" if expected_role == "user" else "user"

            # Check content is not empty
            if not msg["content"] or not msg["content"].strip():
                raise ValidationError(f"Message {i}: empty content")

        # Check last message is from model
        if messages[-1]["role"] != "model":
            raise ValidationError(
                "Last message must be from model, "
                f"got: {messages[-1]['role']}"
            )

        return True
