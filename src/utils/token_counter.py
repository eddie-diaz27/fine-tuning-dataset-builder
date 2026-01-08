"""
Token counter for provider-specific tokenization.
Supports: OpenAI (tiktoken), Anthropic, Google, Mistral, Llama
"""

from typing import Dict, List, Optional
import warnings


class TokenCounterError(Exception):
    """Raised when token counting fails"""
    pass


class TokenCounter:
    """Count tokens per provider specification"""

    def __init__(self):
        """Initialize token counter with cached tokenizers"""
        self._tokenizer_cache: Dict[str, any] = {}
        self._encoding_cache: Dict[str, any] = {}

    def count_tokens(self, text: str, provider: str, model: str) -> int:
        """
        Count tokens using provider-specific method.

        Args:
            text: Text to count tokens for
            provider: Provider name (openai, anthropic, google, mistral, llama)
            model: Model name

        Returns:
            Token count (int)

        Raises:
            TokenCounterError: If token counting fails
        """
        if not text:
            return 0

        provider = provider.lower()

        if provider == "openai":
            return self._count_openai(text, model)
        elif provider == "anthropic":
            return self._count_anthropic(text)
        elif provider == "google":
            return self._count_google(text, model)
        elif provider == "mistral":
            return self._count_mistral(text)
        elif provider == "llama":
            return self._count_llama(text)
        else:
            raise TokenCounterError(f"Unknown provider: {provider}")

    def _count_openai(self, text: str, model: str) -> int:
        """
        Count tokens using tiktoken for OpenAI models.

        Args:
            text: Text to count
            model: OpenAI model name

        Returns:
            Token count
        """
        try:
            import tiktoken
        except ImportError:
            raise TokenCounterError(
                "tiktoken is required for OpenAI token counting.\n"
                "Install with: pip install tiktoken"
            )

        try:
            # Get or create encoding
            if model not in self._encoding_cache:
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback to cl100k_base for newer models
                    encoding = tiktoken.get_encoding("cl100k_base")
                self._encoding_cache[model] = encoding
            else:
                encoding = self._encoding_cache[model]

            # Count tokens
            tokens = encoding.encode(text)
            return len(tokens)

        except Exception as e:
            raise TokenCounterError(f"Failed to count OpenAI tokens: {e}")

    def _count_anthropic(self, text: str) -> int:
        """
        Count tokens for Anthropic Claude models.
        Uses approximation: ~4 characters per token.

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        # Anthropic approximation: ~4 chars per token
        # This is a conservative estimate
        return len(text) // 4

    def _count_google(self, text: str, model: str) -> int:
        """
        Count tokens for Google Gemini models.
        Falls back to approximation if API unavailable.

        Args:
            text: Text to count
            model: Gemini model name

        Returns:
            Token count
        """
        try:
            import google.generativeai as genai

            # Try to use API token counting if available
            # Note: This requires an API key to be configured
            try:
                # Create model instance if not cached
                cache_key = f"google_{model}"
                if cache_key not in self._tokenizer_cache:
                    self._tokenizer_cache[cache_key] = genai.GenerativeModel(model)

                model_instance = self._tokenizer_cache[cache_key]
                result = model_instance.count_tokens(text)
                return result.total_tokens

            except Exception:
                # Fallback to approximation if API call fails
                return self._count_approximation(text)

        except ImportError:
            # google-generativeai not installed, use approximation
            return self._count_approximation(text)

    def _count_mistral(self, text: str) -> int:
        """
        Count tokens for Mistral models.
        Uses tiktoken as proxy with gpt-3.5-turbo encoding.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        try:
            import tiktoken

            # Use gpt-3.5-turbo encoding as proxy
            if "mistral_encoding" not in self._encoding_cache:
                encoding = tiktoken.get_encoding("cl100k_base")
                self._encoding_cache["mistral_encoding"] = encoding
            else:
                encoding = self._encoding_cache["mistral_encoding"]

            tokens = encoding.encode(text)
            return len(tokens)

        except ImportError:
            # Fallback to approximation
            return self._count_approximation(text)

    def _count_llama(self, text: str) -> int:
        """
        Count tokens for Meta Llama models.
        Uses tiktoken as proxy with gpt-3.5-turbo encoding.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        try:
            import tiktoken

            # Use gpt-3.5-turbo encoding as proxy
            if "llama_encoding" not in self._encoding_cache:
                encoding = tiktoken.get_encoding("cl100k_base")
                self._encoding_cache["llama_encoding"] = encoding
            else:
                encoding = self._encoding_cache["llama_encoding"]

            tokens = encoding.encode(text)
            return len(tokens)

        except ImportError:
            # Fallback to approximation
            return self._count_approximation(text)

    def _count_approximation(self, text: str) -> int:
        """
        Fallback approximation for token counting.
        Uses ~4 characters per token.

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        return len(text) // 4

    def count_message_tokens(
        self,
        messages: List[Dict[str, str]],
        provider: str,
        model: str
    ) -> int:
        """
        Count tokens in formatted message structure.
        Accounts for role tokens and formatting overhead.

        Args:
            messages: List of message dicts with 'role' and 'content'
            provider: Provider name
            model: Model name

        Returns:
            Total token count including formatting overhead

        Example:
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        """
        # Count content tokens
        total_tokens = 0

        for message in messages:
            content = message.get('content', '')
            role = message.get('role', '')

            # Count content
            content_tokens = self.count_tokens(content, provider, model)
            total_tokens += content_tokens

            # Add overhead for role and formatting
            # Typical overhead: 4 tokens per message for formatting
            total_tokens += 4

        # Add overhead for message delimiters
        # Typical overhead: 3 tokens for overall structure
        total_tokens += 3

        return total_tokens

    def count_conversation_tokens(
        self,
        system: str,
        user: str,
        assistant: str,
        provider: str,
        model: str
    ) -> int:
        """
        Count tokens for a complete conversation.

        Args:
            system: System message
            user: User message
            assistant: Assistant message
            provider: Provider name
            model: Model name

        Returns:
            Total token count
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]

        return self.count_message_tokens(messages, provider, model)

    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """
        Quick estimation from character count.
        Uses ~4 characters per token average.

        Args:
            char_count: Number of characters

        Returns:
            Estimated token count
        """
        return char_count // 4

    def get_encoding_name(self, provider: str, model: str) -> str:
        """
        Get the encoding name used for a provider/model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Encoding name (e.g., 'cl100k_base')
        """
        if provider == "openai":
            try:
                import tiktoken
                try:
                    encoding = tiktoken.encoding_for_model(model)
                    return encoding.name
                except KeyError:
                    return "cl100k_base"
            except ImportError:
                return "unknown"

        elif provider in ["mistral", "llama"]:
            return "cl100k_base (proxy)"

        elif provider == "anthropic":
            return "approximation (~4 chars/token)"

        elif provider == "google":
            return "gemini_tokenizer"

        return "unknown"
