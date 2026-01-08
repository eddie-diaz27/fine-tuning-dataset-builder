"""
AI-powered conversation detector for identifying dialogue patterns in documents.
Uses LLM to intelligently detect and extract conversations with chunking support.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class Turn:
    """Single turn in a conversation"""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """Detected conversation with metadata"""
    turns: List[Turn]
    source_pattern: str  # AI-detected or regex pattern name
    confidence: float  # 0.0-1.0
    text_range: Tuple[int, int]  # Character positions in source
    context_before: str = ""  # Optional surrounding context
    context_after: str = ""


class ConversationDetector:
    """
    AI-powered conversation detection with optional regex tool.

    Primary approach: Use LLM to detect conversations in chunks
    Fallback/tool: Regex patterns for simple, well-structured formats
    """

    # Optional regex patterns as a tool for the AI or fallback
    SIMPLE_PATTERNS = {
        'qa_colon': r'Q:\s*(.+?)\n+A:\s*(.+?)(?=\n\n|Q:|$)',
        'user_assistant': r'(?:User|Human):\s*(.+?)\n+(?:Assistant|AI|Bot):\s*(.+?)(?=\n\n|(?:User|Human):|$)',
    }

    def __init__(
        self,
        llm_provider: str = None,
        llm_model: str = None,
        use_regex_tool: bool = True,
        chunk_size: int = 3000
    ):
        """
        Initialize AI-powered conversation detector.

        Args:
            llm_provider: LLM provider (google, openai, anthropic)
            llm_model: LLM model name
            use_regex_tool: Whether to provide regex as a tool to AI
            chunk_size: Max characters per chunk for processing
        """
        self.llm_provider = llm_provider or os.getenv('AGENT_PROVIDER', 'google')
        self.llm_model = llm_model or os.getenv('AGENT_MODEL', 'gemini-2.0-flash-exp')
        self.use_regex_tool = use_regex_tool
        self.chunk_size = chunk_size
        self._llm_client = None

    def _init_llm_client(self):
        """Initialize LLM client on-demand."""
        if self._llm_client is not None:
            return

        if self.llm_provider == 'google':
            try:
                import google.generativeai as genai
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment")

                genai.configure(api_key=api_key)
                self._llm_client = genai.GenerativeModel(self.llm_model)
            except ImportError:
                raise ImportError(
                    "google-generativeai is required for Google provider.\n"
                    "Install with: pip install google-generativeai"
                )

        elif self.llm_provider == 'openai':
            try:
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")

                self._llm_client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI provider.\n"
                    "Install with: pip install openai"
                )

        elif self.llm_provider == 'anthropic':
            try:
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")

                self._llm_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "anthropic is required for Anthropic provider.\n"
                    "Install with: pip install anthropic"
                )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def detect_conversations(
        self,
        text: str,
        min_turns: int = 2,
        confidence_threshold: float = 0.7,
        include_context: bool = False,
        max_context_paragraphs: int = 2
    ) -> List[Conversation]:
        """
        Detect conversations using AI with intelligent chunking.

        Args:
            text: Text to search
            min_turns: Minimum conversation turns (1 turn = 1 Q&A pair)
            confidence_threshold: Minimum confidence to include
            include_context: Include surrounding paragraphs
            max_context_paragraphs: Max context paragraphs to include

        Returns:
            List of Conversation objects
        """
        # Initialize LLM client
        self._init_llm_client()

        # Split text into processable chunks
        chunks = self._chunk_text(text)

        all_conversations = []

        for chunk_idx, (chunk_text, chunk_offset) in enumerate(chunks):
            # Use AI to detect conversations in this chunk
            chunk_conversations = self._ai_detect_chunk(
                chunk_text,
                chunk_offset,
                include_context,
                max_context_paragraphs
            )
            all_conversations.extend(chunk_conversations)

        # Filter by confidence and min turns
        filtered = [
            conv for conv in all_conversations
            if conv.confidence >= confidence_threshold
            and len(conv.turns) >= min_turns * 2  # 2 turns per exchange
        ]

        # Deduplicate overlapping matches
        deduplicated = self._deduplicate_matches(filtered)

        return deduplicated

    def _chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into chunks for processing.

        Args:
            text: Full text to chunk

        Returns:
            List of (chunk_text, offset) tuples
        """
        if len(text) <= self.chunk_size:
            return [(text, 0)]

        chunks = []
        # Split on paragraph boundaries
        paragraphs = text.split('\n\n')

        current_chunk = ""
        current_offset = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append((current_chunk, current_offset))
                    current_offset += len(current_chunk) + 2

                # Start new chunk with overlap (last paragraph of previous chunk)
                if len(para) > self.chunk_size:
                    # Paragraph too large, split it
                    chunks.append((para[:self.chunk_size], current_offset))
                    current_offset += self.chunk_size
                    current_chunk = ""
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_offset))

        return chunks

    def _ai_detect_chunk(
        self,
        chunk_text: str,
        chunk_offset: int,
        include_context: bool,
        max_context_paragraphs: int
    ) -> List[Conversation]:
        """
        Use AI to detect conversations in a text chunk.

        Args:
            chunk_text: Chunk to analyze
            chunk_offset: Character offset of chunk in original text
            include_context: Include surrounding context
            max_context_paragraphs: Max context paragraphs

        Returns:
            List of detected conversations
        """
        # Construct AI prompt
        prompt = self._build_detection_prompt(chunk_text)

        try:
            # Call LLM
            if self.llm_provider == 'google':
                response = self._llm_client.generate_content(prompt)
                response_text = response.text

            elif self.llm_provider == 'openai':
                response = self._llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                response_text = response.choices[0].message.content

            elif self.llm_provider == 'anthropic':
                response = self._llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=4000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text

            else:
                response_text = ""

            # Parse AI response to extract conversations
            conversations = self._parse_ai_response(
                response_text,
                chunk_text,
                chunk_offset
            )

            return conversations

        except Exception as e:
            # If AI fails, optionally fall back to regex
            print(f"Warning: AI detection failed: {e}")
            if self.use_regex_tool:
                return self._regex_fallback(chunk_text, chunk_offset)
            return []

    def _build_detection_prompt(self, text: str) -> str:
        """
        Build prompt for AI conversation detection.

        Args:
            text: Text to analyze

        Returns:
            Prompt string
        """
        regex_tool_info = ""
        if self.use_regex_tool:
            regex_tool_info = """
You also have access to simple regex patterns as a tool:
- 'qa_colon': Q: ... A: ... format
- 'user_assistant': User: ... Assistant: ... format

Use these only for very simple, well-structured patterns.
"""

        prompt = f"""You are an expert at identifying conversational dialogues in text documents.

Your task: Analyze the following text and identify ALL conversations or dialogue exchanges.

A conversation can be:
- Q&A format (Question/Answer pairs)
- Dialogue between two or more people
- Chat logs between customer and agent
- Interview transcripts
- Sales call transcripts
- Support conversations
- Any format where two parties are exchanging messages

IMPORTANT:
- Look for dialogue markers like "Rep:", "Prospect:", "Customer:", "Agent:", "User:", "Assistant:", etc.
- Detect conversations even if formatting varies (e.g., **Rep:** vs Rep: vs REP:)
- Each conversation should have at least 2 turns (one exchange)
- Extract the EXACT text verbatim - do not modify or summarize
- Provide a confidence score (0.0-1.0) for each detected conversation

{regex_tool_info}

TEXT TO ANALYZE:
{text}

RESPONSE FORMAT (JSON):
{{
  "conversations": [
    {{
      "turns": [
        {{"role": "user", "content": "exact text from user"}},
        {{"role": "assistant", "content": "exact text from assistant"}}
      ],
      "pattern_detected": "description of pattern (e.g., 'Rep/Prospect dialogue')",
      "confidence": 0.95,
      "start_char": 0,
      "end_char": 100
    }}
  ]
}}

Extract ALL conversations found in the text."""

        return prompt

    def _parse_ai_response(
        self,
        response_text: str,
        chunk_text: str,
        chunk_offset: int
    ) -> List[Conversation]:
        """
        Parse AI response to extract conversation objects.

        Args:
            response_text: AI response
            chunk_text: Original chunk text
            chunk_offset: Offset in original document

        Returns:
            List of Conversation objects
        """
        import json

        conversations = []

        try:
            # Try to parse JSON response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            data = json.loads(response_text)

            for conv_data in data.get("conversations", []):
                turns = []

                for turn_data in conv_data.get("turns", []):
                    turn = Turn(
                        role=turn_data.get("role", "user"),
                        content=turn_data.get("content", "")
                    )
                    turns.append(turn)

                if len(turns) >= 2:  # At least one exchange
                    conversation = Conversation(
                        turns=turns,
                        source_pattern=conv_data.get("pattern_detected", "ai_detected"),
                        confidence=float(conv_data.get("confidence", 0.8)),
                        text_range=(
                            chunk_offset + conv_data.get("start_char", 0),
                            chunk_offset + conv_data.get("end_char", len(chunk_text))
                        )
                    )
                    conversations.append(conversation)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse AI response as JSON: {e}")
            # Try to extract conversations using fallback parsing
            conversations = self._fallback_parse(response_text, chunk_text, chunk_offset)

        return conversations

    def _fallback_parse(
        self,
        response_text: str,
        chunk_text: str,
        chunk_offset: int
    ) -> List[Conversation]:
        """
        Fallback parsing if JSON fails.

        Args:
            response_text: AI response text
            chunk_text: Original chunk
            chunk_offset: Offset in document

        Returns:
            List of conversations (best effort)
        """
        # If JSON parsing fails, try to extract turn patterns from response
        conversations = []

        # Look for user/assistant patterns in the response
        turn_pattern = r'"role":\s*"(user|assistant)",\s*"content":\s*"(.+?)"'
        matches = re.findall(turn_pattern, response_text, re.DOTALL)

        if matches:
            turns = [Turn(role=role, content=content) for role, content in matches]

            if len(turns) >= 2:
                conversation = Conversation(
                    turns=turns,
                    source_pattern="ai_detected_fallback",
                    confidence=0.6,  # Lower confidence for fallback
                    text_range=(chunk_offset, chunk_offset + len(chunk_text))
                )
                conversations.append(conversation)

        return conversations

    def _regex_fallback(
        self,
        text: str,
        offset: int
    ) -> List[Conversation]:
        """
        Regex-based fallback detection.

        Args:
            text: Text to search
            offset: Character offset

        Returns:
            List of detected conversations
        """
        conversations = []

        for pattern_name, pattern in self.SIMPLE_PATTERNS.items():
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)

            for match in matches:
                try:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()

                    if question and answer:
                        turns = [
                            Turn(role="user", content=question),
                            Turn(role="assistant", content=answer)
                        ]

                        conversation = Conversation(
                            turns=turns,
                            source_pattern=f"regex_{pattern_name}",
                            confidence=0.9,  # High confidence for clear regex match
                            text_range=(offset + match.start(), offset + match.end())
                        )
                        conversations.append(conversation)

                except (IndexError, AttributeError):
                    continue

        return conversations

    def _deduplicate_matches(
        self,
        conversations: List[Conversation]
    ) -> List[Conversation]:
        """
        Remove overlapping conversations.

        Args:
            conversations: List of conversations

        Returns:
            Deduplicated list
        """
        if not conversations:
            return []

        # Sort by start position
        sorted_convs = sorted(conversations, key=lambda c: c.text_range[0])

        deduplicated = []
        last_end = -1

        for conv in sorted_convs:
            start, end = conv.text_range

            # Check for overlap
            if start >= last_end:
                # No overlap, add it
                deduplicated.append(conv)
                last_end = end
            else:
                # Overlap detected - keep the one with higher confidence
                if deduplicated and conv.confidence > deduplicated[-1].confidence:
                    deduplicated[-1] = conv
                    last_end = end

        return deduplicated

    def is_conversational(self, text: str) -> bool:
        """
        Quick check if text contains conversational patterns.

        Args:
            text: Text to check

        Returns:
            True if conversational patterns detected
        """
        conversations = self.detect_conversations(
            text,
            min_turns=1,
            confidence_threshold=0.6
        )
        return len(conversations) > 0

    def get_pattern_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about detected patterns.

        Args:
            text: Text to analyze

        Returns:
            Dict with pattern counts
        """
        all_convs = self.detect_conversations(
            text,
            min_turns=1,
            confidence_threshold=0.5
        )

        stats = {
            "total_conversations": len(all_convs),
            "patterns": {},
            "avg_confidence": 0.0
        }

        if all_convs:
            for conv in all_convs:
                pattern = conv.source_pattern
                if pattern not in stats["patterns"]:
                    stats["patterns"][pattern] = 0
                stats["patterns"][pattern] += 1

            stats["avg_confidence"] = sum(c.confidence for c in all_convs) / len(all_convs)

        return stats
