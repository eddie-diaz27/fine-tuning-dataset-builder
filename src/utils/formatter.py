"""
Formatter for converting samples to provider-specific JSONL.
"""

import json
from typing import List
from providers.base_provider import BaseProvider


class Formatter:
    """Format samples to provider-specific JSONL"""

    def __init__(self, provider: BaseProvider):
        """
        Initialize formatter.

        Args:
            provider: Provider instance
        """
        self.provider = provider

    def format_samples(self, samples: List) -> List[str]:
        """
        Format samples to JSONL strings.

        Args:
            samples: List of Sample objects

        Returns:
            List of JSON strings (one per line)
        """
        jsonl_lines = []

        for sample in samples:
            # Format using provider
            formatted = self.provider.format_example(
                system=sample.content.get('system', ''),
                user=sample.content.get('user', ''),
                assistant=sample.content.get('assistant', '')
            )

            # Convert to JSON string
            json_str = json.dumps(formatted, ensure_ascii=False)
            jsonl_lines.append(json_str)

        return jsonl_lines

    def write_jsonl(
        self,
        samples: List,
        output_path: str,
        pretty: bool = False
    ):
        """
        Write samples to JSONL file.

        Args:
            samples: List of Sample objects
            output_path: Path to output file
            pretty: Whether to pretty-print JSON (increases file size)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Format using provider
                formatted = self.provider.format_example(
                    system=sample.content.get('system', ''),
                    user=sample.content.get('user', ''),
                    assistant=sample.content.get('assistant', '')
                )

                # Convert to JSON
                if pretty:
                    json_str = json.dumps(formatted, ensure_ascii=False, indent=2)
                else:
                    json_str = json.dumps(formatted, ensure_ascii=False)

                # Write to file
                f.write(json_str + '\n')

    def format_single(self, sample) -> str:
        """
        Format a single sample to JSON string.

        Args:
            sample: Sample object

        Returns:
            JSON string
        """
        formatted = self.provider.format_example(
            system=sample.content.get('system', ''),
            user=sample.content.get('user', ''),
            assistant=sample.content.get('assistant', '')
        )

        return json.dumps(formatted, ensure_ascii=False)

    def validate_jsonl(self, file_path: str) -> bool:
        """
        Validate that a JSONL file is properly formatted.

        Args:
            file_path: Path to JSONL file

        Returns:
            True if valid

        Raises:
            ValueError: If file is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Try to parse JSON
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Line {line_num}: Invalid JSON - {e}"
                        )

                    # Validate format
                    try:
                        self.provider.validate_example(data)
                    except Exception as e:
                        raise ValueError(
                            f"Line {line_num}: Format error - {e}"
                        )

            return True

        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
