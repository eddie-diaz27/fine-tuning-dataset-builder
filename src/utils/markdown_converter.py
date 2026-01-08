"""
Markdown converter for dataset export/import.
Allows human-readable editing of datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from utils.validator import Sample


class MarkdownConverter:
    """Convert datasets between JSONL and Markdown formats"""

    def __init__(self):
        """Initialize markdown converter"""
        self.separator = "\n\n---\n\n"
        self.sample_header = "## Sample {index}\n\n"

    def export_to_markdown(
        self,
        samples: List[Sample],
        output_path: str,
        include_metadata: bool = True
    ):
        """
        Export samples to markdown format.

        Args:
            samples: List of Sample objects
            output_path: Path to output .md file
            include_metadata: Whether to include metadata section

        Format:
            ## Sample 1

            **ID:** abc-123
            **Tokens:** 150
            **Keywords:** sales, negotiation
            **Source:** input.txt
            **Mode:** create

            ### System
            You are a helpful sales assistant.

            ### User
            How do I handle objections?

            ### Assistant
            Here are key strategies...

            ---

            ## Sample 2
            ...
        """
        md_content = []
        md_content.append("# Dataset Samples\n")
        md_content.append(f"Total Samples: {len(samples)}\n")
        md_content.append("=" * 60 + "\n")

        for idx, sample in enumerate(samples, 1):
            # Sample header
            md_content.append(f"\n{self.sample_header.format(index=idx)}")

            # Metadata section
            if include_metadata:
                md_content.append("**Metadata:**\n")
                md_content.append(f"- **ID:** `{sample.id}`\n")
                md_content.append(f"- **Tokens:** {sample.token_length}\n")
                md_content.append(f"- **Keywords:** {', '.join(sample.topic_keywords)}\n")
                md_content.append(f"- **Source:** {sample.source_file}\n")
                md_content.append(f"- **Mode:** {sample.mode}\n")
                if hasattr(sample, 'split'):
                    md_content.append(f"- **Split:** {sample.split}\n")
                md_content.append("\n")

            # Content sections
            if sample.content.get('system'):
                md_content.append("### System\n")
                md_content.append(f"{sample.content['system']}\n\n")

            md_content.append("### User\n")
            md_content.append(f"{sample.content['user']}\n\n")

            md_content.append("### Assistant\n")
            md_content.append(f"{sample.content['assistant']}\n")

            # Separator between samples
            if idx < len(samples):
                md_content.append(self.separator)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(md_content))

        print(f"✓ Exported {len(samples)} samples to {output_path}")

    def import_from_markdown(
        self,
        md_path: str,
        preserve_metadata: bool = True
    ) -> List[Sample]:
        """
        Import samples from markdown format.

        Args:
            md_path: Path to .md file
            preserve_metadata: Whether to preserve existing IDs/metadata

        Returns:
            List of Sample objects

        Notes:
            - Parses markdown sections (System, User, Assistant)
            - Preserves metadata if present
            - Generates new IDs if metadata missing
        """
        md_path = Path(md_path)

        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by sample separator
        sample_blocks = content.split(self.separator)

        samples = []
        for block in sample_blocks:
            if not block.strip() or block.strip().startswith('#'):
                continue

            sample = self._parse_sample_block(block, preserve_metadata)
            if sample:
                samples.append(sample)

        print(f"✓ Imported {len(samples)} samples from {md_path}")
        return samples

    def _parse_sample_block(
        self,
        block: str,
        preserve_metadata: bool
    ) -> Sample:
        """
        Parse a single sample block from markdown.

        Args:
            block: Markdown text block
            preserve_metadata: Whether to use existing metadata

        Returns:
            Sample object or None
        """
        import uuid
        import re

        lines = block.split('\n')

        # Extract metadata
        metadata = {}
        if preserve_metadata:
            metadata = self._extract_metadata(block)

        # Extract content sections
        content = {
            'system': '',
            'user': '',
            'assistant': ''
        }

        current_section = None
        section_content = []

        for line in lines:
            # Check for section headers
            if line.strip() == '### System':
                if current_section and section_content:
                    content[current_section] = '\n'.join(section_content).strip()
                current_section = 'system'
                section_content = []
            elif line.strip() == '### User':
                if current_section and section_content:
                    content[current_section] = '\n'.join(section_content).strip()
                current_section = 'user'
                section_content = []
            elif line.strip() == '### Assistant':
                if current_section and section_content:
                    content[current_section] = '\n'.join(section_content).strip()
                current_section = 'assistant'
                section_content = []
            elif current_section and not line.startswith('##') and not line.startswith('**'):
                # Content line
                section_content.append(line)

        # Capture last section
        if current_section and section_content:
            content[current_section] = '\n'.join(section_content).strip()

        # Validate content
        if not content['user'] or not content['assistant']:
            return None

        # Create Sample object
        sample_id = metadata.get('id', str(uuid.uuid4()))

        # Count tokens (approximate if not in metadata)
        token_length = metadata.get('token_length', len(json.dumps(content)) // 4)

        sample = Sample(
            id=sample_id,
            content=content,
            token_length=token_length,
            topic_keywords=metadata.get('keywords', []),
            source_file=metadata.get('source_file', 'imported_from_markdown'),
            mode=metadata.get('mode', 'edited')
        )

        # Add split if present
        if 'split' in metadata:
            sample.split = metadata['split']

        return sample

    def _extract_metadata(self, block: str) -> Dict[str, Any]:
        """
        Extract metadata from markdown block.

        Args:
            block: Markdown text block

        Returns:
            Dict with metadata fields
        """
        import re

        metadata = {}

        # Extract ID
        id_match = re.search(r'\*\*ID:\*\*\s*`([^`]+)`', block)
        if id_match:
            metadata['id'] = id_match.group(1)

        # Extract token length
        tokens_match = re.search(r'\*\*Tokens:\*\*\s*(\d+)', block)
        if tokens_match:
            metadata['token_length'] = int(tokens_match.group(1))

        # Extract keywords
        keywords_match = re.search(r'\*\*Keywords:\*\*\s*([^\n]+)', block)
        if keywords_match:
            keywords_str = keywords_match.group(1).strip()
            metadata['keywords'] = [k.strip() for k in keywords_str.split(',')]

        # Extract source file
        source_match = re.search(r'\*\*Source:\*\*\s*([^\n]+)', block)
        if source_match:
            metadata['source_file'] = source_match.group(1).strip()

        # Extract mode
        mode_match = re.search(r'\*\*Mode:\*\*\s*([^\n]+)', block)
        if mode_match:
            metadata['mode'] = mode_match.group(1).strip()

        # Extract split
        split_match = re.search(r'\*\*Split:\*\*\s*([^\n]+)', block)
        if split_match:
            metadata['split'] = split_match.group(1).strip()

        return metadata

    def sync_markdown_to_jsonl(
        self,
        md_path: str,
        jsonl_path: str,
        provider,
        backup: bool = True
    ):
        """
        Sync changes from markdown back to JSONL.

        Args:
            md_path: Path to edited .md file
            jsonl_path: Path to .jsonl file to update
            provider: Provider instance for formatting
            backup: Whether to backup original JSONL

        Process:
            1. Backup original JSONL
            2. Import samples from markdown
            3. Format samples for provider
            4. Write to JSONL
        """
        from utils.formatter import Formatter

        jsonl_path = Path(jsonl_path)

        # Backup original
        if backup and jsonl_path.exists():
            backup_path = jsonl_path.with_suffix('.jsonl.backup')
            import shutil
            shutil.copy(jsonl_path, backup_path)
            print(f"  ✓ Backed up original to {backup_path}")

        # Import from markdown
        samples = self.import_from_markdown(md_path, preserve_metadata=True)

        # Format and write JSONL
        formatter = Formatter(provider)
        formatter.write_jsonl(samples, str(jsonl_path))

        print(f"✓ Synced {len(samples)} samples from markdown to JSONL")

    def create_diff_report(
        self,
        original_jsonl: str,
        edited_md: str,
        output_path: str = None
    ):
        """
        Create a diff report showing changes between original and edited.

        Args:
            original_jsonl: Path to original JSONL
            edited_md: Path to edited markdown
            output_path: Path to save diff report (optional)

        Returns:
            Diff report string
        """
        from utils.formatter import Formatter
        import difflib

        # Load original samples
        formatter = Formatter(None)
        original_samples = []

        with open(original_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Convert to Sample (simplified)
                    # This is a placeholder - full implementation would reconstruct Sample objects
                    original_samples.append(data)

        # Load edited samples
        edited_samples = self.import_from_markdown(edited_md, preserve_metadata=True)

        # Create diff report
        report = []
        report.append("# Dataset Diff Report\n")
        report.append(f"Original: {original_jsonl}\n")
        report.append(f"Edited: {edited_md}\n")
        report.append("=" * 60 + "\n\n")

        report.append(f"Original samples: {len(original_samples)}\n")
        report.append(f"Edited samples: {len(edited_samples)}\n\n")

        # Sample-by-sample comparison
        for idx in range(min(len(original_samples), len(edited_samples))):
            orig = json.dumps(original_samples[idx], indent=2)
            edited = json.dumps(edited_samples[idx].content, indent=2)

            if orig != edited:
                report.append(f"## Sample {idx + 1} - MODIFIED\n\n")
                diff = difflib.unified_diff(
                    orig.splitlines(),
                    edited.splitlines(),
                    lineterm='',
                    fromfile='original',
                    tofile='edited'
                )
                report.append('\n'.join(diff))
                report.append("\n\n")

        report_str = ''.join(report)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            print(f"✓ Diff report saved to {output_path}")

        return report_str
