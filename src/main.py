"""
Main orchestrator for Fine-Tuning Dataset Builder.
Coordinates workflow: config → processing → generation → validation → export
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import uuid
import pickle
from datetime import datetime
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from utils.config_loader import load_config
from utils.file_processor import FileProcessor
from utils.token_counter import TokenCounter
from utils.validator import Validator
from utils.formatter import Formatter
from utils.splitter import Splitter
from providers import get_provider
from agents import get_agent
from persistence.sqlite_client import SQLiteClient


class DatasetBuilder:
    """Main orchestrator for dataset generation"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize dataset builder.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.runtime_config = None
        self.checkpoint_dir = Path("memory/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self, runtime_config: Dict[str, Any] = None):
        """
        Run complete dataset generation workflow.

        Args:
            runtime_config: Runtime configuration (overrides config.yaml)
        """
        print("\n" + "="*60)
        print(" "*15 + "Dataset Generation Workflow")
        print("="*60)

        try:
            # Step 1: Load configuration
            print("\n[1/8] Loading configuration...")
            self._load_configuration(runtime_config)

            # Step 2: Load input files
            print("\n[2/8] Processing input files...")
            processed_files = self._process_input_files()

            # Step 3: Initialize components
            print("\n[3/8] Initializing components...")
            provider, agent, token_counter, validator = self._initialize_components()

            # Step 4: Generate samples
            print("\n[4/8] Generating samples...")
            samples = self._generate_samples(agent, processed_files, provider, token_counter)

            # Save checkpoint after generation
            self._save_checkpoint('generation', {'samples': samples})

            # Step 5: Validate dataset
            print("\n[5/8] Validating dataset...")
            validated_samples = self._validate_dataset(validator, samples)

            # Save checkpoint after validation
            self._save_checkpoint('validation', {'samples': validated_samples})

            # Step 6: Split dataset
            print("\n[6/8] Splitting dataset...")
            train_samples, val_samples = self._split_dataset(validated_samples)

            # Step 7: Save to database
            print("\n[7/8] Saving to database...")
            dataset_id = self._save_to_database(train_samples, val_samples)

            # Step 8: Export files
            print("\n[8/8] Exporting files...")
            self._export_files(dataset_id, train_samples, val_samples, provider)

            # Success summary
            self._print_success_summary(dataset_id, train_samples, val_samples)

        except KeyboardInterrupt:
            print("\n\n⚠️  Generation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _load_configuration(self, runtime_config: Dict[str, Any] = None):
        """Load configuration from file or runtime"""
        if runtime_config:
            self.runtime_config = runtime_config
            print("✓ Using runtime configuration")
        else:
            self.config = load_config(self.config_path)
            print(f"✓ Loaded configuration from {self.config_path}")

    def _process_input_files(self) -> List:
        """Process all input files"""
        input_dir = Path("input_files")

        if not input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {input_dir}\n"
                f"Please create the directory and add source files."
            )

        # Get all files
        files = list(input_dir.glob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith('.')]

        if not files:
            raise FileNotFoundError(
                f"No files found in {input_dir}\n"
                f"Please add source files (PDF, DOCX, TXT, etc.)"
            )

        print(f"Found {len(files)} input files")

        # Process files
        processor = FileProcessor()
        processed = []

        for file_path in files:
            try:
                processed_file = processor.process_file(str(file_path))
                processed.append(processed_file)
                print(f"  ✓ Processed {file_path.name} ({len(processed_file.text)} chars)")
            except Exception as e:
                print(f"  ⚠️  Skipped {file_path.name}: {e}")

        if not processed:
            raise ValueError("No files were successfully processed")

        print(f"✓ Successfully processed {len(processed)} files")
        return processed

    def _initialize_components(self):
        """Initialize provider, agent, and utilities"""
        # Get configuration values
        if self.runtime_config:
            provider_name = self.runtime_config['dataset']['provider']
            model = self.runtime_config['dataset']['model']
            mode = self.runtime_config['agent']['mode']
            agent_config = self.runtime_config['agent']
        else:
            provider_name = self.config.dataset.provider
            model = self.config.dataset.model
            mode = self.config.agent.mode
            agent_config = {
                'variety': self.config.agent.variety,
                'length': self.config.agent.length,
                'model': model
            }

        # Get API key
        api_key_env = f"{provider_name.upper()}_API_KEY"
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(
                f"API key not found for {provider_name}.\n"
                f"Please set {api_key_env} in your .env file."
            )

        # Initialize provider
        provider = get_provider(provider_name)
        print(f"✓ Initialized {provider_name} provider")

        # Initialize agent
        agent = get_agent(mode, provider_name, model, api_key, agent_config)
        print(f"✓ Initialized {mode.upper()} mode agent")

        # Initialize utilities
        token_counter = TokenCounter()
        validator = Validator(provider, model, token_counter)
        print(f"✓ Initialized utilities")

        return provider, agent, token_counter, validator

    def _generate_samples(self, agent, processed_files, provider, token_counter):
        """Generate samples using agent"""
        if self.runtime_config:
            target_count = self.runtime_config['dataset']['target_samples']
        else:
            target_count = self.config.dataset.target_samples

        samples = agent.generate_samples(
            processed_files,
            target_count,
            provider,
            token_counter
        )

        print(f"✓ Generated {len(samples)} samples")
        return samples

    def _validate_dataset(self, validator, samples):
        """Validate dataset"""
        validation_result = validator.validate_dataset(samples)

        if not validation_result.valid:
            print("\n⚠️  Validation warnings:")
            for error in validation_result.errors:
                print(f"  - {error}")

            # Check if errors are critical
            critical = any('exceeds' in e.lower() or 'invalid' in e.lower()
                          for e in validation_result.errors)

            if critical:
                raise ValueError(
                    "Critical validation errors found. Cannot proceed.\n"
                    "Please review the errors above and adjust configuration."
                )

        print(f"✓ Validation passed ({len(samples)} samples)")
        return samples

    def _split_dataset(self, samples):
        """Split dataset into train/val"""
        if self.runtime_config:
            split_config = self.runtime_config['split']
        else:
            split_config = {
                'enabled': self.config.split.enabled,
                'ratio': self.config.split.ratio,
                'strategy': self.config.split.strategy
            }

        if not split_config['enabled']:
            # No split, all samples are "train"
            for sample in samples:
                sample.split = 'train'
            print(f"✓ No split (all {len(samples)} samples in train set)")
            return samples, []

        # Perform split
        splitter = Splitter()

        # Map strategy names
        strategy_map = {
            'random': 'random',
            'weighted_source': 'weighted',
            'weighted_keywords': 'weighted'
        }

        strategy = strategy_map.get(split_config['strategy'], 'random')
        weight_by = 'source' if split_config['strategy'] == 'weighted_source' else 'keywords'

        train, val = splitter.split_dataset(
            samples,
            ratio=split_config['ratio'],
            strategy=strategy,
            weight_by=weight_by if strategy == 'weighted' else None
        )

        # Set split attribute
        for sample in train:
            sample.split = 'train'
        for sample in val:
            sample.split = 'validation'

        print(f"✓ Split complete: {len(train)} train, {len(val)} validation")

        # Print split summary
        splitter.print_split_summary(
            train,
            val,
            weight_by=weight_by if strategy == 'weighted' else None
        )

        return train, val

    def _save_to_database(self, train_samples, val_samples):
        """Save dataset to SQLite database"""
        # Ensure memory directory exists
        Path("memory").mkdir(exist_ok=True)

        db_path = "memory/dataset_builder.db"
        client = SQLiteClient(db_path)

        # Generate dataset ID
        dataset_id = str(uuid.uuid4())

        # Get dataset name
        if self.runtime_config:
            dataset_name = self.runtime_config['dataset']['name']
            provider_name = self.runtime_config['dataset']['provider']
            model = self.runtime_config['dataset']['model']
            mode = self.runtime_config['agent']['mode']
            config_dict = self.runtime_config
        else:
            dataset_name = self.config.dataset.name or f"dataset_{dataset_id[:8]}"
            provider_name = self.config.dataset.provider
            model = self.config.dataset.model
            mode = self.config.agent.mode
            config_dict = {
                'variety': self.config.agent.variety,
                'length': self.config.agent.length,
                'split_ratio': self.config.split.ratio
            }

        # Combine samples
        all_samples = train_samples + val_samples

        # Save to database
        client.save_dataset(
            dataset_id=dataset_id,
            name=dataset_name,
            provider=provider_name,
            model=model,
            mode=mode,
            samples=all_samples,
            config=config_dict
        )

        print(f"✓ Saved to database (ID: {dataset_id})")

        # Backup database
        try:
            client.backup_database()
            print(f"✓ Database backed up")
        except Exception as e:
            print(f"⚠️  Backup failed: {e}")

        return dataset_id

    def _export_files(self, dataset_id, train_samples, val_samples, provider):
        """Export JSONL and CSV files"""
        # Get dataset name
        if self.runtime_config:
            dataset_name = self.runtime_config['dataset']['name']
        else:
            dataset_name = self.config.dataset.name or f"dataset_{dataset_id[:8]}"

        # Create output directory
        output_dir = Path("output_datasets") / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize formatter
        formatter = Formatter(provider)

        # Export train JSONL
        if train_samples:
            train_path = output_dir / "train.jsonl"
            formatter.write_jsonl(train_samples, str(train_path))
            print(f"  ✓ Exported train.jsonl ({len(train_samples)} samples)")

        # Export validation JSONL
        if val_samples:
            val_path = output_dir / "validation.jsonl"
            formatter.write_jsonl(val_samples, str(val_path))
            print(f"  ✓ Exported validation.jsonl ({len(val_samples)} samples)")

        # Export CSV metadata
        db_path = "memory/dataset_builder.db"
        client = SQLiteClient(db_path)
        csv_path = output_dir / "metadata.csv"
        client.export_csv(dataset_id, str(csv_path))
        print(f"  ✓ Exported metadata.csv")

        print(f"✓ Files saved to {output_dir}")

    def _print_success_summary(self, dataset_id, train_samples, val_samples):
        """Print success summary"""
        total_samples = len(train_samples) + len(val_samples)
        total_tokens = sum(s.token_length for s in train_samples + val_samples)

        print("\n" + "="*60)
        print(" "*20 + "SUCCESS!")
        print("="*60)
        print(f"\nDataset ID: {dataset_id}")
        print(f"Total Samples: {total_samples}")
        print(f"  - Train: {len(train_samples)}")
        print(f"  - Validation: {len(val_samples)}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Average Tokens: {total_tokens/total_samples:.0f}" if total_samples > 0 else "")
        print("\nFiles exported to output_datasets/")
        print("="*60)

    def _save_checkpoint(self, stage: str, data: Dict[str, Any]):
        """
        Save checkpoint to allow resuming after failure.

        Args:
            stage: Workflow stage (e.g., 'generation', 'validation')
            data: Data to checkpoint (samples, config, etc.)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{stage}_{timestamp}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            'stage': stage,
            'timestamp': timestamp,
            'runtime_config': self.runtime_config,
            'data': data
        }

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  💾 Checkpoint saved: {checkpoint_name}")
        except Exception as e:
            print(f"  ⚠️  Failed to save checkpoint: {e}")

    def _load_checkpoint(self, stage: str = None) -> Dict[str, Any]:
        """
        Load the most recent checkpoint for a given stage.

        Args:
            stage: Workflow stage to load (None = most recent)

        Returns:
            Checkpoint data dict or None
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pkl"))

        if not checkpoints:
            return None

        if stage:
            checkpoints = [c for c in checkpoints if c.stem.startswith(stage)]

        if not checkpoints:
            return None

        # Get most recent
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"  📂 Loaded checkpoint: {latest.name}")
            return checkpoint
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
            return None

    def _list_checkpoints(self):
        """List available checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not checkpoints:
            print("No checkpoints found")
            return

        print("\nAvailable checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            size = cp.stat().st_size / 1024  # KB
            mtime = datetime.fromtimestamp(cp.stat().st_mtime)
            print(f"  {i}. {cp.stem} ({size:.1f} KB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


def generate_dataset_workflow(config: Dict[str, Any], input_files: List = None):
    """
    Workflow function for menu system.

    Args:
        config: Runtime configuration dict
        input_files: List of input file paths (optional)
    """
    builder = DatasetBuilder()
    builder.run(runtime_config=config)


def main():
    """Main entry point"""
    # Check for CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Direct CLI mode (future enhancement)
        print("Direct CLI mode not yet implemented")
        print("Please run without arguments to use interactive menu")
        sys.exit(1)
    else:
        # Interactive menu mode
        from cli.menu import run_menu
        run_menu()


if __name__ == '__main__':
    main()
