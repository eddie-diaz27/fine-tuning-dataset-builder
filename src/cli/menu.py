"""
Interactive menu system for Fine-Tuning Dataset Builder.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

from cli import prompts
from utils.config_loader import load_config, Config
from utils.file_processor import FileProcessor


class Menu:
    """Interactive CLI menu system"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize menu system.

        Args:
            config_path: Path to config.yaml
        """
        self.config_path = config_path
        self.config: Optional[Config] = None
        self.runtime_config: Dict[str, Any] = {}
        self.config_name: Optional[str] = None  # Track current config name

    def run(self):
        """Run the interactive menu loop"""
        self.print_header()

        # Load initial config
        try:
            self.config = load_config(self.config_path)
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
            print("⚠️  Using default configuration\n")

        while True:
            self.print_main_menu()
            choice = input("\nEnter choice: ").strip()

            if choice == '1':
                self.configure_dataset()
            elif choice == '2':
                self.load_saved_config()
            elif choice == '3':
                self.generate_dataset()
            elif choice == '4':
                self.view_statistics()
            elif choice == '5':
                self.export_dataset()
            elif choice == '6':
                self.import_markdown()
            elif choice == '7':
                self.delete_dataset()
            elif choice == '8':
                self.settings()
            elif choice == '9':
                self.print_goodbye()
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-9.")

    def print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print(" "*15 + "Fine-Tuning Dataset Builder")
        print("="*60)
        print("\nGenerate high-quality datasets for LLM fine-tuning")
        print("Supports: OpenAI, Anthropic, Google, Mistral, Meta Llama")
        print("="*60)

    def print_main_menu(self):
        """Print main menu options"""
        print("\n" + "="*60)
        print("Main Menu")
        print("="*60)
        print("\n1. Configure Dataset")
        print("2. Load Saved Configuration")
        print("3. Generate Dataset")
        print("4. View Dataset Statistics")
        print("5. Export Dataset")
        print("6. Import Edited Markdown")
        print("7. Delete Dataset")
        print("8. Settings")
        print("9. Exit")

    def configure_dataset(self):
        """Configure dataset generation settings"""
        print("\n" + "="*60)
        print("Configure Dataset")
        print("="*60)

        # Ask if user wants to create new or edit existing
        from persistence.sqlite_client import SQLiteClient
        db = SQLiteClient()
        configs = db.list_saved_configs()

        existing_config_name = None
        if configs:
            print("\n1. Create new configuration")
            print("2. Edit existing configuration")
            choice = input("\nEnter choice (1-2): ").strip()

            if choice == '2':
                print("\nExisting Configurations:")
                print("-" * 60)
                for i, cfg in enumerate(configs, 1):
                    print(f"{i}. {cfg['name']} - {cfg['provider']}/{cfg['model']}/{cfg['mode'].upper()}")

                config_choice = input("\nSelect configuration to edit (number): ").strip()
                try:
                    config_idx = int(config_choice) - 1
                    if 0 <= config_idx < len(configs):
                        existing_config_name = configs[config_idx]['name']
                        config_data = db.get_config(existing_config_name)
                        if config_data:
                            self.runtime_config = config_data['config']
                            self.config_name = existing_config_name
                            print(f"\n✓ Loaded '{existing_config_name}' for editing")
                            prompts.print_configuration_summary(self.runtime_config, existing_config_name)
                            print("\nNow you can modify the settings:")
                except (ValueError, IndexError):
                    print("Invalid selection, creating new configuration instead")

        db.close()

        # Prompt for configuration (or use defaults from loaded config)
        provider = prompts.prompt_provider()
        model = prompts.prompt_model(provider)
        mode = prompts.prompt_mode()
        target_size = prompts.prompt_dataset_size()

        # Optional settings
        variety = prompts.prompt_variety()
        length = prompts.prompt_length()

        # Split configuration
        split_enabled = prompts.prompt_yes_no("Enable train/validation split?", default=True)

        if split_enabled:
            split_ratio = prompts.prompt_train_val_split()
            split_strategy = prompts.prompt_split_strategy()
        else:
            split_ratio = None
            split_strategy = 'random'

        # Dataset name
        dataset_name = prompts.prompt_dataset_name()

        # Build runtime config
        self.runtime_config = {
            'dataset': {
                'name': dataset_name,
                'provider': provider,
                'model': model,
                'target_samples': target_size
            },
            'agent': {
                'mode': mode,
                'variety': variety,
                'length': length
            },
            'split': {
                'enabled': split_enabled,
                'ratio': split_ratio,
                'strategy': split_strategy
            }
        }

        # Print summary
        prompts.print_configuration_summary(self.runtime_config)

        # Confirm
        if prompts.confirm_configuration():
            print("\n✓ Configuration saved to memory")

            # Generate default name for tracking (even if not saved to DB)
            from datetime import datetime
            dataset_name = self.runtime_config['dataset']['name']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_config_name = f"{dataset_name}:{timestamp}"

            # Set default name if not already set
            if not self.config_name:
                self.config_name = default_config_name

            # Save or update configuration in database
            if prompts.prompt_yes_no("\nSave this configuration for future use?", default=True):
                from persistence.sqlite_client import SQLiteClient
                from datetime import datetime

                db = SQLiteClient()

                # Generate default name with format: {dataset-name}:{timestamp}
                dataset_name = self.runtime_config['dataset']['name']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_config_name = f"{dataset_name}:{timestamp}"

                if existing_config_name:
                    # Editing existing config
                    print(f"\nUpdating existing configuration: {existing_config_name}")
                    if prompts.prompt_yes_no("Keep the same name?", default=True):
                        config_name = existing_config_name
                    else:
                        config_name = input(f"Enter new name (default: {default_config_name}): ").strip()
                        if not config_name:
                            config_name = default_config_name
                else:
                    # New configuration
                    config_name = input(f"Enter configuration name (default: {default_config_name}): ").strip()
                    if not config_name:
                        config_name = default_config_name

                if config_name:
                    try:
                        if existing_config_name and config_name == existing_config_name:
                            # Update existing
                            if db.update_config(config_name, self.runtime_config):
                                print(f"✓ Configuration '{config_name}' updated in database")
                                self.config_name = config_name
                            else:
                                print(f"⚠️  Failed to update configuration")
                        else:
                            # Save new
                            db.save_config(config_name, self.runtime_config)
                            print(f"✓ Configuration '{config_name}' saved to database")
                            self.config_name = config_name
                    except ValueError as e:
                        print(f"⚠️  {e}")
                db.close()
        else:
            print("\n❌ Configuration discarded")
            self.runtime_config = {}

    def load_saved_config(self):
        """Load a saved configuration from database"""
        print("\n" + "="*60)
        print("Load Saved Configuration")
        print("="*60)

        from persistence.sqlite_client import SQLiteClient
        db = SQLiteClient()

        # List saved configurations
        configs = db.list_saved_configs()

        if not configs:
            print("\n⚠️  No saved configurations found.")
            print("Create and save a configuration first (Option 1).")
            db.close()
            return

        print("\nSaved Configurations:")
        print("-" * 60)
        for i, cfg in enumerate(configs, 1):
            print(f"\n{i}. {cfg['name']}")
            print(f"   Provider: {cfg['provider']} | Model: {cfg['model']} | Mode: {cfg['mode'].upper()}")
            print(f"   Created: {cfg['created_at']}")
            if cfg['last_used']:
                print(f"   Last used: {cfg['last_used']}")

        print("\n" + "-" * 60)
        choice = input("\nEnter configuration number to load (or 0 to cancel): ").strip()

        try:
            choice_num = int(choice)
            if choice_num == 0:
                print("❌ Cancelled")
                db.close()
                return

            if 1 <= choice_num <= len(configs):
                selected_config = configs[choice_num - 1]
                config_data = db.get_config(selected_config['name'])

                if config_data:
                    self.runtime_config = config_data['config']
                    self.config_name = selected_config['name']
                    print(f"\n✓ Loaded configuration: {selected_config['name']}")
                    prompts.print_configuration_summary(self.runtime_config, selected_config['name'])
                else:
                    print(f"\n❌ Failed to load configuration")
            else:
                print("❌ Invalid selection")

        except ValueError:
            print("❌ Invalid input")

        db.close()

    def generate_dataset(self):
        """Generate dataset with current configuration"""
        print("\n" + "="*60)
        print("Generate Dataset")
        print("="*60)

        # Show loaded configuration name
        if self.config_name:
            print(f"\nUsing configuration: {self.config_name}")

        # Check if configured
        if not self.runtime_config:
            print("\n⚠️  No configuration found. Please configure dataset first.")
            return

        # Check for input files
        input_dir = Path("input_files")
        if not input_dir.exists():
            print(f"\n❌ Input directory not found: {input_dir}")
            return

        # Get input files
        input_files = list(input_dir.glob("*"))
        input_files = [f for f in input_files if f.is_file() and not f.name.startswith('.')]

        if not input_files:
            print(f"\n❌ No files found in {input_dir}")
            print("Please add source files to input_files/ directory")
            return

        print(f"\nFound {len(input_files)} input files:")
        for f in input_files:
            print(f"  - {f.name}")

        # Confirm
        if not prompts.prompt_yes_no(f"\nProceed with generation using {len(input_files)} files?"):
            print("❌ Generation cancelled")
            return

        # Import generation logic here to avoid circular imports
        print("\n" + "="*60)
        print("Starting Generation...")
        print("="*60)

        try:
            # Delegate to main.py logic
            from main import generate_dataset_workflow

            generate_dataset_workflow(
                config=self.runtime_config,
                input_files=input_files
            )

            print("\n✓ Dataset generation complete!")

        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

    def view_statistics(self):
        """View dataset statistics from database"""
        print("\n" + "="*60)
        print("View Dataset Statistics")
        print("="*60)

        try:
            from persistence import SQLiteClient

            db_path = "memory/dataset_builder.db"
            if not Path(db_path).exists():
                print("\n⚠️  No database found. Generate a dataset first.")
                return

            client = SQLiteClient(db_path)

            # Get all datasets
            datasets = client.list_datasets()

            if not datasets:
                print("\n⚠️  No datasets found in database.")
                return

            print(f"\nFound {len(datasets)} dataset(s):\n")

            for i, ds in enumerate(datasets, 1):
                print(f"{i}. {ds['name']}")
                print(f"   Provider: {ds['provider']} ({ds['model']})")
                print(f"   Mode: {ds['mode']}")
                print(f"   Samples: {ds['total_samples']}")
                print(f"   Tokens: {ds['total_tokens']:,}")
                print(f"   Created: {ds['created_at']}")
                print()

            # Prompt for details
            if prompts.prompt_yes_no("View detailed statistics for a dataset?"):
                choice = input(f"Enter dataset number (1-{len(datasets)}): ").strip()

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(datasets):
                        dataset_id = datasets[idx]['id']
                        self._show_dataset_details(client, dataset_id)
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Invalid input")

        except Exception as e:
            print(f"\n❌ Failed to load statistics: {e}")

    def _show_dataset_details(self, client, dataset_id: str):
        """Show detailed dataset statistics"""
        samples = client.get_samples(dataset_id)

        if not samples:
            print("\n⚠️  No samples found for this dataset")
            return

        print("\n" + "="*60)
        print("Dataset Details")
        print("="*60)

        # Calculate statistics
        total_samples = len(samples)
        train_samples = sum(1 for s in samples if s.get('split') == 'train')
        val_samples = sum(1 for s in samples if s.get('split') == 'validation')

        total_tokens = sum(s.get('token_length', 0) for s in samples)
        avg_tokens = total_tokens / total_samples if total_samples > 0 else 0

        # Source distribution
        sources = {}
        for s in samples:
            source = s.get('source_file', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        print(f"\nTotal Samples: {total_samples}")
        print(f"  Train: {train_samples}")
        print(f"  Validation: {val_samples}")

        print(f"\nTokens:")
        print(f"  Total: {total_tokens:,}")
        print(f"  Average: {avg_tokens:.0f}")

        print(f"\nSource Distribution:")
        for source, count in sources.items():
            print(f"  {source}: {count} samples")

        print("="*60)

    def export_dataset(self):
        """Export dataset to JSONL and CSV"""
        print("\n" + "="*60)
        print("Export Dataset")
        print("="*60)

        try:
            from persistence import SQLiteClient

            db_path = "memory/dataset_builder.db"
            if not Path(db_path).exists():
                print("\n⚠️  No database found. Generate a dataset first.")
                return

            client = SQLiteClient(db_path)
            datasets = client.list_datasets()

            if not datasets:
                print("\n⚠️  No datasets found.")
                return

            print(f"\nAvailable datasets:\n")
            for i, ds in enumerate(datasets, 1):
                print(f"{i}. {ds['name']} ({ds['total_samples']} samples)")

            choice = input(f"\nEnter dataset number to export (1-{len(datasets)}): ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    dataset_id = datasets[idx]['id']
                    dataset_name = datasets[idx]['name']

                    # Export
                    print(f"\nExporting {dataset_name}...")
                    self._export_dataset_files(client, dataset_id, dataset_name)
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Invalid input")

        except Exception as e:
            print(f"\n❌ Export failed: {e}")

    def _export_dataset_files(self, client, dataset_id: str, dataset_name: str):
        """Export dataset files"""
        from utils.formatter import Formatter
        from utils.markdown_converter import MarkdownConverter
        from utils.validator import Sample
        from providers import get_provider

        # Get samples from database (returns list of dicts)
        sample_dicts = client.get_samples(dataset_id)

        if not sample_dicts:
            print("⚠️  No samples to export")
            return

        # Get dataset info
        dataset_info = client.get_dataset(dataset_id)
        provider_name = dataset_info['provider']

        # Initialize provider
        provider = get_provider(provider_name)
        formatter = Formatter(provider)

        # Convert dict samples to Sample objects
        samples = []
        for s in sample_dicts:
            sample_obj = Sample(
                id=s.get('id', ''),
                content=json.loads(s['content']),
                token_length=s.get('token_length', 0),
                topic_keywords=s.get('topic_keywords', '').split(',') if s.get('topic_keywords') else [],
                source_file=s.get('source_file', ''),
                mode=s.get('mode', '')
            )
            if s.get('split'):
                sample_obj.split = s['split']
            samples.append(sample_obj)

        # Create output directory
        output_dir = Path("output_datasets") / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export JSONL
        train_samples = [s for s in samples if hasattr(s, 'split') and s.split == 'train']
        val_samples = [s for s in samples if hasattr(s, 'split') and s.split == 'validation']

        if train_samples:
            train_path = output_dir / "train.jsonl"
            formatter.write_jsonl(train_samples, str(train_path))
            print(f"✓ Exported {len(train_samples)} train samples to {train_path}")

        if val_samples:
            val_path = output_dir / "validation.jsonl"
            formatter.write_jsonl(val_samples, str(val_path))
            print(f"✓ Exported {len(val_samples)} validation samples to {val_path}")

        # Export CSV metadata
        csv_path = output_dir / "metadata.csv"
        client.export_csv(dataset_id, str(csv_path))
        print(f"✓ Exported metadata to {csv_path}")

        # Ask if user wants markdown export
        print("\n📄 Export readable markdown versions?")
        if prompts.prompt_yes_no("This allows editing samples in human-readable format"):
            md_converter = MarkdownConverter()

            # Export markdown versions (samples already converted to Sample objects above)
            if train_samples:
                train_md_path = output_dir / "train.md"
                md_converter.export_to_markdown(train_samples, str(train_md_path))

            if val_samples:
                val_md_path = output_dir / "validation.md"
                md_converter.export_to_markdown(val_samples, str(val_md_path))

            print(f"\n📝 Markdown files exported!")
            print(f"   To edit: Open .md files in any text editor")
            print(f"   To sync changes back: Use menu option '6. Import Edited Markdown'")

        print(f"\n✓ Export complete! Files saved to {output_dir}")

    def import_markdown(self):
        """Import edited markdown and sync back to JSONL"""
        print("\n" + "="*60)
        print("Import Edited Markdown")
        print("="*60)

        try:
            # List available markdown files
            output_dir = Path("output_datasets")
            if not output_dir.exists():
                print("\n⚠️  No output_datasets directory found")
                return

            md_files = list(output_dir.rglob("*.md"))
            if not md_files:
                print("\n⚠️  No markdown files found in output_datasets/")
                print("   Export a dataset with markdown option first (menu option 5)")
                return

            print(f"\nFound {len(md_files)} markdown file(s):\n")
            for i, md_file in enumerate(md_files, 1):
                rel_path = md_file.relative_to(output_dir)
                print(f"{i}. {rel_path}")

            choice = input(f"\nEnter file number to import (1-{len(md_files)}): ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(md_files):
                    md_path = md_files[idx]
                    self._sync_markdown_to_jsonl(md_path)
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Invalid input")

        except Exception as e:
            print(f"\n❌ Import failed: {e}")
            import traceback
            traceback.print_exc()

    def _sync_markdown_to_jsonl(self, md_path: Path):
        """Sync markdown edits back to JSONL"""
        from utils.markdown_converter import MarkdownConverter
        from utils.formatter import Formatter
        from providers import get_provider
        from persistence import SQLiteClient

        print(f"\n📥 Importing from {md_path.name}...")

        # Determine corresponding JSONL path
        jsonl_path = md_path.with_suffix('.jsonl')

        if not jsonl_path.exists():
            print(f"⚠️  Corresponding JSONL not found: {jsonl_path}")
            print("   Creating new JSONL file from markdown...")

        # Get provider from parent directory or prompt user
        dataset_name = md_path.parent.name
        db_path = "memory/dataset_builder.db"

        provider_name = None
        if Path(db_path).exists():
            client = SQLiteClient(db_path)
            datasets = client.list_datasets()

            # Find matching dataset
            for ds in datasets:
                if ds['name'] == dataset_name:
                    provider_name = ds['provider']
                    break

        if not provider_name:
            print("\n⚠️  Could not determine provider from database")
            print("Available providers: google, openai, anthropic, mistral, llama")
            provider_name = input("Enter provider name: ").strip().lower()

        try:
            provider = get_provider(provider_name)
        except ValueError as e:
            print(f"❌ {e}")
            return

        # Import from markdown
        md_converter = MarkdownConverter()
        md_converter.sync_markdown_to_jsonl(
            str(md_path),
            str(jsonl_path),
            provider,
            backup=True
        )

        print(f"\n✅ Successfully synced changes from markdown to JSONL!")
        print(f"   Original JSONL backed up to {jsonl_path}.backup")
        print(f"   Updated JSONL: {jsonl_path}")

    def delete_dataset(self):
        """Delete a dataset from database and optionally from disk"""
        print("\n" + "="*60)
        print("Delete Dataset")
        print("="*60)

        try:
            from persistence import SQLiteClient
            import shutil

            db_path = "memory/dataset_builder.db"
            if not Path(db_path).exists():
                print("\n⚠️  No database found.")
                return

            client = SQLiteClient(db_path)
            datasets = client.list_datasets()

            if not datasets:
                print("\n⚠️  No datasets found.")
                return

            print(f"\nAvailable datasets:\n")
            for i, ds in enumerate(datasets, 1):
                print(f"{i}. {ds['name']} ({ds['total_samples']} samples, {ds['mode']} mode)")

            choice = input(f"\nEnter dataset number to delete (1-{len(datasets)}, or 0 to cancel): ").strip()

            try:
                idx = int(choice) - 1
                if idx == -1:  # User entered 0
                    print("Cancelled")
                    return

                if 0 <= idx < len(datasets):
                    dataset_id = datasets[idx]['id']
                    dataset_name = datasets[idx]['name']

                    # Confirm deletion
                    print(f"\n⚠️  WARNING: This will delete '{dataset_name}' from the database.")
                    if prompts.prompt_yes_no("Are you sure you want to delete this dataset?"):
                        # Delete from database
                        if client.delete_dataset(dataset_id):
                            print(f"✓ Deleted '{dataset_name}' from database")

                            # Ask if user wants to delete exported files
                            output_dir = Path("output_datasets") / dataset_name
                            if output_dir.exists():
                                if prompts.prompt_yes_no(f"Also delete exported files in {output_dir}?"):
                                    shutil.rmtree(output_dir)
                                    print(f"✓ Deleted exported files")

                            print(f"\n✅ Dataset '{dataset_name}' successfully deleted!")
                        else:
                            print(f"❌ Failed to delete dataset")
                    else:
                        print("Deletion cancelled")
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Invalid input")

        except Exception as e:
            print(f"\n❌ Delete failed: {e}")
            import traceback
            traceback.print_exc()

    def settings(self):
        """Settings menu"""
        print("\n" + "="*60)
        print("Settings")
        print("="*60)
        print("\n1. View Current Configuration")
        print("2. Edit config.yaml")
        print("3. Check API Keys")
        print("4. Back to Main Menu")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            self._view_config()
        elif choice == '2':
            self._edit_config()
        elif choice == '3':
            self._check_api_keys()
        elif choice == '4':
            return
        else:
            print("❌ Invalid choice")

    def _view_config(self):
        """View current configuration"""
        if self.config:
            print("\n" + "="*60)
            print("Current Configuration")
            print("="*60)

            # Print key settings
            print(f"\nDataset:")
            print(f"  Provider: {self.config.dataset.provider}")
            print(f"  Model: {self.config.dataset.model}")

            print(f"\nAgent:")
            print(f"  Mode: {self.config.agent.mode}")
            print(f"  Variety: {self.config.agent.variety}")
            print(f"  Length: {self.config.agent.length}")

            print("="*60)
        else:
            print("\n⚠️  No configuration loaded")

    def _edit_config(self):
        """Edit config.yaml"""
        print(f"\nTo edit configuration, open: {self.config_path}")
        print("After editing, restart the application to apply changes.")

    def _check_api_keys(self):
        """Check API keys in .env"""
        import os
        from dotenv import load_dotenv

        load_dotenv()

        print("\n" + "="*60)
        print("API Keys Status")
        print("="*60)

        keys = {
            'Google': 'GOOGLE_API_KEY',
            'OpenAI': 'OPENAI_API_KEY',
            'Anthropic': 'ANTHROPIC_API_KEY',
            'Mistral': 'MISTRAL_API_KEY'
        }

        for provider, env_var in keys.items():
            value = os.getenv(env_var)
            status = "✓ Set" if value else "❌ Not set"
            print(f"{provider:15} {status}")

        print("="*60)
        print("\nTo add API keys, edit the .env file in the project root")

    def print_goodbye(self):
        """Print goodbye message"""
        print("\n" + "="*60)
        print("Thank you for using Fine-Tuning Dataset Builder!")
        print("="*60)
        print()


def run_menu():
    """Entry point for menu system"""
    menu = Menu()
    menu.run()
