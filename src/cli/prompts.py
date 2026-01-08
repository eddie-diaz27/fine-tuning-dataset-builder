"""
User input prompts and validation helpers for CLI.
"""

from typing import List, Tuple, Optional, Dict, Any


def prompt_provider() -> str:
    """
    Prompt user to select an LLM provider.

    Returns:
        Provider name
    """
    print("\n" + "="*60)
    print("Select LLM Provider")
    print("="*60)
    print("\nAvailable providers:")
    print("  1. Google Gemini (FREE TIER - Recommended)")
    print("  2. OpenAI (Requires API key)")
    print("  3. Anthropic (Requires API key)")
    print("  4. Mistral (Requires API key)")
    print("  5. Meta Llama (Requires setup)")

    while True:
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            return 'google'
        elif choice == '2':
            return 'openai'
        elif choice == '3':
            return 'anthropic'
        elif choice == '4':
            return 'mistral'
        elif choice == '5':
            return 'llama'
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def prompt_model(provider: str) -> str:
    """
    Prompt user to select a model for the given provider.

    Args:
        provider: Provider name

    Returns:
        Model name
    """
    print("\n" + "="*60)
    print(f"Select Model ({provider})")
    print("="*60)

    models = get_provider_models(provider)

    print("\nAvailable models:")
    for i, (model, desc) in enumerate(models, 1):
        print(f"  {i}. {model} - {desc}")

    while True:
        choice = input(f"\nEnter choice (1-{len(models)}): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx][0]
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(models)}.")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number 1-{len(models)}.")


def get_provider_models(provider: str) -> List[Tuple[str, str]]:
    """
    Get available models for a provider.

    Args:
        provider: Provider name

    Returns:
        List of (model_name, description) tuples
    """
    models = {
        'google': [
            ('gemini-2.5-pro', 'Most capable (FREE)'),
            ('gemini-2.5-flash', 'Fast and efficient (FREE)'),
            ('gemini-2.5-flash-lite', 'Lightweight (FREE)'),
            ('gemini-2.0-flash-exp', 'Experimental (FREE)')
        ],
        'openai': [
            ('gpt-4o', 'Most capable GPT-4'),
            ('gpt-4o-mini', 'Efficient GPT-4'),
            ('gpt-4.1', 'Latest GPT-4'),
            ('gpt-4.1-mini', 'Efficient GPT-4.1'),
            ('gpt-3.5-turbo', 'Fast and affordable')
        ],
        'anthropic': [
            ('claude-3-haiku-20240307', 'Fast and efficient (Bedrock only)')
        ],
        'mistral': [
            ('mistral-7b', 'Small model'),
            ('mistral-large-v2', 'Large model'),
            ('mistral-nemo', 'Efficient model'),
            ('mixtral-8x7b', 'Mixture of experts')
        ],
        'llama': [
            ('llama-3.1-8b', 'Small model'),
            ('llama-3.1-70b', 'Medium model'),
            ('llama-3.1-405b', 'Large model')
        ]
    }

    return models.get(provider, [])


def prompt_mode() -> str:
    """
    Prompt user to select generation mode.

    Returns:
        Mode name
    """
    print("\n" + "="*60)
    print("Select Generation Mode")
    print("="*60)
    print("\nAvailable modes:")
    print("  1. CREATE - Generate synthetic conversations from source files")
    print("  2. IDENTIFY - Extract existing conversations verbatim")
    print("  3. HYBRID - Combine both extraction and generation")
    print("  4. CUSTOM - User-defined custom behavior")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            return 'create'
        elif choice == '2':
            return 'identify'
        elif choice == '3':
            return 'hybrid'
        elif choice == '4':
            return 'custom'
        else:
            print("❌ Invalid choice. Please enter 1-4.")


def prompt_dataset_size() -> int:
    """
    Prompt user for target dataset size.

    Returns:
        Number of samples
    """
    print("\n" + "="*60)
    print("Dataset Size")
    print("="*60)

    while True:
        size_input = input("\nEnter target number of samples (e.g., 100): ").strip()

        try:
            size = int(size_input)
            if size <= 0:
                print("❌ Size must be greater than 0.")
                continue
            if size > 10000:
                confirm = input(f"⚠️  {size} is a large dataset. Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue

            return size
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def prompt_train_val_split() -> Optional[float]:
    """
    Prompt user for train/validation split ratio.

    Returns:
        Train ratio (e.g., 0.8) or None for no split
    """
    print("\n" + "="*60)
    print("Train/Validation Split")
    print("="*60)
    print("\nOptions:")
    print("  1. No split (all samples in one file)")
    print("  2. 80/20 split (recommended)")
    print("  3. 70/30 split")
    print("  4. 90/10 split")
    print("  5. Custom ratio")

    while True:
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            return None
        elif choice == '2':
            return 0.8
        elif choice == '3':
            return 0.7
        elif choice == '4':
            return 0.9
        elif choice == '5':
            while True:
                custom = input("Enter train ratio (e.g., 0.75): ").strip()
                try:
                    ratio = float(custom)
                    if 0 < ratio < 1:
                        return ratio
                    else:
                        print("❌ Ratio must be between 0 and 1.")
                except ValueError:
                    print("❌ Invalid input. Please enter a decimal number.")
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def prompt_split_strategy() -> str:
    """
    Prompt user for split strategy.

    Returns:
        Strategy name
    """
    print("\n" + "="*60)
    print("Split Strategy")
    print("="*60)
    print("\nOptions:")
    print("  1. Random - Uniform random split")
    print("  2. Weighted by Source - Equal representation per source file")
    print("  3. Weighted by Keywords - Equal representation per topic")

    while True:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            return 'random'
        elif choice == '2':
            return 'weighted_source'
        elif choice == '3':
            return 'weighted_keywords'
        else:
            print("❌ Invalid choice. Please enter 1-3.")


def prompt_variety() -> str:
    """
    Prompt user for generation variety/temperature.

    Returns:
        Variety level
    """
    print("\n" + "="*60)
    print("Generation Variety")
    print("="*60)
    print("\nOptions:")
    print("  1. Low - More consistent, focused responses")
    print("  2. Medium - Balanced (recommended)")
    print("  3. High - More creative, diverse responses")

    while True:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            return 'low'
        elif choice == '2':
            return 'medium'
        elif choice == '3':
            return 'high'
        else:
            print("❌ Invalid choice. Please enter 1-3.")


def prompt_length() -> str:
    """
    Prompt user for conversation length preference.

    Returns:
        Length level
    """
    print("\n" + "="*60)
    print("Conversation Length")
    print("="*60)
    print("\nOptions:")
    print("  1. Short - 2-3 exchanges")
    print("  2. Medium - 4-6 exchanges (recommended)")
    print("  3. Long - 7+ exchanges")

    while True:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            return 'short'
        elif choice == '2':
            return 'medium'
        elif choice == '3':
            return 'long'
        else:
            print("❌ Invalid choice. Please enter 1-3.")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """
    Prompt user for yes/no confirmation.

    Args:
        question: Question to ask
        default: Default answer if user presses Enter

    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{question} ({default_str}): ").strip().lower()

    if not response:
        return default

    return response in ['y', 'yes']


def prompt_dataset_name() -> str:
    """
    Prompt user for dataset name.

    Returns:
        Dataset name
    """
    while True:
        name = input("\nEnter dataset name (or press Enter for auto-generated): ").strip()

        if not name:
            # Auto-generate name
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"dataset_{timestamp}"
            print(f"✓ Using auto-generated name: {name}")
            return name

        # Validate name
        if len(name) < 3:
            print("❌ Name must be at least 3 characters.")
            continue

        # Check for invalid characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            print("❌ Name can only contain letters, numbers, underscores, and hyphens.")
            continue

        return name


def print_configuration_summary(config: Dict[str, Any], config_name: str = None):
    """
    Print configuration summary for user review.

    Args:
        config: Configuration dict
        config_name: Optional config name to display
    """
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)

    # Display config name if provided
    if config_name:
        print(f"\nConfiguration: {config_name}")
    else:
        # Generate default name from dataset name and timestamp
        from datetime import datetime
        dataset_name = config.get('dataset', {}).get('name', 'dataset')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{dataset_name}:{timestamp}"
        print(f"\nConfiguration: {default_name} (unsaved)")

    dataset = config.get('dataset', {})
    agent = config.get('agent', {})
    split = config.get('split', {})

    print(f"\nDataset:")
    print(f"  Name: {dataset.get('name', 'N/A')}")
    print(f"  Provider: {dataset.get('provider', 'N/A')}")
    print(f"  Model: {dataset.get('model', 'N/A')}")
    print(f"  Mode: {agent.get('mode', 'N/A')}")
    print(f"  Target Size: {dataset.get('target_samples', 'N/A')} samples")

    print(f"\nGeneration:")
    print(f"  Variety: {agent.get('variety', 'N/A')}")
    print(f"  Length: {agent.get('length', 'N/A')}")

    if split.get('enabled'):
        print(f"\nSplit:")
        print(f"  Ratio: {split.get('ratio', 'N/A')}")
        print(f"  Strategy: {split.get('strategy', 'N/A')}")
    else:
        print(f"\nSplit: Disabled")

    print("="*60)


def confirm_configuration() -> bool:
    """
    Confirm configuration with user.

    Returns:
        True if confirmed
    """
    return prompt_yes_no("\nProceed with this configuration?", default=True)
