"""
Configuration loader for Fine-Tuning Dataset Builder.
Loads and validates config.yaml and .env files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid"""
    pass


class EnvironmentError(Exception):
    """Raised when environment variables are missing"""
    pass


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    size: int = 100
    output_name: str = "my_dataset"
    min_quality: int = 7


@dataclass
class SplitConfig:
    """Train/validation split configuration"""
    enabled: bool = True
    ratio: float = 0.8
    strategy: str = "weighted"  # random or weighted
    weight_by: str = "source"  # source or keywords


@dataclass
class AgentCreateConfig:
    """CREATE mode agent configuration"""
    temperature: float = 0.7
    variety: str = "high"  # low, medium, high
    focus_topics: list = field(default_factory=list)
    length: str = "medium"  # short, medium, long


@dataclass
class AgentIdentifyConfig:
    """IDENTIFY mode agent configuration"""
    min_turns: int = 2
    include_context: bool = True
    max_context_paragraphs: int = 2
    confidence_threshold: float = 0.8


@dataclass
class AgentHybridConfig:
    """HYBRID mode agent configuration"""
    generation_ratio: float = 0.5
    deduplicate: bool = True


@dataclass
class AgentCustomConfig:
    """CUSTOM mode agent configuration"""
    use_prompt_file: bool = True
    allow_advanced: bool = True


@dataclass
class AgentConfig:
    """Agent configuration for all modes"""
    create: AgentCreateConfig = field(default_factory=AgentCreateConfig)
    identify: AgentIdentifyConfig = field(default_factory=AgentIdentifyConfig)
    hybrid: AgentHybridConfig = field(default_factory=AgentHybridConfig)
    custom: AgentCustomConfig = field(default_factory=AgentCustomConfig)


@dataclass
class OutputConfig:
    """Output configuration"""
    format: str = "jsonl"
    include_csv: bool = True
    include_stats: bool = True
    pretty_print: bool = False
    validate_before_save: bool = True


@dataclass
class SystemConfig:
    """System configuration"""
    agentic_llm: str = "google"
    agentic_model: str = "gemini-2.0-flash-exp"
    max_retries: int = 3
    timeout: int = 60
    verbose: bool = False
    debug: bool = False
    batch_size: int = 10
    max_concurrent: int = 5


@dataclass
class FilesConfig:
    """File processing configuration"""
    supported_formats: list = field(default_factory=lambda: [
        'pdf', 'docx', 'txt', 'csv', 'xlsx', 'pptx', 'md', 'html'
    ])
    max_size_mb: int = 50
    skip_empty: bool = True
    encoding: str = "utf-8"


@dataclass
class PersistenceConfig:
    """Persistence configuration"""
    database_path: str = "memory/dataset_builder.db"
    keep_history: bool = True
    auto_backup: bool = True
    backup_dir: str = "memory/backups"
    max_backups: int = 10


@dataclass
class Config:
    """Main configuration object"""
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    mode: str = "create"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)


def load_config(path: str = "config.yaml") -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to config.yaml file

    Returns:
        Config object with validated settings

    Raises:
        ConfigError: If configuration is invalid
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"Please ensure config.yaml exists in the project root."
        )

    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}")

    if data is None:
        data = {}

    # Create config object with defaults
    config = Config()

    # Update with values from file
    if 'provider' in data:
        config.provider = data['provider']
    if 'model' in data:
        config.model = data['model']
    if 'mode' in data:
        config.mode = data['mode']

    # Dataset config
    if 'dataset' in data:
        d = data['dataset']
        config.dataset = DatasetConfig(
            size=d.get('size', 100),
            output_name=d.get('output_name', 'my_dataset'),
            min_quality=d.get('min_quality', 7)
        )

    # Split config
    if 'split' in data:
        s = data['split']
        config.split = SplitConfig(
            enabled=s.get('enabled', True),
            ratio=s.get('ratio', 0.8),
            strategy=s.get('strategy', 'weighted'),
            weight_by=s.get('weight_by', 'source')
        )

    # Agent config
    if 'agent' in data:
        a = data['agent']

        create_cfg = AgentCreateConfig()
        if 'create' in a:
            c = a['create']
            create_cfg = AgentCreateConfig(
                temperature=c.get('temperature', 0.7),
                variety=c.get('variety', 'high'),
                focus_topics=c.get('focus_topics', []),
                length=c.get('length', 'medium')
            )

        identify_cfg = AgentIdentifyConfig()
        if 'identify' in a:
            i = a['identify']
            identify_cfg = AgentIdentifyConfig(
                min_turns=i.get('min_turns', 2),
                include_context=i.get('include_context', True),
                max_context_paragraphs=i.get('max_context_paragraphs', 2),
                confidence_threshold=i.get('confidence_threshold', 0.8)
            )

        hybrid_cfg = AgentHybridConfig()
        if 'hybrid' in a:
            h = a['hybrid']
            hybrid_cfg = AgentHybridConfig(
                generation_ratio=h.get('generation_ratio', 0.5),
                deduplicate=h.get('deduplicate', True)
            )

        custom_cfg = AgentCustomConfig()
        if 'custom' in a:
            c = a['custom']
            custom_cfg = AgentCustomConfig(
                use_prompt_file=c.get('use_prompt_file', True),
                allow_advanced=c.get('allow_advanced', True)
            )

        config.agent = AgentConfig(
            create=create_cfg,
            identify=identify_cfg,
            hybrid=hybrid_cfg,
            custom=custom_cfg
        )

    # Output config
    if 'output' in data:
        o = data['output']
        config.output = OutputConfig(
            format=o.get('format', 'jsonl'),
            include_csv=o.get('include_csv', True),
            include_stats=o.get('include_stats', True),
            pretty_print=o.get('pretty_print', False),
            validate_before_save=o.get('validate_before_save', True)
        )

    # System config
    if 'system' in data:
        sys = data['system']
        config.system = SystemConfig(
            agentic_llm=sys.get('agentic_llm', 'google'),
            agentic_model=sys.get('agentic_model', 'gemini-2.0-flash-exp'),
            max_retries=sys.get('max_retries', 3),
            timeout=sys.get('timeout', 60),
            verbose=sys.get('verbose', False),
            debug=sys.get('debug', False),
            batch_size=sys.get('batch_size', 10),
            max_concurrent=sys.get('max_concurrent', 5)
        )

    # Files config
    if 'files' in data:
        files = data['files']
        config.files = FilesConfig(
            supported_formats=files.get('supported_formats', [
                'pdf', 'docx', 'txt', 'csv', 'xlsx', 'pptx', 'md', 'html'
            ]),
            max_size_mb=files.get('max_size_mb', 50),
            skip_empty=files.get('skip_empty', True),
            encoding=files.get('encoding', 'utf-8')
        )

    # Persistence config
    if 'persistence' in data:
        p = data['persistence']
        config.persistence = PersistenceConfig(
            database_path=p.get('database_path', 'memory/dataset_builder.db'),
            keep_history=p.get('keep_history', True),
            auto_backup=p.get('auto_backup', True),
            backup_dir=p.get('backup_dir', 'memory/backups'),
            max_backups=p.get('max_backups', 10)
        )

    # Validate configuration
    _validate_config(config)

    return config


def _validate_config(config: Config) -> None:
    """
    Validate configuration values.

    Args:
        config: Config object to validate

    Raises:
        ConfigError: If configuration is invalid
    """
    # Validate provider
    valid_providers = ['openai', 'anthropic', 'google', 'mistral', 'llama']
    if config.provider not in valid_providers:
        raise ConfigError(
            f"Invalid provider: {config.provider}\n"
            f"Valid providers: {', '.join(valid_providers)}"
        )

    # Validate mode
    valid_modes = ['create', 'identify', 'hybrid', 'custom']
    if config.mode not in valid_modes:
        raise ConfigError(
            f"Invalid mode: {config.mode}\n"
            f"Valid modes: {', '.join(valid_modes)}"
        )

    # Validate dataset size
    if config.dataset.size <= 0:
        raise ConfigError("Dataset size must be positive")

    # Validate split ratio
    if not 0 < config.split.ratio < 1:
        raise ConfigError("Split ratio must be between 0 and 1")

    # Validate split strategy
    if config.split.strategy not in ['random', 'weighted']:
        raise ConfigError("Split strategy must be 'random' or 'weighted'")

    # Validate weight_by
    if config.split.weight_by not in ['source', 'keywords']:
        raise ConfigError("weight_by must be 'source' or 'keywords'")

    # Validate temperature
    if not 0 <= config.agent.create.temperature <= 1:
        raise ConfigError("Temperature must be between 0 and 1")

    # Validate variety
    if config.agent.create.variety not in ['low', 'medium', 'high']:
        raise ConfigError("Variety must be 'low', 'medium', or 'high'")

    # Validate length
    if config.agent.create.length not in ['short', 'medium', 'long']:
        raise ConfigError("Length must be 'short', 'medium', or 'long'")


def load_env() -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Returns:
        Dictionary of environment variables

    Raises:
        EnvironmentError: If .env file is missing or required keys are missing
    """
    env_path = Path('.env')

    if not env_path.exists():
        raise EnvironmentError(
            ".env file not found.\n\n"
            "Steps to fix:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your API keys to .env\n"
            "  3. Ensure .env is in the project root directory"
        )

    # Load environment variables
    load_dotenv()

    # Get all API keys
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
        'LLAMA_API_KEY': os.getenv('LLAMA_API_KEY'),
        'AGENT_PROVIDER': os.getenv('AGENT_PROVIDER'),
        'AGENT_MODEL': os.getenv('AGENT_MODEL'),
    }

    return env_vars


def validate_provider_model(provider: str, model: str) -> bool:
    """
    Validate that model is available for the specified provider.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        True if valid

    Raises:
        ConfigError: If model is not available for provider
    """
    # Provider-model mapping
    provider_models = {
        'openai': [
            'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-3.5-turbo'
        ],
        'anthropic': [
            'claude-3-haiku'
        ],
        'google': [
            'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
            'gemini-2.0-flash-exp'
        ],
        'mistral': [
            'mistral-7b', 'mistral-large-v2', 'mistral-nemo', 'mixtral-8x7b'
        ],
        'llama': [
            'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b'
        ]
    }

    if provider not in provider_models:
        raise ConfigError(f"Unknown provider: {provider}")

    if model not in provider_models[provider]:
        raise ConfigError(
            f"Model '{model}' not available for provider '{provider}'.\n"
            f"Available models: {', '.join(provider_models[provider])}"
        )

    return True
