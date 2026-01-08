"""
Agent system for dataset generation.
"""

from agents.base_agent import BaseAgent, AgentError
from agents.create_agent import CreateAgent
from agents.identify_agent import IdentifyAgent
from agents.hybrid_agent import HybridAgent
from agents.custom_agent import CustomAgent

__all__ = [
    'BaseAgent',
    'AgentError',
    'CreateAgent',
    'IdentifyAgent',
    'HybridAgent',
    'CustomAgent'
]


def get_agent(mode: str, provider: str, model: str, api_key: str, config: dict):
    """
    Factory function to get agent instance.

    Args:
        mode: Generation mode (create, identify, hybrid, custom)
        provider: LLM provider name
        model: Model name
        api_key: API key
        config: Agent configuration dict

    Returns:
        Agent instance

    Raises:
        ValueError: If mode is invalid
    """
    mode = mode.lower()

    if mode == 'create':
        return CreateAgent(provider, model, api_key, config)
    elif mode == 'identify':
        return IdentifyAgent(provider, model, api_key, config)
    elif mode == 'hybrid':
        return HybridAgent(provider, model, api_key, config)
    elif mode == 'custom':
        return CustomAgent(provider, model, api_key, config)
    else:
        raise ValueError(
            f"Invalid mode: {mode}. "
            f"Must be one of: create, identify, hybrid, custom"
        )
