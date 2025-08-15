#!/usr/bin/env python3
"""
Provider Factory for ArXiv Recommendation System.

This factory creates the appropriate LLM service instances based on configuration.
"""

import logging
from typing import Union, Optional

from config import config
from services.query_service import QueryService
from services.gemini_query_service import GeminiQueryService

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating LLM service instances based on configuration."""
    
    @staticmethod
    def create_query_service(
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Union[QueryService, GeminiQueryService]:
        """
        Create a query service instance based on the provider.
        
        Args:
            provider: LLM provider ("openai" or "gemini"). If None, uses config default.
            api_key: Optional API key override
            
        Returns:
            Query service instance
            
        Raises:
            ValueError: If provider is invalid or API key is missing
        """
        provider = provider or config.llm_provider
        
        if provider == "openai":
            return QueryService(api_key=api_key)
        elif provider == "gemini":
            return GeminiQueryService(api_key=api_key)
        else:
            raise ValueError(f"Invalid provider: {provider}. Must be 'openai' or 'gemini'")
    
    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available LLM providers."""
        return ["openai", "gemini"]
    
    @staticmethod
    def validate_provider(provider: str) -> bool:
        """Validate if a provider is supported."""
        return provider in ProviderFactory.get_available_providers()
    
    @staticmethod
    def get_provider_info(provider: str) -> dict:
        """Get information about a specific provider."""
        provider_info = {
            "openai": {
                "name": "OpenAI",
                "query_model": config.openai_query_model,
                "embedding_model": config.openai_embedding_model,
                "features": [
                    "Structured JSON outputs",
                    "High-quality query generation",
                    "Proven performance"
                ],
                "cost_per_1k_tokens": {
                    "input": 2.5,  # GPT-4o input cost
                    "output": 10.0,  # GPT-4o output cost
                    "embedding": 0.02  # text-embedding-3-small
                }
            },
            "gemini": {
                "name": "Google Gemini",
                "query_model": config.gemini_query_model,
                "embedding_model": config.gemini_embedding_model,
                "features": [
                    "Structured JSON outputs",
                    "Native thinking mode",
                    "Extremely cost-effective",
                    "Multilingual support",
                    "Long context (2M tokens)"
                ],
                "cost_per_1k_tokens": {
                    "input": 0.075,  # Gemini 2.5 Flash input cost
                    "output": 0.3,   # Gemini 2.5 Flash output cost
                    "embedding": 0.025  # text-embedding-004
                }
            }
        }
        
        if provider not in provider_info:
            raise ValueError(f"Unknown provider: {provider}")
            
        return provider_info[provider]
    
    @staticmethod
    def compare_providers() -> dict:
        """Compare available providers side by side."""
        providers = ProviderFactory.get_available_providers()
        comparison = {}
        
        for provider in providers:
            comparison[provider] = ProviderFactory.get_provider_info(provider)
        
        # Add cost comparison
        openai_cost = comparison["openai"]["cost_per_1k_tokens"]
        gemini_cost = comparison["gemini"]["cost_per_1k_tokens"]
        
        comparison["cost_savings"] = {
            "input_tokens": f"Gemini is {openai_cost['input'] / gemini_cost['input']:.1f}x cheaper",
            "output_tokens": f"Gemini is {openai_cost['output'] / gemini_cost['output']:.1f}x cheaper",
            "embeddings": f"Gemini is {openai_cost['embedding'] / gemini_cost['embedding']:.1f}x cheaper"
        }
        
        return comparison
    
    @staticmethod
    def recommend_provider(
        cost_sensitive: bool = True,
        multilingual: bool = False,
        long_context: bool = False
    ) -> str:
        """
        Recommend the best provider based on requirements.
        
        Args:
            cost_sensitive: Whether cost is a primary concern
            multilingual: Whether multilingual support is needed
            long_context: Whether long context processing is needed
            
        Returns:
            Recommended provider name
        """
        # Score each provider
        scores = {"openai": 0, "gemini": 0}
        
        # Cost factor (heavily weighted if cost_sensitive is True)
        if cost_sensitive:
            scores["gemini"] += 5  # Gemini is much cheaper
            scores["openai"] += 1
        else:
            scores["gemini"] += 2
            scores["openai"] += 2
        
        # Multilingual support
        if multilingual:
            scores["gemini"] += 3  # Gemini has better multilingual support
            scores["openai"] += 2
        
        # Long context processing
        if long_context:
            scores["gemini"] += 3  # Gemini supports 2M tokens
            scores["openai"] += 1  # OpenAI has shorter context limits
        
        # Quality and reliability (OpenAI slight edge in proven performance)
        scores["openai"] += 2
        scores["gemini"] += 2
        
        # Return the provider with highest score
        return max(scores, key=scores.get)


# Convenience functions for backward compatibility
def create_query_service(provider: Optional[str] = None, api_key: Optional[str] = None):
    """Convenience function to create a query service."""
    return ProviderFactory.create_query_service(provider, api_key)


def get_current_provider() -> str:
    """Get the currently configured provider."""
    return config.llm_provider


def switch_provider(provider: str) -> bool:
    """
    Switch the global provider configuration.
    
    Args:
        provider: New provider to use
        
    Returns:
        True if switch was successful
        
    Note:
        This only affects the global config for the current session.
        To persist the change, update the environment variable.
    """
    if not ProviderFactory.validate_provider(provider):
        logger.error(f"Invalid provider: {provider}")
        return False
    
    old_provider = config.llm_provider
    config.llm_provider = provider
    
    logger.info(f"Switched LLM provider from {old_provider} to {provider}")
    return True