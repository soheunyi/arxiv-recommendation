#!/usr/bin/env python3
"""
LLM-powered arXiv search query generator.

This script provides a CLI interface for generating intelligent search queries 
for arXiv based on user-provided research topics. Supports both OpenAI and Gemini providers.
"""

import logging
import sys

from script_utils import (
    setup_backend_path, print_success, print_error, print_progress, print_info
)

# Setup backend imports
setup_backend_path()
from services.provider_factory import ProviderFactory, get_current_provider

logger = logging.getLogger(__name__)


def preview_queries(service, topic: str, max_queries: int = 10, provider: str = None) -> None:
    """Preview generated queries without saving."""
    provider_name = provider or get_current_provider()
    print_progress(f"Generating search queries for: '{topic}' using {provider_name.title()}")
    
    queries = service.generate_search_queries(topic, max_queries)
    
    print(f"\n📊 Generated {len(queries['search_queries'])} queries:")
    
    for i, query_info in enumerate(queries['search_queries'], 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        emoji = priority_emoji.get(query_info['priority'], "⚪")
        
        print(f"\n{i:2d}. {emoji} {query_info['priority'].upper()}")
        print(f"    Query: {query_info['query']}")
        print(f"    Purpose: {query_info['description']}")
    
    print_info("Suggested arXiv categories:")
    for category in queries['categories']:
        print(f"    - {category}")
    
    print_info("Filter keywords:")
    print(f"    {', '.join(queries['filter_keywords'])}")
    
    print_info("Related terms:")
    print(f"    {', '.join(queries['related_terms'])}")


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate intelligent arXiv search queries using LLMs (OpenAI or Gemini)"
    )
    
    parser.add_argument("topic", help="Research topic to generate queries for")
    parser.add_argument("--max-queries", type=int, default=15,
                       help="Maximum number of queries to generate (default: 15)")
    parser.add_argument("--preview", action="store_true",
                       help="Preview queries without saving")
    parser.add_argument("--save", type=str,
                       help="Save queries to configuration file")
    parser.add_argument("--provider", choices=["openai", "gemini"], 
                       help=f"LLM provider to use (default: {get_current_provider()})")
    parser.add_argument("--compare-providers", action="store_true",
                       help="Show comparison between available providers")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Handle provider comparison
        if args.compare_providers:
            comparison = ProviderFactory.compare_providers()
            print_info("Provider Comparison:")
            print()
            
            for provider, info in comparison.items():
                if provider == "cost_savings":
                    print_info("💰 Cost Savings with Gemini:")
                    for metric, saving in info.items():
                        print(f"    {metric}: {saving}")
                    continue
                    
                print(f"🤖 {info['name']}")
                print(f"    Query Model: {info['query_model']}")
                print(f"    Embedding Model: {info['embedding_model']}")
                print(f"    Features: {', '.join(info['features'])}")
                
                costs = info['cost_per_1k_tokens']
                print(f"    Cost per 1K tokens:")
                print(f"      Input: ${costs['input']}")
                print(f"      Output: ${costs['output']}")
                print(f"      Embedding: ${costs['embedding']}")
                print()
            
            # Show recommendation
            recommended = ProviderFactory.recommend_provider(
                cost_sensitive=True, multilingual=False, long_context=False
            )
            print_success(f"💡 Recommended provider: {recommended}")
            return
        
        # Determine provider
        provider = args.provider or get_current_provider()
        
        # Create service using factory
        service = ProviderFactory.create_query_service(provider)
        
        if args.preview:
            preview_queries(service, args.topic, args.max_queries, provider)
        else:
            provider_name = provider.title()
            print_progress(f"Generating search queries for: '{args.topic}' using {provider_name}")
            queries = service.generate_search_queries(args.topic, args.max_queries)
            
            print_success(f"Generated {len(queries['search_queries'])} search queries using {provider_name}")
            print_info(f"Target categories: {', '.join(queries['categories'])}")
            
            if args.save:
                success = service.save_queries_config(queries, args.save)
                if success:
                    print_success(f"Saved configuration to: {args.save}")
                else:
                    print_error(f"Failed to save configuration to: {args.save}")
            else:
                # Display the queries
                preview_queries(service, args.topic, args.max_queries, provider)
                
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()