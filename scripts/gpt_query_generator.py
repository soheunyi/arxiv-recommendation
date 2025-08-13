#!/usr/bin/env python3
"""
GPT-powered arXiv search query generator.

This script provides a CLI interface to the QueryService for generating
intelligent search queries for arXiv based on user-provided research topics.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from arxiv_recommendation.services import QueryService

logger = logging.getLogger(__name__)


class GPTQueryGenerator:
    """Generate arXiv search queries using GPT."""
    
    def __init__(self, api_key=None):
        """Initialize the GPT query generator."""
        self.query_service = QueryService(api_key)
    
    def generate_search_queries(self, topic: str, max_queries: int = 15) -> Dict:
        """Generate comprehensive arXiv search queries for a given topic."""
        return self.query_service.generate_search_queries(topic, max_queries)

    def save_queries_config(self, queries: Dict, filepath: str) -> None:
        """Save generated queries to a configuration file."""
        success = self.query_service.save_queries_config(queries, filepath)
        if success:
            logger.info(f"Saved query configuration to: {filepath}")
        else:
            logger.error(f"Failed to save configuration to: {filepath}")

    def load_queries_config(self, filepath: str) -> Dict:
        """Load queries from a configuration file."""
        config = self.query_service.load_queries_config(filepath)
        if config is None:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        logger.info(f"Loaded query configuration from: {filepath}")
        return config

    def preview_queries(self, topic: str, max_queries: int = 10) -> None:
        """Preview generated queries without saving."""
        print(f"🤖 Generating search queries for: '{topic}'")
        
        queries = self.generate_search_queries(topic, max_queries)
        
        print(f"\n📊 Generated {len(queries['search_queries'])} queries:")
        
        for i, query_info in enumerate(queries['search_queries'], 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            emoji = priority_emoji.get(query_info['priority'], "⚪")
            
            print(f"\n{i:2d}. {emoji} {query_info['priority'].upper()}")
            print(f"    Query: {query_info['query']}")
            print(f"    Purpose: {query_info['description']}")
        
        print(f"\n🏷️ Suggested arXiv categories:")
        for category in queries['categories']:
            print(f"    - {category}")
        
        print(f"\n🔍 Filter keywords:")
        print(f"    {', '.join(queries['filter_keywords'])}")
        
        print(f"\n🔗 Related terms:")
        print(f"    {', '.join(queries['related_terms'])}")


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate intelligent arXiv search queries using GPT"
    )
    
    parser.add_argument("topic", help="Research topic to generate queries for")
    parser.add_argument("--max-queries", type=int, default=15,
                       help="Maximum number of queries to generate (default: 15)")
    parser.add_argument("--preview", action="store_true",
                       help="Preview queries without saving")
    parser.add_argument("--save", type=str,
                       help="Save queries to configuration file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        generator = GPTQueryGenerator()
        
        if args.preview:
            generator.preview_queries(args.topic, args.max_queries)
        else:
            print(f"🤖 Generating search queries for: '{args.topic}'")
            queries = generator.generate_search_queries(args.topic, args.max_queries)
            
            print(f"✅ Generated {len(queries['search_queries'])} search queries")
            print(f"🏷️ Target categories: {', '.join(queries['categories'])}")
            
            if args.save:
                generator.save_queries_config(queries, args.save)
                print(f"💾 Saved configuration to: {args.save}")
            else:
                # Display the queries
                generator.preview_queries(args.topic, args.max_queries)
                
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()