#!/usr/bin/env python3
"""
Query Service for ArXiv Recommendation System.

This service provides GPT-powered query generation functionality
that can be used by both API endpoints and CLI scripts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

from ..config import config

logger = logging.getLogger(__name__)


class QueryService:
    """Service for generating arXiv search queries using GPT."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the query service."""
        self.client = OpenAI(api_key=api_key or config.openai_api_key)
        self.model = "gpt-4o"  # Use GPT-4o for structured outputs support
    
    def generate_search_queries(self, topic: str, max_queries: int = 15) -> Dict:
        """
        Generate comprehensive arXiv search queries for a given topic.
        
        Args:
            topic: The research topic to search for
            max_queries: Maximum number of queries to generate
            
        Returns:
            Dictionary with structured search information
        """
        prompt = self._build_prompt(topic, max_queries)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "arxiv_search_queries",
                        "strict": True,
                        "schema": self._get_json_schema()
                    }
                }
            )
            
            result = json.loads(response.choices[0].message.content)
            
            logger.info(f"Generated {len(result.get('search_queries', []))} queries for topic '{topic}'")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate queries for topic '{topic}': {e}")
            return self._get_fallback_queries(topic)
    
    def save_queries_config(self, queries_config: Dict, filepath: str) -> bool:
        """
        Save query configuration to a JSON file.
        
        Args:
            queries_config: The queries configuration dictionary
            filepath: Path to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = Path(filepath)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(queries_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved query configuration to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")
            return False
    
    def load_queries_config(self, filepath: str) -> Optional[Dict]:
        """
        Load query configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Loaded configuration dictionary or None if failed
        """
        try:
            config_path = Path(filepath)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {filepath}")
                return None
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            logger.info(f"Loaded query configuration from {filepath}")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            return None
    
    def _get_json_schema(self) -> Dict:
        """Get the JSON schema for structured outputs."""
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The original research topic"
                },
                "search_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The arXiv API query string with proper syntax"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Priority level of this query"
                            },
                            "description": {
                                "type": "string",
                                "description": "Human-readable description of what this query searches for"
                            }
                        },
                        "required": ["query", "priority", "description"],
                        "additionalProperties": False
                    },
                    "minItems": 5,
                    "maxItems": 20,
                    "description": "Array of search queries with metadata"
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Valid arXiv category code like cs.LG, math.OC, quant-ph, etc."
                    },
                    "minItems": 2,
                    "maxItems": 10,
                    "description": "Relevant arXiv subject categories"
                },
                "filter_keywords": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 2,
                        "description": "Keywords for relevance filtering"
                    },
                    "minItems": 3,
                    "maxItems": 15,
                    "description": "Keywords to use for paper relevance filtering"
                },
                "related_terms": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 2,
                        "description": "Related terms and synonyms"
                    },
                    "minItems": 3,
                    "maxItems": 15,
                    "description": "Related terms, synonyms, and concepts"
                }
            },
            "required": ["topic", "search_queries", "categories", "filter_keywords", "related_terms"],
            "additionalProperties": False
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT query generation."""
        return """You are an expert in scientific literature search, specifically for arXiv. Your task is to generate high-quality, diverse search queries that will comprehensively cover a research topic.

Key guidelines:
1. Use proper arXiv API query syntax (ti:, abs:, au:, cat:, all:, AND, OR)
2. Create queries of varying specificity - from broad to very specific
3. Include author-based queries for well-known researchers in the field
4. Use category filtering for relevant arXiv subject classes
5. Consider different terminologies and synonyms
6. Prioritize queries by expected relevance and coverage

Query syntax examples:
- ti:"exact phrase" - title contains exact phrase
- abs:"phrase" - abstract contains phrase  
- au:"Author Name" - specific author
- cat:cs.LG - category filtering
- all:"term" - anywhere in paper
- Combine with AND, OR: ti:"machine learning" AND cat:cs.LG

Generate diverse, comprehensive queries that will capture the full breadth of research in the given topic."""

    def _build_prompt(self, topic: str, max_queries: int) -> str:
        """Build the user prompt for query generation."""
        return f"""Generate comprehensive arXiv search queries for the research topic: "{topic}"

Requirements:
- Create {max_queries} diverse search queries with varying specificity
- Include queries targeting: titles, abstracts, authors, and specific categories
- Cover both broad and narrow aspects of the topic
- Use proper arXiv API syntax
- Prioritize queries as high/medium/low based on expected relevance
- Include relevant arXiv subject categories
- Provide filtering keywords for relevance assessment
- Add related terms and synonyms

Focus on creating queries that will discover the most relevant and comprehensive set of papers for someone researching "{topic}"."""

    def _get_fallback_queries(self, topic: str) -> Dict:
        """Get fallback queries when GPT fails."""
        return {
            "topic": topic,
            "search_queries": [
                {
                    "query": f'ti:"{topic}"',
                    "priority": "high",
                    "description": f"Papers with '{topic}' in title"
                },
                {
                    "query": f'abs:"{topic}"',
                    "priority": "medium", 
                    "description": f"Papers with '{topic}' in abstract"
                },
                {
                    "query": f'all:"{topic}"',
                    "priority": "low",
                    "description": f"Papers mentioning '{topic}' anywhere"
                }
            ],
            "categories": ["cs.LG", "stat.ML"],
            "filter_keywords": [topic.lower()],
            "related_terms": [topic.lower()]
        }