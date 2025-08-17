#!/usr/bin/env python3
"""
Gemini Query Service for ArXiv Recommendation System.

This service provides Gemini-powered query generation functionality
that can be used by both API endpoints and CLI scripts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from google import genai

from config import config

logger = logging.getLogger(__name__)


class GeminiQueryService:
    """Service for generating arXiv search queries using Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini query service."""
        self.api_key = api_key or config.gemini_api_key
        self.model = config.gemini_query_model
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
            
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
    
    def generate_search_queries(
        self, 
        topic: str, 
        max_queries: int = 15,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive arXiv search queries for a given topic.
        
        Args:
            topic: The research topic to search for
            max_queries: Maximum number of queries to generate
            date_from: Start date for papers (YYYY-MM-DD format)
            date_to: End date for papers (YYYY-MM-DD format)
            
        Returns:
            Dictionary with structured search information
        """
        prompt = self._build_prompt(topic, max_queries, date_from, date_to)
        
        try:
            # Use the new Gemini API with structured outputs
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": f"{self._get_system_prompt()}\n\n{prompt}"}]
                    }
                ],
                config=genai.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent structured output
                    top_k=40,
                    top_p=0.8,
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                    response_schema=self._get_json_schema()
                )
            )
            
            # Parse the JSON response
            result = json.loads(response.text)
            
            # Validate the response structure
            if not self._validate_response(result):
                logger.warning(f"Invalid response structure from Gemini for topic '{topic}'")
                return self._get_fallback_queries(topic)
            
            logger.info(f"Generated {len(result.get('search_queries', []))} queries for topic '{topic}' using Gemini")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Gemini for topic '{topic}': {e}")
            return self._get_fallback_queries(topic)
        except Exception as e:
            logger.error(f"Failed to generate queries for topic '{topic}' using Gemini: {e}")
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
                        "required": ["query", "priority", "description"]
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
            "required": ["topic", "search_queries", "categories", "filter_keywords", "related_terms"]
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Gemini query generation."""
        return """You are an expert in scientific literature search, specifically for arXiv. Your task is to generate high-quality, diverse search queries that will comprehensively cover a research topic.

Key guidelines:
1. Use proper arXiv API query syntax (ti:, abs:, au:, cat:, all:, AND, OR)
2. AVOID exact phrase matching - search for individual words instead
3. DO NOT include year constraints in queries
4. Create queries of varying specificity - from broad to very specific
5. Include author-based queries for well-known researchers in the field
6. Use category filtering for relevant arXiv subject classes
7. Consider different terminologies and synonyms
8. Prioritize queries by expected relevance and coverage

Query syntax examples:
- ti:machine AND ti:learning (NOT ti:"machine learning")
- abs:neural AND abs:network (NOT abs:"neural networks")  
- au:Smith (for author searches)
- cat:cs.LG - category filtering
- all:quantum AND all:computing (NOT all:"quantum computing")
- Combine with AND, OR: ti:deep AND ti:learning AND cat:cs.LG

IMPORTANT: Never use quotes for phrase matching. Always break phrases into individual word searches with AND operators.

Generate diverse, comprehensive queries that will capture the full breadth of research in the given topic.

IMPORTANT: Return only valid JSON that matches the specified schema. Do not include any additional text or explanations outside the JSON response."""

    def _build_prompt(
        self, 
        topic: str, 
        max_queries: int,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> str:
        """Build the user prompt for query generation."""
        
        # Build date constraint text
        date_constraint = ""
        if date_from or date_to:
            date_constraint = "\n\nDate Constraints:\n"
            if date_from and date_to:
                # Convert YYYY-MM-DD to YYYYMMDD format for arXiv
                from_formatted = date_from.replace('-', '')
                to_formatted = date_to.replace('-', '')
                date_constraint += f"- Include date filtering in queries using: submittedDate:[{from_formatted} TO {to_formatted}]\n"
                date_constraint += f"- Apply date range: {date_from} to {date_to}\n"
            elif date_from:
                from_formatted = date_from.replace('-', '')
                date_constraint += f"- Include date filtering for papers from {date_from} onwards: submittedDate:[{from_formatted} TO *]\n"
            elif date_to:
                to_formatted = date_to.replace('-', '')
                date_constraint += f"- Include date filtering for papers up to {date_to}: submittedDate:[* TO {to_formatted}]\n"
            
            date_constraint += "- Combine date filters with topic queries using AND operator\n"
            date_constraint += "- Example: ti:\"machine learning\" AND submittedDate:[20240101 TO 20241231]\n"
        
        return f"""Generate comprehensive arXiv search queries for the research topic: "{topic}"

Requirements:
- Create {max_queries} diverse search queries with varying specificity
- Include queries targeting: titles, abstracts, authors, and specific categories
- Cover both broad and narrow aspects of the topic
- Use proper arXiv API syntax
- Prioritize queries as high/medium/low based on expected relevance
- Include relevant arXiv subject categories
- Provide filtering keywords for relevance assessment
- Add related terms and synonyms{date_constraint}

Focus on creating queries that will discover the most relevant and comprehensive set of papers for someone researching "{topic}".

Return the response as valid JSON matching the specified schema."""

    def _validate_response(self, response: Dict) -> bool:
        """Validate that the response has the expected structure."""
        required_fields = ["topic", "search_queries", "categories", "filter_keywords", "related_terms"]
        
        # Check all required fields are present
        for field in required_fields:
            if field not in response:
                return False
        
        # Check search_queries structure
        if not isinstance(response["search_queries"], list) or len(response["search_queries"]) < 3:
            return False
            
        for query in response["search_queries"]:
            if not isinstance(query, dict):
                return False
            if not all(field in query for field in ["query", "priority", "description"]):
                return False
            if query["priority"] not in ["high", "medium", "low"]:
                return False
        
        # Check other arrays
        for field in ["categories", "filter_keywords", "related_terms"]:
            if not isinstance(response[field], list) or len(response[field]) < 1:
                return False
        
        return True

    def _get_fallback_queries(self, topic: str) -> Dict:
        """Get fallback queries when Gemini fails."""
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