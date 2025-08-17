#!/usr/bin/env python3
"""
Query Refinement Service for ArXiv Recommendation System.

This service refines ArXiv search queries using GPT-5-nano when initial
searches return zero results, providing intelligent query reformulation.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


class QueryRefinementService:
    """
    Service for refining ArXiv search queries using GPT-5-nano.

    Features:
    - Cost-effective query refinement using GPT-5-nano
    - Generates 2-3 refined query variations
    - Maintains refinement history for analytics
    - Intelligent trigger based on search results
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the query refinement service."""
        self.client = OpenAI(api_key=api_key or config.openai_api_key)
        self.model = "gpt-4o-nano"  # Ultra cost-effective for refinement tasks

    def should_refine(self, search_results: List, min_results: int = 1) -> bool:
        """
        Determine if query refinement is needed.

        Args:
            search_results: List of papers from initial search
            min_results: Minimum results threshold

        Returns:
            True if refinement should be attempted
        """
        return len(search_results) < min_results

    async def refine_query(
        self, original_query: str, topic_context: str, max_variations: int = 3
    ) -> Dict:
        """
        Generate refined query variations using GPT-5-nano.

        Args:
            original_query: The original ArXiv search query that failed
            topic_context: Context about the research topic
            max_variations: Maximum number of query variations to generate

        Returns:
            Dictionary with refined queries and metadata
        """
        try:
            prompt = self._build_refinement_prompt(
                original_query, topic_context, max_variations
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query_refinement",
                        "strict": True,
                        "schema": self._get_refinement_schema(),
                    },
                },
                temperature=0.7,  # Slightly higher for creative refinement
            )

            result = json.loads(response.choices[0].message.content)

            # Add metadata
            result["original_query"] = original_query
            result["topic_context"] = topic_context
            result["refinement_timestamp"] = datetime.now().isoformat()
            result["model_used"] = self.model

            logger.info(
                f"🔄 Query refinement generated {len(result.get('refined_queries', []))} variations for: {original_query[:50]}..."
            )

            return result

        except Exception as e:
            logger.error(f"Query refinement failed for '{original_query}': {e}")
            return self._get_fallback_refinement(original_query, topic_context)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for query refinement."""
        return """You are an expert in ArXiv query syntax correction. Your task is to make MINIMAL changes to fix ArXiv search queries that returned zero results.

CRITICAL Rules:
1. Make only minimal syntax corrections (typos, punctuation, spacing)
2. Remove quotes around titles - search for individual words instead
3. Remove year constraints from queries 
4. Use ti: field for title word searches (not exact phrases)
5. Keep the core search intent unchanged

Common fixes:
- Fix spelling errors in technical terms
- Remove quotes: ti:"deep learning" → ti:deep AND ti:learning
- Remove years: "2023" → remove entirely
- Fix spacing: ti: "term" → ti:term
- Simple boolean fixes: AND/OR corrections

AVOID:
- Major strategy changes
- Adding new concepts or synonyms
- Changing search scope dramatically
- Complex query restructuring"""

    def _build_refinement_prompt(
        self, original_query: str, topic_context: str, max_variations: int
    ) -> str:
        """Build the refinement prompt."""
        return f"""Fix this ArXiv search query that returned zero results with MINIMAL changes:

Original Query: {original_query}
Topic Context: {topic_context}

Make only small syntax corrections. Generate {max_variations} minimal fixes:

1. **Syntax Fix**: Correct spelling, spacing, punctuation errors
2. **Remove Quotes**: Change ti:"phrase" to individual words: ti:word1 AND ti:word2  
3. **Remove Years**: Delete any year constraints (2023, 2024, etc.)

For each fix:
- Make the smallest possible change
- Keep the original search intent
- Focus on syntax and formatting issues
- Don't add new concepts or synonyms

Example fixes:
- ti:"deep leraning" → ti:deep AND ti:learning (fix typo + remove quotes)
- ti:"neural networks 2023" → ti:neural AND ti:networks (remove quotes + year)
- ti: quantum → ti:quantum (fix spacing)"""

    def _get_refinement_schema(self) -> Dict:
        """Get the JSON schema for refinement results."""
        return {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Analysis of why the original query might have failed",
                },
                "refined_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The refined ArXiv search query",
                            },
                            "strategy": {
                                "type": "string",
                                "enum": [
                                    "syntax_fix",
                                    "remove_quotes",
                                    "remove_years",
                                    "spacing_fix",
                                ],
                                "description": "The minimal refinement strategy used",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Why this refinement might work better",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence that this query will find results",
                            },
                        },
                        "required": ["query", "strategy", "explanation", "confidence"],
                        "additionalProperties": False,
                    },
                    "minItems": 2,
                    "maxItems": 5,
                    "description": "Array of refined query variations",
                },
            },
            "required": ["analysis", "refined_queries"],
            "additionalProperties": False,
        }

    def _get_fallback_refinement(self, original_query: str, topic_context: str) -> Dict:
        """Get fallback refinement when GPT-5-nano fails."""
        # Simple fallback strategies
        fallback_queries = []

        # Extract potential keywords from original query
        keywords = []
        if '"' in original_query:
            # Extract quoted phrases
            import re

            quoted_phrases = re.findall(r'"([^"]*)"', original_query)
            keywords.extend(quoted_phrases)

        # Add topic context as keyword
        if topic_context:
            keywords.append(topic_context.lower())

        # Generate simple variations
        for i, keyword in enumerate(keywords[:3]):
            if keyword:
                fallback_queries.append(
                    {
                        "query": f'ti:{keyword.replace(" ", " AND ti:")}',
                        "strategy": "remove_quotes", 
                        "explanation": f"Remove quotes and search title for individual words: {keyword}",
                        "confidence": 0.6,
                    }
                )

        # Ensure we have at least one fallback
        if not fallback_queries:
            topic_words = (topic_context or "research").split()
            if len(topic_words) > 1:
                query = "ti:" + " AND ti:".join(topic_words)
            else:
                query = f"ti:{topic_words[0] if topic_words else 'research'}"
                
            fallback_queries.append(
                {
                    "query": query,
                    "strategy": "remove_quotes",
                    "explanation": "Search title for individual words without quotes",
                    "confidence": 0.3,
                }
            )

        return {
            "analysis": "Automatic fallback refinement due to GPT-5-nano failure",
            "refined_queries": fallback_queries,
            "original_query": original_query,
            "topic_context": topic_context,
            "refinement_timestamp": datetime.now().isoformat(),
            "model_used": "fallback",
        }


# Convenience function for single-use refinement
async def refine_failed_query(
    original_query: str, topic_context: str, max_variations: int = 3
) -> Dict:
    """
    Convenience function to refine a single failed query.

    Args:
        original_query: The original query that failed
        topic_context: Context about the research topic
        max_variations: Maximum variations to generate

    Returns:
        Refinement result dictionary
    """
    service = QueryRefinementService()
    return await service.refine_query(original_query, topic_context, max_variations)
