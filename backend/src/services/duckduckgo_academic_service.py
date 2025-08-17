#!/usr/bin/env python3
"""
DuckDuckGo Academic Search Service for ArXiv Recommendation System.

This service provides academic paper search functionality using DuckDuckGo
as a fallback when ArXiv searches return insufficient results. It focuses on
extracting ArXiv papers that may not have been found through direct ArXiv queries.

Key features:
- Academic-focused search patterns
- ArXiv link extraction and ID resolution
- Rate limiting and error handling
- Domain authority scoring for quality filtering
- Integration with existing ArXiv infrastructure
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    from ddgs.exceptions import (
        RatelimitException,
        TimeoutException,
        DDGSException as DuckDuckGoSearchException
    )
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    logger.warning("ddgs not available. Install with: uv add ddgs")
    DUCKDUCKGO_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class DDGS:
        pass
    class RatelimitException(Exception):
        pass
    class TimeoutException(Exception):
        pass
    class DuckDuckGoSearchException(Exception):
        pass


class AcademicDomain(Enum):
    """Academic domain classifications with authority scores."""
    ARXIV = {"domain": "arxiv.org", "authority": 10, "type": "preprint_repository"}
    GOOGLE_SCHOLAR = {"domain": "scholar.google.com", "authority": 9, "type": "citation_database"}
    RESEARCHGATE = {"domain": "researchgate.net", "authority": 8, "type": "academic_network"}
    IEEE = {"domain": "ieee.org", "authority": 9, "type": "publisher"}
    ACM = {"domain": "acm.org", "authority": 9, "type": "publisher"}
    SEMANTIC_SCHOLAR = {"domain": "semanticscholar.org", "authority": 8, "type": "citation_database"}
    PUBMED = {"domain": "pubmed.ncbi.nlm.nih.gov", "authority": 9, "type": "medical_database"}
    SPRINGER = {"domain": "springer.com", "authority": 7, "type": "publisher"}
    NATURE = {"domain": "nature.com", "authority": 10, "type": "publisher"}
    SCIENCE = {"domain": "science.org", "authority": 10, "type": "publisher"}


@dataclass
class AcademicSearchResult:
    """Structure for academic search results from DuckDuckGo."""
    title: str
    url: str
    snippet: str
    domain: str
    authority_score: int
    result_type: str  # 'arxiv_link', 'academic_pdf', 'citation_page'
    arxiv_id: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class SearchMetrics:
    """Metrics for tracking search performance."""
    total_queries: int = 0
    successful_queries: int = 0
    arxiv_links_found: int = 0
    academic_results_found: int = 0
    rate_limit_hits: int = 0
    timeout_errors: int = 0
    total_search_time: float = 0.0


class DuckDuckGoAcademicService:
    """
    Academic search service using DuckDuckGo with intelligent ArXiv link extraction.
    
    This service provides fallback search capabilities when ArXiv direct searches
    don't return sufficient results. It prioritizes finding ArXiv papers through
    alternative search strategies and academic domain filtering.
    """
    
    # Conservative rate limiting for DuckDuckGo
    RATE_LIMIT_DELAY = 5.0  # 5 seconds between requests
    MAX_RETRIES = 3
    BACKOFF_MULTIPLIER = 2.0
    DEFAULT_TIMEOUT = 15
    
    # ArXiv ID patterns (enhanced from existing arxiv_client.py patterns)
    ARXIV_PATTERNS = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf',
        r'(\d{4}\.\d{4,5}(?:v\d+)?)'  # Just the ID itself
    ]
    
    def __init__(self):
        """Initialize the DuckDuckGo academic search service."""
        if not DUCKDUCKGO_AVAILABLE:
            raise ImportError("DuckDuckGo search library not available. Install with: uv add ddgs")
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
        self.metrics = SearchMetrics()
        
        # Academic domain mapping for quick lookups
        self.domain_authority = {}
        for domain_enum in AcademicDomain:
            domain_info = domain_enum.value
            self.domain_authority[domain_info["domain"]] = domain_info["authority"]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT),
            connector=aiohttp.TCPConnector(limit=1)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    async def _rate_limit(self):
        """Enforce conservative rate limiting for DuckDuckGo requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.RATE_LIMIT_DELAY:
            delay = self.RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"DuckDuckGo rate limiting: waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def extract_arxiv_ids(self, text: str) -> List[str]:
        """
        Extract ArXiv IDs from text using enhanced patterns.
        
        Args:
            text: Text to search for ArXiv IDs
            
        Returns:
            List of clean ArXiv IDs found in the text
        """
        arxiv_ids = set()
        
        for pattern in self.ARXIV_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the ID (remove version if present)
                clean_id = re.sub(r'v\d+$', '', match)
                if self._is_valid_arxiv_id(clean_id):
                    arxiv_ids.add(clean_id)
        
        return list(arxiv_ids)
    
    def _is_valid_arxiv_id(self, arxiv_id: str) -> bool:
        """Validate ArXiv ID format."""
        # ArXiv IDs follow format: YYMM.NNNN or YYMM.NNNNN
        pattern = r'^\d{4}\.\d{4,5}$'
        return bool(re.match(pattern, arxiv_id))
    
    def _calculate_domain_authority(self, url: str) -> int:
        """Calculate domain authority score for academic relevance."""
        for domain, authority in self.domain_authority.items():
            if domain in url.lower():
                return authority
        
        # Check for other academic indicators
        academic_indicators = [
            'edu', 'ac.uk', 'ac.', 'university', 'institute', 'research',
            'journal', 'proceedings', 'conference'
        ]
        
        for indicator in academic_indicators:
            if indicator in url.lower():
                return 5  # Moderate academic authority
        
        return 1  # Low authority for general domains
    
    def _classify_result_type(self, result: Dict[str, str]) -> str:
        """Classify the type of academic search result."""
        url = result.get('href', '').lower()
        title = result.get('title', '').lower()
        
        # Check for ArXiv links
        if 'arxiv.org' in url:
            return 'arxiv_link'
        
        # Check for academic PDFs
        if url.endswith('.pdf') and any(domain in url for domain in self.domain_authority.keys()):
            return 'academic_pdf'
        
        # Check for citation pages
        if any(term in url for term in ['scholar.google', 'semanticscholar', 'researchgate']):
            return 'citation_page'
        
        # Check for academic publishers
        academic_publishers = ['ieee.org', 'acm.org', 'springer.com', 'nature.com', 'science.org']
        if any(publisher in url for publisher in academic_publishers):
            return 'publisher_page'
        
        return 'general_result'
    
    def _calculate_confidence_score(self, result: Dict[str, str], query: str) -> float:
        """Calculate confidence score for search result relevance."""
        title = result.get('title', '').lower()
        snippet = result.get('body', '').lower()
        url = result.get('href', '').lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Domain authority contributes 40%
        domain_authority = self._calculate_domain_authority(url)
        score += (domain_authority / 10.0) * 0.4
        
        # Query terms in title contribute 30%
        query_words = query_lower.split()
        title_matches = sum(1 for word in query_words if word in title)
        if query_words:
            score += (title_matches / len(query_words)) * 0.3
        
        # Query terms in snippet contribute 20%
        snippet_matches = sum(1 for word in query_words if word in snippet)
        if query_words:
            score += (snippet_matches / len(query_words)) * 0.2
        
        # Result type contributes 10%
        result_type = self._classify_result_type(result)
        type_scores = {
            'arxiv_link': 1.0,
            'academic_pdf': 0.8,
            'citation_page': 0.7,
            'publisher_page': 0.6,
            'general_result': 0.2
        }
        score += type_scores.get(result_type, 0.0) * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def search_academic_papers(
        self,
        query: str,
        max_results: int = 20,
        focus_on_arxiv: bool = True
    ) -> List[AcademicSearchResult]:
        """
        Search for academic papers using DuckDuckGo with academic-focused patterns.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            focus_on_arxiv: Whether to prioritize ArXiv results
            
        Returns:
            List of AcademicSearchResult objects
        """
        if not DUCKDUCKGO_AVAILABLE:
            logger.error("DuckDuckGo search not available")
            return []
        
        logger.info(f"🔍 DuckDuckGo academic search: {query}")
        search_start_time = asyncio.get_event_loop().time()
        
        try:
            await self._rate_limit()
            
            # Create single optimized search query  
            # Parse query to extract individual terms (no quoted phrases)
            query_terms = query.replace('"', '').split()
            optimized_query = ' '.join(query_terms) + ' arxiv'
            
            logger.info(f"🌐 Optimized DuckDuckGo Query: \"{optimized_query}\"")
            
            # Execute single DuckDuckGo search
            ddgs_results = await self._execute_ddgs_search(
                optimized_query, 
                max_results=max_results
            )
            
            # Log search results summary
            if ddgs_results:
                logger.info(f"📊 DuckDuckGo Results: {len(ddgs_results)} results found")
                # Log first few results for debugging
                for j, result in enumerate(ddgs_results[:3]):
                    title = result.get('title', 'No title')[:50]
                    url = result.get('href', 'No URL')
                    logger.debug(f"   {j+1}. {title}... → {url}")
            else:
                logger.info(f"📊 DuckDuckGo Results: No results for query '{optimized_query}'")
            
            # Process and filter results
            all_results = []
            seen_urls = set()
            
            for result in ddgs_results:
                url = result.get('href', '')
                
                # Skip duplicates
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Convert to AcademicSearchResult
                academic_result = self._convert_to_academic_result(result, query)
                
                # Filter by minimum quality threshold
                if academic_result.confidence_score >= 0.3:
                    all_results.append(academic_result)
            
            self.metrics.successful_queries += 1
            
            # Sort results by confidence score and authority
            all_results.sort(key=lambda x: (x.confidence_score, x.authority_score), reverse=True)
            
            # Limit final results
            final_results = all_results[:max_results]
            
            # Update metrics
            search_time = asyncio.get_event_loop().time() - search_start_time
            self.metrics.total_search_time += search_time
            self.metrics.total_queries += 1  # Single optimized query
            self.metrics.academic_results_found += len(final_results)
            self.metrics.arxiv_links_found += sum(1 for r in final_results if r.arxiv_id)
            
            logger.info(f"📊 DuckDuckGo search completed: {len(final_results)} results in {search_time:.2f}s")
            logger.info(f"📄 ArXiv papers found: {sum(1 for r in final_results if r.arxiv_id)}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"DuckDuckGo academic search failed: {e}")
            return []
    
    def _generate_academic_search_patterns(self, query: str, focus_on_arxiv: bool = True) -> List[str]:
        """Generate academic-focused search patterns for DuckDuckGo."""
        patterns = []
        
        if focus_on_arxiv:
            # ArXiv-focused patterns (highest priority)
            patterns.extend([
                f'"{query}" site:arxiv.org',
                f'{query} site:arxiv.org filetype:pdf',
                f'{query} arxiv.org/abs',
                f'{query} arxiv preprint'
            ])
        
        # Academic site patterns
        patterns.extend([
            f'"{query}" site:scholar.google.com',
            f'"{query}" site:researchgate.net filetype:pdf',
            f'"{query}" site:semanticscholar.org',
            f'{query} filetype:pdf (site:ieee.org OR site:acm.org)',
            f'"{query}" (site:nature.com OR site:science.org)'
        ])
        
        # General academic patterns
        patterns.extend([
            f'"{query}" filetype:pdf academic paper',
            f'{query} research paper PDF',
            f'"{query}" conference proceedings',
            f'{query} journal article PDF'
        ])
        
        return patterns
    
    async def _execute_ddgs_search(self, query: str, max_results: int = 30) -> List[Dict[str, str]]:
        """Execute a single DuckDuckGo search with comprehensive error handling."""
        retry_count = 0
        last_exception = None
        
        while retry_count < self.MAX_RETRIES:
            try:
                # Use conservative search parameters and disable problematic engines
                ddgs = DDGS(
                    timeout=self.DEFAULT_TIMEOUT,
                    verify=True
                )
                
                # Add site restriction to avoid Wikipedia and other problematic engines
                # Focus on academic sites and ArXiv
                academic_query = f"{query} (site:arxiv.org OR site:scholar.google.com OR site:researchgate.net OR site:semanticscholar.org OR site:ieee.org OR site:acm.org OR filetype:pdf)"
                
                logger.debug(f"🔍 Enhanced academic query: {academic_query}")
                
                results = ddgs.text(
                    academic_query,  # Use enhanced academic-focused query
                    region="wt-wt",  # No regional bias
                    safesearch="moderate",
                    timelimit=None,  # No time limit for academic papers
                    max_results=max_results
                )
                
                return results if results else []
                
            except RatelimitException as e:
                logger.warning(f"DuckDuckGo rate limit hit (attempt {retry_count + 1})")
                last_exception = e
                self.metrics.rate_limit_hits += 1
                
                # Exponential backoff for rate limits
                backoff_time = self.RATE_LIMIT_DELAY * (self.BACKOFF_MULTIPLIER ** retry_count)
                logger.info(f"Backing off for {backoff_time:.1f}s due to rate limit")
                await asyncio.sleep(backoff_time)
                
            except TimeoutException as e:
                logger.warning(f"DuckDuckGo search timeout (attempt {retry_count + 1})")
                last_exception = e
                self.metrics.timeout_errors += 1
                
                # Shorter backoff for timeouts
                backoff_time = 2.0 * (retry_count + 1)
                await asyncio.sleep(backoff_time)
                
            except DuckDuckGoSearchException as e:
                error_msg = str(e)
                logger.error(f"DuckDuckGo search exception (attempt {retry_count + 1}): {error_msg}")
                last_exception = e
                
                # Check for specific Wikipedia/DNS errors and try fallback
                if "wikipedia" in error_msg.lower() or "dns error" in error_msg.lower() or "failed to lookup" in error_msg.lower():
                    logger.warning("⚠️  Wikipedia/DNS error detected, trying fallback search without site restrictions")
                    
                    # Try a simpler query without site restrictions as fallback
                    try:
                        fallback_results = ddgs.text(
                            query,  # Use original query without site restrictions
                            region="wt-wt",
                            safesearch="moderate",
                            timelimit=None,
                            max_results=max_results
                        )
                        logger.info(f"✅ Fallback search succeeded with {len(fallback_results) if fallback_results else 0} results")
                        return fallback_results if fallback_results else []
                    except Exception as fallback_e:
                        logger.error(f"❌ Fallback search also failed: {fallback_e}")
                
                # Don't retry on general DuckDuckGo exceptions
                break
                
            except Exception as e:
                logger.error(f"Unexpected DuckDuckGo search error (attempt {retry_count + 1}): {e}")
                last_exception = e
                
                # Brief delay before retry
                await asyncio.sleep(1.0)
                
            retry_count += 1
        
        # All retries exhausted
        logger.error(f"DuckDuckGo search failed after {self.MAX_RETRIES} attempts. Last error: {last_exception}")
        
        # Re-raise specific exceptions for upstream handling
        if isinstance(last_exception, (RatelimitException, TimeoutException)):
            raise last_exception
        
        return []
    
    def _convert_to_academic_result(self, ddg_result: Dict[str, str], query: str) -> AcademicSearchResult:
        """Convert DuckDuckGo result to AcademicSearchResult."""
        title = ddg_result.get('title', '')
        url = ddg_result.get('href', '')
        snippet = ddg_result.get('body', '')
        
        # Extract domain
        domain = ''
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
        except:
            domain = url.split('/')[2] if '/' in url else ''
        
        # Calculate scores
        authority_score = self._calculate_domain_authority(url)
        confidence_score = self._calculate_confidence_score(ddg_result, query)
        result_type = self._classify_result_type(ddg_result)
        
        # Extract ArXiv ID if present
        arxiv_id = None
        arxiv_ids = self.extract_arxiv_ids(f"{title} {url} {snippet}")
        if arxiv_ids:
            arxiv_id = arxiv_ids[0]  # Take the first valid ArXiv ID
        
        return AcademicSearchResult(
            title=title,
            url=url,
            snippet=snippet,
            domain=domain,
            authority_score=authority_score,
            result_type=result_type,
            arxiv_id=arxiv_id,
            confidence_score=confidence_score
        )
    
    async def extract_arxiv_papers_from_search(
        self,
        query: str,
        max_results: int = 15
    ) -> List[str]:
        """
        Search for ArXiv papers using DuckDuckGo and extract ArXiv IDs.
        
        This method is optimized for finding ArXiv papers that may not appear
        in direct ArXiv API searches due to different indexing or query processing.
        
        Args:
            query: Search query string
            max_results: Maximum number of ArXiv IDs to return
            
        Returns:
            List of ArXiv IDs found through DuckDuckGo search
        """
        logger.info(f"🎯 Extracting ArXiv papers via DuckDuckGo: {query}")
        
        # Search with ArXiv focus
        search_results = await self.search_academic_papers(
            query=query,
            max_results=max_results * 3,  # Search more to find ArXiv papers
            focus_on_arxiv=True
        )
        
        # Extract ArXiv IDs from results
        arxiv_ids = []
        for result in search_results:
            if result.arxiv_id and result.arxiv_id not in arxiv_ids:
                arxiv_ids.append(result.arxiv_id)
                
                if len(arxiv_ids) >= max_results:
                    break
        
        logger.info(f"📄 Found {len(arxiv_ids)} ArXiv papers via DuckDuckGo")
        return arxiv_ids
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search metrics."""
        return {
            "total_queries": self.metrics.total_queries,
            "successful_queries": self.metrics.successful_queries,
            "success_rate": (
                self.metrics.successful_queries / self.metrics.total_queries * 100
                if self.metrics.total_queries > 0 else 0
            ),
            "arxiv_links_found": self.metrics.arxiv_links_found,
            "academic_results_found": self.metrics.academic_results_found,
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "timeout_errors": self.metrics.timeout_errors,
            "average_search_time": (
                self.metrics.total_search_time / self.metrics.successful_queries
                if self.metrics.successful_queries > 0 else 0
            ),
            "total_search_time": self.metrics.total_search_time
        }
    
    def reset_metrics(self):
        """Reset search metrics."""
        self.metrics = SearchMetrics()


# Convenience functions for integration
async def search_arxiv_papers_via_duckduckgo(query: str, max_results: int = 15) -> List[str]:
    """
    Convenience function to search for ArXiv papers using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of ArXiv IDs to return
        
    Returns:
        List of ArXiv IDs found through DuckDuckGo search
    """
    try:
        async with DuckDuckGoAcademicService() as service:
            return await service.extract_arxiv_papers_from_search(query, max_results)
    except ImportError:
        logger.error("DuckDuckGo search not available. Install with: uv add ddgs")
        return []
    except Exception as e:
        logger.error(f"DuckDuckGo ArXiv search failed: {e}")
        return []


async def search_academic_papers_via_duckduckgo(
    query: str, 
    max_results: int = 20
) -> List[AcademicSearchResult]:
    """
    Convenience function to search for academic papers using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of AcademicSearchResult objects
    """
    try:
        async with DuckDuckGoAcademicService() as service:
            return await service.search_academic_papers(query, max_results)
    except ImportError:
        logger.error("DuckDuckGo search not available. Install with: uv add ddgs")
        return []
    except Exception as e:
        logger.error(f"DuckDuckGo academic search failed: {e}")
        return []