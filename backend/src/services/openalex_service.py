#!/usr/bin/env python3
"""
OpenAlex Service for ArXiv Recommendation System.

This service provides integration with the OpenAlex API for:
- Paper metadata enrichment
- Citation tracking and analysis
- Cross-database reference linking
- Topic classification
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenAlexService:
    """Service for interacting with OpenAlex API"""
    
    def __init__(self, email: Optional[str] = None):
        """Initialize OpenAlex service with polite pool access."""
        try:
            import pyalex
            from pyalex import Works, Authors
            
            # Configure polite pool for faster access
            if email:
                pyalex.config.email = email
                logger.info(f"OpenAlex configured with email: {email}")
            
            self.Works = Works
            self.Authors = Authors
            
        except ImportError:
            logger.error("PyAlex not installed. Install with: pip install pyalex")
            raise
    
    async def check_arxiv_availability(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if ArXiv paper is indexed in OpenAlex.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            Dict with OpenAlex metadata or None if not found
        """
        try:
            # Query OpenAlex by ArXiv ID
            work = self.Works()[f"arxiv:{arxiv_id}"]
            
            if not work:
                return None
                
            return {
                "openalex_id": work.id,
                "title": work.title,
                "cited_by_count": work.cited_by_count or 0,
                "referenced_works": work.referenced_works or [],
                "topics": work.topics or [],
                "publication_date": work.publication_date,
                "created_date": work.created_date,
                "authorships": work.authorships or [],
                "abstract": getattr(work, 'abstract', None),
                "doi": work.doi,
                "is_retracted": work.is_retracted,
                "is_paratext": work.is_paratext,
                "open_access": work.open_access
            }
            
        except Exception as e:
            logger.debug(f"ArXiv paper {arxiv_id} not found in OpenAlex: {e}")
            return None
    
    async def get_paper_references(self, openalex_id: str) -> List[Dict[str, Any]]:
        """
        Get references for a paper from OpenAlex.
        
        Args:
            openalex_id: OpenAlex work ID
            
        Returns:
            List of reference dictionaries
        """
        try:
            work = self.Works()[openalex_id]
            
            if not work or not work.referenced_works:
                return []
            
            references = []
            
            # Get details for each referenced work
            for i, ref_id in enumerate(work.referenced_works[:100]):  # Limit to 100 refs
                try:
                    ref_work = self.Works()[ref_id]
                    
                    # Extract ArXiv ID if available
                    arxiv_id = None
                    if ref_work.ids and ref_work.ids.get('openalex'):
                        # Look for ArXiv ID in external IDs
                        for external_id in (ref_work.ids.get('external', []) or []):
                            if 'arxiv' in external_id.lower():
                                arxiv_id = external_id.split(':')[-1]
                                break
                    
                    reference = {
                        "openalex_work_id": ref_work.id,
                        "cited_paper_id": arxiv_id,
                        "cited_title": ref_work.title,
                        "cited_authors": self._format_authors(ref_work.authorships),
                        "cited_year": ref_work.publication_year,
                        "reference_context": f"Reference {i+1} from OpenAlex",
                        "citation_number": i + 1,
                        "is_arxiv_paper": bool(arxiv_id),
                        "source": "openalex",
                        "confidence_score": 0.95  # High confidence for OpenAlex data
                    }
                    
                    references.append(reference)
                    
                except Exception as e:
                    logger.debug(f"Failed to get reference details for {ref_id}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(references)} references from OpenAlex for {openalex_id}")
            return references
            
        except Exception as e:
            logger.error(f"Failed to get references for {openalex_id}: {e}")
            return []
    
    async def get_citation_network(self, openalex_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get papers citing and cited by this work.
        
        Args:
            openalex_id: OpenAlex work ID
            depth: Citation network depth (1 = direct citations only)
            
        Returns:
            Dict with citing and cited papers
        """
        try:
            work = self.Works()[openalex_id]
            
            # Get citing papers (limited to recent ones)
            citing_papers = []
            try:
                citing_works = self.Works().filter(
                    cites=openalex_id,
                    from_publication_date="2020-01-01"  # Recent papers only
                ).get()[:50]  # Limit to 50 citing papers
                
                for citing_work in citing_works:
                    citing_papers.append({
                        "openalex_id": citing_work.id,
                        "title": citing_work.title,
                        "authors": self._format_authors(citing_work.authorships),
                        "publication_year": citing_work.publication_year,
                        "cited_by_count": citing_work.cited_by_count or 0,
                        "topics": citing_work.topics or []
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to get citing papers for {openalex_id}: {e}")
            
            # Get cited papers (references)
            cited_papers = []
            if work.referenced_works:
                for ref_id in work.referenced_works[:20]:  # Limit to 20 cited papers
                    try:
                        ref_work = self.Works()[ref_id]
                        cited_papers.append({
                            "openalex_id": ref_work.id,
                            "title": ref_work.title,
                            "authors": self._format_authors(ref_work.authorships),
                            "publication_year": ref_work.publication_year,
                            "cited_by_count": ref_work.cited_by_count or 0,
                            "topics": ref_work.topics or []
                        })
                    except Exception as e:
                        logger.debug(f"Failed to get cited paper details for {ref_id}: {e}")
                        continue
            
            return {
                "paper": {
                    "openalex_id": work.id,
                    "title": work.title,
                    "cited_by_count": work.cited_by_count or 0
                },
                "citing_papers": citing_papers,
                "cited_papers": cited_papers,
                "citation_count": work.cited_by_count or 0,
                "reference_count": len(work.referenced_works) if work.referenced_works else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get citation network for {openalex_id}: {e}")
            return {
                "paper": None,
                "citing_papers": [],
                "cited_papers": [],
                "citation_count": 0,
                "reference_count": 0
            }
    
    async def find_similar_papers(self, openalex_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find papers with similar topics/concepts.
        
        Args:
            openalex_id: OpenAlex work ID
            limit: Maximum number of similar papers to return
            
        Returns:
            List of similar papers
        """
        try:
            work = self.Works()[openalex_id]
            
            if not work or not work.topics:
                return []
            
            # Get the primary topic
            primary_topic = work.topics[0] if work.topics else None
            if not primary_topic:
                return []
            
            # Search for papers with similar topics
            similar_works = self.Works().filter(
                topics={"id": primary_topic["id"]},
                from_publication_date="2020-01-01",  # Recent papers
                cited_by_count=">5"  # Well-cited papers
            ).get()[:limit*2]  # Get more to filter out the original
            
            similar_papers = []
            for similar_work in similar_works:
                # Skip the original paper
                if similar_work.id == openalex_id:
                    continue
                    
                # Extract ArXiv ID if available
                arxiv_id = None
                if hasattr(similar_work, 'ids') and similar_work.ids:
                    for external_id in (similar_work.ids.get('external', []) or []):
                        if 'arxiv' in str(external_id).lower():
                            arxiv_id = str(external_id).split(':')[-1]
                            break
                
                similar_papers.append({
                    "openalex_id": similar_work.id,
                    "arxiv_id": arxiv_id,
                    "title": similar_work.title,
                    "authors": self._format_authors(similar_work.authorships),
                    "publication_year": similar_work.publication_year,
                    "cited_by_count": similar_work.cited_by_count or 0,
                    "topics": similar_work.topics or [],
                    "similarity_reason": f"Shared topic: {primary_topic.get('display_name', 'Unknown')}"
                })
                
                if len(similar_papers) >= limit:
                    break
            
            logger.info(f"Found {len(similar_papers)} similar papers for {openalex_id}")
            return similar_papers
            
        except Exception as e:
            logger.error(f"Failed to find similar papers for {openalex_id}: {e}")
            return []
    
    async def batch_check_papers(self, arxiv_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Check multiple ArXiv papers for OpenAlex availability.
        
        Args:
            arxiv_ids: List of ArXiv paper IDs
            
        Returns:
            Dict mapping arxiv_id to OpenAlex data (or None if not found)
        """
        results = {}
        
        for arxiv_id in arxiv_ids:
            results[arxiv_id] = await self.check_arxiv_availability(arxiv_id)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        logger.info(f"Batch checked {len(arxiv_ids)} papers, found {sum(1 for r in results.values() if r)} in OpenAlex")
        return results
    
    def _format_authors(self, authorships: List[Dict]) -> str:
        """Format author list from OpenAlex authorship data."""
        if not authorships:
            return ""
        
        authors = []
        for authorship in authorships[:5]:  # Limit to first 5 authors
            author = authorship.get('author', {})
            display_name = author.get('display_name', 'Unknown Author')
            authors.append(display_name)
        
        if len(authorships) > 5:
            authors.append("et al.")
        
        return ", ".join(authors)


# Convenience function for quick checks
async def check_arxiv_in_openalex(arxiv_id: str, email: Optional[str] = None) -> Optional[Dict]:
    """
    Quick check if an ArXiv paper is in OpenAlex.
    
    Args:
        arxiv_id: ArXiv paper ID
        email: Email for polite pool access
        
    Returns:
        OpenAlex data or None
    """
    service = OpenAlexService(email=email)
    return await service.check_arxiv_availability(arxiv_id)