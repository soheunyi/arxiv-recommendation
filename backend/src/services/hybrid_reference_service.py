#!/usr/bin/env python3
"""
Hybrid Reference Service for ArXiv Recommendation System.

This service implements a two-stage reference fetching approach:
1. Stage 1: Immediate ArXiv HTML parsing for quick results
2. Stage 2: OpenAlex enrichment for comprehensive metadata

This provides the best of both worlds: immediate access to new papers
and rich citation data when available.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from .reference_service import ReferenceService
from .openalex_service import OpenAlexService
from database import DatabaseManager

logger = logging.getLogger(__name__)


class HybridReferenceService:
    """
    Two-stage reference fetching service combining ArXiv and OpenAlex data.
    """
    
    def __init__(self, email: Optional[str] = None):
        """Initialize hybrid service with both ArXiv and OpenAlex capabilities."""
        self.db_manager = DatabaseManager()
        self.openalex_service = OpenAlexService(email=email)
        
    async def fetch_references_stage1(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Stage 1: Immediate ArXiv HTML parsing for quick reference extraction.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            List of references from ArXiv HTML parsing
        """
        try:
            logger.info(f"Stage 1: Fetching references from ArXiv HTML for {arxiv_id}")
            
            # Use existing ArXiv HTML parser
            async with ReferenceService() as ref_service:
                references = await ref_service.fetch_and_parse_references(arxiv_id)
            
            if references:
                # Store with source marked as 'arxiv'
                await self._store_references_with_source(arxiv_id, references, source='arxiv')
                logger.info(f"Stage 1: Stored {len(references)} references from ArXiv for {arxiv_id}")
            else:
                logger.warning(f"Stage 1: No references found in ArXiv HTML for {arxiv_id}")
            
            return references
            
        except Exception as e:
            logger.error(f"Stage 1 failed for {arxiv_id}: {e}")
            return []
    
    async def enrich_with_openalex(self, arxiv_id: str) -> bool:
        """
        Stage 2: Enrich paper and references with OpenAlex data when available.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            True if enrichment was successful, False otherwise
        """
        try:
            logger.info(f"Stage 2: Attempting OpenAlex enrichment for {arxiv_id}")
            
            # Check if paper exists in OpenAlex
            openalex_data = await self.openalex_service.check_arxiv_availability(arxiv_id)
            
            if not openalex_data:
                logger.info(f"Stage 2: Paper {arxiv_id} not yet indexed in OpenAlex")
                await self.db_manager.update_openalex_check_time(arxiv_id)
                return False
            
            # Update paper metadata with OpenAlex data
            success = await self.db_manager.update_paper_openalex_data(
                paper_id=arxiv_id,
                openalex_id=openalex_data['openalex_id'],
                citation_count=openalex_data['cited_by_count'],
                topics=openalex_data.get('topics')
            )
            
            if not success:
                logger.error(f"Stage 2: Failed to update paper metadata for {arxiv_id}")
                return False
            
            # Get references from OpenAlex
            openalex_refs = await self.openalex_service.get_paper_references(
                openalex_data['openalex_id']
            )
            
            if openalex_refs:
                # Enrich existing references or add new ones
                await self._enrich_references(arxiv_id, openalex_refs)
                logger.info(f"Stage 2: Enriched with {len(openalex_refs)} OpenAlex references for {arxiv_id}")
            
            logger.info(f"Stage 2: Successfully enriched {arxiv_id} with OpenAlex data")
            return True
            
        except Exception as e:
            logger.error(f"Stage 2 enrichment failed for {arxiv_id}: {e}")
            await self.db_manager.update_openalex_check_time(arxiv_id)
            return False
    
    async def fetch_references_hybrid(self, arxiv_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Complete hybrid reference fetching with both stages.
        
        Args:
            arxiv_id: ArXiv paper ID
            force_refresh: Force re-fetching even if data exists
            
        Returns:
            Dict with fetching results and metadata
        """
        result = {
            "arxiv_id": arxiv_id,
            "stage1_success": False,
            "stage2_success": False,
            "references_found": 0,
            "source": "none",
            "openalex_available": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Stage 1: ArXiv HTML parsing
            arxiv_refs = await self.fetch_references_stage1(arxiv_id)
            
            if arxiv_refs:
                result["stage1_success"] = True
                result["references_found"] = len(arxiv_refs)
                result["source"] = "arxiv"
            
            # Stage 2: OpenAlex enrichment
            openalex_success = await self.enrich_with_openalex(arxiv_id)
            
            if openalex_success:
                result["stage2_success"] = True
                result["openalex_available"] = True
                result["source"] = "hybrid" if result["stage1_success"] else "openalex"
                
                # Get updated reference count after enrichment
                refs = await self.db_manager.get_paper_references(arxiv_id)
                result["references_found"] = len(refs)
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid reference fetching failed for {arxiv_id}: {e}")
            result["error"] = str(e)
            return result
    
    async def get_enhanced_references(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Get all references for a paper, with OpenAlex enrichment if available.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            List of enhanced reference dictionaries
        """
        try:
            references = await self.db_manager.get_paper_references(arxiv_id)
            
            # Enhance each reference with additional metadata
            enhanced_refs = []
            for ref in references:
                enhanced_ref = dict(ref)
                
                # Add source indicator
                enhanced_ref["has_openalex_data"] = bool(ref.get("openalex_work_id"))
                enhanced_ref["confidence"] = ref.get("confidence_score", 0.7)
                
                # Add citation metrics if available from OpenAlex
                if ref.get("openalex_work_id"):
                    try:
                        citation_data = await self._get_citation_metrics(ref["openalex_work_id"])
                        enhanced_ref.update(citation_data)
                    except Exception as e:
                        logger.debug(f"Failed to get citation metrics for {ref['openalex_work_id']}: {e}")
                
                enhanced_refs.append(enhanced_ref)
            
            return enhanced_refs
            
        except Exception as e:
            logger.error(f"Failed to get enhanced references for {arxiv_id}: {e}")
            return []
    
    async def get_citation_network(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get citation network for a paper (requires OpenAlex data).
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            Citation network data or empty dict if not available
        """
        try:
            openalex_id = await self.db_manager.get_openalex_id(arxiv_id)
            
            if not openalex_id:
                return {
                    "error": "Paper not yet indexed in OpenAlex",
                    "arxiv_id": arxiv_id,
                    "openalex_available": False
                }
            
            network = await self.openalex_service.get_citation_network(openalex_id)
            network["arxiv_id"] = arxiv_id
            network["openalex_available"] = True
            
            return network
            
        except Exception as e:
            logger.error(f"Failed to get citation network for {arxiv_id}: {e}")
            return {
                "error": str(e),
                "arxiv_id": arxiv_id,
                "openalex_available": False
            }
    
    async def find_similar_papers(self, arxiv_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find papers similar to the given paper (requires OpenAlex data).
        
        Args:
            arxiv_id: ArXiv paper ID
            limit: Maximum number of similar papers
            
        Returns:
            List of similar papers or empty list if not available
        """
        try:
            openalex_id = await self.db_manager.get_openalex_id(arxiv_id)
            
            if not openalex_id:
                logger.info(f"Paper {arxiv_id} not yet in OpenAlex for similarity search")
                return []
            
            similar = await self.openalex_service.find_similar_papers(openalex_id, limit)
            return similar
            
        except Exception as e:
            logger.error(f"Failed to find similar papers for {arxiv_id}: {e}")
            return []
    
    async def batch_enrich_papers(self, arxiv_ids: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """
        Batch enrich multiple papers with OpenAlex data.
        
        Args:
            arxiv_ids: List of ArXiv paper IDs
            batch_size: Number of papers to process in parallel
            
        Returns:
            Summary of enrichment results
        """
        results = {
            "total_papers": len(arxiv_ids),
            "enriched_count": 0,
            "not_found_count": 0,
            "error_count": 0,
            "enriched_papers": [],
            "not_found_papers": [],
            "errors": []
        }
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i + batch_size]
            batch_tasks = []
            
            for arxiv_id in batch:
                task = self.enrich_with_openalex(arxiv_id)
                batch_tasks.append((arxiv_id, task))
            
            # Execute batch
            for arxiv_id, task in batch_tasks:
                try:
                    success = await task
                    if success:
                        results["enriched_count"] += 1
                        results["enriched_papers"].append(arxiv_id)
                    else:
                        results["not_found_count"] += 1
                        results["not_found_papers"].append(arxiv_id)
                except Exception as e:
                    results["error_count"] += 1
                    results["errors"].append({"arxiv_id": arxiv_id, "error": str(e)})
            
            # Small delay between batches
            if i + batch_size < len(arxiv_ids):
                await asyncio.sleep(1)
        
        logger.info(f"Batch enrichment completed: {results['enriched_count']}/{results['total_papers']} papers enriched")
        return results
    
    async def _store_references_with_source(
        self, 
        citing_paper_id: str, 
        references: List[Dict[str, Any]], 
        source: str = 'arxiv'
    ) -> bool:
        """Store references with source tracking."""
        try:
            # Add source to each reference
            enhanced_refs = []
            for ref in references:
                enhanced_ref = dict(ref)
                enhanced_ref["source"] = source
                enhanced_refs.append(enhanced_ref)
            
            return await self.db_manager.store_paper_references(citing_paper_id, enhanced_refs)
            
        except Exception as e:
            logger.error(f"Failed to store references with source for {citing_paper_id}: {e}")
            return False
    
    async def _enrich_references(self, citing_paper_id: str, openalex_refs: List[Dict[str, Any]]):
        """Enrich existing references with OpenAlex data."""
        for ref in openalex_refs:
            try:
                await self.db_manager.enrich_reference(
                    citing_paper_id=citing_paper_id,
                    openalex_work_id=ref["openalex_work_id"],
                    reference_data={
                        "title": ref["cited_title"],
                        "authors": ref["cited_authors"],
                        "publication_year": ref["cited_year"],
                        "confidence": ref.get("confidence_score", 0.9)
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to enrich reference {ref.get('cited_title', 'Unknown')}: {e}")
    
    async def _get_citation_metrics(self, openalex_work_id: str) -> Dict[str, Any]:
        """Get citation metrics for an OpenAlex work."""
        try:
            work = self.openalex_service.Works()[openalex_work_id]
            return {
                "cited_by_count": work.cited_by_count or 0,
                "publication_year": work.publication_year,
                "is_open_access": work.open_access.get("is_oa", False) if work.open_access else False,
                "topics": work.topics[:3] if work.topics else []  # Top 3 topics
            }
        except Exception:
            return {}