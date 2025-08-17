"""
Enhanced reference service with GROBID-only approach.

This service provides high-quality reference extraction using:
1. GROBID PDF parsing for maximum accuracy and complete metadata
2. Enhanced title extraction patterns for better coverage
3. Database storage with comprehensive metadata
"""

import asyncio
import logging
import aiosqlite
from typing import Any, Dict, List, Optional
from datetime import datetime

from backend.src.services.grobid_service import GrobidService, extract_arxiv_references
from backend.src.database import DatabaseManager

logger = logging.getLogger(__name__)


class EnhancedReferenceService:
    """
    Enhanced reference service with GROBID-only extraction strategy.
    
    Features:
    - GROBID PDF parsing for maximum accuracy and complete metadata
    - Enhanced title extraction with multiple TEI patterns
    - Unified database storage with source tracking
    - Intelligent error handling and recovery
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    async def fetch_references_enhanced(self, arxiv_id: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        GROBID-only reference extraction for high-quality results.
        
        Args:
            arxiv_id: ArXiv paper ID
            force_refresh: Force re-fetching even if references exist
            
        Returns:
            List of reference dictionaries with enhanced metadata
        """
        try:
            # Check if references already exist (unless force refresh)
            if not force_refresh:
                existing_refs = await self.db_manager.get_paper_references(arxiv_id)
                if existing_refs:
                    logger.info(f"Using existing references for {arxiv_id} ({len(existing_refs)} found)")
                    return existing_refs
            
            logger.info(f"Starting GROBID-only reference extraction for {arxiv_id}")
            
            # GROBID-only extraction for high-quality results
            grobid_refs = await self._try_grobid_extraction(arxiv_id)
            
            if grobid_refs:
                logger.info(f"GROBID extraction successful: {len(grobid_refs)} references found")
                # Store with GROBID source
                success = await self.db_manager.store_paper_references(arxiv_id, grobid_refs)
                if success:
                    return grobid_refs
                else:
                    logger.error(f"Failed to store GROBID references for {arxiv_id}")
            else:
                logger.warning(f"GROBID extraction failed for {arxiv_id} - no references extracted")
            
            return []
            
        except Exception as e:
            logger.error(f"GROBID reference extraction failed for {arxiv_id}: {e}")
            return []
    
    async def _try_grobid_extraction(self, arxiv_id: str) -> Optional[List[Dict[str, Any]]]:
        """Try to extract references using GROBID service."""
        try:
            # Use the GROBID service
            grobid_refs = await extract_arxiv_references(arxiv_id)
            
            if grobid_refs:
                # Mark all references as GROBID source
                for ref in grobid_refs:
                    ref["source"] = "grobid"
                    
                logger.debug(f"GROBID extracted {len(grobid_refs)} references for {arxiv_id}")
                return grobid_refs
            
            return None
            
        except Exception as e:
            logger.debug(f"GROBID extraction failed for {arxiv_id}: {e}")
            return None
    
    # HTML extraction method removed - using GROBID-only approach for better quality
    
    async def get_reference_summary(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get a summary of references for a paper.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            Summary dict with counts, sources, and metadata
        """
        try:
            references = await self.db_manager.get_paper_references(arxiv_id)
            
            if not references:
                return {
                    "arxiv_id": arxiv_id,
                    "total_references": 0,
                    "sources": {},
                    "arxiv_papers_found": 0,
                    "has_enhanced_metadata": False
                }
            
            # Analyze sources
            sources = {}
            arxiv_count = 0
            has_enhanced = False
            
            for ref in references:
                source = ref.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
                
                if ref.get("is_arxiv_paper"):
                    arxiv_count += 1
                
                # Check for enhanced metadata (GROBID fields)
                if ref.get("doi") or ref.get("journal") or ref.get("publisher"):
                    has_enhanced = True
            
            return {
                "arxiv_id": arxiv_id,
                "total_references": len(references),
                "sources": sources,
                "arxiv_papers_found": arxiv_count,
                "has_enhanced_metadata": has_enhanced,
                "primary_source": max(sources.keys(), key=sources.get) if sources else "none"
            }
            
        except Exception as e:
            logger.error(f"Failed to get reference summary for {arxiv_id}: {e}")
            return {"error": str(e)}
    
    async def batch_extract_references(
        self, 
        arxiv_ids: List[str], 
        max_concurrent: int = 3,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Batch extract references for multiple papers.
        
        Args:
            arxiv_ids: List of ArXiv paper IDs
            max_concurrent: Maximum concurrent extractions
            force_refresh: Force re-extraction even if references exist
            
        Returns:
            Summary dict with results and statistics
        """
        results = {
            "total_papers": len(arxiv_ids),
            "successful_extractions": 0,
            "grobid_extractions": 0,
            "html_extractions": 0,
            "failed_extractions": 0,
            "total_references": 0,
            "papers_processed": [],
            "errors": []
        }
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_single(arxiv_id: str):
            async with semaphore:
                try:
                    logger.info(f"Processing {arxiv_id}")
                    refs = await self.fetch_references_enhanced(arxiv_id, force_refresh)
                    
                    if refs:
                        results["successful_extractions"] += 1
                        results["total_references"] += len(refs)
                        
                        # Determine primary source
                        source = refs[0].get("source", "unknown") if refs else "unknown"
                        if source == "grobid":
                            results["grobid_extractions"] += 1
                        elif source == "html":
                            results["html_extractions"] += 1
                        
                        results["papers_processed"].append({
                            "arxiv_id": arxiv_id,
                            "references_found": len(refs),
                            "source": source
                        })
                        
                        logger.info(f"Successfully processed {arxiv_id}: {len(refs)} references ({source})")
                    else:
                        results["failed_extractions"] += 1
                        logger.warning(f"No references found for {arxiv_id}")
                        
                except Exception as e:
                    results["failed_extractions"] += 1
                    results["errors"].append({"arxiv_id": arxiv_id, "error": str(e)})
                    logger.error(f"Failed to process {arxiv_id}: {e}")
        
        # Process all papers concurrently
        await asyncio.gather(*[extract_single(arxiv_id) for arxiv_id in arxiv_ids])
        
        logger.info(f"Batch extraction completed: {results['successful_extractions']}/{results['total_papers']} papers processed")
        logger.info(f"Sources: GROBID: {results['grobid_extractions']}, HTML: {results['html_extractions']}")
        
        return results
    
    async def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about reference extraction across all papers."""
        try:
            # Get database stats
            db_stats = await self.db_manager.get_database_stats()
            
            # Get reference source breakdown
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Source breakdown
                cursor = await db.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM paper_references 
                    WHERE source IS NOT NULL 
                    GROUP BY source
                """)
                source_stats = {row["source"]: row["count"] for row in await cursor.fetchall()}
                
                # Papers with references
                cursor = await db.execute("""
                    SELECT COUNT(DISTINCT citing_paper_id) as papers_with_refs
                    FROM paper_references
                """)
                papers_with_refs = (await cursor.fetchone())["papers_with_refs"]
                
                # Enhanced metadata coverage
                cursor = await db.execute("""
                    SELECT COUNT(*) as enhanced_count
                    FROM paper_references 
                    WHERE doi IS NOT NULL OR journal IS NOT NULL OR publisher IS NOT NULL
                """)
                enhanced_count = (await cursor.fetchone())["enhanced_count"]
                
                total_refs = sum(source_stats.values())
                
                return {
                    "total_papers": db_stats["total_papers"],
                    "papers_with_references": papers_with_refs,
                    "total_references": total_refs,
                    "source_breakdown": source_stats,
                    "enhanced_metadata_coverage": {
                        "count": enhanced_count,
                        "percentage": (enhanced_count / total_refs * 100) if total_refs > 0 else 0
                    },
                    "average_references_per_paper": total_refs / papers_with_refs if papers_with_refs > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get extraction statistics: {e}")
            return {"error": str(e)}


# Convenience functions for backward compatibility
async def fetch_enhanced_references(arxiv_id: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Convenience function for enhanced reference extraction.
    
    Args:
        arxiv_id: ArXiv paper ID
        force_refresh: Force re-extraction
        
    Returns:
        List of reference dictionaries
    """
    service = EnhancedReferenceService()
    return await service.fetch_references_enhanced(arxiv_id, force_refresh)


async def get_reference_summary(arxiv_id: str) -> Dict[str, Any]:
    """
    Convenience function for reference summary.
    
    Args:
        arxiv_id: ArXiv paper ID
        
    Returns:
        Reference summary dictionary
    """
    service = EnhancedReferenceService()
    return await service.get_reference_summary(arxiv_id)