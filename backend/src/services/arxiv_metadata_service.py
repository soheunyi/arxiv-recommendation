#!/usr/bin/env python3
"""
ArXiv Metadata Service for Reference Validation.

This service fetches metadata for ArXiv papers to enable validation
of reference matches found through DuckDuckGo search.
"""

import asyncio
import logging
import re
from typing import Dict, Optional, Any
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime

logger = logging.getLogger(__name__)


class ArXivMetadataService:
    """Service for fetching ArXiv paper metadata."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = 3.0  # ArXiv API rate limit
        self._last_request_time = 0.0
    
    async def get_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for an ArXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "1706.03762")
            
        Returns:
            Dictionary with paper metadata or None if not found
        """
        try:
            # Ensure rate limiting
            await self._rate_limit()
            
            # Clean ArXiv ID
            clean_id = self._clean_arxiv_id(arxiv_id)
            if not clean_id:
                logger.warning(f"Invalid ArXiv ID: {arxiv_id}")
                return None
            
            # Build query URL
            query_url = f"{self.base_url}?id_list={clean_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(query_url) as response:
                    if response.status != 200:
                        logger.warning(f"ArXiv API returned status {response.status} for {arxiv_id}")
                        return None
                    
                    xml_content = await response.text()
                    return self._parse_arxiv_response(xml_content)
        
        except Exception as e:
            logger.error(f"Failed to fetch ArXiv metadata for {arxiv_id}: {e}")
            return None
    
    async def _rate_limit(self):
        """Ensure ArXiv API rate limiting compliance."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_delay:
            delay = self.rate_limit_delay - time_since_last
            logger.debug(f"ArXiv API rate limiting: waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def _clean_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """Clean and validate ArXiv ID."""
        if not arxiv_id:
            return None
        
        # Remove common prefixes and clean up
        arxiv_id = re.sub(r'^(arxiv:|arXiv:)', '', arxiv_id.strip())
        arxiv_id = re.sub(r'\.pdf$', '', arxiv_id)
        
        # Validate format (new format: YYMM.NNNNN or old format: subject-class/YYMMnnn)
        new_format = re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id)
        old_format = re.match(r'^[a-z-]+/\d{7}(v\d+)?$', arxiv_id)
        
        if new_format or old_format:
            return arxiv_id
        
        return None
    
    def _parse_arxiv_response(self, xml_content: str) -> Optional[Dict[str, Any]]:
        """Parse ArXiv API XML response."""
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find the entry (paper)
            entry = root.find('atom:entry', namespaces)
            if entry is None:
                logger.warning("No entry found in ArXiv response")
                return None
            
            # Extract basic metadata
            title = self._get_text(entry, 'atom:title', namespaces)
            summary = self._get_text(entry, 'atom:summary', namespaces)
            published = self._get_text(entry, 'atom:published', namespaces)
            updated = self._get_text(entry, 'atom:updated', namespaces)
            
            # Extract ArXiv ID from URL
            arxiv_id = None
            id_element = entry.find('atom:id', namespaces)
            if id_element is not None:
                url = id_element.text
                match = re.search(r'arxiv\.org/abs/(.+)$', url)
                if match:
                    arxiv_id = match.group(1)
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = self._get_text(author, 'atom:name', namespaces)
                if name:
                    authors.append(name)
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract year from published date
            year = None
            if published:
                try:
                    date_obj = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    year = date_obj.year
                except ValueError:
                    logger.warning(f"Could not parse published date: {published}")
            
            # Clean up title and summary
            if title:
                title = re.sub(r'\s+', ' ', title.strip())
            if summary:
                summary = re.sub(r'\s+', ' ', summary.strip())
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'authors': ', '.join(authors) if authors else '',
                'year': year,
                'abstract': summary,
                'categories': categories,
                'published': published,
                'updated': updated,
                'author_list': authors  # Individual author names
            }
        
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing ArXiv response: {e}")
            return None
    
    def _get_text(self, element, path: str, namespaces: Dict[str, str]) -> Optional[str]:
        """Safely extract text from XML element."""
        try:
            found = element.find(path, namespaces)
            return found.text if found is not None else None
        except Exception:
            return None


# Convenience function for integration
async def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for an ArXiv paper.
    
    Args:
        arxiv_id: ArXiv paper ID
        
    Returns:
        Dictionary with paper metadata or None if not found
    """
    service = ArXivMetadataService()
    return await service.get_paper_metadata(arxiv_id)