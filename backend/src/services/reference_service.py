#!/usr/bin/env python3
"""
Reference Service for ArXiv Recommendation System.

This service handles fetching and parsing references from arXiv papers
by extracting citation information from HTML content.
"""

import re
import logging
from typing import List, Dict, Optional, Any
import aiohttp
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedReference:
    """Structure for parsed reference information."""
    cited_title: Optional[str] = None
    cited_authors: Optional[str] = None
    cited_year: Optional[int] = None
    reference_context: Optional[str] = None
    is_arxiv_paper: bool = False
    cited_paper_id: Optional[str] = None


class ReferenceService:
    """
    Service for parsing references from arXiv papers.
    
    Features:
    - Fetches HTML content from arXiv
    - Parses references section from HTML
    - Extracts citation information
    - Identifies arXiv papers in references
    """
    
    RATE_LIMIT_DELAY = 3.1  # Same as ArXivClient for consistency
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=1)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _ensure_session(self):
        """Ensure we have an active session."""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=1)
            )
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.RATE_LIMIT_DELAY:
            delay = self.RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"Rate limiting: waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def _construct_html_url(self, arxiv_id: str) -> str:
        """Construct HTML URL from arXiv ID."""
        # Remove version number if present (e.g., "2301.12345v1" -> "2301.12345")
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        return f"https://arxiv.org/abs/{clean_id}"
    
    async def fetch_paper_html(self, arxiv_id: str) -> Optional[str]:
        """
        Fetch HTML content for an arXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            HTML content as string, or None if failed
        """
        await self._ensure_session()
        await self._rate_limit()
        
        url = self._construct_html_url(arxiv_id)
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    logger.debug(f"Successfully fetched HTML for {arxiv_id}")
                    return html_content
                else:
                    logger.warning(f"Failed to fetch HTML for {arxiv_id}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching HTML for {arxiv_id}: {e}")
            return None
    
    def _extract_references_section(self, html_content: str) -> Optional[str]:
        """
        Extract the references section from HTML content.
        
        Args:
            html_content: Full HTML content of the paper
            
        Returns:
            References section HTML, or None if not found
        """
        # Common patterns for references sections
        ref_patterns = [
            r'<h2[^>]*>\s*References\s*</h2>(.*?)(?=<h[12]|$)',
            r'<h3[^>]*>\s*References\s*</h3>(.*?)(?=<h[123]|$)',
            r'<div[^>]*class="[^"]*references[^"]*"[^>]*>(.*?)</div>',
            r'<section[^>]*>\s*<h[23][^>]*>\s*References\s*</h[23]>(.*?)</section>',
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
            if match:
                logger.debug("Found references section using pattern")
                return match.group(1)
        
        # Fallback: look for common reference indicators (must be proper section headers)
        fallback_pattern = r'(?:^|\n)\s*(?:References|Bibliography|Works Cited)\s*(?:\n|$).*?(?=\n\s*\n|\Z)'
        match = re.search(fallback_pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            # Additional validation: check if this looks like a real references section
            matched_text = match.group(0)
            # Must contain some typical reference patterns like citations [1], years, etc.
            if (re.search(r'\[\d+\]', matched_text) or 
                re.search(r'\d{4}', matched_text) or 
                len(matched_text.split('\n')) >= 3):  # At least a few lines
                logger.debug("Found references section using fallback pattern")
                return matched_text
        
        logger.warning("Could not find references section in HTML")
        return None
    
    def _parse_reference_entry(self, ref_text: str, ref_number: int) -> ParsedReference:
        """
        Parse a single reference entry.
        
        Args:
            ref_text: Text content of the reference
            ref_number: Reference number in the list
            
        Returns:
            ParsedReference object
        """
        ref = ParsedReference()
        ref.reference_context = ref_text.strip()
        
        # Extract year (4-digit number, prioritizing publication years over page numbers)
        year_patterns = [
            r',\s*(\d{4})\.?$',  # Year at end of citation
            r'\((\d{4})\)',  # Year in parentheses
            r',\s*(19\d{2}|20\d{2})\.',  # Specific year range 1900-2099
            r'(19\d{2}|20\d{2})',  # Any year in range as fallback
        ]
        
        for pattern in year_patterns:
            year_match = re.search(pattern, ref_text)
            if year_match:
                year = int(year_match.group(1))
                # Validate reasonable publication year range for academic papers
                if 1950 <= year <= 2025:
                    ref.cited_year = year
                    break
        
        # Check if it's an arXiv paper
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', ref_text, re.IGNORECASE)
        if arxiv_match:
            ref.is_arxiv_paper = True
            ref.cited_paper_id = arxiv_match.group(1)
        
        # Extract title (try multiple patterns in order of reliability)
        title_patterns = [
            r'"([^"]+)"',  # Title in quotes
            r'[''"]([^''\"]+)[''"]',  # Title in smart quotes
            # Pattern for "[num] Authors. Title. Venue year."
            r'\[\d+\]\s+[^.]+\.\s+([^.]+)\.\s+[^.]*\d{4}',
            # Pattern for "Authors. Title. Venue year."
            r'(?:et al\.?|[A-Z][a-z]+,\s*[A-Z]\.?)\s+([^.]+)\.\s+[^.]*\d{4}',
            # General pattern: title after authors, before venue
            r'(?:[A-Z][a-zA-Z\s,]+\.)\s+([A-Z][^.]*)\.\s*(?:[A-Z]|arXiv|\d{4})',
            # Fallback: anything that looks like a title
            r'([A-Z][A-Za-z\s:]+(?:is|of|for|and|the|with)[A-Za-z\s]*)',
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, ref_text)
            if title_match:
                potential_title = title_match.group(1).strip()
                    
                # Filter out false positives and validate
                if (len(potential_title) > 5 and 
                    not re.match(r'^\d+$', potential_title) and
                    # Shouldn't be just author names
                    not re.match(r'^[A-Z][a-z]+,\s*[A-Z]', potential_title) and
                    # Should contain meaningful words
                    any(word in potential_title.lower() for word in ['is', 'of', 'for', 'and', 'the', 'with', 'on', 'in', 'to', 'learning', 'neural', 'deep'])):
                    ref.cited_title = potential_title
                    break
        
        # Extract authors (usually at the beginning, before the title)
        author_patterns = [
            # Pattern for "[num] Authors. Title..." - capture everything until title is found
            r'\[\d+\]\s+([^.]+(?:et al\.?)?)\.\s+[A-Z]',
            # Pattern for full author list with et al
            r'\[\d+\]\s+([A-Z][^.]*?(?:et al\.?))\.',
            # Pattern for author names followed by period then title
            r'^([A-Z][^.]+?)\.\s+[A-Z][a-z]+',
            # Pattern for author names with initials and et al
            r'([A-Z][a-z]+(?:,\s*[A-Z]\.?)*(?:,\s*[A-Z][a-z]+)*(?:\s*et al\.?)?)',
        ]
        
        for pattern in author_patterns:
            author_match = re.search(pattern, ref_text)
            if author_match:
                authors_text = author_match.group(1).strip()
                # Clean up and validate
                if (authors_text and len(authors_text) > 3 and len(authors_text) < 200 and
                    # Should contain author-like patterns
                    (re.search(r'[A-Z][a-z]+', authors_text) or 'et al' in authors_text) and
                    # Should not be the title (check for title keywords)
                    not any(word in authors_text.lower() for word in ['attention', 'learning', 'neural', 'deep', 'bert', 'transformer', 'pre-training']) and
                    # Should not end with common title words
                    not authors_text.lower().endswith(('need', 'transformers', 'learning'))):
                    ref.cited_authors = authors_text
                    break
        
        return ref
    
    def parse_references(self, html_content: str) -> List[ParsedReference]:
        """
        Parse all references from HTML content.
        
        Args:
            html_content: Full HTML content of the paper
            
        Returns:
            List of ParsedReference objects
        """
        references_section = self._extract_references_section(html_content)
        if not references_section:
            return []
        
        # Split into individual references
        # Try multiple splitting patterns
        ref_entries = []
        
        # Pattern 1: HTML list items (ArXiv format with <li> elements)
        li_pattern = r'<li[^>]*class="[^"]*ltx_bibitem[^"]*"[^>]*>(.*?)</li>'
        li_matches = re.findall(li_pattern, references_section, re.DOTALL)
        
        if li_matches:
            for i, li_content in enumerate(li_matches, 1):
                # Extract text content from the <li> element
                # Remove HTML tags and clean up
                text_content = re.sub(r'<[^>]+>', ' ', li_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                
                if text_content and len(text_content) > 20:
                    ref_entries.append((i, text_content))
        
        # Pattern 2: Numbered references [1], [2], etc. (fallback)
        if not ref_entries:
            numbered_pattern = r'\[(\d+)\]\s*([^[]+?)(?=\[\d+\]|$)'
            numbered_matches = re.findall(numbered_pattern, references_section, re.DOTALL)
            
            if numbered_matches:
                for ref_num, ref_text in numbered_matches:
                    if ref_text.strip():
                        ref_entries.append((int(ref_num), ref_text.strip()))
        
        # Pattern 3: Line-by-line splitting (final fallback)
        if not ref_entries:
            lines = references_section.split('\n')
            current_ref = ""
            ref_num = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_ref:
                        ref_entries.append((ref_num, current_ref))
                        current_ref = ""
                        ref_num += 1
                else:
                    if current_ref:
                        current_ref += " " + line
                    else:
                        current_ref = line
            
            # Don't forget the last reference
            if current_ref:
                ref_entries.append((ref_num, current_ref))
        
        # Parse each reference entry
        parsed_refs = []
        for ref_num, ref_text in ref_entries:
            if len(ref_text) > 20:  # Filter out very short entries
                parsed_ref = self._parse_reference_entry(ref_text, ref_num)
                parsed_refs.append(parsed_ref)
        
        logger.info(f"Parsed {len(parsed_refs)} references from HTML")
        return parsed_refs
    
    async def fetch_and_parse_references(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Fetch HTML and parse references for an arXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            List of reference dictionaries suitable for database storage
        """
        html_content = await self.fetch_paper_html(arxiv_id)
        if not html_content:
            return []
        
        parsed_refs = self.parse_references(html_content)
        
        # Convert to database format
        ref_dicts = []
        for ref in parsed_refs:
            ref_dict = {
                "cited_paper_id": ref.cited_paper_id,
                "cited_title": ref.cited_title,
                "cited_authors": ref.cited_authors,
                "cited_year": ref.cited_year,
                "reference_context": ref.reference_context,
                "is_arxiv_paper": ref.is_arxiv_paper
            }
            ref_dicts.append(ref_dict)
        
        return ref_dicts


# Convenience function for single-use operations
async def fetch_paper_references(arxiv_id: str) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch and parse references for a single paper.
    
    Args:
        arxiv_id: ArXiv paper ID
        
    Returns:
        List of reference dictionaries
    """
    async with ReferenceService() as service:
        return await service.fetch_and_parse_references(arxiv_id)