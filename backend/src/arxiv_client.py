"""ArXiv API client for fetching research papers."""

import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import logging
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structure for arXiv paper metadata."""
    id: str
    title: str
    abstract: str
    authors: List[str]
    category: str
    published_date: datetime
    updated_date: datetime
    arxiv_url: str
    pdf_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None


class ArXivClient:
    """
    Async ArXiv API client with rate limiting and error handling.
    
    Features:
    - Respects arXiv API rate limits (1 request per 3 seconds)
    - Category filtering and validation
    - Robust XML parsing with error handling
    - Pagination support for large result sets
    """
    
    BASE_URL = "https://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3.1  # Seconds between requests (slightly over 3s for safety)
    
    # Valid arXiv categories for validation
    VALID_CATEGORIES = {
        # Computer Science
        "cs.AI", "cs.CL", "cs.CC", "cs.CE", "cs.CG", "cs.GT", "cs.CV", "cs.CY",
        "cs.CR", "cs.DS", "cs.DB", "cs.DL", "cs.DM", "cs.DC", "cs.ET", "cs.FL",
        "cs.GL", "cs.GR", "cs.AR", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
        "cs.MS", "cs.MA", "cs.MM", "cs.NI", "cs.NE", "cs.NA", "cs.OS", "cs.OH",
        "cs.PF", "cs.PL", "cs.RO", "cs.SI", "cs.SE", "cs.SD", "cs.SC", "cs.SY",
        # Statistics
        "stat.AP", "stat.CO", "stat.ML", "stat.ME", "stat.OT", "stat.TH",
        # Mathematics
        "math.AG", "math.AT", "math.AP", "math.CA", "math.CO", "math.AC",
        "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN",
        "math.GT", "math.GR", "math.HO", "math.IT", "math.KT", "math.LO",
        "math.MP", "math.MG", "math.NT", "math.NA", "math.OA", "math.OC",
        "math.PR", "math.QA", "math.RT", "math.RA", "math.SP", "math.ST",
        "math.SG"
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=1)  # Single connection for rate limiting
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
        """Enforce rate limiting between API requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.RATE_LIMIT_DELAY:
            delay = self.RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"Rate limiting: waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def validate_category(self, category: str) -> bool:
        """Validate if category is a recognized arXiv category."""
        return category in self.VALID_CATEGORIES
    
    async def fetch_recent_papers(
        self, 
        category: str, 
        max_results: int = 100,
        days_back: int = 1
    ) -> List[PaperMetadata]:
        """
        Fetch recent papers from a specific arXiv category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG')
            max_results: Maximum number of papers to retrieve
            days_back: Number of days to look back for papers
            
        Returns:
            List of PaperMetadata objects
        """
        if not self.validate_category(category):
            logger.warning(f"Invalid arXiv category: {category}")
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build search query for recent papers in category
        search_query = f"cat:{category}"
        
        logger.info(f"Fetching recent papers: category={category}, max_results={max_results}, days_back={days_back}")
        
        return await self._search_papers(
            query=search_query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending"
        )
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 100,
        category: Optional[str] = None
    ) -> List[PaperMetadata]:
        """
        Search arXiv papers by query string.
        
        Args:
            query: Search query string
            max_results: Maximum number of papers to retrieve  
            category: Optional category filter
            
        Returns:
            List of PaperMetadata objects
        """
        search_query = query
        
        # Add category filter if specified
        if category:
            if self.validate_category(category):
                search_query = f"({query}) AND cat:{category}"
            else:
                logger.warning(f"Invalid category {category}, ignoring filter")
        
        logger.info(f"Searching papers: query='{query}', max_results={max_results}")
        
        return await self._search_papers(
            query=search_query,
            max_results=max_results,
            sort_by="relevance",
            sort_order="descending"
        )
    
    async def _search_papers(
        self,
        query: str,
        max_results: int,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[PaperMetadata]:
        """Internal method to execute arXiv API search."""
        await self._ensure_session()
        
        # arXiv API parameters
        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, 2000),  # arXiv API limit
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        # Debug logging for ArXiv queries with flush for real-time visibility
        logger.info(f"🔍 ArXiv Search Query: {query}")
        logger.debug(f"📡 ArXiv API URL: {url}")
        
        # Force log flush for real-time debugging
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        try:
            await self._rate_limit()
            
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.error(f"arXiv API request failed: {response.status}")
                    return []
                
                xml_content = await response.text()
                return self._parse_arxiv_response(xml_content)
        
        except asyncio.TimeoutError:
            logger.error("arXiv API request timed out")
            return []
        except Exception as e:
            logger.error(f"arXiv API request failed: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[PaperMetadata]:
        """Parse arXiv API XML response into PaperMetadata objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # arXiv uses Atom namespace
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    paper = self._parse_entry(entry, namespaces)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper entry: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(papers)} papers from arXiv response")
            return papers
        
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {e}")
            return []
    
    def _parse_entry(self, entry, namespaces: Dict[str, str]) -> Optional[PaperMetadata]:
        """Parse a single arXiv entry into PaperMetadata."""
        try:
            # Extract basic fields
            id_elem = entry.find('atom:id', namespaces)
            title_elem = entry.find('atom:title', namespaces)
            summary_elem = entry.find('atom:summary', namespaces)
            published_elem = entry.find('atom:published', namespaces)
            updated_elem = entry.find('atom:updated', namespaces)
            
            if not all([id_elem is not None, title_elem is not None, summary_elem is not None, published_elem is not None]):
                missing_fields = []
                if id_elem is None:
                    missing_fields.append("id")
                if title_elem is None:
                    missing_fields.append("title")
                if summary_elem is None:
                    missing_fields.append("summary")
                if published_elem is None:
                    missing_fields.append("published")
                logger.warning(f"Missing required fields in arXiv entry: {', '.join(missing_fields)}")
                return None
            
            # Parse arXiv ID from URL
            arxiv_url = id_elem.text.strip()
            arxiv_id = arxiv_url.split('/')[-1]
            
            # Clean title and abstract
            title = title_elem.text.strip().replace('\n', ' ')
            abstract = summary_elem.text.strip().replace('\n', ' ')
            
            # Parse dates
            published_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
            updated_date = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00')) if updated_elem is not None else published_date
            
            # Extract authors
            authors = []
            author_elems = entry.findall('atom:author', namespaces)
            for author_elem in author_elems:
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract primary category
            category_elem = entry.find('arxiv:primary_category', namespaces)
            if category_elem is not None:
                category = category_elem.get('term', '')
            else:
                # Fallback to first category
                category_elems = entry.findall('atom:category', namespaces)
                category = category_elems[0].get('term', '') if category_elems else ''
            
            # Generate PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Extract optional fields
            doi = None
            journal_ref = None
            
            # Look for DOI in links
            link_elems = entry.findall('atom:link', namespaces)
            for link_elem in link_elems:
                if link_elem.get('title') == 'doi':
                    doi = link_elem.get('href')
                    break
            
            # Look for journal reference
            journal_elem = entry.find('arxiv:journal_ref', namespaces)
            if journal_elem is not None:
                journal_ref = journal_elem.text.strip()
            
            return PaperMetadata(
                id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                category=category,
                published_date=published_date,
                updated_date=updated_date,
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
                doi=doi,
                journal_ref=journal_ref
            )
        
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None

    async def fetch_paper_references(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Fetch and parse references for an arXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            List of reference dictionaries
        """
        from services.reference_service import ReferenceService
        
        try:
            async with ReferenceService() as ref_service:
                references = await ref_service.fetch_and_parse_references(arxiv_id)
                logger.info(f"Fetched {len(references)} references for paper {arxiv_id}")
                return references
        except Exception as e:
            logger.error(f"Failed to fetch references for {arxiv_id}: {e}")
            return []

    def extract_arxiv_id(self, arxiv_url: str) -> Optional[str]:
        """
        Extract arXiv ID from various URL formats.
        
        Args:
            arxiv_url: ArXiv URL in any common format
            
        Returns:
            Clean arXiv ID or None if not found
        """
        import re
        
        # Enhanced patterns for different arXiv URL formats (including versions)
        patterns = [
            r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf',  # PDF filename pattern
            r'(\d{4}\.\d{4,5}(?:v\d+)?)'  # Just the ID itself
        ]
        
        for pattern in patterns:
            match = re.search(pattern, arxiv_url, re.IGNORECASE)
            if match:
                # Clean up the ID (remove version if present for consistency)
                arxiv_id = match.group(1)
                clean_id = re.sub(r'v\d+$', '', arxiv_id)
                return clean_id
        
        return None
    
    def extract_arxiv_ids_from_text(self, text: str) -> List[str]:
        """
        Extract multiple ArXiv IDs from text content (useful for DuckDuckGo results).
        
        Args:
            text: Text content to search for ArXiv IDs
            
        Returns:
            List of unique ArXiv IDs found in the text
        """
        import re
        
        arxiv_ids = set()
        
        # Enhanced patterns for text extraction
        patterns = [
            r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf',
            # More liberal pattern for standalone IDs in academic text
            r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the ID and validate
                clean_id = re.sub(r'v\d+$', '', match)
                if self._is_valid_arxiv_id(clean_id):
                    arxiv_ids.add(clean_id)
        
        return list(arxiv_ids)
    
    def _is_valid_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Validate ArXiv ID format.
        
        Args:
            arxiv_id: ArXiv ID to validate
            
        Returns:
            True if valid ArXiv ID format, False otherwise
        """
        import re
        # ArXiv IDs follow format: YYMM.NNNN or YYMM.NNNNN
        pattern = r'^\d{4}\.\d{4,5}$'
        return bool(re.match(pattern, arxiv_id))
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """
        Fetch a single paper's metadata by arXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            PaperMetadata object or None if not found
        """
        try:
            logger.info(f"Fetching paper metadata for arXiv ID: {arxiv_id}")
            
            # Search for the specific arXiv ID
            search_query = f"id:{arxiv_id}"
            papers = await self._search_papers(
                query=search_query,
                max_results=1,
                sort_by="submittedDate",
                sort_order="descending"
            )
            
            if papers:
                logger.info(f"Found paper: {papers[0].title}")
                return papers[0]
            else:
                logger.warning(f"Paper not found for arXiv ID: {arxiv_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            return None


# Convenience functions for common use cases
async def fetch_daily_papers(categories: List[str], max_per_category: int = 50) -> List[PaperMetadata]:
    """Fetch daily papers from multiple categories."""
    async with ArXivClient() as client:
        all_papers = []
        
        for category in categories:
            papers = await client.fetch_recent_papers(
                category=category,
                max_results=max_per_category,
                days_back=1
            )
            all_papers.extend(papers)
        
        return all_papers


async def search_by_keywords(keywords: List[str], max_results: int = 100) -> List[PaperMetadata]:
    """Search papers by keywords."""
    query = " OR ".join(f'all:"{keyword}"' for keyword in keywords)
    
    async with ArXivClient() as client:
        return await client.search_papers(query=query, max_results=max_results)