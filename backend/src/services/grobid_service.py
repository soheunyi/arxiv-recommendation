"""
GROBID service for extracting references from arXiv PDFs.

This service provides async HTTP client functionality for communicating with
GROBID service to extract references from PDF content with TEI XML parsing.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import aiohttp
import aiofiles
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GrobidReference:
    """Structure for parsed reference from GROBID TEI XML."""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    conference: Optional[str] = None
    url: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: float = 1.0


class TEINamespaces:
    """TEI XML namespaces used by GROBID."""
    
    TEI = "http://www.tei-c.org/ns/1.0"
    XML = "http://www.w3.org/XML/1998/namespace"
    
    @classmethod
    def get_nsmap(cls) -> Dict[str, str]:
        """Get namespace map for XML parsing."""
        return {
            "tei": cls.TEI,
            "xml": cls.XML
        }


class GrobidService:
    """Service for extracting references from PDFs using GROBID."""
    
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        self.grobid_url = grobid_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=60)  # 1 minute timeout for PDF processing
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure we have an active session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def is_alive(self) -> bool:
        """Check if GROBID service is alive."""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.grobid_url}/api/isalive") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"GROBID health check failed: {e}")
            return False
    
    async def fetch_arxiv_pdf(self, arxiv_id: str) -> Optional[Path]:
        """
        Fetch PDF from arXiv for the given paper ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            Path to downloaded PDF file, or None if failed
        """
        try:
            # Clean arxiv ID (remove version if present)
            clean_id = re.sub(r'v\d+$', '', arxiv_id)
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            
            # Create cache directory
            cache_dir = Path("data/pdfs")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_path = cache_dir / f"{clean_id}.pdf"
            
            # Check if already cached
            if pdf_path.exists():
                logger.debug(f"Using cached PDF for {arxiv_id}")
                return pdf_path
            
            await self._ensure_session()
            
            # Download PDF
            logger.info(f"Downloading PDF for {arxiv_id}")
            async with self.session.get(pdf_url) as response:
                if response.status == 200:
                    async with aiofiles.open(pdf_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    logger.debug(f"Downloaded PDF for {arxiv_id}: {pdf_path}")
                    return pdf_path
                else:
                    logger.error(f"Failed to download PDF for {arxiv_id}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
            return None
    
    async def extract_references_from_pdf(self, pdf_path: Path) -> List[GrobidReference]:
        """
        Extract references from PDF using GROBID.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted references
        """
        try:
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return []
            
            await self._ensure_session()
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('input', 
                          open(pdf_path, 'rb'), 
                          filename=pdf_path.name,
                          content_type='application/pdf')
            
            # Call GROBID processReferences endpoint
            url = f"{self.grobid_url}/api/processReferences"
            
            logger.info(f"Processing PDF with GROBID: {pdf_path.name}")
            start_time = datetime.now()
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    tei_xml = await response.text()
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    logger.debug(f"GROBID processing completed in {processing_time:.2f}s")
                    
                    # Parse TEI XML
                    references = self._parse_tei_references(tei_xml)
                    logger.info(f"Extracted {len(references)} references from {pdf_path.name}")
                    
                    return references
                else:
                    error_text = await response.text()
                    logger.error(f"GROBID processing failed: HTTP {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error processing PDF with GROBID: {e}")
            return []
    
    def _parse_tei_references(self, tei_xml: str) -> List[GrobidReference]:
        """
        Parse references from GROBID TEI XML output.
        
        Args:
            tei_xml: TEI XML string from GROBID
            
        Returns:
            List of parsed references
        """
        try:
            # Parse XML
            root = ET.fromstring(tei_xml)
            nsmap = TEINamespaces.get_nsmap()
            
            references = []
            
            # Find all reference elements
            ref_elements = root.findall(".//tei:biblStruct", nsmap)
            
            for ref_elem in ref_elements:
                ref = self._parse_single_reference(ref_elem, nsmap)
                if ref:
                    references.append(ref)
            
            return references
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse TEI XML: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing TEI references: {e}")
            return []
    
    def _parse_single_reference(self, ref_elem: ET.Element, nsmap: Dict[str, str]) -> Optional[GrobidReference]:
        """Parse a single reference from TEI XML element."""
        try:
            ref = GrobidReference()
            
            # Extract title - try multiple title levels for better coverage
            title_elem = None
            title_patterns = [
                ".//tei:title[@level='a']",  # Article title (most common)
                ".//tei:title[@level='m']",  # Monograph/book title
                ".//tei:title[@level='j']",  # Journal title (fallback)
                ".//tei:title[not(@level)]", # Title without level attribute
                ".//tei:title"               # Any title element
            ]
            
            for pattern in title_patterns:
                title_elem = ref_elem.find(pattern, nsmap)
                if title_elem is not None and title_elem.text and title_elem.text.strip():
                    ref.title = title_elem.text.strip()
                    break
            
            # Extract authors
            authors = []
            author_elems = ref_elem.findall(".//tei:author", nsmap)
            for author_elem in author_elems:
                forename_elem = author_elem.find(".//tei:forename", nsmap)
                surname_elem = author_elem.find(".//tei:surname", nsmap)
                
                author_name = ""
                if forename_elem is not None and forename_elem.text:
                    author_name += forename_elem.text.strip()
                if surname_elem is not None and surname_elem.text:
                    if author_name:
                        author_name += " "
                    author_name += surname_elem.text.strip()
                
                if author_name:
                    authors.append(author_name)
            
            if authors:
                ref.authors = authors
            
            # Extract publication year
            date_elem = ref_elem.find(".//tei:date[@when]", nsmap)
            if date_elem is not None:
                when_attr = date_elem.get("when")
                if when_attr:
                    # Extract year from date
                    year_match = re.search(r'(\d{4})', when_attr)
                    if year_match:
                        ref.year = int(year_match.group(1))
            
            # Extract journal/venue
            journal_elem = ref_elem.find(".//tei:title[@level='j']", nsmap)
            if journal_elem is not None and journal_elem.text:
                ref.journal = journal_elem.text.strip()
            
            # Extract conference
            meeting_elem = ref_elem.find(".//tei:title[@level='m']", nsmap)
            if meeting_elem is not None and meeting_elem.text:
                ref.conference = meeting_elem.text.strip()
            
            # Extract volume
            vol_elem = ref_elem.find(".//tei:biblScope[@unit='volume']", nsmap)
            if vol_elem is not None and vol_elem.text:
                ref.volume = vol_elem.text.strip()
            
            # Extract pages
            page_elem = ref_elem.find(".//tei:biblScope[@unit='page']", nsmap)
            if page_elem is not None:
                if page_elem.text:
                    ref.pages = page_elem.text.strip()
                elif page_elem.get("from") and page_elem.get("to"):
                    ref.pages = f"{page_elem.get('from')}-{page_elem.get('to')}"
            
            # Extract DOI
            doi_elem = ref_elem.find(".//tei:idno[@type='DOI']", nsmap)
            if doi_elem is not None and doi_elem.text:
                ref.doi = doi_elem.text.strip()
            
            # Extract arXiv ID
            arxiv_elem = ref_elem.find(".//tei:idno[@type='arXiv']", nsmap)
            if arxiv_elem is not None and arxiv_elem.text:
                ref.arxiv_id = arxiv_elem.text.strip()
            
            # Extract ISBN
            isbn_elem = ref_elem.find(".//tei:idno[@type='ISBN']", nsmap)
            if isbn_elem is not None and isbn_elem.text:
                ref.isbn = isbn_elem.text.strip()
            
            # Extract publisher
            publisher_elem = ref_elem.find(".//tei:publisher", nsmap)
            if publisher_elem is not None and publisher_elem.text:
                ref.publisher = publisher_elem.text.strip()
            
            # Extract URL
            url_elem = ref_elem.find(".//tei:ptr[@target]", nsmap)
            if url_elem is not None:
                target = url_elem.get("target")
                if target:
                    ref.url = target
            
            # Get raw text representation
            ref.raw_text = ET.tostring(ref_elem, encoding='unicode', method='text').strip()
            
            # Only return if we have at least title or authors
            if ref.title or ref.authors:
                return ref
            
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing single reference: {e}")
            return None
    
    async def extract_references_from_arxiv(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """
        Complete workflow: fetch PDF and extract references for an arXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            List of reference dictionaries suitable for database storage
        """
        try:
            # Check GROBID service health
            if not await self.is_alive():
                logger.error("GROBID service is not available")
                return []
            
            # Fetch PDF
            pdf_path = await self.fetch_arxiv_pdf(arxiv_id)
            if not pdf_path:
                logger.error(f"Failed to fetch PDF for {arxiv_id}")
                return []
            
            # Extract references
            grobid_refs = await self.extract_references_from_pdf(pdf_path)
            
            # Convert to database format
            db_refs = []
            for ref in grobid_refs:
                db_ref = {
                    "cited_paper_id": ref.arxiv_id,
                    "cited_title": ref.title,
                    "cited_authors": ", ".join(ref.authors) if ref.authors else None,
                    "cited_year": ref.year,
                    "reference_context": ref.raw_text,
                    "is_arxiv_paper": bool(ref.arxiv_id),
                    "source": "grobid",
                    "doi": ref.doi,
                    "journal": ref.journal,
                    "conference": ref.conference,
                    "volume": ref.volume,
                    "pages": ref.pages,
                    "isbn": ref.isbn,
                    "publisher": ref.publisher,
                    "url": ref.url,
                    "confidence_score": ref.confidence
                }
                db_refs.append(db_ref)
            
            logger.info(f"Successfully extracted {len(db_refs)} references for {arxiv_id}")
            return db_refs
            
        except Exception as e:
            logger.error(f"Error in complete reference extraction for {arxiv_id}: {e}")
            return []


# Convenience function for single-use operations
async def extract_arxiv_references(arxiv_id: str) -> List[Dict[str, Any]]:
    """
    Convenience function to extract references for a single arXiv paper.
    
    Args:
        arxiv_id: ArXiv paper ID
        
    Returns:
        List of reference dictionaries
    """
    async with GrobidService() as service:
        return await service.extract_references_from_arxiv(arxiv_id)