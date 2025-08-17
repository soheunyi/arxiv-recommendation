#!/usr/bin/env python3
"""
Academic Quality Filter for ArXiv Recommendation System.

This module provides quality assessment and filtering for academic search results
from various sources including DuckDuckGo, with focus on academic relevance,
domain authority, and content quality indicators.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Academic content type classifications."""
    ARXIV_PAPER = "arxiv_paper"
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    THESIS = "thesis"
    BOOK_CHAPTER = "book_chapter"
    GENERAL_ACADEMIC = "general_academic"
    NON_ACADEMIC = "non_academic"


@dataclass
class QualityMetrics:
    """Quality assessment metrics for academic content."""
    domain_authority: float  # 0-10 scale
    content_relevance: float  # 0-1 scale
    academic_indicators: float  # 0-1 scale
    accessibility: float  # 0-1 scale (PDF availability, etc.)
    overall_score: float  # 0-1 scale
    content_type: ContentType
    reasons: List[str]  # Quality assessment reasoning


@dataclass
class AcademicDomain:
    """Academic domain information."""
    domain: str
    authority: int
    type_category: str
    trust_level: str
    specialties: List[str]


class AcademicQualityFilter:
    """
    Quality filter for academic search results with domain authority and content assessment.
    
    This filter evaluates academic search results from various sources and provides
    quality scores based on domain authority, content indicators, and academic relevance.
    """
    
    # Academic domain database with authority scores
    ACADEMIC_DOMAINS = {
        # Preprint repositories
        "arxiv.org": AcademicDomain("arxiv.org", 10, "preprint_repository", "high", ["physics", "cs", "math", "bio"]),
        "biorxiv.org": AcademicDomain("biorxiv.org", 9, "preprint_repository", "high", ["biology", "life_sciences"]),
        "medrxiv.org": AcademicDomain("medrxiv.org", 9, "preprint_repository", "high", ["medicine", "health"]),
        
        # Citation databases
        "scholar.google.com": AcademicDomain("scholar.google.com", 9, "citation_database", "high", ["general"]),
        "semanticscholar.org": AcademicDomain("semanticscholar.org", 8, "citation_database", "high", ["general"]),
        "pubmed.ncbi.nlm.nih.gov": AcademicDomain("pubmed.ncbi.nlm.nih.gov", 10, "citation_database", "high", ["medicine", "biology"]),
        "dblp.org": AcademicDomain("dblp.org", 9, "citation_database", "high", ["computer_science"]),
        
        # Publishers
        "ieee.org": AcademicDomain("ieee.org", 9, "publisher", "high", ["engineering", "cs", "electronics"]),
        "acm.org": AcademicDomain("acm.org", 9, "publisher", "high", ["computer_science"]),
        "springer.com": AcademicDomain("springer.com", 8, "publisher", "high", ["general"]),
        "nature.com": AcademicDomain("nature.com", 10, "publisher", "high", ["science"]),
        "science.org": AcademicDomain("science.org", 10, "publisher", "high", ["science"]),
        "elsevier.com": AcademicDomain("elsevier.com", 8, "publisher", "high", ["general"]),
        "wiley.com": AcademicDomain("wiley.com", 8, "publisher", "high", ["general"]),
        "cambridge.org": AcademicDomain("cambridge.org", 8, "publisher", "high", ["general"]),
        "oxford.org": AcademicDomain("oxford.org", 8, "publisher", "high", ["general"]),
        
        # Academic networks
        "researchgate.net": AcademicDomain("researchgate.net", 7, "academic_network", "medium", ["general"]),
        "academia.edu": AcademicDomain("academia.edu", 6, "academic_network", "medium", ["general"]),
        
        # Repositories
        "hal.archives-ouvertes.fr": AcademicDomain("hal.archives-ouvertes.fr", 8, "repository", "high", ["general"]),
        "core.ac.uk": AcademicDomain("core.ac.uk", 8, "repository", "high", ["general"]),
    }
    
    # Academic file types and their quality scores
    ACADEMIC_FILE_TYPES = {
        "pdf": 0.9,
        "ps": 0.7,
        "doc": 0.5,
        "docx": 0.5,
        "tex": 0.8,
        "html": 0.3
    }
    
    # Academic keywords that indicate quality content
    ACADEMIC_KEYWORDS = {
        "high_quality": {
            "journal", "conference", "proceedings", "ieee", "acm", "arxiv",
            "research", "study", "analysis", "algorithm", "method", "approach",
            "experiment", "evaluation", "paper", "article", "publication"
        },
        "medium_quality": {
            "technical", "report", "survey", "review", "tutorial", "workshop",
            "symposium", "seminar", "thesis", "dissertation", "preprint"
        },
        "domain_specific": {
            "cs": {"computer science", "machine learning", "ai", "neural", "deep learning"},
            "physics": {"quantum", "relativity", "particle", "cosmology"},
            "math": {"theorem", "proof", "mathematics", "statistics"},
            "biology": {"genetics", "molecular", "cell", "protein", "dna"}
        }
    }
    
    def __init__(self):
        """Initialize the academic quality filter."""
        self.domain_cache = {}
        self.quality_cache = {}
    
    def assess_quality(
        self,
        url: str,
        title: str,
        snippet: str,
        query_context: Optional[str] = None
    ) -> QualityMetrics:
        """
        Assess the academic quality of a search result.
        
        Args:
            url: URL of the search result
            title: Title of the content
            snippet: Content snippet/description
            query_context: Original search query for relevance assessment
            
        Returns:
            QualityMetrics object with comprehensive quality assessment
        """
        try:
            # Parse domain information
            domain_info = self._analyze_domain(url)
            domain_authority = domain_info["authority"] / 10.0  # Normalize to 0-1
            
            # Assess content relevance
            content_relevance = self._assess_content_relevance(title, snippet, query_context)
            
            # Check academic indicators
            academic_indicators = self._check_academic_indicators(url, title, snippet)
            
            # Assess accessibility (PDF availability, etc.)
            accessibility = self._assess_accessibility(url, title, snippet)
            
            # Determine content type
            content_type = self._classify_content_type(url, title, snippet)
            
            # Calculate overall score with weights
            weights = {
                "domain_authority": 0.3,
                "content_relevance": 0.25,
                "academic_indicators": 0.25,
                "accessibility": 0.2
            }
            
            overall_score = (
                domain_authority * weights["domain_authority"] +
                content_relevance * weights["content_relevance"] +
                academic_indicators * weights["academic_indicators"] +
                accessibility * weights["accessibility"]
            )
            
            # Generate reasoning
            reasons = self._generate_quality_reasoning(
                domain_info, content_relevance, academic_indicators, 
                accessibility, content_type
            )
            
            return QualityMetrics(
                domain_authority=domain_authority * 10,  # Back to 0-10 scale for display
                content_relevance=content_relevance,
                academic_indicators=academic_indicators,
                accessibility=accessibility,
                overall_score=overall_score,
                content_type=content_type,
                reasons=reasons
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed for {url}: {e}")
            return QualityMetrics(
                domain_authority=1.0,
                content_relevance=0.1,
                academic_indicators=0.1,
                accessibility=0.1,
                overall_score=0.1,
                content_type=ContentType.NON_ACADEMIC,
                reasons=[f"Assessment failed: {str(e)}"]
            )
    
    def _analyze_domain(self, url: str) -> Dict[str, Any]:
        """Analyze domain information and authority."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check cache first
            if domain in self.domain_cache:
                return self.domain_cache[domain]
            
            # Check known academic domains
            if domain in self.ACADEMIC_DOMAINS:
                domain_data = self.ACADEMIC_DOMAINS[domain]
                result = {
                    "domain": domain,
                    "authority": domain_data.authority,
                    "type": domain_data.type_category,
                    "trust_level": domain_data.trust_level,
                    "specialties": domain_data.specialties
                }
            else:
                # Assess unknown domain
                result = self._assess_unknown_domain(domain)
            
            # Cache the result
            self.domain_cache[domain] = result
            return result
            
        except Exception as e:
            logger.warning(f"Domain analysis failed for {url}: {e}")
            return {
                "domain": "unknown",
                "authority": 1,
                "type": "unknown",
                "trust_level": "low",
                "specialties": []
            }
    
    def _assess_unknown_domain(self, domain: str) -> Dict[str, Any]:
        """Assess quality of unknown academic domains."""
        authority = 1  # Default low authority
        domain_type = "unknown"
        trust_level = "low"
        specialties = []
        
        # Check for academic indicators in domain
        academic_indicators = [
            ("edu", 6, "university"),
            ("ac.uk", 6, "university"),
            ("ac.", 5, "university"),
            ("university", 5, "university"),
            ("institute", 4, "research_institute"),
            ("research", 4, "research_institute"),
            ("gov", 7, "government"),
            ("org", 3, "organization")
        ]
        
        for indicator, score, type_name in academic_indicators:
            if indicator in domain:
                authority = max(authority, score)
                domain_type = type_name
                trust_level = "medium" if score >= 5 else "low"
                break
        
        return {
            "domain": domain,
            "authority": authority,
            "type": domain_type,
            "trust_level": trust_level,
            "specialties": specialties
        }
    
    def _assess_content_relevance(
        self, 
        title: str, 
        snippet: str, 
        query_context: Optional[str]
    ) -> float:
        """Assess content relevance to the search query."""
        if not query_context:
            return 0.5  # Neutral score if no query context
        
        try:
            query_words = set(query_context.lower().split())
            content_text = f"{title} {snippet}".lower()
            content_words = set(content_text.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(content_words))
            max_possible = len(query_words)
            
            if max_possible == 0:
                return 0.5
            
            # Base relevance score
            relevance = overlap / max_possible
            
            # Boost for exact phrase matches
            if query_context.lower() in content_text:
                relevance += 0.2
            
            # Boost for title matches (titles are more important)
            title_overlap = len(query_words.intersection(set(title.lower().split())))
            if title_overlap > 0:
                relevance += (title_overlap / max_possible) * 0.3
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.warning(f"Content relevance assessment failed: {e}")
            return 0.3
    
    def _check_academic_indicators(self, url: str, title: str, snippet: str) -> float:
        """Check for academic quality indicators in content."""
        try:
            content_text = f"{title} {snippet} {url}".lower()
            score = 0.0
            
            # Check for high-quality academic keywords
            high_quality_matches = sum(
                1 for keyword in self.ACADEMIC_KEYWORDS["high_quality"]
                if keyword in content_text
            )
            score += min(high_quality_matches * 0.15, 0.6)
            
            # Check for medium-quality academic keywords
            medium_quality_matches = sum(
                1 for keyword in self.ACADEMIC_KEYWORDS["medium_quality"]
                if keyword in content_text
            )
            score += min(medium_quality_matches * 0.1, 0.3)
            
            # Check for citation patterns (academic references)
            citation_patterns = [
                r'\[\d+\]',  # [1], [23], etc.
                r'\(\d{4}\)',  # (2023), (2024), etc.
                r'et al\.',  # "et al."
                r'doi:',  # DOI references
                r'arxiv:',  # ArXiv references
            ]
            
            for pattern in citation_patterns:
                if re.search(pattern, content_text, re.IGNORECASE):
                    score += 0.1
            
            # Check for academic file type indicators
            if url.endswith('.pdf'):
                score += 0.2
            
            # Check for academic publishing terms
            publishing_terms = [
                "published", "journal", "conference", "proceedings",
                "volume", "issue", "pages", "abstract", "introduction",
                "methodology", "results", "conclusion", "references"
            ]
            
            publishing_matches = sum(1 for term in publishing_terms if term in content_text)
            score += min(publishing_matches * 0.05, 0.25)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Academic indicators check failed: {e}")
            return 0.2
    
    def _assess_accessibility(self, url: str, title: str, snippet: str) -> float:
        """Assess how accessible the academic content is."""
        try:
            score = 0.5  # Base accessibility score
            
            # PDF availability (high accessibility for academic content)
            if url.endswith('.pdf'):
                score += 0.4
            elif 'pdf' in url.lower() or 'pdf' in snippet.lower():
                score += 0.3
            
            # Open access indicators
            open_access_indicators = [
                "open access", "free", "public", "available", "download",
                "arxiv.org", "pubmed", "pmc", "repository"
            ]
            
            content_text = f"{url} {title} {snippet}".lower()
            for indicator in open_access_indicators:
                if indicator in content_text:
                    score += 0.1
                    break
            
            # Paywall/subscription indicators (reduce accessibility)
            paywall_indicators = [
                "subscription", "paywall", "purchase", "buy", "login required",
                "register", "membership", "access denied"
            ]
            
            for indicator in paywall_indicators:
                if indicator in content_text:
                    score -= 0.2
                    break
            
            return max(min(score, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Accessibility assessment failed: {e}")
            return 0.3
    
    def _classify_content_type(self, url: str, title: str, snippet: str) -> ContentType:
        """Classify the type of academic content."""
        try:
            content_text = f"{url} {title} {snippet}".lower()
            
            # ArXiv papers
            if "arxiv.org" in url:
                return ContentType.ARXIV_PAPER
            
            # Journal articles
            journal_indicators = ["journal", "vol.", "volume", "issue", "pp.", "pages"]
            if any(indicator in content_text for indicator in journal_indicators):
                return ContentType.JOURNAL_ARTICLE
            
            # Conference papers
            conference_indicators = ["conference", "proceedings", "symposium", "workshop"]
            if any(indicator in content_text for indicator in conference_indicators):
                return ContentType.CONFERENCE_PAPER
            
            # Preprints
            preprint_indicators = ["preprint", "biorxiv", "medrxiv", "arxiv"]
            if any(indicator in content_text for indicator in preprint_indicators):
                return ContentType.PREPRINT
            
            # Thesis/Dissertation
            thesis_indicators = ["thesis", "dissertation", "phd", "master"]
            if any(indicator in content_text for indicator in thesis_indicators):
                return ContentType.THESIS
            
            # Book chapters
            book_indicators = ["chapter", "book", "handbook", "textbook"]
            if any(indicator in content_text for indicator in book_indicators):
                return ContentType.BOOK_CHAPTER
            
            # Check if generally academic
            academic_indicators = list(self.ACADEMIC_KEYWORDS["high_quality"]) + \
                                list(self.ACADEMIC_KEYWORDS["medium_quality"])
            
            academic_score = sum(1 for indicator in academic_indicators if indicator in content_text)
            
            if academic_score >= 3:
                return ContentType.GENERAL_ACADEMIC
            else:
                return ContentType.NON_ACADEMIC
                
        except Exception as e:
            logger.warning(f"Content type classification failed: {e}")
            return ContentType.NON_ACADEMIC
    
    def _generate_quality_reasoning(
        self,
        domain_info: Dict[str, Any],
        content_relevance: float,
        academic_indicators: float,
        accessibility: float,
        content_type: ContentType
    ) -> List[str]:
        """Generate human-readable reasoning for quality assessment."""
        reasons = []
        
        # Domain authority reasoning
        authority = domain_info["authority"]
        if authority >= 8:
            reasons.append(f"High authority domain ({domain_info['domain']}) - {domain_info['type']}")
        elif authority >= 5:
            reasons.append(f"Medium authority domain ({domain_info['domain']}) - {domain_info['type']}")
        else:
            reasons.append(f"Low authority domain ({domain_info['domain']})")
        
        # Content relevance reasoning
        if content_relevance >= 0.7:
            reasons.append("High relevance to search query")
        elif content_relevance >= 0.4:
            reasons.append("Medium relevance to search query")
        else:
            reasons.append("Low relevance to search query")
        
        # Academic indicators reasoning
        if academic_indicators >= 0.6:
            reasons.append("Strong academic indicators (citations, keywords, structure)")
        elif academic_indicators >= 0.3:
            reasons.append("Some academic indicators present")
        else:
            reasons.append("Few academic indicators")
        
        # Accessibility reasoning
        if accessibility >= 0.7:
            reasons.append("High accessibility (PDF/open access)")
        elif accessibility >= 0.4:
            reasons.append("Medium accessibility")
        else:
            reasons.append("Limited accessibility (potential paywall)")
        
        # Content type reasoning
        if content_type == ContentType.ARXIV_PAPER:
            reasons.append("ArXiv preprint - high academic quality")
        elif content_type == ContentType.JOURNAL_ARTICLE:
            reasons.append("Journal article - peer-reviewed content")
        elif content_type == ContentType.CONFERENCE_PAPER:
            reasons.append("Conference paper - academic venue")
        elif content_type == ContentType.NON_ACADEMIC:
            reasons.append("Non-academic content")
        
        return reasons
    
    def filter_results(
        self,
        results: List[Dict[str, Any]],
        min_quality_threshold: float = 0.4,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank academic search results by quality.
        
        Args:
            results: List of search results to filter
            min_quality_threshold: Minimum quality score to include
            max_results: Maximum number of results to return
            
        Returns:
            Filtered and ranked list of search results with quality metrics
        """
        try:
            scored_results = []
            
            for result in results:
                url = result.get("url", result.get("href", ""))
                title = result.get("title", "")
                snippet = result.get("snippet", result.get("body", ""))
                
                # Assess quality
                quality_metrics = self.assess_quality(url, title, snippet)
                
                # Add quality metrics to result
                enhanced_result = result.copy()
                enhanced_result["quality_metrics"] = quality_metrics
                enhanced_result["quality_score"] = quality_metrics.overall_score
                
                # Filter by minimum threshold
                if quality_metrics.overall_score >= min_quality_threshold:
                    scored_results.append(enhanced_result)
            
            # Sort by quality score (descending)
            scored_results.sort(key=lambda x: x["quality_score"], reverse=True)
            
            # Limit results if requested
            if max_results:
                scored_results = scored_results[:max_results]
            
            logger.info(f"Quality filtering: {len(scored_results)}/{len(results)} results passed threshold {min_quality_threshold}")
            
            return scored_results
            
        except Exception as e:
            logger.error(f"Quality filtering failed: {e}")
            return results  # Return original results if filtering fails
    
    def get_quality_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get quality summary statistics for a set of results."""
        try:
            if not results:
                return {"error": "No results to analyze"}
            
            quality_scores = []
            content_types = {}
            domain_authorities = []
            
            for result in results:
                if "quality_metrics" in result:
                    metrics = result["quality_metrics"]
                    quality_scores.append(metrics.overall_score)
                    domain_authorities.append(metrics.domain_authority)
                    
                    content_type = metrics.content_type.value
                    content_types[content_type] = content_types.get(content_type, 0) + 1
            
            if not quality_scores:
                return {"error": "No quality metrics found"}
            
            return {
                "total_results": len(results),
                "average_quality": sum(quality_scores) / len(quality_scores),
                "max_quality": max(quality_scores),
                "min_quality": min(quality_scores),
                "average_domain_authority": sum(domain_authorities) / len(domain_authorities),
                "content_type_distribution": content_types,
                "high_quality_count": sum(1 for score in quality_scores if score >= 0.7),
                "medium_quality_count": sum(1 for score in quality_scores if 0.4 <= score < 0.7),
                "low_quality_count": sum(1 for score in quality_scores if score < 0.4)
            }
            
        except Exception as e:
            logger.error(f"Quality summary generation failed: {e}")
            return {"error": str(e)}


# Convenience functions
def assess_academic_quality(url: str, title: str, snippet: str, query_context: Optional[str] = None) -> QualityMetrics:
    """Convenience function to assess academic quality of a single result."""
    filter_service = AcademicQualityFilter()
    return filter_service.assess_quality(url, title, snippet, query_context)


def filter_academic_results(
    results: List[Dict[str, Any]], 
    min_quality: float = 0.4,
    max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Convenience function to filter academic search results by quality."""
    filter_service = AcademicQualityFilter()
    return filter_service.filter_results(results, min_quality, max_results)