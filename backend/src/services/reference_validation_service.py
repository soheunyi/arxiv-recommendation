#!/usr/bin/env python3
"""
Reference Validation Service for ArXiv Paper Matching.

This service validates whether a found ArXiv paper actually matches a citation reference
by comparing author names, publication years, and title similarity to prevent false positives
in the DuckDuckGo enhancement pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of reference validation."""
    is_valid: bool
    confidence_score: float
    author_match: bool
    year_match: bool
    title_similarity: float
    reasons: List[str]


class AuthorNameNormalizer:
    """Handles author name normalization and comparison."""
    
    # Special character mappings for normalization
    CHAR_MAPPINGS = {
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u',
        'ñ': 'n', 'ç': 'c', 'š': 's', 'ž': 'z', 'č': 'c',
        'ř': 'r', 'ý': 'y', 'ď': 'd', 'ť': 't', 'ľ': 'l',
        'ø': 'o', 'å': 'a', 'æ': 'ae', 'þ': 'th', 'ð': 'd',
        'ß': 'ss', 'œ': 'oe'
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize author name for comparison.
        
        - Converts to lowercase
        - Removes special characters and diacritics
        - Handles common variations
        - Keeps only letters and basic punctuation
        """
        if not name:
            return ""
        
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(dr|prof|professor|phd|md|jr|sr|ii|iii|iv)\b\.?', '', name)
        
        # Apply character mappings
        for char, replacement in AuthorNameNormalizer.CHAR_MAPPINGS.items():
            name = name.replace(char, replacement)
        
        # Remove unicode diacritics
        name = unicodedata.normalize('NFD', name)
        name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
        
        # Keep only letters, spaces, hyphens, and periods
        name = re.sub(r'[^a-z\s\-\.]', '', name)
        
        # Normalize spaces
        name = ' '.join(name.split())
        
        return name
    
    @staticmethod
    def extract_last_name(full_name: str) -> str:
        """Extract the last name from a full name."""
        normalized = AuthorNameNormalizer.normalize_name(full_name)
        if not normalized:
            return ""
        
        # Handle "Last, First" format
        if ',' in normalized:
            return normalized.split(',')[0].strip()
        
        # Handle "First Last" format
        parts = normalized.split()
        if parts:
            return parts[-1]  # Last word is typically the surname
        
        return normalized
    
    @staticmethod
    def compare_author_lists(
        reference_authors: str, 
        arxiv_authors: str,
        min_matches: int = 1
    ) -> Tuple[bool, float, List[str]]:
        """
        Compare two author lists and check for matches.
        
        Args:
            reference_authors: Authors from original reference
            arxiv_authors: Authors from ArXiv paper
            min_matches: Minimum number of matching last names required
            
        Returns:
            Tuple of (match_found, similarity_score, matched_names)
        """
        if not reference_authors or not arxiv_authors:
            return False, 0.0, []
        
        # Parse author lists
        ref_names = AuthorNameNormalizer._parse_author_list(reference_authors)
        arxiv_names = AuthorNameNormalizer._parse_author_list(arxiv_authors)
        
        if not ref_names or not arxiv_names:
            return False, 0.0, []
        
        # Extract last names
        ref_last_names = [AuthorNameNormalizer.extract_last_name(name) for name in ref_names]
        arxiv_last_names = [AuthorNameNormalizer.extract_last_name(name) for name in arxiv_names]
        
        # Filter out empty names
        ref_last_names = [name for name in ref_last_names if name]
        arxiv_last_names = [name for name in arxiv_last_names if name]
        
        # Find exact matches
        exact_matches = []
        for ref_name in ref_last_names:
            for arxiv_name in arxiv_last_names:
                if ref_name == arxiv_name:
                    exact_matches.append(ref_name)
        
        # Find fuzzy matches (for similar names)
        fuzzy_matches = []
        for ref_name in ref_last_names:
            for arxiv_name in arxiv_last_names:
                similarity = SequenceMatcher(None, ref_name, arxiv_name).ratio()
                if similarity >= 0.85 and ref_name not in exact_matches:
                    fuzzy_matches.append(f"{ref_name}~{arxiv_name}")
        
        all_matches = exact_matches + fuzzy_matches
        
        # Calculate similarity score
        total_names = max(len(ref_last_names), len(arxiv_last_names))
        similarity_score = len(all_matches) / total_names if total_names > 0 else 0.0
        
        # Check if minimum matches requirement is met
        has_sufficient_matches = len(exact_matches) >= min_matches
        
        return has_sufficient_matches, similarity_score, all_matches
    
    @staticmethod
    def _parse_author_list(authors_str: str) -> List[str]:
        """Parse author string into individual names."""
        if not authors_str:
            return []
        
        # Common separators for author lists
        separators = [';', ' and ', ' & ', ',']
        
        # Start with the full string
        names = [authors_str]
        
        # Split by each separator
        for sep in separators:
            new_names = []
            for name in names:
                new_names.extend(name.split(sep))
            names = new_names
        
        # Clean up names
        cleaned_names = []
        for name in names:
            name = name.strip()
            if name and len(name) > 1:  # Avoid single characters
                cleaned_names.append(name)
        
        return cleaned_names


class ReferenceValidator:
    """Main validation service for reference matching."""
    
    def __init__(self):
        self.normalizer = AuthorNameNormalizer()
    
    def validate_reference_match(
        self,
        reference: Dict[str, Any],
        arxiv_paper: Dict[str, Any],
        strict_year: bool = False,
        year_tolerance: int = 5
    ) -> ValidationResult:
        """
        Validate if an ArXiv paper matches a reference citation.
        
        Args:
            reference: Original reference data
            arxiv_paper: ArXiv paper metadata
            strict_year: Whether to require year matching
            year_tolerance: Allowed year difference for fuzzy matching
            
        Returns:
            ValidationResult with match decision and details
        """
        reasons = []
        
        # Extract data from inputs
        ref_authors = reference.get('cited_authors', '')
        ref_year = reference.get('cited_year')
        ref_title = reference.get('cited_title', '')
        
        arxiv_authors = arxiv_paper.get('authors', '')
        arxiv_year = arxiv_paper.get('year')
        arxiv_title = arxiv_paper.get('title', '')
        
        # 1. Author validation
        author_match, author_similarity, matched_names = self.normalizer.compare_author_lists(
            ref_authors, arxiv_authors, min_matches=1
        )
        
        if author_match:
            reasons.append(f"Author match found: {matched_names}")
        else:
            reasons.append(f"No author matches: '{ref_authors}' vs '{arxiv_authors}'")
        
        # 2. Year validation
        year_match = self._validate_years(ref_year, arxiv_year, year_tolerance, strict_year)
        
        if year_match:
            reasons.append(f"Year match: {ref_year} ≈ {arxiv_year}")
        else:
            reasons.append(f"Year mismatch: {ref_year} vs {arxiv_year}")
        
        # 3. Title similarity (optional, for additional confidence)
        title_similarity = self._calculate_title_similarity(ref_title, arxiv_title)
        
        if title_similarity > 0.7:
            reasons.append(f"High title similarity: {title_similarity:.2f}")
        elif title_similarity > 0.4:
            reasons.append(f"Moderate title similarity: {title_similarity:.2f}")
        else:
            reasons.append(f"Low title similarity: {title_similarity:.2f}")
        
        # 4. Overall validation decision
        if strict_year:
            is_valid = author_match and year_match
        else:
            # More lenient: author match OR (year match + decent title similarity)
            is_valid = author_match or (year_match and title_similarity > 0.5)
        
        # 5. Calculate confidence score
        confidence_score = self._calculate_confidence(
            author_match, author_similarity, year_match, title_similarity
        )
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            author_match=author_match,
            year_match=year_match,
            title_similarity=title_similarity,
            reasons=reasons
        )
    
    def _validate_years(
        self, 
        ref_year: Optional[int], 
        arxiv_year: Optional[int], 
        tolerance: int,
        strict: bool
    ) -> bool:
        """Validate year matching with tolerance."""
        if not ref_year or not arxiv_year:
            return not strict  # Allow missing years if not strict
        
        try:
            ref_year = int(ref_year)
            arxiv_year = int(arxiv_year)
            return abs(ref_year - arxiv_year) <= tolerance
        except (ValueError, TypeError):
            return not strict
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        title1 = self._normalize_title(title1)
        title2 = self._normalize_title(title2)
        
        # Use sequence matcher for similarity
        return SequenceMatcher(None, title1, title2).ratio()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        
        # Convert to lowercase and normalize unicode
        title = title.lower()
        title = unicodedata.normalize('NFD', title)
        title = ''.join(c for c in title if unicodedata.category(c) != 'Mn')
        
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.sub(r'[^\w\s]', ' ', title).split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return ' '.join(words)
    
    def _calculate_confidence(
        self,
        author_match: bool,
        author_similarity: float,
        year_match: bool,
        title_similarity: float
    ) -> float:
        """Calculate overall confidence score for the match."""
        score = 0.0
        
        # Author matching is most important (50%)
        if author_match:
            score += 0.5
        else:
            score += author_similarity * 0.2  # Partial credit for similarity
        
        # Year matching is second most important (30%)
        if year_match:
            score += 0.3
        
        # Title similarity provides additional confidence (20%)
        score += title_similarity * 0.2
        
        return min(score, 1.0)  # Cap at 1.0


# Convenience function for integration
async def validate_arxiv_reference_match(
    reference: Dict[str, Any],
    arxiv_paper: Dict[str, Any],
    strict_validation: bool = False
) -> ValidationResult:
    """
    Validate if an ArXiv paper matches a reference citation.
    
    Args:
        reference: Original reference data with cited_authors, cited_year, cited_title
        arxiv_paper: ArXiv paper metadata with authors, year, title
        strict_validation: Whether to require both author and year matches
        
    Returns:
        ValidationResult with match decision and confidence
    """
    validator = ReferenceValidator()
    return validator.validate_reference_match(
        reference, 
        arxiv_paper, 
        strict_year=strict_validation
    )