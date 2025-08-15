"""
Test configuration and fixtures for ArXiv Recommendation System.

This file provides shared test utilities, fixtures, and educational helpers
that demonstrate how the system components work together.
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Add backend source to Python path for imports
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

# Educational imports to show system structure
from database import DatabaseManager
from config import Config
from arxiv_client import ArXivClient, PaperMetadata
from services.query_service import QueryService
from services.gemini_query_service import GeminiQueryService
from services.collaborative_service import CollaborativeService
from services.reference_service import ReferenceService
from services.openalex_service import OpenAlexService
from services.hybrid_reference_service import HybridReferenceService


# =============================================================================
# Educational Test Fixtures
# =============================================================================

@pytest.fixture
def educational_print():
    """
    EDUCATIONAL: Helper fixture for verbose test output.
    
    This demonstrates how to create test utilities that help with learning.
    Use this in tests to show step-by-step process execution.
    """
    def print_section(title: str, content: str = ""):
        """Print formatted test section for educational purposes."""
        print(f"\n{'='*60}")
        print(f"📚 EDUCATIONAL: {title}")
        print(f"{'='*60}")
        if content:
            print(content)
    
    return print_section


@pytest_asyncio.fixture
async def temp_database():
    """
    EDUCATIONAL: Creates a temporary test database.
    
    This shows how to:
    - Create isolated test environments
    - Set up database schemas for testing
    - Ensure clean state between tests
    
    Returns:
        Database: Configured test database instance
    """
    # Create temporary file for test database
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_file.close()
    
    # Initialize database with test configuration
    db = DatabaseManager(db_path=temp_file.name)
    
    print(f"\n📊 Test Database: Created at {temp_file.name}")
    print(f"🔧 Database Schema: Setting up tables...")
    
    # FIXED: Actually initialize the database tables
    await db.initialize()
    print(f"✅ Database tables created successfully")
    
    yield db
    
    # Cleanup: Remove temporary database file
    print(f"🧹 Cleanup: Removing test database")
    os.unlink(temp_file.name)


@pytest.fixture  
def mock_config():
    """
    EDUCATIONAL: Mock configuration for testing.
    
    This demonstrates how to:
    - Override system configuration for tests
    - Use dependency injection patterns
    - Control external service interactions
    """
    config = Mock(spec=Config)
    config.openai_api_key = "test_openai_key"
    config.gemini_api_key = "test_gemini_key" 
    config.llm_provider = "gemini"
    config.gemini_query_model = "gemini-2.5-flash"
    config.openai_query_model = "gpt-4o"
    config.openai_budget_limit = 20.0
    
    print(f"\n⚙️  Mock Config: LLM Provider = {config.llm_provider}")
    print(f"🔑 API Keys: OpenAI = {config.openai_api_key[:10]}..., Gemini = {config.gemini_api_key[:10]}...")
    
    return config


@pytest.fixture
def sample_paper():
    """
    EDUCATIONAL: Sample paper data for testing.
    
    This shows the structure of paper data throughout the system.
    Real papers have this exact format from ArXiv API.
    """
    paper = PaperMetadata(
        id="2301.12345",
        title="Large Language Models for Scientific Discovery",
        abstract="This paper explores the application of large language models to accelerate scientific discovery processes. We demonstrate how LLMs can assist researchers in literature review, hypothesis generation, and experimental design.",
        authors=["John Smith", "Jane Doe", "Alice Johnson"],
        category="cs.AI",
        published_date=datetime(2023, 1, 15),
        updated_date=datetime(2023, 1, 16),
        arxiv_url="https://arxiv.org/abs/2301.12345",
        pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
        doi="10.1000/test-doi",
        journal_ref="Test Journal 2023"
    )
    
    print(f"\n📄 Sample Paper: {paper.id}")
    print(f"📝 Title: {paper.title[:50]}...")
    print(f"👥 Authors: {', '.join(paper.authors)}")
    print(f"🏷️  Category: {paper.category}")
    
    return paper


@pytest.fixture
def sample_references():
    """
    EDUCATIONAL: Sample reference data showing citation relationships.
    
    This demonstrates the reference tracking system structure and
    how papers connect to each other through citations.
    """
    references = [
        {
            "cited_paper_id": "2012.11841",
            "cited_title": "Attention Is All You Need",
            "cited_authors": "Vaswani, A., Shazeer, N., Parmar, N.",
            "cited_year": 2017,
            "reference_context": "[1] Vaswani, A., Shazeer, N., Parmar, N., et al. Attention is all you need. In Advances in neural information processing systems, pages 5998-6008, 2017.",
            "is_arxiv_paper": True,
            "citation_number": 1
        },
        {
            "cited_paper_id": None,
            "cited_title": "Deep Learning",
            "cited_authors": "Goodfellow, I., Bengio, Y., Courville, A.",
            "cited_year": 2016,
            "reference_context": "[2] Goodfellow, I., Bengio, Y., Courville, A. Deep Learning. MIT Press, 2016.",
            "is_arxiv_paper": False,
            "citation_number": 2
        }
    ]
    
    print(f"\n🔗 Sample References: {len(references)} citations")
    for ref in references:
        print(f"   {ref['citation_number']}. {ref['cited_title']} ({'ArXiv' if ref['is_arxiv_paper'] else 'External'})")
    
    return references


@pytest.fixture
def mock_arxiv_response():
    """
    EDUCATIONAL: Mock ArXiv API response.
    
    This shows the actual XML structure returned by ArXiv API
    and how it's parsed into our PaperMetadata objects.
    """
    xml_response = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/2301.12345v1</id>
            <title>Large Language Models for Scientific Discovery</title>
            <summary>This paper explores the application of large language models...</summary>
            <author>
                <name>John Smith</name>
            </author>
            <author>
                <name>Jane Doe</name>
            </author>
            <published>2023-01-15T00:00:00Z</published>
            <updated>2023-01-16T00:00:00Z</updated>
            <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/>
        </entry>
    </feed>"""
    
    print(f"\n🌐 Mock ArXiv Response: Valid XML with 1 paper")
    print(f"📋 Response Structure: Atom feed format with paper metadata")
    
    return xml_response


@pytest.fixture
def mock_openai_response():
    """
    EDUCATIONAL: Mock OpenAI API response for query generation.
    
    This shows the structured JSON format expected from GPT-4
    and how it's used to generate ArXiv search queries.
    """
    response = {
        "topic": "machine learning",
        "search_queries": [
            {
                "query": 'ti:"machine learning" AND cat:cs.LG',
                "priority": "high", 
                "description": "Papers with machine learning in title, CS.LG category"
            },
            {
                "query": 'abs:"deep learning" AND submittedDate:[20240101 TO 20241231]',
                "priority": "medium",
                "description": "Recent deep learning papers in abstracts"
            }
        ],
        "categories": ["cs.LG", "cs.AI", "stat.ML"],
        "filter_keywords": ["machine learning", "deep learning", "neural networks"],
        "related_terms": ["artificial intelligence", "ML", "data science"]
    }
    
    print(f"\n🤖 Mock OpenAI Response: {len(response['search_queries'])} queries generated")
    print(f"🎯 Categories: {', '.join(response['categories'])}")
    print(f"🔍 Keywords: {', '.join(response['filter_keywords'])}")
    
    return response


@pytest.fixture
def mock_gemini_response():
    """
    EDUCATIONAL: Mock Gemini API response for query generation.
    
    This shows how Gemini's response format compares to OpenAI
    and demonstrates the provider abstraction layer.
    """
    response = {
        "topic": "quantum computing",
        "search_queries": [
            {
                "query": 'ti:"quantum computing" AND cat:quant-ph',
                "priority": "high",
                "description": "Quantum computing papers in quantum physics category"
            },
            {
                "query": 'abs:"quantum algorithm" AND au:"Peter Shor"',
                "priority": "medium", 
                "description": "Quantum algorithms by notable researchers"
            },
            {
                "query": 'cat:quant-ph AND quantum',
                "priority": "low",
                "description": "General quantum physics papers"
            },
            {
                "query": 'abs:"quantum" AND cat:cs.ET',
                "priority": "medium",
                "description": "Quantum papers in computer science"
            },
            {
                "query": 'ti:"qubit" OR ti:"superposition"',
                "priority": "high",
                "description": "Papers about qubits and superposition"
            },
            {
                "query": 'au:"John Preskill" AND quantum',
                "priority": "low",
                "description": "Quantum papers by notable researchers"
            },
            {
                "query": 'abs:"quantum gate" AND cat:quant-ph',
                "priority": "medium",
                "description": "Quantum gate implementations"
            },
            {
                "query": 'ti:"quantum error" OR abs:"error correction"',
                "priority": "high",
                "description": "Quantum error correction research"
            }
        ],
        "categories": ["quant-ph", "cs.ET", "math-ph"],
        "filter_keywords": ["quantum", "qubit", "superposition"],
        "related_terms": ["quantum mechanics", "entanglement", "quantum gate"]
    }
    
    print(f"\n💎 Mock Gemini Response: {len(response['search_queries'])} queries generated")
    print(f"🏷️  Categories: {', '.join(response['categories'])}")
    print(f"🔎 Keywords: {', '.join(response['filter_keywords'])}")
    
    return response


@pytest.fixture
def sample_openalex_data():
    """
    EDUCATIONAL: Sample OpenAlex API response data.
    
    This shows the structure of data returned by OpenAlex API
    and how it's used in the two-stage reference system.
    """
    openalex_work = {
        "id": "https://openalex.org/W2345678901",
        "title": "Large Language Models for Scientific Discovery",
        "cited_by_count": 42,
        "referenced_works": [
            "https://openalex.org/W1234567890",
            "https://openalex.org/W0987654321"
        ],
        "topics": [
            {"id": "T10", "display_name": "Machine Learning"},
            {"id": "T15", "display_name": "Natural Language Processing"}
        ],
        "publication_date": "2023-01-15",
        "created_date": "2023-01-16",
        "authorships": [
            {"author": {"display_name": "John Smith"}},
            {"author": {"display_name": "Jane Doe"}}
        ],
        "doi": "10.1000/test-doi",
        "is_retracted": False,
        "is_paratext": False,
        "open_access": {"is_oa": True}
    }
    
    print(f"\n🌐 Sample OpenAlex Work: {openalex_work['id']}")
    print(f"📊 Citations: {openalex_work['cited_by_count']}")
    print(f"🔗 References: {len(openalex_work['referenced_works'])}")
    print(f"🏷️  Topics: {len(openalex_work['topics'])}")
    
    return openalex_work


@pytest.fixture
def sample_hybrid_references():
    """
    EDUCATIONAL: Sample hybrid reference data from two-stage system.
    
    This demonstrates the enhanced reference structure that combines
    ArXiv HTML parsing with OpenAlex enrichment data.
    """
    references = [
        {
            # ArXiv-only reference (Stage 1)
            "cited_paper_id": "2012.11841",
            "cited_title": "Attention Is All You Need",
            "cited_authors": "Vaswani, A., Shazeer, N., Parmar, N.",
            "cited_year": 2017,
            "reference_context": "[1] Vaswani et al. introduced the Transformer architecture...",
            "citation_number": 1,
            "is_arxiv_paper": True,
            "source": "arxiv",
            "confidence_score": 0.85,
            "openalex_work_id": None,
            "has_openalex_data": False
        },
        {
            # Enhanced reference (Stage 1 + Stage 2)
            "cited_paper_id": "1706.03762",
            "cited_title": "Attention Is All You Need (Enhanced)",
            "cited_authors": "Vaswani, A., Shazeer, N., Parmar, N.",
            "cited_year": 2017,
            "reference_context": "[1] Vaswani et al. introduced the Transformer architecture...",
            "citation_number": 1,
            "is_arxiv_paper": True,
            "source": "hybrid",
            "confidence_score": 0.95,
            "openalex_work_id": "https://openalex.org/W2963650001",
            "has_openalex_data": True,
            "cited_by_count": 28749,
            "publication_year": 2017,
            "is_open_access": True,
            "topics": [{"id": "T10", "display_name": "Machine Learning"}]
        },
        {
            # External reference (Stage 1 only)
            "cited_paper_id": None,
            "cited_title": "Deep Learning",
            "cited_authors": "Goodfellow, I., Bengio, Y., Courville, A.",
            "cited_year": 2016,
            "reference_context": "[2] Goodfellow et al. provide comprehensive coverage...",
            "citation_number": 2,
            "is_arxiv_paper": False,
            "source": "arxiv",
            "confidence_score": 0.80,
            "openalex_work_id": None,
            "has_openalex_data": False
        }
    ]
    
    print(f"\n🔗 Sample Hybrid References: {len(references)} references")
    enhanced_count = sum(1 for r in references if r['has_openalex_data'])
    arxiv_count = sum(1 for r in references if r['is_arxiv_paper'])
    
    print(f"   ✨ Enhanced with OpenAlex: {enhanced_count}")
    print(f"   📝 ArXiv papers: {arxiv_count}")
    print(f"   📖 External papers: {len(references) - arxiv_count}")
    
    return references


@pytest.fixture
async def mock_hybrid_service():
    """
    EDUCATIONAL: Mock hybrid reference service for testing.
    
    This demonstrates how to mock complex service compositions
    while maintaining realistic interface behavior.
    """
    service = Mock(spec=HybridReferenceService)
    
    # Mock async methods with realistic return values
    service.fetch_references_stage1 = AsyncMock(return_value=[])
    service.enrich_with_openalex = AsyncMock(return_value=False)
    service.fetch_references_hybrid = AsyncMock(return_value={
        "stage1_success": True,
        "stage2_success": False,
        "references_found": 0,
        "source": "arxiv",
        "openalex_available": False
    })
    service.get_enhanced_references = AsyncMock(return_value=[])
    service.get_citation_network = AsyncMock(return_value={
        "citing_papers": [],
        "cited_papers": [],
        "citation_count": 0
    })
    service.batch_enrich_papers = AsyncMock(return_value={
        "total_papers": 0,
        "enriched_count": 0,
        "not_found_count": 0,
        "error_count": 0
    })
    
    print(f"\n🎭 Mock Hybrid Service: Ready for testing")
    print(f"⚡ Async methods: All mocked with realistic responses")
    
    return service


# =============================================================================
# Async Test Utilities
# =============================================================================

@pytest.fixture
def event_loop():
    """
    EDUCATIONAL: Event loop for async tests.
    
    This is required for testing async/await functions.
    It shows how modern Python applications use asyncio for concurrency.
    """
    print(f"\n🔄 Event Loop: Setting up asyncio for async tests")
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_arxiv_client():
    """
    EDUCATIONAL: Mock ArXiv client for testing.
    
    This demonstrates how to mock external API calls to:
    - Make tests independent of external services
    - Control response timing and content
    - Test error conditions safely
    """
    client = Mock(spec=ArXivClient)
    
    # Mock async methods with realistic responses
    client.search_papers = AsyncMock(return_value=[])
    client.fetch_recent_papers = AsyncMock(return_value=[])
    client.fetch_paper_references = AsyncMock(return_value=[])
    
    print(f"\n📡 Mock ArXiv Client: Ready for testing")
    print(f"🎭 Mocked Methods: search_papers, fetch_recent_papers, fetch_paper_references")
    
    return client


# =============================================================================
# Test Data Generators  
# =============================================================================

def generate_test_papers(count: int = 5) -> list[PaperMetadata]:
    """
    EDUCATIONAL: Generate test paper data.
    
    This shows how to create realistic test data that mimics
    actual ArXiv paper structure and content patterns.
    """
    papers = []
    categories = ["cs.AI", "cs.LG", "cs.CL", "stat.ML", "quant-ph"]
    
    for i in range(count):
        paper = PaperMetadata(
            id=f"2024.{1000 + i:05d}",
            title=f"Test Paper {i+1}: Advanced Research in Category {categories[i % len(categories)]}",
            abstract=f"This is a test abstract for paper {i+1}. It demonstrates the structure and content of academic papers in the ArXiv recommendation system.",
            authors=[f"Author {j+1}" for j in range((i % 3) + 1)],
            category=categories[i % len(categories)],
            published_date=datetime.now() - timedelta(days=i),
            updated_date=datetime.now() - timedelta(days=i-1),
            arxiv_url=f"https://arxiv.org/abs/2024.{1000 + i:05d}",
            pdf_url=f"https://arxiv.org/pdf/2024.{1000 + i:05d}.pdf"
        )
        papers.append(paper)
    
    print(f"\n📚 Generated {count} test papers")
    print(f"🏷️  Categories: {', '.join(set(p.category for p in papers))}")
    
    return papers


# =============================================================================
# Performance Testing Utilities
# =============================================================================

@pytest.fixture
def performance_timer():
    """
    EDUCATIONAL: Timer for performance testing.
    
    This demonstrates how to measure and validate system performance
    to ensure the application meets speed requirements.
    """
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            print(f"\n⏱️  Performance Timer: Started")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print(f"⏱️  Performance Timer: Completed in {duration:.3f} seconds")
            
        @property
        def duration(self) -> float:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
    
    return Timer


# =============================================================================
# Error Testing Utilities
# =============================================================================

@pytest.fixture
def error_scenarios():
    """
    EDUCATIONAL: Common error scenarios for testing.
    
    This shows how to test error handling and system resilience
    by simulating various failure conditions.
    """
    scenarios = {
        "network_timeout": {
            "exception": asyncio.TimeoutError,
            "message": "Network request timed out",
            "description": "Simulates slow or unresponsive external services"
        },
        "api_key_invalid": {
            "exception": ValueError,
            "message": "Invalid API key provided", 
            "description": "Tests authentication failure handling"
        },
        "rate_limit_exceeded": {
            "exception": Exception,
            "message": "Rate limit exceeded",
            "description": "Tests API rate limiting responses"
        },
        "malformed_response": {
            "exception": ValueError,
            "message": "Invalid response format",
            "description": "Tests parsing of unexpected API responses"
        }
    }
    
    print(f"\n❌ Error Scenarios: {len(scenarios)} test cases available")
    for name, scenario in scenarios.items():
        print(f"   • {name}: {scenario['description']}")
    
    return scenarios


# =============================================================================
# Integration Test Helpers
# =============================================================================

@pytest.fixture
def integration_environment(temp_database, mock_config):
    """
    EDUCATIONAL: Complete test environment for integration tests.
    
    This demonstrates how to set up a full system environment
    that includes all necessary components working together.
    """
    environment = {
        "database": temp_database,
        "config": mock_config,
        "arxiv_client": None,  # Will be mocked in actual tests
        "query_service": None,  # Will be initialized in tests
        "collaborative_service": None,  # Will be initialized in tests
    }
    
    print(f"\n🏗️  Integration Environment: Ready")
    print(f"💾 Database: {temp_database.db_path}")
    print(f"⚙️  Config: Mocked with test values")
    
    return environment


# =============================================================================
# Custom Test Markers
# =============================================================================

def pytest_configure(config):
    """
    EDUCATIONAL: Custom test markers configuration.
    
    This shows how to organize tests by type and complexity
    for better test management and execution control.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"  
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and timing tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "external: Tests that require external services"
    )


# =============================================================================
# Test Session Hooks
# =============================================================================

def pytest_sessionstart(session):
    """
    EDUCATIONAL: Test session startup hook.
    
    This demonstrates how to set up global test configuration
    and display helpful information about the test run.
    """
    print("\n" + "="*80)
    print("🧪 ArXiv Recommendation System - Test Suite Starting")
    print("="*80)
    print("📚 EDUCATIONAL PURPOSE: These tests demonstrate system architecture")
    print("🎯 LEARNING GOALS:")
    print("   • Understand data flow through the system")
    print("   • See how components interact and depend on each other")
    print("   • Learn testing patterns for async Python applications")
    print("   • Observe error handling and edge case management")
    print("="*80)


def pytest_sessionfinish(session, exitstatus):
    """
    EDUCATIONAL: Test session cleanup hook.
    
    This shows how to perform cleanup and provide summary
    information about test execution results.
    """
    print("\n" + "="*80)
    print("🏁 Test Suite Completed")
    print("="*80)
    print(f"📊 Exit Status: {exitstatus}")
    print("💡 EDUCATIONAL SUMMARY:")
    print("   • Tests demonstrate real system behavior")
    print("   • Mocks show how to isolate components")
    print("   • Fixtures provide reusable test infrastructure")
    print("   • Integration tests show component relationships")
    print("="*80)