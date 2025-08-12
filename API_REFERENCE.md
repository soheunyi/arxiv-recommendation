# API Reference - ArXiv Recommendation System

Complete API documentation for all implemented components.

## Table of Contents

- [ArXiv Client](#arxiv-client)
- [Database Manager](#database-manager)  
- [Embedding Manager](#embedding-manager)
- [Recommendation Engine](#recommendation-engine)
- [Multi-Agent System](#multi-agent-system)
- [Configuration](#configuration)

---

## ArXiv Client

**Module**: `src.arxiv_recommendation.arxiv_client`

### ArXivClient

Async client for fetching papers from arXiv API with rate limiting.

```python
from arxiv_recommendation.arxiv_client import ArXivClient

async with ArXivClient() as client:
    papers = await client.fetch_recent_papers("cs.AI", max_results=10)
```

#### Methods

##### `fetch_recent_papers(category, max_results=100, days_back=1)`

Fetch recent papers from a specific arXiv category.

**Parameters:**
- `category` (str): arXiv category (e.g., 'cs.AI', 'cs.LG')
- `max_results` (int): Maximum papers to retrieve (default: 100)
- `days_back` (int): Days to look back (default: 1)

**Returns:** `List[PaperMetadata]`

**Example:**
```python
papers = await client.fetch_recent_papers("cs.AI", max_results=50, days_back=2)
```

##### `search_papers(query, max_results=100, category=None)`

Search arXiv papers by query string.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum papers to retrieve
- `category` (str, optional): Category filter

**Returns:** `List[PaperMetadata]`

##### `validate_category(category)`

Validate if category is a recognized arXiv category.

**Parameters:**
- `category` (str): Category to validate

**Returns:** `bool`

### PaperMetadata

Data structure for arXiv paper metadata.

```python
@dataclass
class PaperMetadata:
    id: str                    # arXiv ID (e.g., "2301.12345")
    title: str                 # Paper title
    abstract: str              # Abstract text
    authors: List[str]         # Author names
    category: str              # Primary category
    published_date: datetime   # Publication date
    updated_date: datetime     # Last update date
    arxiv_url: str            # arXiv abstract URL
    pdf_url: str              # PDF download URL
    doi: Optional[str]        # DOI if available
    journal_ref: Optional[str] # Journal reference
```

---

## Database Manager

**Module**: `src.arxiv_recommendation.database`

### DatabaseManager

Async SQLite database manager for papers, ratings, and preferences.

```python
from arxiv_recommendation.database import DatabaseManager

db = DatabaseManager()
await db.initialize()
```

#### Core Methods

##### `initialize()`

Initialize database with tables and indexes.

```python
await db.initialize()
```

##### `store_papers(papers)`

Store paper metadata in database.

**Parameters:**
- `papers` (List[PaperMetadata]): Papers to store

**Returns:** `int` - Number of papers stored

##### `get_unprocessed_papers(limit=100)`

Get papers not yet processed for recommendations.

**Parameters:**
- `limit` (int): Maximum papers to return

**Returns:** `List[Dict[str, Any]]` - Unprocessed papers

##### `store_user_rating(paper_id, rating, notes=None)`

Store user rating for a paper.

**Parameters:**
- `paper_id` (str): Paper ID
- `rating` (int): Rating 1-5
- `notes` (str, optional): Optional notes

**Returns:** `bool` - Success status

##### `get_user_ratings(min_rating=None)`

Get user ratings with paper information.

**Parameters:**
- `min_rating` (int, optional): Minimum rating filter

**Returns:** `List[Dict[str, Any]]` - Ratings with paper data

#### Preferences Methods

##### `get_user_preferences()`

Get all user preferences.

**Returns:** `Dict[str, Any]` - Preferences dictionary

##### `update_user_preference(key, value, value_type)`

Update a single user preference.

**Parameters:**
- `key` (str): Preference key
- `value` (Any): Preference value
- `value_type` (str): Type ('string', 'integer', 'float', 'json', 'boolean')

#### Statistics Methods

##### `get_database_stats()`

Get database statistics.

**Returns:** `Dict[str, Any]` - Statistics including paper counts, ratings, etc.

---

## Embedding Manager

**Module**: `src.arxiv_recommendation.embeddings`

### EmbeddingManager

OpenAI embedding manager with local caching and cost tracking.

```python
from arxiv_recommendation.embeddings import EmbeddingManager

embedding_manager = EmbeddingManager()
embedding = await embedding_manager.get_embedding("paper abstract text")
```

#### Core Methods

##### `get_embedding(text, paper_id=None)`

Get embedding for text with caching.

**Parameters:**
- `text` (str): Text to embed
- `paper_id` (str, optional): Paper ID for logging

**Returns:** `Optional[List[float]]` - Embedding vector or None

##### `get_embeddings_batch(texts, paper_ids=None)`

Get embeddings for multiple texts efficiently.

**Parameters:**
- `texts` (List[str]): Texts to embed
- `paper_ids` (List[str], optional): Paper IDs for logging

**Returns:** `List[Optional[List[float]]]` - List of embeddings

#### Similarity Methods

##### `cosine_similarity(embedding1, embedding2)`

Calculate cosine similarity between embeddings.

**Parameters:**
- `embedding1` (List[float]): First embedding
- `embedding2` (List[float]): Second embedding

**Returns:** `float` - Similarity score (-1 to 1)

##### `find_similar_embeddings(query_embedding, embeddings, top_k=10, min_similarity=0.5)`

Find most similar embeddings to query.

**Parameters:**
- `query_embedding` (List[float]): Query embedding
- `embeddings` (List[Dict]): List of embeddings with metadata
- `top_k` (int): Number of results
- `min_similarity` (float): Minimum similarity threshold

**Returns:** `List[Dict[str, Any]]` - Similar embeddings with scores

#### Utility Methods

##### `get_cache_stats()`

Get cache statistics.

**Returns:** `Dict[str, Any]` - Cache stats including hit rate, size, cost

##### `clear_cache(older_than_days=None)`

Clear embedding cache files.

**Parameters:**
- `older_than_days` (int, optional): Only clear files older than X days

---

## Recommendation Engine

**Module**: `src.arxiv_recommendation.recommendations`

### RecommendationEngine

Personalized recommendation engine with MMR diversity optimization.

```python
from arxiv_recommendation.recommendations import RecommendationEngine

engine = RecommendationEngine()
recommendations = await engine.generate_recommendations(
    paper_embeddings=embeddings,
    user_preferences=preferences,
    top_k=10
)
```

#### Core Methods

##### `generate_recommendations(paper_embeddings, user_preferences, top_k=10)`

Generate personalized recommendations.

**Parameters:**
- `paper_embeddings` (List[Dict]): Paper embeddings with metadata
- `user_preferences` (Dict): User preference settings
- `top_k` (int): Number of recommendations

**Returns:** `List[Dict[str, Any]]` - Recommendations with scores

##### `update_recommendations_feedback(feedback)`

Update algorithm based on user feedback.

**Parameters:**
- `feedback` (List[Dict]): User feedback with ratings

##### `explain_recommendation(paper_recommendation)`

Generate explanation for recommendation.

**Parameters:**
- `paper_recommendation` (Dict): Recommendation to explain

**Returns:** `str` - Human-readable explanation

### RecommendationScore

Scoring breakdown for recommendations.

```python
@dataclass
class RecommendationScore:
    similarity: float    # Semantic similarity (0-1)
    novelty: float      # Temporal novelty (0-1)  
    diversity: float    # Diversity within set (0-1)
    quality: float      # Quality indicators (0-1)
    final_score: float  # Weighted combination
```

---

## Multi-Agent System

**Module**: `src.arxiv_recommendation.agents`

### Agents

#### DataAgent

Handles arXiv paper collection and preprocessing.

```python
from arxiv_recommendation.agents import DataAgent

agent = DataAgent()
message = await agent.process(collect_message)
```

#### RecommendationAgent

Generates embeddings and recommendations.

```python
from arxiv_recommendation.agents import RecommendationAgent

agent = RecommendationAgent()
message = await agent.process(papers_message)
```

#### Coordinator

Orchestrates workflow between agents.

```python
from arxiv_recommendation.agents import Coordinator

coordinator = Coordinator()
result = await coordinator.process(workflow_message)
```

### MultiAgentSystem

Complete system orchestration.

```python
from arxiv_recommendation.agents import MultiAgentSystem

system = MultiAgentSystem()
results = await system.run_daily_workflow()
```

#### Methods

##### `run_daily_workflow()`

Run complete daily recommendation workflow.

**Returns:** `Dict[str, Any]` - Workflow results

##### `process_user_feedback(ratings)`

Process user ratings and update preferences.

**Parameters:**
- `ratings` (List[Dict]): User ratings

**Returns:** `Dict[str, Any]` - Processing results

### Convenience Functions

##### `run_recommendation_system()`

Simple function to run complete system.

**Returns:** `Dict[str, Any]` - System results

```python
from arxiv_recommendation import run_recommendation_system

results = await run_recommendation_system()
print(f"Generated {len(results.get('recommendations', []))} recommendations")
```

---

## Configuration

**Module**: `src.arxiv_recommendation.config`

### Config

Global configuration object loaded from environment.

```python
from arxiv_recommendation.config import config

print(f"API Key: {config.openai_api_key}")
print(f"Budget: ${config.openai_budget_limit}")
```

#### Properties

- `openai_api_key` (str): OpenAI API key
- `embedding_model` (str): Model name (default: 'text-embedding-3-small')
- `embedding_cost_per_token` (float): Cost per token
- `arxiv_categories` (List[str]): Categories to monitor
- `max_daily_papers` (int): Maximum papers per day
- `database_path` (str): Database file path
- `embeddings_path` (str): Embedding cache directory
- `openai_budget_limit` (float): Monthly budget limit

#### Methods

##### `estimate_daily_cost()`

Estimate daily operational cost.

**Returns:** `float` - Estimated daily cost in USD

---

## Error Handling

All async methods may raise:

- `aiohttp.ClientError`: Network/HTTP errors
- `sqlite3.Error`: Database errors  
- `OpenAIError`: API errors
- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing files/directories

## Usage Examples

### Complete Workflow

```python
import asyncio
from arxiv_recommendation import run_recommendation_system

async def main():
    # Run complete system
    results = await run_recommendation_system()
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    recommendations = results.get("recommendations", [])
    print(f"Generated {len(recommendations)} recommendations")
    
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec['title'][:60]}... (Score: {rec['score']:.3f})")

asyncio.run(main())
```

### Custom Pipeline

```python
from arxiv_recommendation.arxiv_client import ArXivClient
from arxiv_recommendation.database import DatabaseManager
from arxiv_recommendation.embeddings import EmbeddingManager
from arxiv_recommendation.recommendations import RecommendationEngine

async def custom_pipeline():
    # Initialize components
    db = DatabaseManager()
    await db.initialize()
    
    embedding_manager = EmbeddingManager()
    engine = RecommendationEngine()
    
    # Fetch papers
    async with ArXivClient() as client:
        papers = await client.fetch_recent_papers("cs.AI", max_results=20)
    
    # Store and process
    await db.store_papers(papers)
    
    # Generate embeddings
    abstracts = [paper.abstract for paper in papers]
    embeddings = await embedding_manager.get_embeddings_batch(abstracts)
    
    # Create recommendations
    paper_embeddings = [
        {"paper_id": paper.id, "embedding": emb}
        for paper, emb in zip(papers, embeddings)
        if emb is not None
    ]
    
    preferences = await db.get_user_preferences()
    recommendations = await engine.generate_recommendations(
        paper_embeddings, preferences, top_k=10
    )
    
    return recommendations
```