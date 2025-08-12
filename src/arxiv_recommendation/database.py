"""Database manager for ArXiv recommendation system using SQLite."""

import sqlite3
import aiosqlite
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict

from .config import config
from .arxiv_client import PaperMetadata

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Async SQLite database manager for ArXiv papers and user preferences.

    Features:
    - Paper storage with full metadata
    - User ratings and preferences
    - Embedding storage and caching
    - Migration system for schema updates
    - Efficient indexing for similarity search
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.database_path
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize database with required tables and indexes."""
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await self._create_indexes(db)
            await self._initialize_preferences(db)
            # Ensure schema migrations (new columns) are applied
            await self._migrate_schema(db)
            await db.commit()

        logger.info(f"Database initialized at {self.db_path}")

    async def _create_tables(self, db: aiosqlite.Connection):
        """Create all required database tables."""

        # Papers table - stores arXiv paper metadata
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,  -- JSON array of author names
                category TEXT NOT NULL,
                published_date DATE NOT NULL,
                updated_date DATE NOT NULL,
                arxiv_url TEXT NOT NULL,
                pdf_url TEXT NOT NULL,
                doi TEXT,
                journal_ref TEXT,
                processed BOOLEAN DEFAULT FALSE,
                embedding_generated BOOLEAN DEFAULT FALSE,
                -- Current scoring snapshot for quick reads
                current_score REAL,
                score_updated_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # User ratings table - stores user feedback on papers
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(paper_id)  -- One rating per paper
            )
        """
        )

        # User preferences table - stores system preferences
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                value_type TEXT NOT NULL,  -- 'string', 'integer', 'float', 'json', 'boolean'
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Embeddings table - stores paper embeddings for similarity search
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_embeddings (
                paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,  -- Serialized numpy array
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Search history table - tracks user search patterns
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                category TEXT,
                results_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Recommendations history - tracks generated recommendations
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                score REAL NOT NULL,
                rank_position INTEGER NOT NULL,
                algorithm_version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    async def _create_indexes(self, db: aiosqlite.Connection):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(category)",
            "CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_processed ON papers(processed)",
            "CREATE INDEX IF NOT EXISTS idx_papers_embedding_generated ON papers(embedding_generated)",
            "CREATE INDEX IF NOT EXISTS idx_user_ratings_paper_id ON user_ratings(paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_ratings_rating ON user_ratings(rating)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations_history(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at DESC)",
        ]

        for index_sql in indexes:
            await db.execute(index_sql)

    async def _initialize_preferences(self, db: aiosqlite.Connection):
        """Initialize default user preferences."""
        default_preferences = [
            (
                "min_rating_threshold",
                "3.5",
                "float",
                "Minimum rating threshold for recommendations",
            ),
            (
                "max_recommendations_per_day",
                "10",
                "integer",
                "Maximum recommendations to show per day",
            ),
            (
                "preferred_categories",
                json.dumps(config.arxiv_categories),
                "json",
                "Preferred arXiv categories",
            ),
            (
                "embedding_similarity_threshold",
                "0.7",
                "float",
                "Minimum similarity score for recommendations",
            ),
            (
                "diversity_weight",
                "0.3",
                "float",
                "Weight for diversity in recommendations (0-1)",
            ),
            (
                "novelty_weight",
                "0.2",
                "float",
                "Weight for novelty in recommendations (0-1)",
            ),
            (
                "last_recommendation_date",
                "",
                "string",
                "Last date recommendations were generated",
            ),
        ]

        for key, value, value_type, description in default_preferences:
            await db.execute(
                """
                INSERT OR IGNORE INTO user_preferences (key, value, value_type, description)
                VALUES (?, ?, ?, ?)
            """,
                (key, value, value_type, description),
            )

    async def _migrate_schema(self, db: aiosqlite.Connection):
        """Apply lightweight schema migrations (idempotent)."""
        # Ensure papers.current_score exists
        await self._ensure_column(db, "papers", "current_score", "REAL")
        await self._ensure_column(db, "papers", "score_updated_at", "TIMESTAMP")

    async def _ensure_column(
        self, db: aiosqlite.Connection, table: str, column: str, col_type: str
    ):
        """Add a column to a table if it doesn't already exist."""
        try:
            cursor = await db.execute(f"PRAGMA table_info({table})")
            rows = await cursor.fetchall()
            col_names = {row[1] for row in rows}  # second field is name
            if column not in col_names:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except Exception as e:
            logger.error(f"Failed ensuring column {table}.{column}: {e}")

    async def store_papers(self, papers: List[PaperMetadata]) -> int:
        """Store papers in database, avoiding duplicates."""
        if not papers:
            return 0

        stored_count = 0

        async with aiosqlite.connect(self.db_path) as db:
            for paper in papers:
                try:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO papers (
                            id, title, abstract, authors, category, published_date, updated_date,
                            arxiv_url, pdf_url, doi, journal_ref, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            paper.id,
                            paper.title,
                            paper.abstract,
                            json.dumps(paper.authors),
                            paper.category,
                            paper.published_date,
                            paper.updated_date,
                            paper.arxiv_url,
                            paper.pdf_url,
                            paper.doi,
                            paper.journal_ref,
                            datetime.now(),
                        ),
                    )
                    stored_count += 1

                except Exception as e:
                    logger.error(f"Failed to store paper {paper.id}: {e}")

            await db.commit()

        logger.info(f"Stored {stored_count} papers in database")
        return stored_count

    async def get_unprocessed_papers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get papers that haven't been processed for recommendations."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT id, title, abstract, authors, category, published_date, arxiv_url
                FROM papers 
                WHERE processed = FALSE 
                ORDER BY published_date DESC 
                LIMIT ?
            """,
                (limit,),
            )

            rows = await cursor.fetchall()

            papers = []
            for row in rows:
                papers.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "authors": json.loads(row["authors"]),
                        "category": row["category"],
                        "published_date": row["published_date"],
                        "arxiv_url": row["arxiv_url"],
                    }
                )

            return papers

    async def mark_papers_processed(self, paper_ids: List[str]):
        """Mark papers as processed for recommendations."""
        if not paper_ids:
            return

        async with aiosqlite.connect(self.db_path) as db:
            placeholders = ",".join("?" * len(paper_ids))
            await db.execute(
                f"""
                UPDATE papers 
                SET processed = TRUE, updated_at = ? 
                WHERE id IN ({placeholders})
            """,
                [datetime.now()] + paper_ids,
            )
            await db.commit()

    async def store_user_rating(
        self, paper_id: str, rating: int, notes: Optional[str] = None
    ) -> bool:
        """Store or update user rating for a paper."""
        if not (1 <= rating <= 5):
            logger.error(f"Invalid rating: {rating}. Must be between 1 and 5.")
            return False

        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO user_ratings (paper_id, rating, notes, updated_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (paper_id, rating, notes, datetime.now()),
                )
                await db.commit()
                logger.info(f"Stored rating {rating} for paper {paper_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to store rating for paper {paper_id}: {e}")
                return False

    async def get_user_ratings(
        self, min_rating: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get user ratings with paper information."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT r.paper_id, r.rating, r.notes, r.created_at,
                       p.title, p.abstract, p.category, p.authors
                FROM user_ratings r
                JOIN papers p ON r.paper_id = p.id
            """
            params = []

            if min_rating is not None:
                query += " WHERE r.rating >= ?"
                params.append(min_rating)

            query += " ORDER BY r.created_at DESC"

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            ratings = []
            for row in rows:
                ratings.append(
                    {
                        "paper_id": row["paper_id"],
                        "rating": row["rating"],
                        "notes": row["notes"],
                        "created_at": row["created_at"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "category": row["category"],
                        "authors": json.loads(row["authors"]),
                    }
                )

            return ratings

    async def get_user_preferences(self) -> Dict[str, Any]:
        """Get all user preferences as a dictionary."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT key, value, value_type FROM user_preferences
            """
            )
            rows = await cursor.fetchall()

            preferences = {}
            for row in rows:
                key, value, value_type = row["key"], row["value"], row["value_type"]

                # Convert values to appropriate types
                if value_type == "integer":
                    preferences[key] = int(value)
                elif value_type == "float":
                    preferences[key] = float(value)
                elif value_type == "boolean":
                    preferences[key] = value.lower() == "true"
                elif value_type == "json":
                    preferences[key] = json.loads(value) if value else {}
                else:
                    preferences[key] = value

            return preferences

    async def update_user_preference(self, key: str, value: Any, value_type: str):
        """Update a single user preference."""
        # Convert value to string for storage
        if value_type == "json":
            str_value = json.dumps(value)
        elif value_type == "boolean":
            str_value = str(value).lower()
        else:
            str_value = str(value)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE user_preferences 
                SET value = ?, updated_at = ? 
                WHERE key = ?
            """,
                (str_value, datetime.now(), key),
            )
            await db.commit()

    async def update_user_preferences(self, preferences_data: Dict[str, Any]):
        """Update multiple user preferences."""
        if "ratings" in preferences_data:
            # Store new ratings
            for rating_data in preferences_data["ratings"]:
                await self.store_user_rating(
                    paper_id=rating_data["paper_id"],
                    rating=rating_data["rating"],
                    notes=rating_data.get("notes"),
                )

        # Update other preferences
        for key, value in preferences_data.items():
            if key != "ratings":
                # Determine value type
                if isinstance(value, bool):
                    value_type = "boolean"
                elif isinstance(value, int):
                    value_type = "integer"
                elif isinstance(value, float):
                    value_type = "float"
                elif isinstance(value, (dict, list)):
                    value_type = "json"
                else:
                    value_type = "string"

                await self.update_user_preference(key, value, value_type)

    async def store_embeddings(self, embeddings: List[Dict[str, Any]]):
        """Store paper embeddings for similarity search."""
        if not embeddings:
            return

        import numpy as np

        async with aiosqlite.connect(self.db_path) as db:
            stored_count = 0

            for emb_data in embeddings:
                try:
                    # Serialize numpy array to bytes
                    embedding_bytes = np.array(emb_data["embedding"]).tobytes()

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO paper_embeddings 
                        (paper_id, embedding, model_name, dimensions)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            emb_data["paper_id"],
                            embedding_bytes,
                            emb_data.get("model_name", config.embedding_model),
                            len(emb_data["embedding"]),
                        ),
                    )
                    stored_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to store embedding for {emb_data['paper_id']}: {e}"
                    )

            await db.commit()
            logger.info(f"Stored {stored_count} embeddings in database")

    async def get_embeddings(
        self, paper_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get stored embeddings."""
        import numpy as np

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                cursor = await db.execute(
                    f"""
                    SELECT paper_id, embedding, model_name, dimensions
                    FROM paper_embeddings 
                    WHERE paper_id IN ({placeholders})
                """,
                    paper_ids,
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT paper_id, embedding, model_name, dimensions
                    FROM paper_embeddings
                """
                )

            rows = await cursor.fetchall()

            embeddings = []
            for row in rows:
                # Deserialize numpy array from bytes
                embedding_array = np.frombuffer(row["embedding"], dtype=np.float32)

                embeddings.append(
                    {
                        "paper_id": row["paper_id"],
                        "embedding": embedding_array.tolist(),
                        "model_name": row["model_name"],
                        "dimensions": row["dimensions"],
                    }
                )

            return embeddings

    async def store_recommendations(
        self, recommendations: List[Dict[str, Any]], algorithm_version: str = "v1.0"
    ):
        """Store generated recommendations for tracking."""
        if not recommendations:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Clear old recommendations (keep last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            await db.execute(
                """
                DELETE FROM recommendations_history 
                WHERE created_at < ?
            """,
                (cutoff_date,),
            )

            # Store new recommendations
            for i, rec in enumerate(recommendations):
                await db.execute(
                    """
                    INSERT INTO recommendations_history 
                    (paper_id, score, rank_position, algorithm_version)
                    VALUES (?, ?, ?, ?)
                """,
                    (rec["paper_id"], rec["score"], i + 1, algorithm_version),
                )

                # Update current score snapshot on papers table
                try:
                    await db.execute(
                        """
                        UPDATE papers 
                        SET current_score = ?, score_updated_at = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            rec["score"],
                            datetime.now(),
                            datetime.now(),
                            rec["paper_id"],
                        ),
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to update current_score for {rec['paper_id']}: {e}"
                    )

            await db.commit()

    async def get_recent_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get papers with most recent current scores for display (sorted by score)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT p.id, p.title, p.abstract, p.category, p.authors,
                       p.published_date, p.arxiv_url, p.pdf_url,
                       r.rating, r.notes,
                       p.current_score, p.score_updated_at
                FROM papers p
                LEFT JOIN user_ratings r ON p.id = r.paper_id
                WHERE p.current_score IS NOT NULL
                ORDER BY p.current_score DESC, p.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()

            papers: List[Dict[str, Any]] = []
            for row in rows:
                papers.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "category": row["category"],
                        "authors": json.loads(row["authors"]) if row["authors"] else [],
                        "published_date": row["published_date"],
                        "arxiv_url": row["arxiv_url"],
                        "pdf_url": row["pdf_url"],
                        "rating": row["rating"] if row["rating"] else 0,
                        "notes": row["notes"],
                        "score": row["current_score"],
                        "score_updated_at": row["score_updated_at"],
                    }
                )

            return papers

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}

            # Paper counts
            cursor = await db.execute("SELECT COUNT(*) FROM papers")
            stats["total_papers"] = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT COUNT(*) FROM papers WHERE processed = TRUE"
            )
            stats["processed_papers"] = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT COUNT(*) FROM papers WHERE embedding_generated = TRUE"
            )
            stats["papers_with_embeddings"] = (await cursor.fetchone())[0]

            # User ratings
            cursor = await db.execute("SELECT COUNT(*) FROM user_ratings")
            stats["user_ratings"] = (await cursor.fetchone())[0]

            # Recent activity
            week_ago = datetime.now() - timedelta(days=7)
            cursor = await db.execute(
                "SELECT COUNT(*) FROM papers WHERE created_at >= ?", (week_ago,)
            )
            stats["papers_last_week"] = (await cursor.fetchone())[0]

            return stats

    async def get_papers_with_ratings(self) -> List[Dict[str, Any]]:
        """Get all papers with their ratings (if any)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT p.id, p.title, p.abstract, p.category, p.authors, 
                       p.published_date, p.arxiv_url, p.pdf_url,
                       r.rating, r.notes, r.created_at as rating_date
                FROM papers p
                LEFT JOIN user_ratings r ON p.id = r.paper_id
                ORDER BY p.created_at DESC
            """
            )
            rows = await cursor.fetchall()

            papers = []
            for row in rows:
                papers.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "category": row["category"],
                        "authors": json.loads(row["authors"]) if row["authors"] else [],
                        "published_date": row["published_date"],
                        "arxiv_url": row["arxiv_url"],
                        "pdf_url": row["pdf_url"],
                        "rating": row["rating"] if row["rating"] else 0,
                        "notes": row["notes"],
                        "rating_date": row["rating_date"],
                    }
                )

            return papers

    async def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers from the database."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT p.id, p.title, p.abstract, p.category, p.authors, 
                       p.published_date, p.arxiv_url, p.pdf_url, p.created_at,
                       p.current_score, p.score_updated_at,
                       r.rating, r.notes
                FROM papers p
                LEFT JOIN user_ratings r ON p.id = r.paper_id
                ORDER BY p.created_at DESC
            """
            )
            rows = await cursor.fetchall()

            papers = []
            for row in rows:
                papers.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "category": row["category"],
                        "authors": json.loads(row["authors"]) if row["authors"] else [],
                        "published_date": row["published_date"],
                        "arxiv_url": row["arxiv_url"],
                        "pdf_url": row["pdf_url"],
                        "created_at": row["created_at"],
                        "current_score": row["current_score"],
                        "score_updated_at": row["score_updated_at"],
                        "rating": row["rating"] if row["rating"] else 0,
                        "notes": row["notes"],
                    }
                )

            return papers

    async def get_recent_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent papers with ratings."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT p.id, p.title, p.abstract, p.category, p.authors, 
                       p.published_date, p.arxiv_url, p.pdf_url, p.created_at,
                       r.rating, r.notes
                FROM papers p
                LEFT JOIN user_ratings r ON p.id = r.paper_id
                ORDER BY p.created_at DESC
                LIMIT ?
            """,
                (limit,),
            )
            rows = await cursor.fetchall()

            papers = []
            for row in rows:
                papers.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "category": row["category"],
                        "authors": json.loads(row["authors"]) if row["authors"] else [],
                        "published_date": row["published_date"],
                        "arxiv_url": row["arxiv_url"],
                        "pdf_url": row["pdf_url"],
                        "created_at": row["created_at"],
                        "rating": row["rating"] if row["rating"] else 0,
                        "notes": row["notes"],
                        "score": 0.5,  # Default score for display
                    }
                )

            return papers
