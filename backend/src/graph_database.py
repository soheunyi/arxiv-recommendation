"""
Graph Database Manager for Citation Networks

This module extends the existing SQLite database with graph-specific tables
and operations for managing citation networks as graph data structures.

Features:
- Node storage for papers with metadata
- Edge storage for citation relationships 
- Graph traversal algorithms using CTEs
- Network analysis metrics (centrality, clustering)
- Temporal graph evolution tracking
"""

import asyncio
import aiosqlite
import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from backend.src.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the citation graph."""
    id: str
    node_type: str  # 'paper', 'author', 'venue', 'topic'
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class GraphEdge:
    """Represents an edge in the citation graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: str  # 'cites', 'authored_by', 'published_in', 'similar_to'
    properties: Dict[str, Any]
    weight: float
    created_at: datetime


@dataclass
class GraphMetrics:
    """Network analysis metrics for nodes."""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    clustering_coefficient: float
    computed_at: datetime


class GraphDatabaseManager:
    """
    Graph database manager that extends SQLite with graph-specific operations.
    
    This class provides graph database functionality on top of the existing
    SQLite database, adding nodes, edges, and graph traversal capabilities
    specifically designed for citation network analysis.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.db_path = db_manager.db_path
        
    async def initialize_graph_schema(self):
        """Initialize graph-specific tables and indexes."""
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_graph_tables(db)
            await self._create_graph_indexes(db)
            await db.commit()
        
        logger.info("Graph database schema initialized")
    
    async def _create_graph_tables(self, db: aiosqlite.Connection):
        """Create graph-specific database tables."""
        
        # Graph nodes table - stores all types of entities
        await db.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL CHECK (node_type IN ('paper', 'author', 'venue', 'topic', 'keyword')),
                properties TEXT NOT NULL,  -- JSON object with node-specific data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Graph edges table - stores relationships between nodes
        await db.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL CHECK (edge_type IN ('cites', 'authored_by', 'published_in', 'similar_to', 'co_authored', 'shares_topic')),
                properties TEXT,  -- JSON object with edge-specific data
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
            )
        """)
        
        # Graph metrics table - stores computed network analysis metrics
        await db.execute("""
            CREATE TABLE IF NOT EXISTS graph_metrics (
                node_id TEXT PRIMARY KEY,
                degree_centrality REAL,
                betweenness_centrality REAL,
                closeness_centrality REAL,
                pagerank REAL,
                clustering_coefficient REAL,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (node_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
            )
        """)
        
        # Graph snapshots table - for temporal analysis
        await db.execute("""
            CREATE TABLE IF NOT EXISTS graph_snapshots (
                id TEXT PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                node_count INTEGER NOT NULL,
                edge_count INTEGER NOT NULL,
                avg_degree REAL,
                density REAL,
                connected_components INTEGER,
                largest_component_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Citation paths table - stores precomputed paths for faster traversal
        await db.execute("""
            CREATE TABLE IF NOT EXISTS citation_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id TEXT NOT NULL,
                target_paper_id TEXT NOT NULL,
                path_length INTEGER NOT NULL,
                path_nodes TEXT NOT NULL,  -- JSON array of node IDs in path
                path_type TEXT CHECK (path_type IN ('shortest', 'influential', 'temporal')),
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_paper_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (target_paper_id) REFERENCES graph_nodes(id)
            )
        """)
    
    async def _create_graph_indexes(self, db: aiosqlite.Connection):
        """Create indexes for graph operations."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type)",
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id)",
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type)",
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_weight ON graph_edges(weight DESC)",
            "CREATE INDEX IF NOT EXISTS idx_graph_metrics_centrality ON graph_metrics(degree_centrality DESC)",
            "CREATE INDEX IF NOT EXISTS idx_citation_paths_source ON citation_paths(source_paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_citation_paths_target ON citation_paths(target_paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_citation_paths_length ON citation_paths(path_length)",
        ]
        
        for index_sql in indexes:
            await db.execute(index_sql)
    
    async def add_paper_node(self, paper_id: str, title: str, authors: List[str], 
                           year: int, categories: List[str], **kwargs) -> bool:
        """Add a paper node to the graph."""
        try:
            properties = {
                "title": title,
                "authors": authors,
                "year": year,
                "categories": categories,
                **kwargs
            }
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO graph_nodes (id, node_type, properties, updated_at)
                    VALUES (?, 'paper', ?, CURRENT_TIMESTAMP)
                """, (paper_id, json.dumps(properties)))
                await db.commit()
            
            logger.debug(f"Added paper node: {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add paper node {paper_id}: {e}")
            return False
    
    async def add_citation_edge(self, citing_paper_id: str, cited_paper_id: str,
                              citation_context: str = None, **kwargs) -> bool:
        """Add a citation edge between two papers."""
        try:
            edge_id = f"{citing_paper_id}-cites-{cited_paper_id}"
            properties = {
                "citation_context": citation_context,
                **kwargs
            }
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO graph_edges 
                    (id, source_id, target_id, edge_type, properties)
                    VALUES (?, ?, ?, 'cites', ?)
                """, (edge_id, citing_paper_id, cited_paper_id, json.dumps(properties)))
                await db.commit()
            
            logger.debug(f"Added citation edge: {citing_paper_id} -> {cited_paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add citation edge {citing_paper_id} -> {cited_paper_id}: {e}")
            return False
    
    async def get_citation_subgraph(self, paper_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get citation subgraph around a paper with specified depth."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Use CTE for graph traversal
                cursor = await db.execute("""
                    WITH RECURSIVE citation_tree(node_id, level, path) AS (
                        -- Base case: start with the target paper
                        SELECT ?, 0, ?
                        
                        UNION ALL
                        
                        -- Recursive case: get papers that cite this paper (backward)
                        SELECT e.source_id, ct.level + 1, ct.path || ',' || e.source_id
                        FROM citation_tree ct
                        JOIN graph_edges e ON ct.node_id = e.target_id
                        WHERE ct.level < ? AND e.edge_type = 'cites'
                        AND e.source_id NOT IN (SELECT value FROM json_each(ct.path))
                        
                        UNION ALL
                        
                        -- Get papers cited by this paper (forward)
                        SELECT e.target_id, ct.level + 1, ct.path || ',' || e.target_id
                        FROM citation_tree ct
                        JOIN graph_edges e ON ct.node_id = e.source_id
                        WHERE ct.level < ? AND e.edge_type = 'cites'
                        AND e.target_id NOT IN (SELECT value FROM json_each(ct.path))
                    )
                    SELECT DISTINCT ct.node_id, ct.level, n.properties
                    FROM citation_tree ct
                    JOIN graph_nodes n ON ct.node_id = n.id
                    WHERE n.node_type = 'paper'
                    ORDER BY ct.level, ct.node_id
                """, (paper_id, paper_id, depth, depth))
                
                nodes = []
                async for row in cursor:
                    node_data = {
                        "id": row[0],
                        "level": row[1],
                        "properties": json.loads(row[2])
                    }
                    nodes.append(node_data)
                
                # Get edges between these nodes
                node_ids = [node["id"] for node in nodes]
                if node_ids:
                    placeholders = ",".join("?" * len(node_ids))
                    edge_cursor = await db.execute(f"""
                        SELECT source_id, target_id, edge_type, properties, weight
                        FROM graph_edges
                        WHERE source_id IN ({placeholders}) 
                        AND target_id IN ({placeholders})
                        AND edge_type = 'cites'
                    """, node_ids + node_ids)
                    
                    edges = []
                    async for row in edge_cursor:
                        edge_data = {
                            "source": row[0],
                            "target": row[1],
                            "type": row[2],
                            "properties": json.loads(row[3]) if row[3] else {},
                            "weight": row[4]
                        }
                        edges.append(edge_data)
                else:
                    edges = []
                
                return {
                    "center_paper": paper_id,
                    "depth": depth,
                    "nodes": nodes,
                    "edges": edges,
                    "node_count": len(nodes),
                    "edge_count": len(edges)
                }
                
        except Exception as e:
            logger.error(f"Failed to get citation subgraph for {paper_id}: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    async def find_shortest_citation_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest citation path between two papers using BFS."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    WITH RECURSIVE citation_path(node_id, path, length) AS (
                        -- Base case
                        SELECT ?, ?, 0
                        
                        UNION ALL
                        
                        -- Recursive case
                        SELECT e.target_id, cp.path || ',' || e.target_id, cp.length + 1
                        FROM citation_path cp
                        JOIN graph_edges e ON cp.node_id = e.source_id
                        WHERE e.edge_type = 'cites'
                        AND cp.length < 6  -- Limit to prevent infinite recursion
                        AND e.target_id NOT IN (SELECT value FROM json_each(cp.path))
                        AND cp.node_id != ?  -- Stop when we reach target
                    )
                    SELECT path, length
                    FROM citation_path
                    WHERE node_id = ?
                    ORDER BY length
                    LIMIT 1
                """, (source_id, source_id, target_id, target_id))
                
                row = await cursor.fetchone()
                if row:
                    path_str, length = row
                    return path_str.split(',')
                return None
                
        except Exception as e:
            logger.error(f"Failed to find citation path from {source_id} to {target_id}: {e}")
            return None
    
    async def compute_node_metrics(self, node_id: str) -> Optional[GraphMetrics]:
        """Compute network analysis metrics for a specific node."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Degree centrality
                cursor = await db.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM graph_edges WHERE source_id = ? AND edge_type = 'cites') as out_degree,
                        (SELECT COUNT(*) FROM graph_edges WHERE target_id = ? AND edge_type = 'cites') as in_degree,
                        (SELECT COUNT(DISTINCT id) FROM graph_nodes WHERE node_type = 'paper') as total_nodes
                """, (node_id, node_id))
                
                row = await cursor.fetchone()
                if not row:
                    return None
                
                out_degree, in_degree, total_nodes = row
                total_degree = out_degree + in_degree
                degree_centrality = total_degree / max(1, total_nodes - 1)
                
                # Store computed metrics
                await db.execute("""
                    INSERT OR REPLACE INTO graph_metrics 
                    (node_id, degree_centrality, computed_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (node_id, degree_centrality))
                
                await db.commit()
                
                return GraphMetrics(
                    node_id=node_id,
                    degree_centrality=degree_centrality,
                    betweenness_centrality=0.0,  # Would need complex computation
                    closeness_centrality=0.0,    # Would need complex computation
                    pagerank=0.0,                # Would need iterative computation
                    clustering_coefficient=0.0,  # Would need neighbor analysis
                    computed_at=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Failed to compute metrics for node {node_id}: {e}")
            return None
    
    async def get_highly_connected_papers(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers with highest citation connections."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        n.id,
                        n.properties,
                        COALESCE(out_deg.count, 0) as citations_made,
                        COALESCE(in_deg.count, 0) as citations_received,
                        COALESCE(out_deg.count, 0) + COALESCE(in_deg.count, 0) as total_connections
                    FROM graph_nodes n
                    LEFT JOIN (
                        SELECT source_id, COUNT(*) as count
                        FROM graph_edges
                        WHERE edge_type = 'cites'
                        GROUP BY source_id
                    ) out_deg ON n.id = out_deg.source_id
                    LEFT JOIN (
                        SELECT target_id, COUNT(*) as count
                        FROM graph_edges
                        WHERE edge_type = 'cites'
                        GROUP BY target_id
                    ) in_deg ON n.id = in_deg.target_id
                    WHERE n.node_type = 'paper'
                    ORDER BY total_connections DESC
                    LIMIT ?
                """, (limit,))
                
                papers = []
                async for row in cursor:
                    paper_data = {
                        "id": row[0],
                        "properties": json.loads(row[1]),
                        "citations_made": row[2],
                        "citations_received": row[3],
                        "total_connections": row[4]
                    }
                    papers.append(paper_data)
                
                return papers
                
        except Exception as e:
            logger.error(f"Failed to get highly connected papers: {e}")
            return []
    
    async def sync_from_references_table(self):
        """Sync graph data from existing paper_references table."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # First, add all papers as nodes
                cursor = await db.execute("""
                    SELECT DISTINCT id, title, authors, category, 
                           strftime('%Y', published_date) as year
                    FROM papers
                """)
                
                papers_added = 0
                async for row in cursor:
                    paper_id, title, authors_json, category, year = row
                    authors = json.loads(authors_json) if authors_json else []
                    categories = [category] if category else []
                    
                    success = await self.add_paper_node(
                        paper_id, title, authors, int(year or 0), categories
                    )
                    if success:
                        papers_added += 1
                
                # Then, add citation edges from paper_references
                ref_cursor = await db.execute("""
                    SELECT citing_paper_id, cited_paper_id, reference_context
                    FROM paper_references
                    WHERE cited_paper_id IS NOT NULL
                """)
                
                edges_added = 0
                async for row in ref_cursor:
                    citing_id, cited_id, context = row
                    success = await self.add_citation_edge(citing_id, cited_id, context)
                    if success:
                        edges_added += 1
                
                logger.info(f"Synced {papers_added} papers and {edges_added} citation edges to graph database")
                return {"papers_added": papers_added, "edges_added": edges_added}
                
        except Exception as e:
            logger.error(f"Failed to sync from references table: {e}")
            return {"error": str(e)}
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Node counts by type
                cursor = await db.execute("""
                    SELECT node_type, COUNT(*) as count
                    FROM graph_nodes
                    GROUP BY node_type
                """)
                node_counts = {row[0]: row[1] async for row in cursor}
                
                # Edge counts by type
                cursor = await db.execute("""
                    SELECT edge_type, COUNT(*) as count
                    FROM graph_edges
                    GROUP BY edge_type
                """)
                edge_counts = {row[0]: row[1] async for row in cursor}
                
                # Overall statistics
                cursor = await db.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM graph_nodes) as total_nodes,
                        (SELECT COUNT(*) FROM graph_edges) as total_edges,
                        (SELECT AVG(degree) FROM (
                            SELECT COUNT(*) as degree
                            FROM (
                                SELECT source_id as node_id FROM graph_edges
                                UNION ALL
                                SELECT target_id as node_id FROM graph_edges
                            )
                            GROUP BY node_id
                        )) as avg_degree
                """)
                
                row = await cursor.fetchone()
                total_nodes, total_edges, avg_degree = row if row else (0, 0, 0)
                
                # Density calculation
                max_edges = total_nodes * (total_nodes - 1)
                density = (2 * total_edges / max_edges) if max_edges > 0 else 0
                
                return {
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "node_counts": node_counts,
                    "edge_counts": edge_counts,
                    "avg_degree": float(avg_degree or 0),
                    "density": density,
                    "computed_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}


# Convenience functions for graph operations
async def initialize_graph_database(db_manager: DatabaseManager) -> GraphDatabaseManager:
    """Initialize graph database with existing database manager."""
    graph_db = GraphDatabaseManager(db_manager)
    await graph_db.initialize_graph_schema()
    return graph_db


async def sync_references_to_graph(db_manager: DatabaseManager) -> Dict[str, Any]:
    """Sync existing references to graph database."""
    graph_db = await initialize_graph_database(db_manager)
    return await graph_db.sync_from_references_table()