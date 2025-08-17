#!/usr/bin/env python3
"""
Generalized ArXiv Paper Collection and Similarity Analysis Script

This script allows for comparative similarity analysis between any two research topics:
1. Collects papers on configurable topic1 from arXiv
2. Collects papers on configurable topic2 from arXiv  
3. Computes embeddings for all collected papers
4. Calculates pairwise similarity scores between papers
5. Provides analysis and visualization of similarities between the two topics

Usage:
    python comparative_similarity_analysis.py "machine learning" "natural language processing"
    python comparative_similarity_analysis.py "optimal transport" "econometrics" --papers-per-topic 50
    python comparative_similarity_analysis.py "quantum computing" "cryptography" --output-dir results
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import asdict
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Add the backend src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "backend" / "src"))

from arxiv_client import ArXivClient, PaperMetadata
from embeddings import EmbeddingManager
from services.query_service import QueryService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComparativeSimilarityAnalyzer:
    """Generalized paper collection and similarity analysis system for comparing any two research topics."""

    def __init__(self, topic1: str, topic2: str, output_dir: str = None, papers_per_topic: int = 100):
        """
        Initialize analyzer for comparing two research topics.
        
        Args:
            topic1: First research topic (content1)
            topic2: Second research topic (content2)
            output_dir: Output directory for results
            papers_per_topic: Number of papers to collect per topic
        """
        self.topic1 = topic1
        self.topic2 = topic2
        self.papers_per_topic = papers_per_topic
        
        # Create output directory with sanitized topic names
        if output_dir is None:
            topic1_clean = self._sanitize_filename(topic1)
            topic2_clean = self._sanitize_filename(topic2)
            output_dir = f"similarity_analysis_{topic1_clean}_vs_{topic2_clean}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.arxiv_client = ArXivClient()
        self.embedding_manager = EmbeddingManager()
        self.query_service = QueryService()

        # Storage for collected papers (generalized from optimal_transport_papers and econometrics_papers)
        self.topic1_papers: List[PaperMetadata] = []
        self.topic2_papers: List[PaperMetadata] = []
        self.all_papers: List[PaperMetadata] = []

        # Storage for embeddings and similarities
        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarity_matrix: np.ndarray = None

        logger.info(
            f"Initialized ComparativeSimilarityAnalyzer:\n"
            f"  Topic 1: '{self.topic1}'\n"
            f"  Topic 2: '{self.topic2}'\n"
            f"  Papers per topic: {self.papers_per_topic}\n"
            f"  Output directory: {self.output_dir}"
        )

    def _sanitize_filename(self, text: str) -> str:
        """Convert topic name to safe filename."""
        import re
        # Replace spaces and special characters with underscores
        safe_name = re.sub(r'[^\w\s-]', '', text)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return safe_name.lower()

    async def collect_papers_for_topic(
        self, topic: str, target_count: int = None
    ) -> List[PaperMetadata]:
        """
        Collect papers for a specific topic using AI-generated queries.

        Args:
            topic: Research topic to search for
            target_count: Target number of papers to collect (defaults to papers_per_topic)

        Returns:
            List of collected papers
        """
        if target_count is None:
            target_count = self.papers_per_topic
            
        logger.info(
            f"Starting paper collection for topic: '{topic}' (target: {target_count} papers)"
        )

        # Generate comprehensive search queries using AI
        query_config = self.query_service.generate_search_queries(topic, max_queries=15)
        logger.info(
            f"Generated {len(query_config['search_queries'])} search queries for '{topic}'"
        )

        # Save query configuration for reference
        topic_clean = self._sanitize_filename(topic)
        query_file = self.output_dir / f"{topic_clean}_queries.json"
        with open(query_file, "w") as f:
            json.dump(query_config, f, indent=2)
        logger.info(f"Saved query configuration to {query_file}")

        collected_papers = []
        paper_ids_seen = set()

        # Execute each query and collect papers
        for i, query_info in enumerate(query_config["search_queries"]):
            query = query_info["query"]
            priority = query_info["priority"]
            description = query_info["description"]

            logger.info(
                f"Executing query {i+1}/{len(query_config['search_queries'])}: {description}"
            )
            logger.info(f"Query: {query} (Priority: {priority})")

            try:
                # Search with higher limit to get diverse results
                search_results = await self.arxiv_client.search_papers(
                    query=query, max_results=50
                )

                # Add unique papers to collection
                new_papers = 0
                for paper in search_results:
                    if (
                        paper.id not in paper_ids_seen
                        and len(collected_papers) < target_count
                    ):
                        collected_papers.append(paper)
                        paper_ids_seen.add(paper.id)
                        new_papers += 1

                logger.info(
                    f"Added {new_papers} new papers from this query (total: {len(collected_papers)})"
                )

                # Break if we've reached our target
                if len(collected_papers) >= target_count:
                    logger.info(
                        f"Reached target of {target_count} papers for '{topic}'"
                    )
                    break

                # Rate limiting - brief pause between queries
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error executing query '{query}': {e}")
                continue

        # Trim to exact target count if we collected more
        if len(collected_papers) > target_count:
            collected_papers = collected_papers[:target_count]

        logger.info(
            f"Successfully collected {len(collected_papers)} papers for '{topic}'"
        )
        return collected_papers

    async def collect_all_papers(self):
        """Collect papers for both topics."""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE PAPER COLLECTION")
        logger.info("=" * 80)

        # Collect topic1 papers
        logger.info(f"\n🔍 PHASE 1: Collecting {self.topic1.title()} Papers")
        self.topic1_papers = await self.collect_papers_for_topic(
            self.topic1, self.papers_per_topic
        )

        # Collect topic2 papers
        logger.info(f"\n📊 PHASE 2: Collecting {self.topic2.title()} Papers")
        self.topic2_papers = await self.collect_papers_for_topic(
            self.topic2, self.papers_per_topic
        )

        # Combine all papers
        self.all_papers = self.topic1_papers + self.topic2_papers

        logger.info(f"\n✅ COLLECTION COMPLETE:")
        logger.info(
            f"   • {self.topic1.title()}: {len(self.topic1_papers)} papers"
        )
        logger.info(f"   • {self.topic2.title()}: {len(self.topic2_papers)} papers")
        logger.info(f"   • Total: {len(self.all_papers)} papers")

        # Save collected papers metadata
        await self.save_papers_metadata()

    async def save_papers_metadata(self):
        """Save collected papers metadata to JSON files."""
        logger.info("\n💾 Saving papers metadata...")

        # Save topic1 papers
        topic1_clean = self._sanitize_filename(self.topic1)
        topic1_file = self.output_dir / f"{topic1_clean}_papers.json"
        topic1_data = [asdict(paper) for paper in self.topic1_papers]
        with open(topic1_file, "w") as f:
            json.dump(topic1_data, f, indent=2, default=str)

        # Save topic2 papers
        topic2_clean = self._sanitize_filename(self.topic2)
        topic2_file = self.output_dir / f"{topic2_clean}_papers.json"
        topic2_data = [asdict(paper) for paper in self.topic2_papers]
        with open(topic2_file, "w") as f:
            json.dump(topic2_data, f, indent=2, default=str)

        # Save combined papers with topic labels
        all_papers_data = []
        for paper in self.topic1_papers:
            paper_dict = asdict(paper)
            paper_dict["topic_category"] = topic1_clean
            all_papers_data.append(paper_dict)

        for paper in self.topic2_papers:
            paper_dict = asdict(paper)
            paper_dict["topic_category"] = topic2_clean
            all_papers_data.append(paper_dict)

        all_file = self.output_dir / "all_papers_with_topics.json"
        with open(all_file, "w") as f:
            json.dump(all_papers_data, f, indent=2, default=str)

        logger.info(f"   • {self.topic1.title()} papers: {topic1_file}")
        logger.info(f"   • {self.topic2.title()} papers: {topic2_file}")
        logger.info(f"   • Combined papers: {all_file}")

    async def compute_embeddings(self):
        """Compute embeddings for all collected papers."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING EMBEDDINGS FOR ALL PAPERS")
        logger.info("=" * 80)

        total_papers = len(self.all_papers)
        logger.info(f"Computing embeddings for {total_papers} papers...")

        for i, paper in enumerate(self.all_papers, 1):
            try:
                # Create text for embedding (title + abstract)
                text_content = f"{paper.title}\n\n{paper.abstract}"

                # Get embedding using the existing embedding manager
                embedding = await self.embedding_manager.get_embedding(text_content)
                self.embeddings[paper.id] = embedding

                if i % 10 == 0:
                    logger.info(
                        f"   Progress: {i}/{total_papers} embeddings computed ({i/total_papers*100:.1f}%)"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to compute embedding for paper {paper.id}: {e}"
                )
                continue

        logger.info(f"✅ Successfully computed {len(self.embeddings)} embeddings")

        # Save embeddings to file
        embeddings_file = self.output_dir / "paper_embeddings.npz"
        embedding_ids = list(self.embeddings.keys())
        
        # Debug: Check embedding shapes
        logger.info(f"Checking embedding dimensions...")
        embedding_shapes = []
        for i, id_ in enumerate(embedding_ids[:5]):  # Check first 5
            emb = self.embeddings[id_]
            shape = np.array(emb).shape if emb is not None else None
            embedding_shapes.append(shape)
            logger.info(f"  Embedding {i}: shape={shape}, type={type(emb)}")
        
        # Filter out None embeddings
        valid_embeddings = []
        valid_ids = []
        for id_ in embedding_ids:
            emb = self.embeddings[id_]
            if emb is not None and len(emb) > 0:
                valid_embeddings.append(emb)
                valid_ids.append(id_)
        
        logger.info(f"Valid embeddings: {len(valid_embeddings)}/{len(embedding_ids)}")
        
        if len(valid_embeddings) == 0:
            raise ValueError("No valid embeddings found")
        
        embedding_vectors = np.array(valid_embeddings)
        embedding_ids = valid_ids

        np.savez(
            embeddings_file,
            embedding_ids=embedding_ids,
            embedding_vectors=embedding_vectors,
        )
        logger.info(f"   • Saved embeddings to: {embeddings_file}")

    def compute_similarity_matrix(self):
        """Compute pairwise similarity scores between all papers."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING PAIRWISE SIMILARITY MATRIX")
        logger.info("=" * 80)

        # Filter out None embeddings (same as in compute_embeddings)
        valid_embeddings = []
        valid_ids = []
        for id_ in self.embeddings.keys():
            emb = self.embeddings[id_]
            if emb is not None and len(emb) > 0:
                valid_embeddings.append(emb)
                valid_ids.append(id_)
        
        if len(valid_embeddings) == 0:
            raise ValueError("No valid embeddings found for similarity computation")
        
        embedding_matrix = np.array(valid_embeddings)
        logger.info(f"Computing similarity matrix for {len(valid_ids)} papers (shape: {embedding_matrix.shape})...")

        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Update paper IDs to only include valid ones
        self.valid_paper_ids = valid_ids

        logger.info(
            f"✅ Computed {self.similarity_matrix.shape[0]}x{self.similarity_matrix.shape[1]} similarity matrix"
        )

        # Create DataFrame for easier analysis
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix, index=self.valid_paper_ids, columns=self.valid_paper_ids
        )

        # Save similarity matrix
        similarity_file = self.output_dir / "similarity_matrix.csv"
        self.similarity_df.to_csv(similarity_file)
        logger.info(f"   • Saved similarity matrix to: {similarity_file}")

        # Save as numpy array too
        np_file = self.output_dir / "similarity_matrix.npz"
        np.savez(np_file, similarity_matrix=self.similarity_matrix, paper_ids=self.valid_paper_ids)
        logger.info(f"   • Saved numpy format to: {np_file}")

    def analyze_similarities(self):
        """Perform comprehensive similarity analysis."""
        logger.info("\n" + "=" * 80)
        logger.info("ANALYZING SIMILARITY PATTERNS")
        logger.info("=" * 80)

        # Create valid papers mapping based on embedding success
        valid_papers = []
        for paper in self.all_papers:
            if paper.id in self.valid_paper_ids:
                valid_papers.append(paper)
        
        logger.info(f"Analyzing {len(valid_papers)} papers with valid embeddings out of {len(self.all_papers)} total")

        # Create paper index mapping for valid papers only
        paper_id_to_index = {
            paper.id: i for i, paper in enumerate(valid_papers)
        }

        # Create topic mappings based on original paper classification
        topic1_paper_ids = {p.id for p in self.topic1_papers}
        topic2_paper_ids = {p.id for p in self.topic2_papers}
        
        # Separate into topic groups based on original topic assignment
        topic1_indices = []
        topic2_indices = []
        
        for i, paper in enumerate(valid_papers):
            if paper.id in topic1_paper_ids:
                topic1_indices.append(i)
            elif paper.id in topic2_paper_ids:
                topic2_indices.append(i)
        
        logger.info(f"Topic distribution: {len(topic1_indices)} {self.topic1}, {len(topic2_indices)} {self.topic2}")

        # Within-topic similarities
        topic1_similarities = self.similarity_matrix[np.ix_(topic1_indices, topic1_indices)]
        topic2_similarities = self.similarity_matrix[np.ix_(topic2_indices, topic2_indices)]

        # Cross-topic similarities
        cross_topic_similarities = self.similarity_matrix[
            np.ix_(topic1_indices, topic2_indices)
        ]

        # Calculate statistics
        stats = {
            f"within_{self._sanitize_filename(self.topic1)}": {
                "mean": np.mean(
                    topic1_similarities[np.triu_indices_from(topic1_similarities, k=1)]
                ),
                "std": np.std(
                    topic1_similarities[np.triu_indices_from(topic1_similarities, k=1)]
                ),
                "min": np.min(
                    topic1_similarities[np.triu_indices_from(topic1_similarities, k=1)]
                ),
                "max": np.max(
                    topic1_similarities[np.triu_indices_from(topic1_similarities, k=1)]
                ),
            },
            f"within_{self._sanitize_filename(self.topic2)}": {
                "mean": np.mean(
                    topic2_similarities[np.triu_indices_from(topic2_similarities, k=1)]
                ),
                "std": np.std(
                    topic2_similarities[np.triu_indices_from(topic2_similarities, k=1)]
                ),
                "min": np.min(
                    topic2_similarities[np.triu_indices_from(topic2_similarities, k=1)]
                ),
                "max": np.max(
                    topic2_similarities[np.triu_indices_from(topic2_similarities, k=1)]
                ),
            },
            "cross_topic": {
                "mean": np.mean(cross_topic_similarities),
                "std": np.std(cross_topic_similarities),
                "min": np.min(cross_topic_similarities),
                "max": np.max(cross_topic_similarities),
            },
        }

        # Print analysis results
        logger.info("\n📊 SIMILARITY STATISTICS:")
        topic1_key = f"within_{self._sanitize_filename(self.topic1)}"
        topic2_key = f"within_{self._sanitize_filename(self.topic2)}"
        
        logger.info(f"Within {self.topic1.title()} papers:")
        logger.info(
            f"   • Mean similarity: {stats[topic1_key]['mean']:.4f}"
        )
        logger.info(
            f"   • Std deviation: {stats[topic1_key]['std']:.4f}"
        )
        logger.info(
            f"   • Range: {stats[topic1_key]['min']:.4f} - {stats[topic1_key]['max']:.4f}"
        )

        logger.info(f"\nWithin {self.topic2.title()} papers:")
        logger.info(f"   • Mean similarity: {stats[topic2_key]['mean']:.4f}")
        logger.info(f"   • Std deviation: {stats[topic2_key]['std']:.4f}")
        logger.info(
            f"   • Range: {stats[topic2_key]['min']:.4f} - {stats[topic2_key]['max']:.4f}"
        )

        logger.info(f"\nCross-topic ({self.topic1.title()} vs {self.topic2.title()}):")
        logger.info(f"   • Mean similarity: {stats['cross_topic']['mean']:.4f}")
        logger.info(f"   • Std deviation: {stats['cross_topic']['std']:.4f}")
        logger.info(
            f"   • Range: {stats['cross_topic']['min']:.4f} - {stats['cross_topic']['max']:.4f}"
        )

        # Save statistics
        stats_file = self.output_dir / "similarity_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\n💾 Saved statistics to: {stats_file}")

        # Find most similar cross-topic pairs
        self.find_most_similar_cross_topic_pairs(
            cross_topic_similarities, topic1_indices, topic2_indices, valid_papers
        )

        return stats

    def find_most_similar_cross_topic_pairs(
        self, cross_similarities, topic1_indices, topic2_indices, valid_papers, top_k=10
    ):
        """Find most similar papers between the two topics."""
        logger.info(f"\n🔍 FINDING TOP {top_k} MOST SIMILAR CROSS-TOPIC PAIRS:")

        # Find top similarities
        flat_indices = np.unravel_index(
            np.argsort(cross_similarities.ravel())[-top_k:], cross_similarities.shape
        )

        similar_pairs = []
        for i in range(top_k):
            topic1_idx = topic1_indices[flat_indices[0][-(i + 1)]]
            topic2_idx = topic2_indices[flat_indices[1][-(i + 1)]]
            similarity = cross_similarities[
                flat_indices[0][-(i + 1)], flat_indices[1][-(i + 1)]
            ]

            topic1_paper = valid_papers[topic1_idx]
            topic2_paper = valid_papers[topic2_idx]

            pair_info = {
                "rank": i + 1,
                "similarity": float(similarity),
                f"{self._sanitize_filename(self.topic1)}_paper": {
                    "arxiv_id": topic1_paper.id,
                    "title": topic1_paper.title,
                    "authors": topic1_paper.authors,
                },
                f"{self._sanitize_filename(self.topic2)}_paper": {
                    "arxiv_id": topic2_paper.id,
                    "title": topic2_paper.title,
                    "authors": topic2_paper.authors,
                },
            }
            similar_pairs.append(pair_info)

            logger.info(f"\n{i+1}. Similarity: {similarity:.4f}")
            logger.info(f"   {self.topic1.title()} Paper: {topic1_paper.title[:80]}...")
            logger.info(f"   {self.topic2.title()} Paper: {topic2_paper.title[:80]}...")

        # Save similar pairs
        pairs_file = self.output_dir / "most_similar_cross_topic_pairs.json"
        with open(pairs_file, "w") as f:
            json.dump(similar_pairs, f, indent=2)
        logger.info(f"\n💾 Saved top similar pairs to: {pairs_file}")

    def create_visualizations(self):
        """Create visualizations of similarity patterns."""
        logger.info("\n" + "=" * 80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 80)

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Similarity matrix heatmap
        plt.figure(figsize=(12, 10))

        # Create custom colormap boundaries for better visualization
        mask = np.triu(self.similarity_matrix, k=1)  # Mask upper triangle

        sns.heatmap(
            self.similarity_matrix,
            cmap="RdYlBu_r",
            center=0.5,
            square=True,
            mask=mask,
            cbar_kws={"label": "Cosine Similarity"},
        )

        plt.title(
            f"Pairwise Similarity Matrix\n({self.topic1.title()} vs {self.topic2.title()} Papers)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Paper Index")
        plt.ylabel("Paper Index")

        # Add topic boundary lines
        topic1_boundary = len([p for p in self.topic1_papers if p.id in self.valid_paper_ids])
        plt.axhline(y=topic1_boundary, color="red", linestyle="--", linewidth=2, alpha=0.7)
        plt.axvline(x=topic1_boundary, color="red", linestyle="--", linewidth=2, alpha=0.7)

        plt.tight_layout()
        heatmap_file = self.output_dir / "similarity_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"   • Saved similarity heatmap: {heatmap_file}")

        # 2. Distribution comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Create valid indices mapping
        valid_papers = [p for p in self.all_papers if p.id in self.valid_paper_ids]
        topic1_paper_ids = {p.id for p in self.topic1_papers}
        topic2_paper_ids = {p.id for p in self.topic2_papers}
        
        topic1_indices = [i for i, p in enumerate(valid_papers) if p.id in topic1_paper_ids]
        topic2_indices = [i for i, p in enumerate(valid_papers) if p.id in topic2_paper_ids]

        # Within topic1
        topic1_similarities = self.similarity_matrix[np.ix_(topic1_indices, topic1_indices)]
        topic1_upper = topic1_similarities[np.triu_indices_from(topic1_similarities, k=1)]

        axes[0].hist(topic1_upper, bins=30, alpha=0.7, color="blue", edgecolor="black")
        axes[0].set_title(f"Within {self.topic1.title()}\nSimilarity Distribution")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(
            np.mean(topic1_upper),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(topic1_upper):.3f}",
        )
        axes[0].legend()

        # Within topic2
        topic2_similarities = self.similarity_matrix[np.ix_(topic2_indices, topic2_indices)]
        topic2_upper = topic2_similarities[np.triu_indices_from(topic2_similarities, k=1)]

        axes[1].hist(topic2_upper, bins=30, alpha=0.7, color="green", edgecolor="black")
        axes[1].set_title(f"Within {self.topic2.title()}\nSimilarity Distribution")
        axes[1].set_xlabel("Cosine Similarity")
        axes[1].set_ylabel("Frequency")
        axes[1].axvline(
            np.mean(topic2_upper),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(topic2_upper):.3f}",
        )
        axes[1].legend()

        # Cross-topic
        cross_similarities = self.similarity_matrix[np.ix_(topic1_indices, topic2_indices)]
        cross_flat = cross_similarities.flatten()

        axes[2].hist(cross_flat, bins=30, alpha=0.7, color="orange", edgecolor="black")
        axes[2].set_title(f"Cross-Topic ({self.topic1.title()} vs {self.topic2.title()})\nSimilarity Distribution")
        axes[2].set_xlabel("Cosine Similarity")
        axes[2].set_ylabel("Frequency")
        axes[2].axvline(
            np.mean(cross_flat),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(cross_flat):.3f}",
        )
        axes[2].legend()

        plt.tight_layout()
        dist_file = self.output_dir / "similarity_distributions.png"
        plt.savefig(dist_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"   • Saved similarity distributions: {dist_file}")

        # 3. Summary statistics comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = [f"Within {self.topic1.title()}", f"Within {self.topic2.title()}", "Cross-Topic"]
        means = [np.mean(topic1_upper), np.mean(topic2_upper), np.mean(cross_flat)]
        stds = [np.std(topic1_upper), np.std(topic2_upper), np.std(cross_flat)]

        x_pos = np.arange(len(categories))
        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=["blue", "green", "orange"],
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_xlabel("Comparison Type")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title(
            "Mean Similarity Scores by Category\n(Error bars show ±1 standard deviation)"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)

        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        stats_file = self.output_dir / "similarity_statistics_plot.png"
        plt.savefig(stats_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"   • Saved statistics plot: {stats_file}")

    async def run_complete_analysis(self):
        """Run the complete similarity analysis pipeline."""
        start_time = datetime.now()
        logger.info(f"\n🚀 STARTING COMPARATIVE SIMILARITY ANALYSIS")
        logger.info(f"Comparing: '{self.topic1}' vs '{self.topic2}'")
        logger.info(f"Start time: {start_time}")
        logger.info("=" * 80)

        try:
            # Step 1: Collect papers
            await self.collect_all_papers()

            # Step 2: Compute embeddings
            await self.compute_embeddings()

            # Step 3: Compute similarity matrix
            self.compute_similarity_matrix()

            # Step 4: Analyze similarities
            stats = self.analyze_similarities()

            # Step 5: Create visualizations
            self.create_visualizations()

            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("\n" + "=" * 80)
            logger.info("🎉 ANALYSIS COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Topics compared: '{self.topic1}' vs '{self.topic2}'")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Papers collected: {len(self.all_papers)}")
            logger.info(f"Embeddings computed: {len(self.embeddings)}")
            logger.info(f"Similarity matrix: {self.similarity_matrix.shape}")
            logger.info(f"Output directory: {self.output_dir.absolute()}")

            # Create summary report
            summary = {
                "analysis_date": start_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "topics": {
                    "topic1": self.topic1,
                    "topic2": self.topic2,
                },
                "papers_collected": {
                    "topic1": len(self.topic1_papers),
                    "topic2": len(self.topic2_papers),
                    "total": len(self.all_papers),
                },
                "embeddings_computed": len(self.embeddings),
                "similarity_matrix_shape": list(self.similarity_matrix.shape),
                "similarity_statistics": stats,
                "output_files": [
                    f"{self._sanitize_filename(self.topic1)}_papers.json",
                    f"{self._sanitize_filename(self.topic2)}_papers.json",
                    "all_papers_with_topics.json",
                    "paper_embeddings.npz",
                    "similarity_matrix.csv",
                    "similarity_matrix.npz",
                    "similarity_statistics.json",
                    "most_similar_cross_topic_pairs.json",
                    "similarity_heatmap.png",
                    "similarity_distributions.png",
                    "similarity_statistics_plot.png",
                ],
            }

            summary_file = self.output_dir / "analysis_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"\n📋 Analysis summary saved to: {summary_file}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comparative Similarity Analysis of ArXiv Papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comparative_similarity_analysis.py "machine learning" "natural language processing"
  python comparative_similarity_analysis.py "optimal transport" "econometrics" --papers-per-topic 50
  python comparative_similarity_analysis.py "quantum computing" "cryptography" --output-dir quantum_vs_crypto
        """
    )
    
    parser.add_argument(
        "topic1",
        help="First research topic to analyze (content1)"
    )
    
    parser.add_argument(
        "topic2", 
        help="Second research topic to analyze (content2)"
    )
    
    parser.add_argument(
        "--papers-per-topic",
        type=int,
        default=100,
        help="Number of papers to collect per topic (default: 100)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: auto-generated from topics)"
    )
    
    return parser.parse_args()


async def main():
    """Main function to run the comparative similarity analysis."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create analyzer instance
        analyzer = ComparativeSimilarityAnalyzer(
            topic1=args.topic1,
            topic2=args.topic2,
            output_dir=args.output_dir,
            papers_per_topic=args.papers_per_topic
        )

        # Run complete analysis
        await analyzer.run_complete_analysis()

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Comparative similarity analysis completed successfully!")
        print(f"📊 Compared: '{args.topic1}' vs '{args.topic2}'")
        print(f"📁 Results saved to: {analyzer.output_dir.absolute()}")
        print("=" * 80)

    except KeyboardInterrupt:
        logger.info("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())