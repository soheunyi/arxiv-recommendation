#!/usr/bin/env python3
"""
ArXiv Paper Collection and Similarity Analysis Script

This script:
1. Collects 100 papers on "optimal transport" from arXiv
2. Collects 100 papers on "econometrics" from arXiv
3. Computes embeddings for all collected papers
4. Calculates pairwise similarity scores between papers
5. Provides analysis and visualization of similarities
"""

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


class SimilarityAnalyzer:
    """Comprehensive paper collection and similarity analysis system."""

    def __init__(self, output_dir: str = "similarity_analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.arxiv_client = ArXivClient()
        self.embedding_manager = EmbeddingManager()
        self.query_service = QueryService()

        # Storage for collected papers
        self.optimal_transport_papers: List[PaperMetadata] = []
        self.econometrics_papers: List[PaperMetadata] = []
        self.all_papers: List[PaperMetadata] = []

        # Storage for embeddings and similarities
        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarity_matrix: np.ndarray = None

        logger.info(
            f"Initialized SimilarityAnalyzer with output directory: {self.output_dir}"
        )

    async def collect_papers_for_topic(
        self, topic: str, target_count: int = 100
    ) -> List[PaperMetadata]:
        """
        Collect papers for a specific topic using AI-generated queries.

        Args:
            topic: Research topic to search for
            target_count: Target number of papers to collect

        Returns:
            List of collected papers
        """
        logger.info(
            f"Starting paper collection for topic: '{topic}' (target: {target_count} papers)"
        )

        # Generate comprehensive search queries using AI
        query_config = self.query_service.generate_search_queries(topic, max_queries=15)
        logger.info(
            f"Generated {len(query_config['search_queries'])} search queries for '{topic}'"
        )

        # Save query configuration for reference
        query_file = self.output_dir / f"{topic.replace(' ', '_')}_queries.json"
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

        # Collect optimal transport papers
        logger.info("\n🔍 PHASE 1: Collecting Optimal Transport Papers")
        self.optimal_transport_papers = await self.collect_papers_for_topic(
            "optimal transport", 100
        )

        # Collect econometrics papers
        logger.info("\n📊 PHASE 2: Collecting Econometrics Papers")
        self.econometrics_papers = await self.collect_papers_for_topic(
            "econometrics", 100
        )

        # Combine all papers
        self.all_papers = self.optimal_transport_papers + self.econometrics_papers

        logger.info(f"\n✅ COLLECTION COMPLETE:")
        logger.info(
            f"   • Optimal Transport: {len(self.optimal_transport_papers)} papers"
        )
        logger.info(f"   • Econometrics: {len(self.econometrics_papers)} papers")
        logger.info(f"   • Total: {len(self.all_papers)} papers")

        # Save collected papers metadata
        await self.save_papers_metadata()

    async def save_papers_metadata(self):
        """Save collected papers metadata to JSON files."""
        logger.info("\n💾 Saving papers metadata...")

        # Save optimal transport papers
        ot_file = self.output_dir / "optimal_transport_papers.json"
        ot_data = [asdict(paper) for paper in self.optimal_transport_papers]
        with open(ot_file, "w") as f:
            json.dump(ot_data, f, indent=2, default=str)

        # Save econometrics papers
        econ_file = self.output_dir / "econometrics_papers.json"
        econ_data = [asdict(paper) for paper in self.econometrics_papers]
        with open(econ_file, "w") as f:
            json.dump(econ_data, f, indent=2, default=str)

        # Save combined papers with topic labels
        all_papers_data = []
        for paper in self.optimal_transport_papers:
            paper_dict = asdict(paper)
            paper_dict["topic_category"] = "optimal_transport"
            all_papers_data.append(paper_dict)

        for paper in self.econometrics_papers:
            paper_dict = asdict(paper)
            paper_dict["topic_category"] = "econometrics"
            all_papers_data.append(paper_dict)

        all_file = self.output_dir / "all_papers_with_topics.json"
        with open(all_file, "w") as f:
            json.dump(all_papers_data, f, indent=2, default=str)

        logger.info(f"   • Optimal Transport papers: {ot_file}")
        logger.info(f"   • Econometrics papers: {econ_file}")
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
        ot_paper_ids = {p.id for p in self.optimal_transport_papers}
        econ_paper_ids = {p.id for p in self.econometrics_papers}
        
        # Separate into topic groups based on original topic assignment
        ot_indices = []
        econ_indices = []
        
        for i, paper in enumerate(valid_papers):
            if paper.id in ot_paper_ids:
                ot_indices.append(i)
            elif paper.id in econ_paper_ids:
                econ_indices.append(i)
        
        logger.info(f"Topic distribution: {len(ot_indices)} optimal transport, {len(econ_indices)} econometrics")

        # Within-topic similarities
        ot_similarities = self.similarity_matrix[np.ix_(ot_indices, ot_indices)]
        econ_similarities = self.similarity_matrix[np.ix_(econ_indices, econ_indices)]

        # Cross-topic similarities
        cross_topic_similarities = self.similarity_matrix[
            np.ix_(ot_indices, econ_indices)
        ]

        # Calculate statistics
        stats = {
            "within_optimal_transport": {
                "mean": np.mean(
                    ot_similarities[np.triu_indices_from(ot_similarities, k=1)]
                ),
                "std": np.std(
                    ot_similarities[np.triu_indices_from(ot_similarities, k=1)]
                ),
                "min": np.min(
                    ot_similarities[np.triu_indices_from(ot_similarities, k=1)]
                ),
                "max": np.max(
                    ot_similarities[np.triu_indices_from(ot_similarities, k=1)]
                ),
            },
            "within_econometrics": {
                "mean": np.mean(
                    econ_similarities[np.triu_indices_from(econ_similarities, k=1)]
                ),
                "std": np.std(
                    econ_similarities[np.triu_indices_from(econ_similarities, k=1)]
                ),
                "min": np.min(
                    econ_similarities[np.triu_indices_from(econ_similarities, k=1)]
                ),
                "max": np.max(
                    econ_similarities[np.triu_indices_from(econ_similarities, k=1)]
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
        logger.info(f"Within Optimal Transport papers:")
        logger.info(
            f"   • Mean similarity: {stats['within_optimal_transport']['mean']:.4f}"
        )
        logger.info(
            f"   • Std deviation: {stats['within_optimal_transport']['std']:.4f}"
        )
        logger.info(
            f"   • Range: {stats['within_optimal_transport']['min']:.4f} - {stats['within_optimal_transport']['max']:.4f}"
        )

        logger.info(f"\nWithin Econometrics papers:")
        logger.info(f"   • Mean similarity: {stats['within_econometrics']['mean']:.4f}")
        logger.info(f"   • Std deviation: {stats['within_econometrics']['std']:.4f}")
        logger.info(
            f"   • Range: {stats['within_econometrics']['min']:.4f} - {stats['within_econometrics']['max']:.4f}"
        )

        logger.info(f"\nCross-topic (OT vs Econometrics):")
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
            cross_topic_similarities, ot_indices, econ_indices, valid_papers
        )

        return stats

    def find_most_similar_cross_topic_pairs(
        self, cross_similarities, ot_indices, econ_indices, valid_papers, top_k=10
    ):
        """Find most similar papers between optimal transport and econometrics."""
        logger.info(f"\n🔍 FINDING TOP {top_k} MOST SIMILAR CROSS-TOPIC PAIRS:")

        # Find top similarities
        flat_indices = np.unravel_index(
            np.argsort(cross_similarities.ravel())[-top_k:], cross_similarities.shape
        )

        similar_pairs = []
        for i in range(top_k):
            ot_idx = ot_indices[flat_indices[0][-(i + 1)]]
            econ_idx = econ_indices[flat_indices[1][-(i + 1)]]
            similarity = cross_similarities[
                flat_indices[0][-(i + 1)], flat_indices[1][-(i + 1)]
            ]

            ot_paper = valid_papers[ot_idx]
            econ_paper = valid_papers[econ_idx]

            pair_info = {
                "rank": i + 1,
                "similarity": float(similarity),
                "optimal_transport_paper": {
                    "arxiv_id": ot_paper.id,
                    "title": ot_paper.title,
                    "authors": ot_paper.authors,
                },
                "econometrics_paper": {
                    "arxiv_id": econ_paper.id,
                    "title": econ_paper.title,
                    "authors": econ_paper.authors,
                },
            }
            similar_pairs.append(pair_info)

            logger.info(f"\n{i+1}. Similarity: {similarity:.4f}")
            logger.info(f"   OT Paper: {ot_paper.title[:80]}...")
            logger.info(f"   Econ Paper: {econ_paper.title[:80]}...")

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

        # Create topic labels for visualization
        topic_labels = ["OT"] * len(self.optimal_transport_papers) + ["Econ"] * len(
            self.econometrics_papers
        )

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
            "Pairwise Similarity Matrix\n(Optimal Transport vs Econometrics Papers)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Paper Index")
        plt.ylabel("Paper Index")

        # Add topic boundary lines
        ot_boundary = len(self.optimal_transport_papers)
        plt.axhline(y=ot_boundary, color="red", linestyle="--", linewidth=2, alpha=0.7)
        plt.axvline(x=ot_boundary, color="red", linestyle="--", linewidth=2, alpha=0.7)

        plt.tight_layout()
        heatmap_file = self.output_dir / "similarity_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"   • Saved similarity heatmap: {heatmap_file}")

        # 2. Distribution comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Within optimal transport
        ot_indices = list(range(len(self.optimal_transport_papers)))
        ot_similarities = self.similarity_matrix[np.ix_(ot_indices, ot_indices)]
        ot_upper = ot_similarities[np.triu_indices_from(ot_similarities, k=1)]

        axes[0].hist(ot_upper, bins=30, alpha=0.7, color="blue", edgecolor="black")
        axes[0].set_title("Within Optimal Transport\nSimilarity Distribution")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(
            np.mean(ot_upper),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(ot_upper):.3f}",
        )
        axes[0].legend()

        # Within econometrics
        econ_indices = list(
            range(len(self.optimal_transport_papers), len(self.all_papers))
        )
        econ_similarities = self.similarity_matrix[np.ix_(econ_indices, econ_indices)]
        econ_upper = econ_similarities[np.triu_indices_from(econ_similarities, k=1)]

        axes[1].hist(econ_upper, bins=30, alpha=0.7, color="green", edgecolor="black")
        axes[1].set_title("Within Econometrics\nSimilarity Distribution")
        axes[1].set_xlabel("Cosine Similarity")
        axes[1].set_ylabel("Frequency")
        axes[1].axvline(
            np.mean(econ_upper),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(econ_upper):.3f}",
        )
        axes[1].legend()

        # Cross-topic
        cross_similarities = self.similarity_matrix[np.ix_(ot_indices, econ_indices)]
        cross_flat = cross_similarities.flatten()

        axes[2].hist(cross_flat, bins=30, alpha=0.7, color="orange", edgecolor="black")
        axes[2].set_title("Cross-Topic (OT vs Econ)\nSimilarity Distribution")
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

        categories = ["Within OT", "Within Econ", "Cross-Topic"]
        means = [np.mean(ot_upper), np.mean(econ_upper), np.mean(cross_flat)]
        stds = [np.std(ot_upper), np.std(econ_upper), np.std(cross_flat)]

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
        logger.info(f"\n🚀 STARTING COMPLETE SIMILARITY ANALYSIS")
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
            logger.info(f"Total duration: {duration}")
            logger.info(f"Papers collected: {len(self.all_papers)}")
            logger.info(f"Embeddings computed: {len(self.embeddings)}")
            logger.info(f"Similarity matrix: {self.similarity_matrix.shape}")
            logger.info(f"Output directory: {self.output_dir.absolute()}")

            # Create summary report
            summary = {
                "analysis_date": start_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "papers_collected": {
                    "optimal_transport": len(self.optimal_transport_papers),
                    "econometrics": len(self.econometrics_papers),
                    "total": len(self.all_papers),
                },
                "embeddings_computed": len(self.embeddings),
                "similarity_matrix_shape": list(self.similarity_matrix.shape),
                "similarity_statistics": stats,
                "output_files": [
                    "optimal_transport_papers.json",
                    "econometrics_papers.json",
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


async def main():
    """Main function to run the similarity analysis."""
    try:
        # Create analyzer instance
        analyzer = SimilarityAnalyzer()

        # Run complete analysis
        await analyzer.run_complete_analysis()

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Similarity analysis completed successfully!")
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
