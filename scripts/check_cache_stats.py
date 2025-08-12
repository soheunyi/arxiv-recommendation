#!/usr/bin/env python3
"""Check embedding cache statistics."""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_recommendation.embeddings import EmbeddingManager

console = Console()


def main():
    """Display cache statistics."""
    console.print("[bold cyan]Embedding Cache Statistics[/bold cyan]\n")

    manager = EmbeddingManager()
    stats = manager.get_cache_stats()

    # Main stats table
    table = Table(title="Cache Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Cache Hits", str(stats.get("cache_hits", 0)))
    table.add_row("Cache Misses", str(stats.get("cache_misses", 0)))
    table.add_row("Hit Rate", f"{stats.get('hit_rate', 0):.1%}")
    table.add_row("Total Embeddings", str(stats.get("total_embeddings", 0)))
    table.add_row("Cache Size (MB)", f"{stats.get('cache_size_mb', 0):.2f}")
    table.add_row("Legacy Pickle Files", str(stats.get("legacy_pickle_files", 0)))
    table.add_row("Daily Cost", f"${stats.get('daily_cost', 0):.6f}")
    table.add_row("Token Count", str(stats.get("token_count", 0)))

    console.print(table)

    # Performance insights
    console.print("\n[bold green]Performance Insights:[/bold green]")

    if stats.get("legacy_pickle_files", 0) > 0:
        console.print(
            f"⚠️  [yellow]{stats['legacy_pickle_files']} pickle files remain - consider migration[/yellow]"
        )
    else:
        console.print("✅ [green]All embeddings migrated to HDF5 format[/green]")

    hit_rate = stats.get("hit_rate", 0)
    if hit_rate > 0.8:
        console.print("✅ [green]Excellent cache hit rate - saving API costs[/green]")
    elif hit_rate > 0.5:
        console.print("📊 [yellow]Good cache hit rate[/yellow]")
    else:
        console.print("📈 [cyan]Cache building up - hit rate will improve[/cyan]")

    total_embeddings = stats.get("total_embeddings", 0)
    if total_embeddings > 50:
        avg_size = (
            stats.get("cache_size_mb", 0) * 1024 / total_embeddings
        )  # KB per embedding
        console.print(
            f"💾 [cyan]Average embedding size: {avg_size:.1f} KB (compressed)[/cyan]"
        )


if __name__ == "__main__":
    main()
