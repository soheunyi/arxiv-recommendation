#!/usr/bin/env python3
"""
Migrate embeddings from pickle to HDF5 format.

This script safely migrates all existing pickle embeddings to the new HDF5 format
with compression and better performance.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_recommendation.embeddings import EmbeddingManager
from src.arxiv_recommendation.config import config

console = Console()


async def main():
    """Run the migration process."""
    console.print("[bold cyan]Embedding Storage Migration Tool[/bold cyan]\n")

    # Initialize the improved embedding manager
    manager = EmbeddingManager()

    # Get initial stats
    stats = manager.get_cache_stats()

    # Display current state
    table = Table(title="Current Cache Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Legacy Pickle Files", str(stats.get("legacy_pickle_files", 0)))
    table.add_row("HDF5 Embeddings", str(stats.get("total_embeddings", 0)))
    table.add_row("Cache Size (MB)", f"{stats.get('cache_size_mb', 0):.2f}")

    console.print(table)
    console.print()

    if stats.get("legacy_pickle_files", 0) == 0:
        console.print("[yellow]No pickle files found to migrate.[/yellow]")
        return

    # Auto-confirm for automation (add --interactive flag for manual use)
    import sys

    auto_confirm = len(sys.argv) > 1 and sys.argv[1] == "--auto"

    console.print(
        f"[bold]Found {stats['legacy_pickle_files']} pickle files to migrate.[/bold]"
    )

    if auto_confirm:
        console.print("[cyan]Auto-confirming migration...[/cyan]")
        confirm = "y"
    else:
        confirm = console.input(
            "[cyan]Do you want to proceed with migration? (y/n): [/cyan]"
        )

    if confirm.lower() != "y":
        console.print("[red]Migration cancelled.[/red]")
        return

    console.print("\n[bold green]Starting migration...[/bold green]\n")

    # Run migration with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Migrating embeddings...", total=stats["legacy_pickle_files"]
        )

        # Run the migration
        await manager.migrate_all_pickle_files()

        progress.update(task, completed=stats["legacy_pickle_files"])

    # Get final stats
    final_stats = manager.get_cache_stats()

    # Display results
    console.print("\n[bold green]Migration Complete![/bold green]\n")

    result_table = Table(title="Migration Results")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Before", style="yellow")
    result_table.add_column("After", style="green")

    result_table.add_row(
        "Pickle Files",
        str(stats.get("legacy_pickle_files", 0)),
        str(final_stats.get("legacy_pickle_files", 0)),
    )
    result_table.add_row(
        "HDF5 Embeddings",
        str(stats.get("total_embeddings", 0)),
        str(final_stats.get("total_embeddings", 0)),
    )
    result_table.add_row(
        "Cache Size (MB)",
        f"{stats.get('cache_size_mb', 0):.2f}",
        f"{final_stats.get('cache_size_mb', 0):.2f}",
    )

    console.print(result_table)

    # Calculate space savings
    if stats.get("cache_size_mb", 0) > 0:
        size_reduction = (
            1 - final_stats.get("cache_size_mb", 0) / stats.get("cache_size_mb", 0)
        ) * 100
        if size_reduction > 0:
            console.print(
                f"\n[bold green]Space saved: {size_reduction:.1f}%[/bold green]"
            )

    console.print(
        "\n[dim]The improved embedding manager will automatically use HDF5 format[/dim]"
    )
    console.print("[dim]and can still read any remaining pickle files if needed.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
