#!/usr/bin/env python3
"""Main entry point for ArXiv Recommendation System CLI."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.arxiv_recommendation import run_recommendation_system
from src.arxiv_recommendation.config import config

app = typer.Typer(help="ArXiv Recommendation System - Personal LLM-powered paper recommendations")
console = Console()


@app.command()
def run(
    categories: Optional[str] = typer.Option(
        None, 
        "--categories", 
        "-c", 
        help="Comma-separated arXiv categories (e.g., cs.AI,cs.LG)"
    ),
    max_papers: Optional[int] = typer.Option(
        None,
        "--max-papers",
        "-m",
        help="Maximum papers to process per day"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run the daily recommendation workflow."""
    
    # Override config if provided
    if categories:
        config.arxiv_categories = [cat.strip() for cat in categories.split(",")]
    if max_papers:
        config.max_daily_papers = max_papers
    
    console.print("[bold blue]ArXiv Recommendation System[/bold blue]")
    console.print(f"Categories: {', '.join(config.arxiv_categories)}")
    console.print(f"Max papers: {config.max_daily_papers}")
    console.print(f"Estimated daily cost: ${config.estimate_daily_cost():.3f}")
    console.print()
    
    async def run_workflow():
        results = await run_recommendation_system()
        
        if "error" in results:
            console.print(f"[bold red]Error:[/bold red] {results['error']}")
            raise typer.Exit(1)
        
        # Display results
        recommendations = results.get("recommendations", [])
        
        if not recommendations:
            console.print("[yellow]No new recommendations found.[/yellow]")
            return
        
        # Create results table
        table = Table(title="Today's Recommendations")
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Title", style="white", max_width=50)
        table.add_column("Authors", style="green", max_width=30)
        table.add_column("Score", style="yellow", justify="right")
        
        for i, rec in enumerate(recommendations[:10], 1):
            table.add_row(
                str(i),
                rec.get("title", "Unknown"),
                ", ".join(rec.get("authors", [])[:2]),
                f"{rec.get('score', 0):.3f}"
            )
        
        console.print(table)
        console.print(f"\n[bold green]Generated {len(recommendations)} recommendations![/bold green]")
    
    try:
        asyncio.run(run_workflow())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def cost():
    """Show cost estimates and usage statistics."""
    table = Table(title="Cost Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    daily_cost = config.estimate_daily_cost()
    monthly_cost = daily_cost * 30
    
    table.add_row("Embedding Model", config.embedding_model)
    table.add_row("Cost per 1K tokens", f"${config.embedding_cost_per_token * 1000:.5f}")
    table.add_row("Max daily papers", str(config.max_daily_papers))
    table.add_row("Estimated daily cost", f"${daily_cost:.3f}")
    table.add_row("Estimated monthly cost", f"${monthly_cost:.2f}")
    table.add_row("Budget limit", f"${config.openai_budget_limit:.2f}")
    
    console.print(table)
    
    if monthly_cost > config.openai_budget_limit:
        console.print(f"[bold red]Warning:[/bold red] Estimated monthly cost (${monthly_cost:.2f}) exceeds budget limit (${config.openai_budget_limit:.2f})")
    else:
        console.print(f"[bold green]✓[/bold green] Monthly cost estimate is within budget")


@app.command()
def config_info():
    """Show current configuration."""
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("OpenAI API Key", "✓ Set" if config.openai_api_key else "✗ Missing")
    table.add_row("Embedding Model", config.embedding_model)
    table.add_row("ArXiv Categories", ", ".join(config.arxiv_categories))
    table.add_row("Max Daily Papers", str(config.max_daily_papers))
    table.add_row("Database Path", config.database_path)
    table.add_row("Embeddings Path", config.embeddings_path)
    table.add_row("Budget Limit", f"${config.openai_budget_limit:.2f}")
    
    console.print(table)


def main():
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()