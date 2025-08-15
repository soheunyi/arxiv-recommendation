#!/usr/bin/env python3
"""
Shared utilities for scripts in the arXiv recommendation system.

This module provides common functionality used across multiple scripts
to reduce code duplication and improve maintainability.
"""

import sys
import time
from pathlib import Path
from typing import Callable, Any
from rich.console import Console
from rich.table import Table

# Global console instance
console = Console()


def setup_backend_path() -> None:
    """Add backend directory to Python path for imports."""
    backend_path = str(Path(__file__).parent.parent / "backend" / "src")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)


def format_size_mb(size_bytes: int) -> str:
    """Format size in bytes to MB with 2 decimal places."""
    return f"{size_bytes / (1024 * 1024):.2f}"


def format_size_kb(size_bytes: int) -> str:
    """Format size in bytes to KB with 1 decimal place.""" 
    return f"{size_bytes / 1024:.1f}"


def create_stats_table(title: str) -> Table:
    """Create a standardized table for displaying statistics."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    return table


def create_comparison_table(title: str) -> Table:
    """Create a standardized table for performance comparisons."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value 1", style="yellow") 
    table.add_column("Value 2", style="green")
    table.add_column("Improvement", style="magenta")
    return table


def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        console.print(f"⏱️ {func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper


def print_success(message: str) -> None:
    """Print a success message with checkmark emoji."""
    console.print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message with X emoji."""
    console.print(f"❌ {message}")


def print_warning(message: str) -> None:
    """Print a warning message with warning emoji."""
    console.print(f"⚠️ {message}")


def print_info(message: str) -> None:
    """Print an info message with info emoji."""
    console.print(f"ℹ️ {message}")


def print_progress(message: str) -> None:
    """Print a progress message with gear emoji."""
    console.print(f"🔄 {message}")


def format_percentage(value: float) -> str:
    """Format a decimal value as a percentage with 1 decimal place."""
    return f"{value * 100:.1f}%"


def calculate_improvement_percentage(old_value: float, new_value: float) -> str:
    """Calculate and format improvement percentage."""
    if old_value == 0:
        return "N/A"
    improvement = (1 - new_value / old_value) * 100
    return f"{improvement:.1f}%"


def safe_cleanup_files(file_paths: list[Path]) -> None:
    """Safely clean up multiple files, ignoring errors."""
    for file_path in file_paths:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print_warning(f"Failed to cleanup {file_path}: {e}")


def safe_cleanup_directory(directory: Path) -> None:
    """Safely remove directory if empty, ignoring errors."""
    try:
        if directory.exists() and directory.is_dir():
            directory.rmdir()
    except Exception as e:
        print_warning(f"Failed to cleanup directory {directory}: {e}")