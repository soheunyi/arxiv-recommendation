#!/usr/bin/env python3
"""
Coverage monitoring and improvement tracking script.

Usage:
    uv run python scripts/coverage_monitor.py
    uv run python scripts/coverage_monitor.py --target 75
    uv run python scripts/coverage_monitor.py --trend
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_coverage():
    """Run tests and generate coverage data"""
    print("🧪 Running test suite with coverage...")
    
    result = subprocess.run([
        "uv", "run", "pytest", "tests/", 
        "--cov=backend", "--cov-report=json", "--cov-report=term", "-q"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Tests failed:\n{result.stderr}")
        return None
        
    return result.stdout


def analyze_coverage():
    """Analyze current coverage and identify improvements"""
    with open("coverage.json") as f:
        data = json.load(f)
    
    total_coverage = data["totals"]["percent_covered"]
    
    # Identify worst components
    files = []
    for filename, info in data["files"].items():
        if filename.startswith("backend/src/"):
            files.append({
                "name": filename.replace("backend/src/", ""),
                "coverage": info["summary"]["percent_covered"], 
                "statements": info["summary"]["num_statements"]
            })
    
    # Priority: 0% coverage with high statement count
    zero_coverage = [f for f in files if f["coverage"] == 0 and f["statements"] > 50]
    low_coverage = [f for f in files if 0 < f["coverage"] < 50 and f["statements"] > 100]
    
    return {
        "total": total_coverage,
        "zero_coverage": sorted(zero_coverage, key=lambda x: -x["statements"]),
        "low_coverage": sorted(low_coverage, key=lambda x: x["coverage"])
    }


def save_trend_data(coverage_data):
    """Save coverage data for trend analysis"""
    trend_file = Path("coverage_trends.json")
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "total_coverage": coverage_data["total"],
        "zero_coverage_files": len(coverage_data["zero_coverage"]),
        "low_coverage_files": len(coverage_data["low_coverage"])
    }
    
    if trend_file.exists():
        with open(trend_file) as f:
            trends = json.load(f)
    else:
        trends = []
    
    trends.append(entry)
    
    # Keep last 50 entries
    trends = trends[-50:]
    
    with open(trend_file, "w") as f:
        json.dump(trends, f, indent=2)


def show_improvement_suggestions(coverage_data):
    """Show actionable improvement suggestions"""
    print(f"\n📊 Current Coverage: {coverage_data['total']:.1f}%")
    print("=" * 50)
    
    if coverage_data["zero_coverage"]:
        print("\n🚨 CRITICAL: Zero Coverage Components")
        for i, comp in enumerate(coverage_data["zero_coverage"][:5], 1):
            potential_gain = (comp["statements"] / 4871) * 100
            print(f"{i}. {comp['name']:<30} (+{potential_gain:.1f}% potential)")
    
    if coverage_data["low_coverage"]:
        print("\n⚠️  IMPROVE: Low Coverage Components")
        for comp in coverage_data["low_coverage"][:3]:
            current = comp["coverage"]
            target = min(80, current + 30)
            potential_gain = ((target - current) / 100) * comp["statements"] / 4871 * 100
            print(f"   {comp['name']:<30} {current:>5.1f}% → {target}% (+{potential_gain:.1f}%)")
    
    # Calculate total potential
    zero_potential = sum(c["statements"] for c in coverage_data["zero_coverage"][:5]) / 4871 * 100
    low_potential = sum(min(30, 80 - c["coverage"]) / 100 * c["statements"] 
                       for c in coverage_data["low_coverage"][:3]) / 4871 * 100
    
    total_potential = coverage_data["total"] + zero_potential * 0.7 + low_potential
    print(f"\n🎯 Achievable Target: {total_potential:.1f}% coverage")
    
    # Next steps
    print(f"\n📋 Next Steps:")
    if coverage_data["zero_coverage"]:
        top_file = coverage_data["zero_coverage"][0]
        print(f"1. Create tests/unit/test_{top_file['name'].replace('.py', '')}.py")
        print(f"2. Target {top_file['statements']} statements for +{(top_file['statements']/4871)*100:.1f}% coverage")
    
    print("3. Run: uv run pytest tests/ --cov=backend --cov-report=html")
    print("4. Monitor: uv run python scripts/coverage_monitor.py --trend")


def show_trends():
    """Show coverage trends over time"""
    trend_file = Path("coverage_trends.json")
    if not trend_file.exists():
        print("❌ No trend data available. Run coverage first.")
        return
    
    with open(trend_file) as f:
        trends = json.load(f)
    
    if len(trends) < 2:
        print("📊 Insufficient data for trend analysis")
        return
        
    print("📈 Coverage Trends:")
    print("-" * 40)
    
    for i, entry in enumerate(trends[-10:]):  # Last 10 entries
        timestamp = datetime.fromisoformat(entry["timestamp"])
        coverage = entry["total_coverage"]
        
        if i > 0:
            prev_coverage = trends[len(trends)-10+i-1]["total_coverage"] 
            change = coverage - prev_coverage
            change_str = f"({change:+.1f}%)" if change != 0 else ""
        else:
            change_str = ""
            
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} | {coverage:5.1f}% {change_str}")


def main():
    parser = argparse.ArgumentParser(description="Coverage monitoring and improvement")
    parser.add_argument("--target", type=float, help="Target coverage percentage")
    parser.add_argument("--trend", action="store_true", help="Show coverage trends")
    parser.add_argument("--no-test", action="store_true", help="Skip running tests")
    
    args = parser.parse_args()
    
    if args.trend:
        show_trends()
        return
    
    if not args.no_test:
        test_output = run_coverage()
        if test_output is None:
            sys.exit(1)
    
    try:
        coverage_data = analyze_coverage()
        save_trend_data(coverage_data)
        show_improvement_suggestions(coverage_data)
        
        if args.target:
            if coverage_data["total"] >= args.target:
                print(f"✅ Target {args.target}% achieved!")
                sys.exit(0)
            else:
                gap = args.target - coverage_data["total"]
                print(f"🎯 Gap to target: {gap:.1f}%")
                sys.exit(1)
                
    except FileNotFoundError:
        print("❌ Coverage data not found. Run tests first.")
        sys.exit(1)


if __name__ == "__main__":
    main()