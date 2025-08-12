#!/usr/bin/env python3
"""
ArXiv Recommendation System - Web Interface Launcher
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit web interface."""
    app_path = Path(__file__).parent / "web" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting ArXiv Recommendation System Web Interface...")
    print("📚 Navigate to the URL shown below to access your recommendations!")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped.")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()