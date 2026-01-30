#!/usr/bin/env python3
"""
Run Script for Langpy Parser UI

Simple script to launch the Streamlit parser interface.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit parser UI."""
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "demos" / "parser" / "streamlit_parser_ui.py"
    
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    print("🚀 Starting Langpy Parser UI...")
    print(f"📄 App location: {app_path}")
    print("🌐 Opening browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Parser UI stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 