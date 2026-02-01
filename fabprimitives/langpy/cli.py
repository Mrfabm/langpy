"""
LangPy CLI - Command-line interface for LangPy.
"""

import sys


def main():
    """Main entry point for the langpy CLI."""
    print("LangPy CLI")
    print("=" * 40)
    print()
    print("Usage:")
    print("  langpy --help     Show this help message")
    print("  langpy --version  Show version")
    print()
    print("Python Usage:")
    print("  from langpy import Langpy")
    print("  lb = Langpy(api_key='...')")
    print("  await lb.agent.run(model='openai:gpt-4', input='Hello!')")

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--version", "-v"):
            from langpy import __version__
            print(f"\nVersion: {__version__}")
        elif sys.argv[1] in ("--help", "-h"):
            pass  # Already printed help above


def init_project():
    """Initialize a new LangPy project."""
    import os

    print("Initializing LangPy project...")

    # Create basic project structure
    dirs = ["src", "tests"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created {d}/")

    # Create a sample file
    sample_code = '''"""Example LangPy application."""

import asyncio
from langpy import Langpy


async def main():
    # Initialize LangPy client
    lb = Langpy()

    # Example: Use the agent primitive
    response = await lb.agent.run(
        model="openai:gpt-4",
        input="Hello! What can you help me with?",
        instructions="Be helpful and concise."
    )

    print(response.output)


if __name__ == "__main__":
    asyncio.run(main())
'''

    if not os.path.exists("src/main.py"):
        with open("src/main.py", "w") as f:
            f.write(sample_code)
        print("  Created src/main.py")

    # Create .env template
    env_template = '''# LangPy Configuration
LANGPY_API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-key-here
'''

    if not os.path.exists(".env.example"):
        with open(".env.example", "w") as f:
            f.write(env_template)
        print("  Created .env.example")

    print("\nDone! Next steps:")
    print("  1. Copy .env.example to .env and add your API keys")
    print("  2. Run: python src/main.py")


if __name__ == "__main__":
    main()
