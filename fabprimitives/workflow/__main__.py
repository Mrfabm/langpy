"""
Main module entry point for LangPy workflow CLI.

Enables usage: python -m workflow run path/to/file.py --debug
"""

import sys
from .cli import main

if __name__ == '__main__':
    sys.exit(main()) 