"""
tests/__init__.py

This file makes the tests directory a Python package and adds
the parent directory to the path so imports work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
