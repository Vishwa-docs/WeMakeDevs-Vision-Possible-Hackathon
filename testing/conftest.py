"""
Shared pytest configuration for the testing/ directory.

Ensures backend modules are importable from the workspace root.
"""

import os
import sys

# Add the backend directory to sys.path so tests can import
# mcp_tools, processors, providers, etc. directly.
BACKEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"
)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
