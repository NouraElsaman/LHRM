"""Pytest configuration for LHRM baseline tests."""
import sys
import os
# Add src to path so tests can import from lhrm_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
