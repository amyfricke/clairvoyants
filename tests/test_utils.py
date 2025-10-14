"""
Utility functions for tests.
"""
import os

def get_example_file_path(filename):
    """Get the correct path to example files from any test file."""
    # Get the project root directory (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "examples", filename)

def get_data_file_path(filename):
    """Get the correct path to data files from any test file."""
    # Get the project root directory (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data", filename)
