"""
Legacy setup.py for backward compatibility.

Modern Python projects should use pyproject.toml for configuration.
This file is maintained for compatibility with older tools that don't support pyproject.toml.

Note: This project uses the src layout with source code in src/pymatlib/
"""

import warnings
from setuptools import setup

# Issue a deprecation warning
warnings.warn(
    "setup.py is deprecated. Use 'pip install .' or 'pip install -e .' "
    "which will use pyproject.toml configuration instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility, use setuptools' automatic discovery
# All configuration should be in pyproject.toml
if __name__ == "__main__":
    setup()
