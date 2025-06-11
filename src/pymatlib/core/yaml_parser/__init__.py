# File: core/yaml_parser/__init__.py
"""
pymatlib.core.yaml_parser
-------------------------
Exposes the main YAML-driven material creation and validation API.
"""

from pymatlib.core.yaml_parser.api import create_material_from_yaml, get_supported_properties, validate_yaml_file
from pymatlib.core.yaml_parser.piecewise_builder import PiecewiseBuilder
from pymatlib.core.yaml_parser.property_type_detector import PropertyType
from pymatlib.data.constants import ProcessingConstants, ErrorMessages, FileConstants

__all__ = [
    'create_material_from_yaml',
    'get_supported_properties',
    'validate_yaml_file',
    'PropertyType',
    'PiecewiseBuilder',
    'ProcessingConstants',
    'ErrorMessages',
    'FileConstants',
]
