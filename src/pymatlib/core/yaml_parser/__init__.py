"""
pymatlib.core.yaml_parser
-------------------------
Exposes the main YAML-driven material creation and validation API.
"""

from pymatlib.core.yaml_parser.api import create_material_from_yaml, get_supported_properties, validate_yaml_file
from pymatlib.core.yaml_parser.property_types import PropertyType

__all__ = [
    'create_material_from_yaml',
    'get_supported_properties',
    'validate_yaml_file',
    'PropertyType',
]
