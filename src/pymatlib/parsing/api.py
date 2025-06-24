import logging
from pathlib import Path
from typing import Union

import sympy as sp

from pymatlib.core.materials import Material
from pymatlib.parsing.config.material_yaml_parser import MaterialYAMLParser

logger = logging.getLogger(__name__)


def create_material(yaml_path: Union[str, Path], T: Union[float, sp.Symbol], enable_plotting: bool = True) -> Material:
    """
    Create material instance from YAML configuration file.
    
    This function serves as the main entry point for creating material (pure metal or alloy) objects
    from YAML configuration files. It handles the parsing of the configuration
    and creation of the material with the specified temperature.
    Args:
        yaml_path: Path to the YAML configuration file
        T: Temperature value or symbol for property evaluation
           - Use a float value for a specific temperature
           - Use a symbolic variable (e.g., sp.Symbol('T') or sp.Symbol('u_C'))
             for symbolic temperature expressions
        enable_plotting: Whether to generate visualization plots (default: True)
    Notes:
        In YAML files, always use 'T' as the temperature variable in equations.
        The system will automatically substitute this with your provided symbol.
    Returns:
        The material instance with all properties initialized
    Examples:
        # Create a material at a specific temperature
        material = create_material('aluminum.yaml', 500.0)

        # Create a material with symbolic temperature expressions
        import sympy as sp
        T = sp.Symbol('T')
        material = create_material('steel.yaml', T)

        # Create a material with a custom temperature symbol
        u_C = sp.Symbol('u_C')
        material = create_material('copper.yaml', u_C)
    """
    logger.info(f"Creating material from: {yaml_path}")
    try:
        parser = MaterialYAMLParser(yaml_path=yaml_path)
        material = parser.create_material(T=T, enable_plotting=enable_plotting)
        logger.info(f"Successfully created material: {material.name}\n")
        return material
    except Exception as e:
        logger.error(f"Failed to create material from {yaml_path}: {e}", exc_info=True)
        raise


def get_supported_properties() -> list:
    """
    Returns a list of all supported material properties.
    Returns:
        List of strings representing valid property names that can be defined in YAML files.
    """
    return list(MaterialYAMLParser.VALID_YAML_PROPERTIES)


def validate_yaml_file(yaml_path: Union[str, Path]) -> bool:
    """
    Validate a YAML file without creating the material.
    Args:
        yaml_path: Path to the YAML configuration file to validate
    Returns:
        True if the file is valid
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the YAML content is invalid
    """
    try:
        _ = MaterialYAMLParser(yaml_path)
        return True
    except FileNotFoundError as e:
        # Re-raise with more context
        raise FileNotFoundError(f"YAML file not found: {yaml_path}") from e
    except ValueError as e:
        # Provide more specific error
        raise ValueError(f"YAML validation failed: {str(e)}") from e
    except Exception as e:
        # Catch other errors
        raise ValueError(f"Unexpected error validating YAML: {str(e)}") from e


# --- Internal/Test Helper ---
def _test_api():
    T = sp.Symbol('T')
    try:
        assert validate_yaml_file('example.yaml') is True
    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"_test_api failed: {e}")
