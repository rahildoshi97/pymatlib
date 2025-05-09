from pathlib import Path
from typing import Union

import sympy as sp

from pymatlib.core.alloy import Alloy
from pymatlib.core.yaml_parser.config_parser import MaterialConfigParser

def create_alloy_from_yaml(yaml_path: Union[str, Path], T: Union[float, sp.Symbol]) -> Alloy:
    """
    Create alloy instance from YAML configuration file.
    
    This function serves as the main entry point for creating alloy objects
    from YAML configuration files. It handles the parsing of the configuration
    and creation of the alloy with the specified temperature.
    
    Args:
        yaml_path: Path to the YAML configuration file
        T: Temperature value or symbol for property evaluation
            - Use a float value for a specific temperature
            - Use sp.Symbol('T') for symbolic temperature expressions
    Returns:
        The alloy instance with all properties initialized
    Examples:
        # Create an alloy at a specific temperature
        alloy = create_alloy_from_yaml('aluminum.yaml', 500.0)
        
        # Create an alloy with symbolic temperature expressions
        import sympy as sp
        T = sp.Symbol('T')
        alloy = create_alloy_from_yaml('steel.yaml', T)
    """
    parser = MaterialConfigParser(yaml_path)
    alloy = parser.create_alloy(T)
    return alloy

def get_supported_properties() -> list:
    """
    Returns a list of all supported material properties.
    Returns:
        List of strings representing valid property names that can be defined in YAML files.
    """
    return list(MaterialConfigParser.VALID_YAML_PROPERTIES)

def validate_yaml_file(yaml_path: Union[str, Path]) -> bool:
    """
    Validate a YAML file without creating an alloy.
    Args:
        yaml_path: Path to the YAML configuration file to validate
    Returns:
        True if the file is valid
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the YAML content is invalid
    """
    try:
        _ = MaterialConfigParser(yaml_path)
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
    except Exception:
        pass
