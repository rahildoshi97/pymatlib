import logging
from pathlib import Path
from typing import Optional, Union

import sympy as sp

from pymatlib.core.materials import Material
from pymatlib.parsing.config.material_yaml_parser import MaterialYAMLParser

logger = logging.getLogger(__name__)


def create_material(yaml_path: Union[str, Path], T: Union[float, sp.Symbol],
                    enable_plotting: bool = True, generate_pdf_report: bool = False) -> Material:
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
        generate_pdf_report: Whether to generate comprehensive PDF report (default: False)
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
        material = parser.create_material(T=T, enable_plotting=enable_plotting,
                                          generate_pdf_report=generate_pdf_report)
        logger.info(f"Successfully created material: {material.name}\n")
        return material
    except Exception as e:
        logger.error(f"Failed to create material from {yaml_path}: {e}", exc_info=True)
        raise


def generate_pdf_report_only(yaml_path: Union[str, Path],
                             T: Union[float, sp.Symbol],
                             output_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Generate only a PDF report without creating plots or processing all properties.

    This is useful when you only need the PDF documentation without the computational overhead
    of full property processing and matplotlib plot generation.
    Args:
        yaml_path: Path to the YAML configuration file
        T: Temperature symbol or value for the report
        output_dir: Optional custom output directory for the PDF report
    Returns:
        Path to the generated PDF file
    Example:
        # Generate PDF report only
        pdf_path = generate_pdf_report_only('steel.yaml', sp.Symbol('T'))
        print(f"Report saved to: {pdf_path}")
    """
    logger.info(f"Generating PDF report for: {yaml_path}")
    try:
        parser = MaterialYAMLParser(yaml_path=yaml_path)
        # Create material without plots
        material = parser.create_material(T=T, enable_plotting=False, generate_pdf_report=False)
        # Generate PDF report
        if output_dir:
            # Temporarily change the PDF directory if custom output is specified
            original_pdf_dir = parser.pdf_reporter.pdf_directory
            parser.pdf_reporter.pdf_directory = Path(output_dir)
            parser.pdf_reporter.pdf_directory.mkdir(exist_ok=True)
        pdf_path = parser.pdf_reporter.generate_pdf_report(material, T)
        if output_dir:
            # Restore original directory
            parser.pdf_reporter.pdf_directory = original_pdf_dir
        logger.info(f"PDF report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Failed to generate PDF report for {yaml_path}: {e}", exc_info=True)
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


def get_material_info(yaml_path: Union[str, Path]) -> dict:
    """
    Get basic information about a material configuration without full processing.
    Args:
        yaml_path: Path to the YAML configuration file
    Returns:
        Dictionary containing material information
    Example:
        info = get_material_info('steel.yaml')
        print(f"Material: {info['name']}")
        print(f"Properties: {info['total_properties']}")
    """
    try:
        parser = MaterialYAMLParser(yaml_path=yaml_path)
        material_type = parser.config.get('material_type', 'Unknown')
        # Base information
        info = {
            'name': parser.config.get('name', 'Unknown'),
            'material_type': material_type,
            'composition': parser.config.get('composition', {})
        }
        # Add temperature properties based on material type
        if material_type == 'pure_metal':
            info['melting_temperature'] = parser.config.get('melting_temperature', 'Undefined')
            info['boiling_temperature'] = parser.config.get('boiling_temperature', 'Undefined')
        elif material_type == 'alloy':
            info['solidus_temperature'] = parser.config.get('solidus_temperature', 'Undefined')
            info['liquidus_temperature'] = parser.config.get('liquidus_temperature', 'Undefined')
            info['initial_boiling_temperature'] = parser.config.get('initial_boiling_temperature', 'Undefined')
            info['final_boiling_temperature'] = parser.config.get('final_boiling_temperature', 'Undefined')
        # Add remaining properties information
        info.update({
            'total_properties': sum(len(props) for props in parser.categorized_properties.values()),
            'property_types': {prop_type: len(props)
                               for prop_type, props in parser.categorized_properties.items()
                               if len(props) > 0},
            'properties': [prop_name for prop_list in parser.categorized_properties.values()
                           for prop_name, _ in prop_list]
        })
        return info
    except Exception as e:
        logger.error(f"Failed to get material info from {yaml_path}: {e}", exc_info=True)
        raise


# --- Internal/Test Helper ---
def _test_api():
    """Test function for API validation."""
    _ = sp.Symbol('T')
    try:
        assert validate_yaml_file('example.yaml') is True
    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"_test_api failed: {e}")
