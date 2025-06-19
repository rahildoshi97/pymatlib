# test_imports.py
from pathlib import Path

import sympy as sp

from pymatlib import create_material


def test_all_imports():
    """Test that all modules can be imported without circular dependencies."""
    try:
        # Test core imports
        from pymatlib.core.materials import Material
        from pymatlib.core.elements import ChemicalElement

        # Test parsing imports
        from pymatlib.parsing.validation.temperature_validator import TemperatureValidator
        from pymatlib.parsing.validation.property_validator import validate_monotonic_energy_density

        T = sp.Symbol('_u_C_')
        current_file = Path(__file__)
        yaml_path_Al = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "materials" / "pure_metals" / "Al" / "Al.yaml"
        yaml_path_SS304L = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "materials" / "alloys" / "SS304L" / "SS304L.yaml"

        mat_Al = create_material(yaml_path=yaml_path_Al, T=T, enable_plotting=True)
        mat_SS304L = create_material(yaml_path=yaml_path_SS304L, T=T, enable_plotting=True)

        print("✅ All imports successful - no circular dependencies detected")
        return True
    except ImportError as e:
        print(f"❌ Import error detected: {e}")
        return False

if __name__ == "__main__":
    test_all_imports()
