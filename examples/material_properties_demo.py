"""Demonstration script for material property evaluation."""
import logging
from pathlib import Path
import sympy as sp

from pymatlib.parsing.api import create_material, get_supported_properties

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -> %(message)s"
    )
    # Silence noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)

def demonstrate_material_properties():
    """Demonstrate material property evaluation."""
    setup_logging()

    T = sp.Symbol('T')

    current_file = Path(__file__)
    yaml_path_Al = current_file.parent.parent / "src" / "pymatlib" / "data" / "materials" / "pure_metals" / "Al" / "Al.yaml"
    yaml_path_SS304L = current_file.parent.parent / "src" / "pymatlib" / "data" / "materials" / "alloys" / "SS304L" / "SS304L.yaml"

    materials = []
    print(f"\n{'='*80}")

    if yaml_path_Al.exists():
        mat_Al = create_material(yaml_path=yaml_path_Al, T=T, enable_plotting=True)
        materials.append(mat_Al)
    else:
        raise FileNotFoundError(f"Aluminum YAML file not found: {yaml_path_Al}")

    if yaml_path_SS304L.exists():
        mat_SS304L = create_material(yaml_path=yaml_path_SS304L, T=T, enable_plotting=True)
        materials.append(mat_SS304L)
    else:
        raise FileNotFoundError(f"SS304L YAML file not found: {yaml_path_SS304L}")

    for mat in materials:
        print(f"\n{'='*80}")
        print(f"MATERIAL: {mat.name}")
        print(f"{'='*80}")
        print(f"Name: {mat.name}")
        print(f"Type: {mat.material_type}")
        print(f"Elements: {[elem.name for elem in mat.elements]}")
        print(f"Composition: {mat.composition}")

        for i in range(len(mat.composition)):
            print(f"  {mat.elements[i].name}: {mat.composition[i]}")

        if hasattr(mat, 'solidus_temperature'):
            print(f"Solidus Temperature: {mat.solidus_temperature}")
        if hasattr(mat, 'liquidus_temperature'):
            print(f"Liquidus Temperature: {mat.liquidus_temperature}")
        if hasattr(mat, 'melting_temperature'):
            print(f"Melting Temperature: {mat.melting_temperature}")
        if hasattr(mat, 'boiling_temperature'):
            print(f"Boiling Temperature: {mat.boiling_temperature}")

        # Test computed properties at specific temperature
        test_temp = 273.15
        valid_properties = get_supported_properties()

        print(f"\n{'='*80}")
        print(f"PROPERTY VALUES AT {test_temp}K")
        print(f"{'='*80}")

        for valid_prop_name in sorted(valid_properties):
            try:
                if hasattr(mat, valid_prop_name):
                    valid_prop_value = getattr(mat, valid_prop_name)
                    if isinstance(valid_prop_value, sp.Expr):
                        numerical_value = valid_prop_value.subs(T, test_temp).evalf()
                        print(f"{valid_prop_name:<30}: {numerical_value} (symbolic)")
                    else:
                        print(f"{valid_prop_name:<30}: {valid_prop_value} (constant)")
                else:
                    print(f"{valid_prop_name:<30}: Not defined in YAML")
            except Exception as e:
                print(f"{valid_prop_name:<30}: Error - {str(e)}")

if __name__ == "__main__":
    demonstrate_material_properties()
