# test_yaml_config.py
import logging
from pathlib import Path

import sympy as sp
from pymatlib.parsing.api import create_material, get_supported_properties

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -> %(message)s"
)
# Silence matplotlib and other noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)

T = sp.Symbol('_u_C_')

current_file = Path(__file__)
yaml_path_Al = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "materials" / "pure_metals" / "Al" / "Al.yaml"
yaml_path_SS304L = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "materials" / "alloys" / "SS304L" / "SS304L.yaml"

mat_Al = create_material(yaml_path=yaml_path_Al, T=T, enable_plotting=True)
mat_SS304L = create_material(yaml_path=yaml_path_SS304L, T=T, enable_plotting=True)

materials = [mat_Al, mat_SS304L] #[1]  # Change index to test different materials
for mat in materials:
    print(f"Name: {mat.name}")
    print(f"Elements: {mat.elements}")
    print(f"Composition: {mat.composition}")
    for i in range(len(mat.composition)):
        print(f"Element {mat.elements[i]}: Composition {mat.composition[i]}")
    print(f"\nSolidus Temperature: {mat.solidus_temperature}")
    print(f"Liquidus Temperature: {mat.liquidus_temperature}")
    print("\nTesting SS304L with symbolic temperature:")
    for field in vars(mat):
        print(f"{field}, {type(field)} = {mat.__getattribute__(field)}, type = {type(mat.__getattribute__(field))}")

    # Test computed properties at specific temperature
    test_temp = 273.15
    # Get all valid property names from the MaterialConfigParser
    valid_properties = get_supported_properties()
    print(f"\n{'='*80}")
    print(f"TESTING ALL MATERIAL PROPERTIES AT {test_temp}K")
    print(f"{'='*80}")
    '''print(f"\nAll possible material properties:")
    for prop in sorted(valid_properties):
        print(f"  - {prop}")
    print(f"\n{'='*80}")'''
    print(f"PROPERTY VALUES AT {test_temp}K")
    print(f"{'='*80}")
    # Test all properties that exist on the material
    for valid_prop_name in sorted(valid_properties):
        try:
            if hasattr(mat, valid_prop_name):
                valid_prop_value = getattr(mat, valid_prop_name)
                # Check if it's a symbolic expression
                if isinstance(valid_prop_value, sp.Expr):
                    # Substitute the temperature and evaluate
                    numerical_value = valid_prop_value.subs(T, test_temp).evalf()
                    print(f"{valid_prop_name:<30}: {numerical_value} (symbolic)")
                else:
                    # It's already a numerical value
                    print(f"{valid_prop_name:<30}: {valid_prop_value} (constant)")
            else:
                print(f"{valid_prop_name:<30}: Not defined in YAML")
        except Exception as e:
            print(f"{valid_prop_name:<30}: Error - {str(e)}")
    print(f"\n{'='*80}\n")
