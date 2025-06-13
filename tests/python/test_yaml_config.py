import logging
from pathlib import Path

import sympy as sp
from pymatlib.core.yaml_parser.api import create_material_from_yaml, get_supported_properties

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -> %(message)s"
)
# Silence matplotlib and other noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)

# Create symbolic temperature variable
T = sp.Symbol('_U_')

current_file = Path(__file__)
# yaml_path = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "pure_metals" / "Al" / "Al.yaml"
yaml_path = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "alloys" / "SS304L" / "SS304L.yaml"
ss316l = create_material_from_yaml(yaml_path=yaml_path, T=T, enable_plotting=True)

print(f"Elements: {ss316l.elements}")
print(f"Composition: {ss316l.composition}")
for i in range(len(ss316l.composition)):
    print(f"Element {ss316l.elements[i]}: Composition {ss316l.composition[i]}")
print(f"\nSolidus Temperature: {ss316l.solidus_temperature}")
print(f"Liquidus Temperature: {ss316l.liquidus_temperature}")
print("\nTesting SS304L with symbolic temperature:")
for field in vars(ss316l):
    print(f"{field}, {type(field)} = {ss316l.__getattribute__(field)}, type = {type(ss316l.__getattribute__(field))}")

# Test computed properties at specific temperature
test_temp = 273.15
# Get all valid property names from the MaterialConfigParser
valid_properties = get_supported_properties()
print(f"\n{'='*80}")
print(f"TESTING ALL MATERIAL PROPERTIES AT {test_temp}K")
print(f"{'='*80}")
print(f"\nAll possible material properties:")
for prop in sorted(valid_properties):
    print(f"  - {prop}")
print(f"\n{'='*80}")
print(f"PROPERTY VALUES AT {test_temp}K")
print(f"{'='*80}")
# Test all properties that exist on the material
for valid_prop_name in sorted(valid_properties):
    try:
        if hasattr(ss316l, valid_prop_name):
            valid_prop_value = getattr(ss316l, valid_prop_name)
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
