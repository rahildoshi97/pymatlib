import logging
from pathlib import Path

import sympy as sp

from pymatlib.core.yaml_parser.api import create_material_from_yaml

logging.basicConfig(
    level=logging.INFO,  # Or INFO for less verbosity
    format="%(asctime)s %(levelname)s %(name)s -> %(message)s"
)
# Silence matplotlib and other noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)

# Create symbolic temperature variable
T = sp.Symbol('T')

# Get the path to the YAML file
current_file = Path(__file__)
# yaml_path = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "alloys" / "SS304L" / "SS304L.yaml"  # SS304L_pwlf_1
# yaml_path = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "alloys" / "SS304L" / "SS304L_comprehensive_2_copy.yaml"
yaml_path = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "alloys" / "SS304L" / "pure_metal.yaml"
# Create alloy from YAML
ss316l = create_material_from_yaml(yaml_path, T=T)
#ss316l_1 = create_material_from_yaml("SS304L_1.yaml", T)

# Test various properties
print(f"Elements: {ss316l.elements}")
print(f"Composition: {ss316l.composition}")
# Print the composition of each element in the alloy
for i in range(len(ss316l.composition)):
    print(f"Element {ss316l.elements[i]}: Composition {ss316l.composition[i]}")

print(f"\nSolidus Temperature: {ss316l.solidus_temperature}")
print(f"Liquidus Temperature: {ss316l.liquidus_temperature}")

print("\nTesting SS304L with symbolic temperature:")
for field in vars(ss316l):
    print(f"{field}, {type(field)} = {ss316l.__getattribute__(field)}, type = {type(ss316l.__getattribute__(field))}")

# Test computed properties at specific temperature
test_temp = 1670
print(f"\nProperties at {test_temp}K:")
print(f"Density: {(ss316l.density, T, test_temp)}")
print(f"Specific enthalpy: {(ss316l.specific_enthalpy, T, test_temp)}")
print(f"Heat Capacity: {(ss316l.heat_capacity, T, test_temp)}")
"""print(f"Heat Conductivity: {ss316l.heat_conductivity.evalf(T, test_temp)}")
print(f"Thermal Diffusivity: {ss316l.thermal_diffusivity.evalf(T, test_temp)}")
print(f"Energy Density: {ss316l.energy_density.evalf(T, test_temp)}")
print(f"Latent heat: {ss316l.latent_heat_of_fusion.evalf(T, test_temp)}")"""

# Test array generation for energy density
if hasattr(ss316l, 'energy_density_array'):
    print(f"\nEnergy Density Array Shape: {ss316l.energy_density_array.shape}")
