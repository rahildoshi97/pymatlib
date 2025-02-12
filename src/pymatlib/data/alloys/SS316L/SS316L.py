import time
import numpy as np
import sympy as sp
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt
from pymatlib.core.alloy import Alloy
from pymatlib.data.element_data import Fe, Cr, Ni, Mo, Mn
from pymatlib.core.models import thermal_diffusivity_by_heat_conductivity, density_by_thermal_expansion, energy_density
from pymatlib.core.data_handler import read_data, celsius_to_kelvin, thousand_times
from pymatlib.core.interpolators import interpolate_property, prepare_interpolation_arrays#, interpolate_binary_search
from pymatlib.core.cpp.fast_interpolation import interpolate_binary_search, interpolate_double_lookup


def create_SS316L(T: Union[float, sp.Symbol]) -> Alloy:
    """
    Creates an Alloy instance for SS316L stainless steel with specific properties.
    Args:
        T (Union[float, sp.Symbol]): Temperature as a symbolic variable or numeric value.
    Returns:
        Alloy: Initialized SS316L alloy with physical properties.
    Notes:
        - **Material Properties**:
            - **Density**: 8.0 g/cm³ (8000 kg/m³) at room temperature
            - **Composition**:
                - Iron (Fe): 67.5 wt% (Balance)
                - Chromium (Cr): 17.0 wt%
                - Nickel (Ni): 12.0 wt%
                - Molybdenum (Mo): 2.5 wt%
                - Manganese (Mn): 1.0 wt%
            - **Phase Transitions**:
                - Solidus: 1658.0 K (1385°C)
                - Liquidus: 1723.0 K (1450°C)
            - **Thermal Expansion**: 16.3 × 10^-6 /K at room temperature
        - **Data Units**: All input data should be in SI units:
            - **Temperature**: Kelvin (K)
            - **Density**: kg/m³
            - **Heat Capacity**: J/(kg·K)
            - **Heat Conductivity**: W/(m·K)
        - **Temperature Range**: Valid from room temperature (273.15K) to 2000K
        - **Property Variations**: Properties are temperature-dependent and implemented as piecewise functions
        - **Data Sources**: Property values based on experimental data and literature
        - **Input Data Files**: Required files in same directory:
            - density_temperature.txt
            - heat_capacity_temperature.txt
            - heat_conductivity_temperature.txt
    Example:
        >>> T = sp.Symbol('T')
        >>> ss316l = create_SS316L(T)
        >>> density_at_1000K = ss316l.density.evalf(T, 1000.0)
    """
    # Define the alloy with specific elemental composition and phase transition temperatures
    SS316L = Alloy(
        elements=[Fe, Cr, Ni, Mo, Mn],
        composition=[0.675, 0.17, 0.12, 0.025, 0.01],  # Fe: 67.5%, Cr: 17%, Ni: 12%, Mo: 2.5%, Mn: 1%
        temperature_solidus=1653.15,  # Solidus temperature in Kelvin (test at 1653.15 K = 1380 C)
        temperature_liquidus=1723.15,  # Liquidus temperature in Kelvin (test at 1723.15 K = 1450 C)
        thermal_expansion_coefficient=16.3e-6  # in 1/K
    )
    # density_data_file_path = "/local/ca00xebo/repos/pymatlib/src/pymatlib/data/alloys/SS316L/density_temperature.txt"
    # Determine the base directory
    base_dir = Path(__file__).parent  # Directory of the current file

    # Paths to data files using relative paths
    density_data_file_path = str(base_dir / 'density_temperature_edited.txt')
    heat_capacity_data_file_path = str(base_dir / 'heat_capacity_temperature_edited.txt')
    heat_conductivity_data_file_path = str(base_dir / '..' / 'SS316L' / 'heat_conductivity_temperature.txt')

    # Read temperature and material property data from the files
    density_temp_array, density_array = read_data(density_data_file_path)
    heat_capacity_temp_array, heat_capacity_array = read_data(heat_capacity_data_file_path)
    heat_conductivity_temp_array, heat_conductivity_array = read_data(heat_conductivity_data_file_path)

    # Ensure the data was loaded correctly
    if any(arr.size == 0 for arr in [density_temp_array, density_array, heat_capacity_temp_array, heat_capacity_array,
                                     heat_conductivity_temp_array, heat_conductivity_array]):
        raise ValueError("Failed to load temperature or material data from the file.")

    # Conversion: temperature to K and/or other material properties to SI units if necessary
    density_temp_array = celsius_to_kelvin(density_temp_array)
    density_array = thousand_times(density_array)  # Density in kg/m³ # gm/cm³ -> kg/m³

    heat_capacity_temp_array = celsius_to_kelvin(heat_capacity_temp_array)
    heat_capacity_array = thousand_times(heat_capacity_array)  # Specific heat capacity in J/(kg·K) # J/g-K -> J/kg-K

    heat_conductivity_temp_array = celsius_to_kelvin(heat_conductivity_temp_array)

    SS316L.heat_conductivity = interpolate_property(T, heat_conductivity_temp_array, heat_conductivity_array)
    SS316L.density = interpolate_property(T, density_temp_array, density_array)
    SS316L.heat_capacity = interpolate_property(T, heat_capacity_temp_array, heat_capacity_array)
    SS316L.thermal_diffusivity = thermal_diffusivity_by_heat_conductivity(SS316L.heat_conductivity, SS316L.density, SS316L.heat_capacity)
    SS316L.latent_heat_of_fusion = interpolate_property(T, SS316L.solidification_interval(), np.array([0.0, 260000.0]))
    SS316L.energy_density = energy_density(T, SS316L.density, SS316L.heat_capacity, SS316L.latent_heat_of_fusion)
    SS316L.energy_density_solidus = SS316L.energy_density.evalf(T, SS316L.temperature_solidus)
    SS316L.energy_density_liquidus = SS316L.energy_density.evalf(T, SS316L.temperature_liquidus)

    # Populate temperature_array and energy_density_array
    SS316L.temperature_array = density_temp_array

    SS316L.energy_density_array = np.array([
        SS316L.energy_density.evalf(T, temp) for temp in density_temp_array
    ])

    args = (SS316L.temperature_array,
            SS316L.energy_density_liquidus,
            SS316L.energy_density_array)

    E = SS316L.energy_density.evalf(T, SS316L.temperature_liquidus)
    T_eq, E_neq, E_eq, inv_delta_E_eq, idx_map = prepare_interpolation_arrays(
        SS316L.temperature_array,
        SS316L.energy_density_array
    )
    args3 = (E, T_eq, E_neq, E_eq, inv_delta_E_eq, idx_map)

    start_time3 = time.perf_counter()
    T_interpolate3 = interpolate_double_lookup(*args3)
    execution_time3 = time.perf_counter() - start_time3
    print(f"Interpolated temperature: {T_interpolate3}")
    print(f"Execution time: {execution_time3:.8f} seconds")

    T_star = interpolate_binary_search(*args)
    print(f"T_star: {T_star}")

    return SS316L


if __name__ == '__main__':
    Temp = sp.Symbol('T')
    alloy = create_SS316L(Temp)

    # Print the composition of each element in the alloy
    for i in range(len(alloy.composition)):
        print(f"Element {alloy.elements[i]}: {alloy.composition[i]}")

    print("\nTesting SS316L with symbolic temperature:")
    for field in vars(alloy):
        print(f"{field} = {alloy.__getattribute__(field)}")

    # Test interpolate_property
    print("\nTesting interpolate_property:")
    test_temp = 2003.15  # Example temperature value

    # Interpolate density, heat capacity, and heat conductivity
    density_result = alloy.density.evalf(Temp, test_temp)
    print(f"Interpolated density at T={test_temp} using evalf: {density_result}")

    heat_capacity_result = alloy.heat_capacity.evalf(Temp, test_temp)
    print(f"Interpolated heat capacity at T={test_temp} using evalf: {heat_capacity_result}")

    heat_conductivity_result = alloy.heat_conductivity.evalf(Temp, test_temp)
    print(f"Interpolated heat conductivity at T={test_temp} using evalf: {heat_conductivity_result}")

    # Test thermal diffusivity calculation
    heat_conductivity = heat_conductivity_result  # Example value for testing
    density = density_result  # Example value for testing
    heat_capacity = heat_capacity_result  # Example value for testing
    thermal_diffusivity = thermal_diffusivity_by_heat_conductivity(
        heat_conductivity, density, heat_capacity)
    print(f"Calculated thermal diffusivity at T={test_temp}: {thermal_diffusivity}")
