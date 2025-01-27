import time
import numpy as np
import sympy as sp
from pathlib import Path
from typing import Union
from pymatlib.core.alloy import Alloy
from pymatlib.data.element_data import Fe, Cr, Ni, Mo, Mn
from pymatlib.core.models import thermal_diffusivity_by_heat_conductivity, density_by_thermal_expansion, energy_density
from pymatlib.core.data_handler import read_data, celsius_to_kelvin, thousand_times, plot_arrays
from pymatlib.core.interpolators import (interpolate_property, check_equidistant, temperature_from_energy_density,
                                         E_eq_from_E_neq, create_idx_mapping, prepare_interpolation_arrays)
from pymatlib.core.cpp.fast_interpolation import temperature_from_energy_density_array, interpolate_double_lookup


def create_alloy(T: Union[float, sp.Symbol]) -> Alloy:
    alloy = Alloy(
        elements=[Fe, Cr, Ni, Mo, Mn],
        composition=[0.675, 0.17, 0.12, 0.025, 0.01],  # Fe: 67.5%, Cr: 17%, Ni: 12%, Mo: 2.5%, Mn: 1%
        temperature_solidus=1653.15,  # Solidus temperature in Kelvin (test at 1653.15 K = 1380 C)
        temperature_liquidus=1723.15,  # Liquidus temperature in Kelvin (test at 1723.15 K = 1450 C)
        thermal_expansion_coefficient=16.3e-6  # in 1/K
    )

    # Determine the base directory
    base_dir = Path(__file__).parent

    # Paths to data files using relative paths
    density_data_file_path = str(base_dir/'..'/'data'/'alloys'/'SS316L'/'density_temperature_edited.txt')
    heat_capacity_data_file_path = str(base_dir/'..'/'data'/'alloys'/'SS316L'/'heat_capacity_temperature_edited.txt')

    # Read temperature and material property data from the files
    density_temp_array, density_array = read_data(density_data_file_path)
    heat_capacity_temp_array, heat_capacity_array = read_data(heat_capacity_data_file_path)

    # Ensure the data was loaded correctly
    if any(arr.size < 2 for arr in [density_temp_array, density_array, heat_capacity_temp_array, heat_capacity_array]):
        raise ValueError("Failed to load temperature or material data from the file.")

    # Conversion: temperature to K and/or other material properties to SI units if necessary
    density_temp_array = celsius_to_kelvin(density_temp_array)
    density_array = thousand_times(density_array)  # Density in kg/m³ # gm/cm³ -> kg/m³
    # plot_arrays(density_temp_array, density_array, "Temperature (K)", "Density (Kg/m^3)")
    heat_capacity_temp_array = celsius_to_kelvin(heat_capacity_temp_array)
    heat_capacity_array = thousand_times(heat_capacity_array)  # Specific heat capacity in J/(kg·K) # J/g-K -> J/kg-K
    # plot_arrays(heat_capacity_temp_array, heat_capacity_array, "Temperature (K)", "Heat Capacity (J/Kg-K)")

    alloy.density = interpolate_property(T, density_temp_array, density_array)
    alloy.heat_capacity = interpolate_property(T, heat_capacity_temp_array, heat_capacity_array)
    alloy.latent_heat_of_fusion = interpolate_property(T, alloy.solidification_interval(), np.array([0.0, 260000.0]))
    alloy.energy_density = energy_density(T, alloy.density, alloy.heat_capacity, alloy.latent_heat_of_fusion)
    print(alloy.energy_density.evalf(T, 1723.15))

    # print("alloy.density:", alloy.density, "type:", type(alloy.density))
    # print("alloy.heat_capacity:", alloy.heat_capacity, "type:", type(alloy.heat_capacity))
    # print("alloy.latent_heat_of_fusion:", alloy.latent_heat_of_fusion, "type:", type(alloy.latent_heat_of_fusion))
    # print("alloy.energy_density:", alloy.energy_density, "type:", type(alloy.energy_density))

    # Populate temperature_array and energy_density_array
    alloy.temperature_array = density_temp_array

    alloy.energy_density_array = np.array([
        alloy.energy_density.evalf(T, temp) for temp in density_temp_array
    ])
    # print(alloy.temperature_array)
    # print(alloy.energy_density_array)
    # plot_arrays(alloy.temperature_array, alloy.energy_density_array, "Temperature (K)", "Energy Density (J/m^3)")

    T_eq, E_neq, E_eq, idx_map = prepare_interpolation_arrays(
        alloy.temperature_array,
        alloy.energy_density_array
    )

    # Online execution
    E = 11789781961.76978
    start_time = time.perf_counter()
    T_interpolate = interpolate_double_lookup(E, T_eq, E_neq, E_eq, idx_map)
    execution_time = time.perf_counter() - start_time
    print(f"Interpolated temperature: {T_interpolate}")
    print(f"Execution time: {execution_time:.8f} seconds")


    def measure_performance(iterations=1):
        all_execution_times = np.zeros(iterations)

        for i in range(iterations):
            start_measure_performance = time.perf_counter()

            results = [temperature_from_energy_density_array(T_eq, E, E_neq) for E in E_neq]
            # results = [interpolate_double_lookup(E, T_eq, E_neq, E_eq, idx_map) for E in E_neq]

            all_execution_times[i] = time.perf_counter() - start_measure_performance

        # Calculate statistics using numpy
        total_time = np.sum(all_execution_times)
        avg_time = np.mean(all_execution_times)
        avg_per_iteration = avg_time / len(E_neq)

        print(f"Total execution time ({iterations} runs): {total_time:.8f} seconds")
        print(f"Average execution time per run: {avg_time:.8f} seconds")
        print(f"Average execution time per iteration: {avg_per_iteration:.8f} seconds")

    # Run the performance test
    measure_performance(iterations=int(1e4))

    return alloy


if __name__ == '__main__':
    Temp = sp.Symbol('T')
    alloy = create_alloy(Temp)

    # Test arrays
    '''T_eq =  np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.])
    E_neq = np.array([1000., 1220., 1650., 2020., 2260., 2609., 3050., 3623., 3960., 4210.])
                    #  0      1      2      3      4      5      6      7      8      9

    # Equidistant energy array with delta_E_eq = 200 (smaller than min_delta = 220)
    E_eq = E_eq_from_E_neq(E_neq)
    # E_eq = np.array([1000., 1209., 1418., 1627., 1836., 2045., 2254., 2463., 2672.,
    #                  2881., 3090., 3299., 3508., 3717., 3926., 4135., 4344.])

    # Index mapping array
    idx_mapping = create_idx_mapping(E_neq, E_eq)
    # idx_map = np.array([0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8])

    # Test target energy values
    E_target = 1.*np.array([1000, 1585, 2688, 3960, 4210])
    print(E_target)
    for target in E_target:
        T_star = temperature_from_energy_density_array(T_eq, target, E_neq)
        print(T_star)
        T_interpolate_double_lookup = interpolate_double_lookup(target, T_eq, E_neq, E_eq, idx_mapping)
        print(T_interpolate_double_lookup)
        if T_star != T_interpolate_double_lookup:
            raise ValueError(f"Value Mismatch. {T_star} != {T_interpolate_double_lookup}")'''


    # Create larger test arrays
    size = 5_000_000
    T_eq = np.linspace(0.0, 5_000_000.0, size)  # Equidistant temperature values
    # Generate non-equidistant energy density array
    # Using cumsum of random values ensures monotonically increasing values
    E_neq = np.cumsum(np.random.uniform(1, 10, size)) + 5_000_000.0

    E_eq = E_eq_from_E_neq(E_neq)

    idx_mapping = create_idx_mapping(E_neq, E_eq)

    # Generate E_target array with specified proportions
    num_points = 1_000_000

    # Calculate points for each category
    below_count = int(num_points * 0.225)  # 22.5% below minimum
    above_count = int(num_points * 0.225)  # 22.5% above maximum
    boundary_count = int(num_points * 0.05)  # 5% boundary points
    inside_count = num_points - (below_count + above_count + boundary_count)  # Remaining points

    # Generate points for each category
    below_points = np.random.uniform(E_neq[0] - 5_000_000, E_neq[0], size=below_count)
    above_points = np.random.uniform(E_neq[-1], E_neq[-1] + 5_000_000, size=above_count)
    boundary_points = np.array([E_neq[0], E_neq[-1]] * (boundary_count // 2), dtype=np.float64)
    inside_points = np.random.uniform(E_neq[0], E_neq[-1], size=inside_count)

    # Combine and shuffle all points
    E_target = np.concatenate([np.float64(inside_points), np.float64(below_points), np.float64(above_points), boundary_points])
    np.random.shuffle(E_target)

    start_execution_1 = time.perf_counter()
    T_star = [temperature_from_energy_density_array(T_eq, float(E), E_neq) for E in E_target]
    execution_time_1 = time.perf_counter() - start_execution_1

    start_execution_2 = time.perf_counter()
    T_interpolate_double_lookup = [interpolate_double_lookup(float(E), T_eq, E_neq, E_eq, idx_mapping) for E in E_target]
    execution_time_2 = time.perf_counter() - start_execution_2

    print(f"execution_time_1 for temperature_from_energy_density_array: {execution_time_1} \n"
          f"execution_time_2 for interpolate_double_lookup: {execution_time_2}")

    # Compare results and show mismatches
    mismatches = [(E, T1, T2) for E, T1, T2 in zip(E_target, T_star, T_interpolate_double_lookup) if abs(T1 - T2) > 1e-8]

    if mismatches:
        print("Mismatches found (E_target, T_star, T_interpolate):")
        for E, T1, T2 in mismatches:
            print(f"E={E:.8f}, T1={T1:.8f}, T2={T2:.8f}, diff={abs(T1-T2):.8f}")
    else:
        print("No mismatches found between the two methods")
