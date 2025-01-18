import time
import numpy as np
import sympy as sp
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt
from pymatlib.core.alloy import Alloy
from pymatlib.data.element_data import Fe, Cr, Ni, Mo, Mn
from pymatlib.core.models import thermal_diffusivity_by_heat_conductivity, density_by_thermal_expansion, energy_density, temperature_from_energy_density
from pymatlib.core.data_handler import read_data, celsius_to_kelvin, thousand_times
from pymatlib.core.interpolators import interpolate_property
from pymatlib.core.cpp.fast_interpolation import temperature_from_energy_density_array


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
    density_data_file_path = str(base_dir / 'density_temperature.txt')
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

    print("SS316L.heat_conductivity:", SS316L.heat_conductivity, "type:", type(SS316L.heat_conductivity))
    print("SS316L.density:", SS316L.density, "type:", type(SS316L.density))
    print("SS316L.heat_capacity:", SS316L.heat_capacity, "type:", type(SS316L.heat_capacity))
    print(f"SS316L.latent_heat_of_fusion: {SS316L.latent_heat_of_fusion}")
    print(f"SS316L.energy_density: {SS316L.energy_density}")
    print(f"SS316L.energy_density_solidus: {SS316L.energy_density_solidus}")
    print(f"SS316L.energy_density_liquidus: {SS316L.energy_density_liquidus}")

    """print("SS316L.heat_conductivity@T_sol/T_liq:", SS316L.heat_conductivity.evalf(T, SS316L.temperature_solidus), SS316L.heat_conductivity.evalf(T, SS316L.temperature_liquidus))
    print("SS316L.density@T_sol/T_liq:", SS316L.density.evalf(T, SS316L.temperature_solidus), SS316L.density.evalf(T, SS316L.temperature_liquidus))
    print("SS316L.heat_capacity@T_sol/T_liq:", SS316L.heat_capacity.evalf(T, SS316L.temperature_solidus), SS316L.heat_capacity.evalf(T, SS316L.temperature_liquidus))
    print("SS316L.latent_heat_of_fusion@T_sol/T_liq:", SS316L.latent_heat_of_fusion.evalf(T, SS316L.temperature_solidus), SS316L.latent_heat_of_fusion.evalf(T, SS316L.temperature_liquidus))
    print("SS316L.energy_density@T_sol/T_liq:", SS316L.energy_density.evalf(T, SS316L.temperature_solidus), SS316L.energy_density.evalf(T, SS316L.temperature_liquidus))"""

    """c_p = []
    density_temp_array = np.array(density_temp_array)
    for temp in density_temp_array:
        cp = SS316L.heat_capacity.evalf(T, temp)
        c_p.append(cp)
    c_p_array = np.array(c_p)
    print(c_p_array)"""

    # Populate temperature_array and energy_density_array
    SS316L.temperature_array = density_temp_array

    SS316L.energy_density_array = np.array([
        SS316L.energy_density.evalf(T, temp) for temp in density_temp_array
    ])

    # Create the plot
    """plt.figure(figsize=(10, 6))
    plt.plot(density_temp_array, energy_density_array, 'b-', linewidth=1)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Energy Density (J/m³)')
    plt.title('Energy Density vs Temperature')
    plt.grid(True)

    # Save the plot
    plt.savefig('energy_density_vs_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()"""

    print("----------" * 10)

    args = (T,
            SS316L.temperature_array,
            # SS316L.energy_density_solidus,  # 9744933767.272629
            # SS316L.energy_density_liquidus,  # 11789781961.769783
            # SS316L.energy_density.evalf(T, SS316L.temperature_liquidus),  # T_star: 1743.1412643772671, expected T_star: 1723.15
            # 10062147268.397945,
            1.01e10,
            SS316L.energy_density)
    # energy density has the same value for both temperatures >> function is not monotonically increasing
    print(SS316L.energy_density.evalf(T, 1723.15))  # 11789781961.769783
    print(SS316L.energy_density.evalf(T, 1743.1412643772671))  # 11789781961.769783

    args1 = (T,
            SS316L.temperature_array,
            SS316L.heat_capacity.evalf(T, SS316L.temperature_liquidus),
            SS316L.heat_capacity)

    start_time = time.time()
    T_star_1 = temperature_from_energy_density(*args)
    time_1 = time.time() - start_time
    print(f"T_star: {T_star_1}")
    print(f"Execution time: {time_1:.6f} seconds\n")

    args2 = (SS316L.temperature_array,
             # 10062147268.397945,
             1.01e10,
             SS316L.energy_density_array)

    start_time = time.time()
    T_star_2 = temperature_from_energy_density_array(*args2)
    time_2 = time.time() - start_time
    print(f"T_star_2: {T_star_2}")
    print(f"Execution time: {time_2:.6f} seconds\n")

    results = []
    execution_times = []

    for h_in in SS316L.energy_density_array:
        args = (T, SS316L.temperature_array, h_in, SS316L.energy_density)
        args_array = (SS316L.temperature_array, h_in, SS316L.energy_density_array)

        start_time = time.time()
        T_star = temperature_from_energy_density_array(*args_array)
        execution_time = time.time() - start_time

        results.append(T_star)
        execution_times.append(execution_time)

        """print(f"Heat Capacity: {h_in}")
        print(f"T_star: {T_star}")
        print(f"Execution time: {execution_time:.6f} seconds\n")"""

    print("Summary:")
    print(f"Total execution time: {sum(execution_times):.8f} seconds")
    print(f"Average execution time: {sum(execution_times)/len(execution_times):.8f} seconds\n")

    # Function to measure performance
    def measure_performance(iterations=1000):
        all_results = []
        all_execution_times = []

        for i in range(iterations):
            results = []
            execution_times = []

            for h_in in SS316L.energy_density_array:
                args_array = (density_temp_array, h_in, SS316L.energy_density_array)

                # Measure execution time
                start_time = time.time()
                T_star = temperature_from_energy_density_array(*args_array)
                execution_time = time.time() - start_time

                results.append(T_star)
                execution_times.append(execution_time)

            # Collect results for this iteration
            all_results.append(results)
            all_execution_times.append(sum(execution_times))

        # Performance summary
        total_time = sum(all_execution_times)
        average_time_per_run = total_time / iterations
        print("Performance Summary:")
        print(f"Total execution time (for {iterations} runs): {total_time:.8f} seconds")
        print(f"Average execution time per run: {average_time_per_run:.8f} seconds")
        print(f"Average execution time per iteration (array loop): {np.mean(all_execution_times) / len(SS316L.energy_density_array):.8f} seconds")

    # Run the performance test
    measure_performance(iterations=100000)

    quit()

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    temp_float = 1400.0
    temp_array = np.array([900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0], dtype=float)

    tec_float = 1.7e-5
    tec_temperature_array = np.array([973.15, 1073.15, 1173.15, 1273.15, 1373.15, 1395.68, 1415.00, 1435.00, 1455.26, 1500.00, 1550.00])
    tec_array = np.array([18.0e-6, 19.0e-6, 19.5e-6, 20.0e-6, 21.0e-6, 21.8e-6, 22.0e-6, 22.3e-6, 22.5e-6, 23.5e-6, 24.0e-6])
    tec = interpolate_property(T, tec_temperature_array, tec_array)

    # Define the corresponding thermal expansion coefficients at these points
    alpha_solidus = 17.0e-6   # 1/K at solidus
    alpha_liquidus = 18.0e-6  # 1/K at liquidus
    alpha_below = 15.0e-6     # 1/K below solidus
    alpha_above = 19.0e-6     # 1/K above liquidus

    # Fit a cubic polynomial for the thermal expansion coefficient
    # Define the symbolic variable for temperature
    Tmp = sp.symbols('Tmp')

    # Define the coefficients for a cubic polynomial: alpha(T) = a*T^3 + b*T^2 + c*T + d
    a, b, c, d = sp.symbols('a b c d')

    # Create a system of equations based on the known values at solidus, liquidus, and boundary conditions
    eq1 = sp.Eq(a*SS316L.temperature_solidus**3 + b*SS316L.temperature_solidus**2 + c*SS316L.temperature_solidus + d, alpha_solidus)  # At solidus
    eq2 = sp.Eq(a*SS316L.temperature_liquidus**3 + b*SS316L.temperature_liquidus**2 + c*SS316L.temperature_liquidus + d, alpha_liquidus)  # At liquidus
    eq3 = sp.Eq(a*973.15**3 + b*973.15**2 + c*973.15 + d, alpha_below)  # Below solidus (starting point)
    eq4 = sp.Eq(a*1550**3 + b*1550**2 + c*1550 + d, alpha_above)  # Above liquidus (ending point)
    # Solve the system for a, b, c, d
    solution = sp.solve([eq1, eq2, eq3, eq4], (a, b, c, d))
    # Substitute the solved values into the cubic expression
    tec_expr = solution[a]*Tmp**3 + solution[b]*Tmp**2 + solution[c]*Tmp + solution[d]
    print("tec_expr:", tec_expr, "type:", type(tec_expr))

    density_float = 8000.0
    density_array = density_array
    # print("density_array:", density_array, "type:", type(density_array))
    density_temp_array = density_temp_array
    # print("density_temp_array:", density_temp_array, "type:", type(density_temp_array))
    density_by_thermal_expansion_float = density_by_thermal_expansion(T, 293.0, 8000.0, tec_float)
    print("density_by_thermal_expansion_float:", density_by_thermal_expansion_float, "type:", type(density_by_thermal_expansion_float))
    print("expr type:", type(density_by_thermal_expansion_float.expr))
    print("assignments type:", type(density_by_thermal_expansion_float.assignments))
    # density_by_thermal_expansion_array = density_by_thermal_expansion(T, 293.0, 8000.0, tec_array)  # testing density_by_thermal_expansion without array support for tec
    # print("density_by_thermal_expansion_array:", density_by_thermal_expansion_array, "type:", type(density_by_thermal_expansion_array))  # Does not return a MaterialProperty when the input is an array
    density_by_thermal_expansion_expr = density_by_thermal_expansion(T, 293.0, 8000.0, tec)
    print("density_by_thermal_expansion_expr:", density_by_thermal_expansion_expr, "type:", type(density_by_thermal_expansion_expr))
    print("expr type:", type(density_by_thermal_expansion_expr.expr))
    print("assignments type:", type(density_by_thermal_expansion_expr.assignments[0].rhs))

    tec_interpolate_float = interpolate_property(T, tec_temperature_array, tec_array)
    print("tec_interpolate_float:", tec_interpolate_float, "type:", type(tec_interpolate_float))
    print("expr type:", type(tec_interpolate_float.expr))
    print("assignments type:", type(tec_interpolate_float.assignments[0].rhs))

    heat_capacity_interpolate_1 = interpolate_property(T, heat_capacity_temp_array, heat_capacity_array)
    print("heat_capacity_interpolate_1:", heat_capacity_interpolate_1, "type:", type(heat_capacity_interpolate_1.expr))
    print("expr type:", type(heat_capacity_interpolate_1.expr))
    print("assignments type:", type(heat_capacity_interpolate_1.assignments[0].rhs))

    # Example values (all floats)
    float_value = 300.15
    k_arr = np.array([15.1, 15.0, 14.9, 14.8, 14.7, 14.5, 14.3, 14.2, 14.0, 13.8, 13.7], dtype=float)  # W/m·K
    rho_arr = np.array([7900.0, 7850.0, 7800.0, 7750.0, 7700.0, 7650.0, 7600.0, 7550.0, 7500.0, 7450.0, 7400.0], dtype=float)  # kg/m³
    c_p_arr = np.array([500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0], dtype=float)  # J/kg·K
    print(np.size(k_arr), np.size(rho_arr), np.size(c_p_arr))

    diffusivity_1 = thermal_diffusivity_by_heat_conductivity(float_value, float_value, float_value)
    print("diffusivity_1:", diffusivity_1, "type:", type(diffusivity_1))
    print("expr type:", type(diffusivity_1.expr))
    print("assignments type:", type(diffusivity_1.assignments))
    print(f"diffusivity_1_evalf:\n{diffusivity_1.evalf(T, density_temp_array)}")

    # diffusivity_2 = thermal_diffusivity_by_heat_conductivity(k_arr, rho_arr, c_p_arr)
    # print("diffusivity_2:", diffusivity_2, "type:", type(diffusivity_2))

    # Thermal Conductivity (W/m·K)
    k_expr = sp.Piecewise(
        (16.3 - (T - 300) * (1.3 / (900 - 300)), T < 900),  # Starting from room temp (300K) to 900K
        (15.0 - (T - 900) * (0.2 / (1000 - 900)), T < 1000),
        (14.8 - (T - 1000) * (0.2 / (1100 - 1000)), (T >= 1000) & (T < 1100)),
        (14.6 - (T - 1100) * (0.2 / (1200 - 1100)), (T >= 1100) & (T < 1200)),
        (14.4 - (T - 1200) * (0.2 / (1300 - 1200)), (T >= 1200) & (T < 1300)),
        (14.2 - (T - 1300) * (0.2 / (1400 - 1300)), (T >= 1300) & (T < 1400)),
        (14.0 - (T - 1400) * (0.2 / (1500 - 1400)), (T >= 1400) & (T < 1500)),
        (13.8 - (T - 1500) * (0.2 / (1600 - 1500)), (T >= 1500) & (T < 1600)),
        (13.6 - (T - 1600) * (0.2 / (1700 - 1600)), (T >= 1600) & (T < 1700)),
        (13.4 - (T - 1700) * (0.2 / (1800 - 1700)), (T >= 1700) & (T < 1800)),
        (13.2, True)  # for T >= 1800
    )

    # Density (kg/m³)
    rho_expr = sp.Piecewise(
        (8000.0 - (T - 300) * (100.0 / (900 - 300)), T < 900),  # Room temp (300K) to 900K
        (7900.0 - (T - 900) * (50.0 / (1000 - 900)), (T >= 900) & (T < 1000)),
        (7850.0 - (T - 1000) * (50.0 / (1100 - 1000)), (T >= 1000) & (T < 1100)),
        (7800.0 - (T - 1100) * (50.0 / (1200 - 1100)), (T >= 1100) & (T < 1200)),
        (7750.0 - (T - 1200) * (50.0 / (1300 - 1200)), (T >= 1200) & (T < 1300)),
        (7700.0 - (T - 1300) * (50.0 / (1400 - 1300)), (T >= 1300) & (T < 1400)),
        (7650.0 - (T - 1400) * (50.0 / (1500 - 1400)), (T >= 1400) & (T < 1500)),
        (7600.0 - (T - 1500) * (50.0 / (1600 - 1500)), (T >= 1500) & (T < 1600)),
        (7550.0 - (T - 1600) * (50.0 / (1700 - 1600)), (T >= 1600) & (T < 1700)),
        (7500.0 - (T - 1700) * (50.0 / (1800 - 1700)), (T >= 1700) & (T < 1800)),
        (7450.0 - (T - 1800) * (50.0 / (1900 - 1800)), (T >= 1800) & (T < 1900)),
        (7400.0, True)  # for T >= 1900
    )

    # Specific Heat Capacity (J/kg·K)
    c_p_expr = sp.Piecewise(
        (500.0 + (T - 900) * (5.0 / (1000 - 900)), T < 1000),
        (505.0 + (T - 1000) * (5.0 / (1100 - 1000)), (T >= 1000) & (T < 1100)),
        (510.0 + (T - 1100) * (5.0 / (1200 - 1100)), (T >= 1100) & (T < 1200)),
        (515.0 + (T - 1200) * (5.0 / (1300 - 1200)), (T >= 1200) & (T < 1300)),
        (520.0 + (T - 1300) * (5.0 / (1400 - 1300)), (T >= 1300) & (T < 1400)),
        (525.0 + (T - 1400) * (5.0 / (1500 - 1400)), (T >= 1400) & (T < 1500)),
        (530.0 + (T - 1500) * (5.0 / (1600 - 1500)), (T >= 1500) & (T < 1600)),
        (535.0 + (T - 1600) * (5.0 / (1700 - 1600)), (T >= 1600) & (T < 1700)),
        (540.0 + (T - 1700) * (5.0 / (1800 - 1700)), (T >= 1700) & (T < 1800)),
        (545.0 + (T - 1800) * (5.0 / (1900 - 1800)), (T >= 1800) & (T < 1900)),
        (550.0, True)  # for T >= 1900
    )

    diffusivity_3 = thermal_diffusivity_by_heat_conductivity(k_expr, rho_expr, c_p_expr)
    print("diffusivity_3:", diffusivity_3, "type:", type(diffusivity_3))
    print("expr type:", type(diffusivity_3.expr))
    print("assignments type:", type(diffusivity_3.assignments))
    print(f"diffusivity_3_evalf:\n{diffusivity_3.evalf(T, density_temp_array)}")

    diffusivity_4 = interpolate_property(T, SS316L.solidification_interval(), (4.81e-6, 4.66e-6))
    print("diffusivity_4:", diffusivity_4, "type:", type(diffusivity_4))
    print("expr type:", type(diffusivity_4.expr))
    print("assignments type:", type(diffusivity_4.assignments[0].rhs))
    print(f"diffusivity_4_evalf:\n{diffusivity_4.evalf(T, density_temp_array)}")

    density_1 = density_by_thermal_expansion(123.4, 293.0, 8000.0, 0.001)
    print("density_1:", density_1)

    T_dbte = sp.Symbol('T_dbte')
    print(f"density_1.evalf:\n{density_1.evalf(T, density_temp_array)}")

    # SS316L.thermal_diffusivity = diffusivity_1

    # Perform interpolation for each property
    print('density')
    # SS316L.density = interpolate_property(T, density_temp_array, density_array)
    print('SS316L.density: ', SS316L.density)
    print('heat_capacity')
    # SS316L.heat_capacity = interpolate_property(T, heat_capacity_temp_array, heat_capacity_array)
    print('heat_conductivity')
    # SS316L.heat_conductivity = interpolate_property(T, heat_conductivity_temp_array, heat_conductivity_array)

    # Calculate thermal diffusivity from thermal conductivity, density, and heat capacity
    print('thermal_diffusivity')
    # print("SS316L.heat_conductivity.evalf(T, density_temp_array):", SS316L.heat_conductivity.evalf(T, density_temp_array))

    SS316L.thermal_diffusivity = thermal_diffusivity_by_heat_conductivity(SS316L.heat_conductivity, SS316L.density, SS316L.heat_capacity)

    # print('SS316L.heat_conductivity.evalf(T, density_temp_array): ', (SS316L.heat_conductivity.evalf(T, density_temp_array)))  # <class 'numpy.ndarray'>
    # print('SS316L.density.expr: ', type(SS316L.density.expr))  # <class 'sympy.core.mul.Mul'>
    # print('SS316L.heat_capacity.evalf(T, heat_conductivity_temp_array): ', (SS316L.heat_capacity.evalf(T, heat_conductivity_temp_array)))  # <class 'numpy.ndarray'>
    print('SS316L.thermal_diffusivity: ', SS316L.thermal_diffusivity)
    print('SS316L.thermal_diffusivity.evalf(T, 2003.15): ', (SS316L.thermal_diffusivity.evalf(T, density_temp_array)))
    print("----------" * 10)

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
