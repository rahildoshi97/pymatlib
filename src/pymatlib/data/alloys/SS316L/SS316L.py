import numpy as np
import sympy as sp
from pathlib import Path
from typing import Union
from pymatlib.core.alloy import Alloy
from pymatlib.data.element_data import Fe, Cr, Ni, Mo, Mn
from pymatlib.core.models import (thermal_diffusivity_by_heat_conductivity, density_by_thermal_expansion,
                                  compute_energy_density_MaterialProperty, compute_energy_density_array)
from pymatlib.core.data_handler import read_data, celsius_to_kelvin, thousand_times
from pymatlib.core.interpolators import interpolate_property


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
        temperature_solidus=1658.0,  # Solidus temperature in Kelvin
        temperature_liquidus=1723.0,  # Liquidus temperature in Kelvin
        thermal_expansion_coefficient=16.3e-6  # in 1/K
    )

    # Determine the base directory
    base_dir = Path(__file__).parent  # Directory of the current file

    # Paths to data files using relative paths
    density_data_file_path = str(base_dir / 'density_temperature.txt')
    heat_capacity_data_file_path = str(base_dir / 'heat_capacity_temperature.txt')
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
    print("SS316L.heat_conductivity:", SS316L.heat_conductivity, "type:", type(SS316L.heat_conductivity))
    print("SS316L.density:", SS316L.density, "type:", type(SS316L.density))
    print("SS316L.heat_capacity:", SS316L.heat_capacity, "type:", type(SS316L.heat_capacity))
    SS316L.latent_heat_of_fusion = interpolate_property(T, SS316L.solidification_interval(), np.array([0.0, 260000.0]))
    print(f"SS316L.latent_heat_of_fusion: {SS316L.latent_heat_of_fusion}")
    # SS316L.energy_density = calc_energy_density(T, SS316L.density, SS316L.heat_capacity, SS316L.latent_heat_of_fusion)
    # print(f"SS316L.energy_density: {SS316L.energy_density}")
    SS316L.energy_density_solidus = compute_energy_density_MaterialProperty(T,
                                                        interpolate_property(SS316L.temperature_solidus, density_temp_array, density_array),
                                                        interpolate_property(T, heat_capacity_temp_array, heat_capacity_array),
                                                        interpolate_property(SS316L.temperature_solidus, SS316L.solidification_interval(), np.array([0.0, 260000.0])))
    print(f"SS316L.energy_density_solidus: {SS316L.energy_density_solidus}")
    SS316L.energy_density_liquidus = compute_energy_density_MaterialProperty(SS316L.temperature_liquidus,
                                                         interpolate_property(SS316L.temperature_liquidus, density_temp_array, density_array),
                                                         interpolate_property(SS316L.temperature_liquidus, heat_capacity_temp_array, heat_capacity_array),
                                                         interpolate_property(SS316L.temperature_liquidus, SS316L.solidification_interval(), np.array([0.0, 260000.0])))
    print(f"SS316L.energy_density_liquidus: {SS316L.energy_density_liquidus}")
    print("----------" * 10)

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

    diffusivity_4 = interpolate_property(T, SS316L.solidification_interval(), (4.81e-6, 4.66e-6))
    print("diffusivity_4:", diffusivity_4, "type:", type(diffusivity_4))
    print("expr type:", type(diffusivity_4.expr))
    print("assignments type:", type(diffusivity_4.assignments[0].rhs))

    # diffusivity_5 = interpolate_property(T, temp_array, density_by_thermal_expansion_array)
    # print("diffusivity_5:", diffusivity_5, "type:", type(diffusivity_5))

    density_1 = density_by_thermal_expansion(SS316L.solidification_interval(), 293.0, 8000.0, 0.001)
    print("density_1:", density_1)

    T_dbte = sp.Symbol('T_dbte')
    print("density_1.expr):", type(density_1.expr))
    print("density_1.assignments[0].rhs):", type(density_1.assignments[0].rhs))
    print("density_1.assignments[0].rhs.subs(T_dbte, 2003.15):", density_1.assignments[0].rhs.subs(T_dbte, 2003.15))

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
    print('SS316L.thermal_diffusivity.evalf(T, 2003.15): ', (heat_capacity_interpolate_1.evalf(T, 1000.0)))
    print("----------" * 10)

    SS316L.energy_density_solidus = compute_energy_density_MaterialProperty(SS316L.temperature_solidus,
                                                        interpolate_property(SS316L.temperature_solidus, density_temp_array, density_array),
                                                        interpolate_property(SS316L.temperature_solidus, heat_capacity_temp_array, heat_capacity_array),
                                                        interpolate_property(SS316L.temperature_solidus, SS316L.solidification_interval(), np.array([0.0, 260000.0])))
    print(f"SS316L.energy_density_solidus: {SS316L.energy_density_solidus}")
    print(interpolate_property(T, density_temp_array, density_array))
    print(interpolate_property(T, heat_capacity_temp_array, heat_capacity_array))

    file_path = "/local/ca00xebo/repos/pymatlib/src/pymatlib/data/alloys/SS316L/density_temperature.txt"
    test_density = 8000.0  # Example density (in kg/m³)
    test_heat_capacity = 500.0  # Example heat capacity (in J/(kg·K))
    test_latent_heat = 250000.0  # Example latent heat (in J/kg)

    print(f"density_by_thermal_expansion_expr: {density_by_thermal_expansion_expr}")
    print(f"density_by_thermal_expansion_float: {density_by_thermal_expansion_float}")
    print(f"heat_capacity_interpolate_1: {heat_capacity_interpolate_1}")
    test_temperatures, test_energy_densities = compute_energy_density_array(
        density_by_thermal_expansion_expr, test_heat_capacity, test_latent_heat, file_path
    )

    # Print the results
    print("Temperatures (K):")
    print(test_temperatures)
    print("\nEnergy Densities (J/m³):")
    print(test_energy_densities)

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
