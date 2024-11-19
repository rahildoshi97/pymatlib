import numpy as np
import sympy as sp
from pathlib import Path
from typing import Union
from pymatlib.core.alloy import Alloy
from pymatlib.data.element_data import Fe, Cr, Mn, Ni
from pymatlib.core.models import thermal_diffusivity_by_heat_conductivity, density_by_thermal_expansion
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
        - **Data Units**: All input data should be in SI units. If data is not in SI units, it must be converted. For example:
            - **Temperature**: Convert Celsius to Kelvin using `celsius_to_kelvin` function.
            - **Density**: Should be in kg/m³.
            - **Heat Capacity**: Should be in J/(kg·K).
            - **Heat Conductivity**: Should be in W/(m·K).
        - **Temperature Array Consistency**: Ensure that all temperature arrays used for interpolation (`density_temp_array`, `heat_capacity_temp_array`, `heat_conductivity_temp_array`) have the same length.
          In this implementation, `density_temp_array` is used as the reference array for all properties to ensure consistency.
        - **Input Data Files**: The data files (`density_temperature.txt`, `heat_capacity_temperature.txt`, `heat_conductivity_temperature.txt`) must be located in the same directory as this script.
          They should contain data in the units specified above.

    Example:
        If you have temperature data in Celsius and property data in non-SI units, convert the temperature to Kelvin and property values to SI units before using them.
    """
    if isinstance(T, float) and (T < 0 or T > 3000):  # Example valid range: 0 to 3000 K
        raise ValueError("Invalid temperature. Temperature must be within the valid range (0 to 3000 K).")
    # Define the alloy with specific elemental composition and phase transition temperatures
    SS316L = Alloy(
        elements=[Fe, Cr, Mn, Ni],
        composition=[0.708, 0.192, 0.018, 0.082],  # Composition: 70.8% Fe, 19.2% Cr, 1.8% Mn, 8.2% Ni
        temperature_solidus=1395.68,  # Solidus temperature in Kelvin
        temperature_liquidus=1455.26,  # Liquidus temperature in Kelvin
        thermal_expansion_coefficient=1.7e-5  # in 1/K
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

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    temp_float = 1400.0
    temp_array = np.array([900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0], dtype=float)

    tec_float = 1.7e-5
    tec_temperature_array = np.array([973.15, 1073.15, 1173.15, 1273.15, 1373.15, 1395.68, 1415.00, 1435.00, 1455.26, 1500.00, 1550.00])
    tec_array = np.array([18.0e-6, 19.0e-6, 19.5e-6, 20.0e-6, 21.0e-6, 21.8e-6, 22.0e-6, 22.3e-6, 22.5e-6, 23.5e-6, 24.0e-6])
    tec = interpolate_property(T, tec_temperature_array, tec_array)

    # Define the corresponding thermal expansion coefficients at these points
    alpha_solidus = 21.8e-6  # 1/K at solidus
    alpha_liquidus = 22.5e-6  # 1/K at liquidus
    alpha_below = 18.0e-6     # 1/K below solidus (approximated starting value)
    alpha_above = 24.0e-6     # 1/K above liquidus (approximated final value)

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
    # density_by_thermal_expansion_array = density_by_thermal_expansion(T, 293.0, 8000.0, tec_array)  # testing density_by_thermal_expansion without array support for tec
    # print("density_by_thermal_expansion_array:", density_by_thermal_expansion_array, "type:", type(density_by_thermal_expansion_array))  # Does not return a MaterialProperty when the input is an array
    density_by_thermal_expansion_expr = density_by_thermal_expansion(T, 293.0, 8000.0, tec)
    print("density_by_thermal_expansion_expr:", density_by_thermal_expansion_expr, "type:", type(density_by_thermal_expansion_expr))

    tec_interpolate_float = interpolate_property(T, tec_temperature_array, tec_array)
    print("tec_interpolate_float:", tec_interpolate_float, "type:", type(tec_interpolate_float))

    heat_capacity_interpolate_1 = interpolate_property(T, heat_capacity_temp_array, heat_capacity_array)
    print("heat_capacity_interpolate_1:", heat_capacity_interpolate_1, "type:", type(heat_capacity_interpolate_1))

    # Example values (all floats)
    float_value = 300.15
    k_arr = np.array([15.1, 15.0, 14.9, 14.8, 14.7, 14.5, 14.3, 14.2, 14.0, 13.8, 13.7], dtype=float)  # W/m·K
    rho_arr = np.array([7900.0, 7850.0, 7800.0, 7750.0, 7700.0, 7650.0, 7600.0, 7550.0, 7500.0, 7450.0, 7400.0], dtype=float)  # kg/m³
    c_p_arr = np.array([500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0], dtype=float)  # J/kg·K
    print(np.size(k_arr), np.size(rho_arr), np.size(c_p_arr))

    diffusivity_1 = thermal_diffusivity_by_heat_conductivity(float_value, float_value, float_value)
    print("diffusivity_1:", diffusivity_1, "type:", type(diffusivity_1))

    # diffusivity_2 = thermal_diffusivity_by_heat_conductivity(k_arr, rho_arr, c_p_arr)
    # print("diffusivity_2:", diffusivity_2, "type:", type(diffusivity_2))

    # Thermal Conductivity (W/m·K)
    k_expr = sp.Piecewise(
        (15.1 - (T - 900) * (0.4 / (1000 - 900)), T < 1000),
        (15.0 - (T - 1000) * (0.4 / (1100 - 1000)), (T >= 1000) & (T < 1100)),
        (14.9 - (T - 1100) * (0.4 / (1200 - 1100)), (T >= 1100) & (T < 1200)),
        (14.8 - (T - 1200) * (0.4 / (1300 - 1200)), (T >= 1200) & (T < 1300)),
        (14.7 - (T - 1300) * (0.4 / (1400 - 1300)), (T >= 1300) & (T < 1400)),
        (14.5 - (T - 1400) * (0.3 / (1500 - 1400)), (T >= 1400) & (T < 1500)),
        (14.3 - (T - 1500) * (0.3 / (1600 - 1500)), (T >= 1500) & (T < 1600)),
        (14.2 - (T - 1600) * (0.2 / (1700 - 1600)), (T >= 1600) & (T < 1700)),
        (14.0 - (T - 1700) * (0.2 / (1800 - 1700)), (T >= 1700) & (T < 1800)),
        (13.8 - (T - 1800) * (0.1 / (1900 - 1800)), (T >= 1800) & (T < 1900)),
        (13.7, True)  # for T >= 1900
    )

    # Density (kg/m³)
    rho_expr = sp.Piecewise(
        (7900.0 - (T - 900) * (50.0 / (1000 - 900)), T < 1000),
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

    print("helloo")
    diffusivity_3 = thermal_diffusivity_by_heat_conductivity(k_expr, rho_expr, c_p_expr)
    print("diffusivity_3:", diffusivity_3, "type:", type(diffusivity_3))

    '''k_inputs = [k_arr, k_expr, float_value]
    rho_inputs = [rho_arr, rho_expr, float_value]
    c_p_inputs = [c_p_arr, c_p_expr, float_value]

    # Test all combinations of inputs using three nested loops
    for i, k in enumerate(k_inputs):
        for j, rho in enumerate(rho_inputs):
            for k, c_p in enumerate(c_p_inputs):
                try:
                    diffusivity = thermal_diffusivity_by_heat_conductivity(k, rho, c_p)
                    print(f"Combination k[{i+1}], rho[{j+1}], c_p[{k+1}]: diffusivity = {diffusivity}, type = {type(diffusivity)}")
                except Exception as e:
                    print(f"Combination k[{i+1}], rho[{j+1}], c_p[{k+1}]: Error - {e}") '''

    diffusivity_4 = interpolate_property(T, SS316L.solidification_interval(), (4.81e-6, 4.66e-6))
    print("diffusivity_4:", diffusivity_4, "type:", type(diffusivity_4))

    # diffusivity_5 = interpolate_property(T, temp_array, density_by_thermal_expansion_array)
    # print("diffusivity_5:", diffusivity_5, "type:", type(diffusivity_5))

    density_1 = density_by_thermal_expansion(SS316L.solidification_interval(), 293.0, 8000.0, 12.0)
    print("density_1:", density_1)

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
        # SS316L.heat_conductivity.expr,
        # SS316L.density.expr,
        # SS316L.heat_capacity.expr
    # )
    # print('SS316L.heat_conductivity.evalf(T, density_temp_array): ', (SS316L.heat_conductivity.evalf(T, density_temp_array)))  # <class 'numpy.ndarray'>
    # print('SS316L.density.expr: ', type(SS316L.density.expr))  # <class 'sympy.core.mul.Mul'>
    # print('SS316L.heat_capacity.evalf(T, heat_conductivity_temp_array): ', (SS316L.heat_capacity.evalf(T, heat_conductivity_temp_array)))  # <class 'numpy.ndarray'>
    print('SS316L.thermal_diffusivity: ', SS316L.thermal_diffusivity)
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
    test_temp = 1400.0  # Example temperature value

    # Interpolate density, heat capacity, and heat conductivity
    density_result = alloy.density.evalf(Temp, test_temp)
    print(f"Interpolated density at T={test_temp} using evalf: {density_result}")

    heat_capacity_result = alloy.heat_capacity.evalf(Temp, test_temp)
    print(f"Interpolated heat capacity at T={test_temp} using evalf: {heat_capacity_result}")

    heat_conductivity_result = alloy.heat_conductivity.evalf(Temp, test_temp)
    print(f"Interpolated heat conductivity at T={test_temp} using evalf: {heat_conductivity_result}")

    # Test thermal diffusivity calculation
    heat_conductivity = 500.  # Example value for testing
    density = 5000.  # Example value for testing
    heat_capacity = 600.  # Example value for testing
    thermal_diffusivity = thermal_diffusivity_by_heat_conductivity(
        heat_conductivity, density, heat_capacity)
    print(f"Calculated thermal diffusivity: {thermal_diffusivity}")
