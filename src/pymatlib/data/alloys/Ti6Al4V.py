import sympy as sp
from typing import Union
from pymatlib.core.alloy import Alloy
from pymatlib.core.constants import Constants
from pymatlib.data.element_data import Ti, Al, V
from pymatlib.core.interpolators import interpolate_equidistant, interpolate_lookup, interpolate_property
from pymatlib.core.models import density_by_thermal_expansion, thermal_diffusivity_by_heat_conductivity


def create_Ti6Al4V(T: Union[float, sp.Symbol]) -> Alloy:
    """
    Creates an Alloy instance for Ti6Al4V with specific properties.
    """
    # Define the Ti6Al4V alloy with its elemental composition and phase transition temperatures
    Ti6Al4V = Alloy(
        elements=[Ti, Al, V],
        composition=[0.90, 0.06, 0.04],         # 90% Ti, 6% Al, 4% V
        temperature_solidus=1878.0,             # Temperature solidus in Kelvin
        temperature_liquidus=1928.0,            # Temperature liquidus in Kelvin
        thermal_expansion_coefficient=8.6e-6,   # 1/K  [Source: MatWeb]
        heat_capacity=526.0,                      # J/(kg·K) [Source: MatWeb]
        latent_heat_of_fusion=290000.0,         # J/kg [Source: MatWeb]
        latent_heat_of_vaporization=8.86e6      # J/kg [Source: MatWeb]
    )

    # Compute density based on thermal expansion and other constants
    Ti6Al4V.density = density_by_thermal_expansion(
        T, Constants.temperature_room, 4430, Ti6Al4V.thermal_expansion_coefficient)
    print("Ti6Al4V.density:", Ti6Al4V.density)

    # Interpolate heat conductivity based on temperature
    Ti6Al4V.heat_conductivity = interpolate_property(
        T, Ti6Al4V.solidification_interval(), (6.7, 32.0))  # W/(m·K)
    print("Ti6Al4V.heat_conductivity:", Ti6Al4V.heat_conductivity)
    # print("Ti6Al4V.heat_conductivity.evalf(T, Ti6Al4V.solidification_interval()):", Ti6Al4V.heat_conductivity.evalf(T, Ti6Al4V.solidification_interval()))
    print("Ti6Al4V.heat_capacity:", Ti6Al4V.heat_capacity)

    k_evalf = Ti6Al4V.heat_conductivity.evalf(T, Ti6Al4V.solidification_interval())
    print("k_evalf:", k_evalf, "type:", type(k_evalf))
    rho_evalf = Ti6Al4V.density.evalf(T, Ti6Al4V.solidification_interval())
    print("rho_evalf:", rho_evalf, "type:", type(rho_evalf))
    # c_p_evalf = Ti6Al4V.heat_capacity.evalf(T, Ti6Al4V.solidification_interval())  # AttributeError: 'float' object has no attribute 'evalf'
    # print("c_p_evalf:", c_p_evalf, "type:", type(c_p_evalf))

    Ti6Al4V.thermal_diffusivity = thermal_diffusivity_by_heat_conductivity(Ti6Al4V.heat_conductivity, Ti6Al4V.density, Ti6Al4V.heat_capacity)
    # Ti6Al4V.thermal_diffusivity = interpolate_property(T, Ti6Al4V.solidification_interval(), k_evalf)  # only works with alloy = create_Ti6Al4V(Temp)
    print("Ti6Al4V.thermal_diffusivity:", Ti6Al4V.thermal_diffusivity, "type:", type(Ti6Al4V.thermal_diffusivity))
    print("----------" * 10)

    return Ti6Al4V


if __name__ == '__main__':
    Temp = sp.Symbol('Temp')
    alloy = create_Ti6Al4V(Temp)

    print("Ti-6Al-4V Alloy Properties:")
    print(f"Density: {alloy.density}")
    print(f"Heat Conductivity: {alloy.heat_conductivity}")
    print(f"Thermal Diffusivity: {alloy.thermal_diffusivity}")

    # Testing Ti6Al4V with symbolic temperature
    print("\nTesting Ti6Al4V with symbolic temperature:")
    for field in vars(alloy):
        print(f"{field} = {alloy.__getattribute__(field)}")

    # Test interpolate_equidistant
    print("\nTesting interpolate_equidistant:")
    test_temp = 1850.0  # Example temperature value
    result = interpolate_equidistant(
        Temp, 1820.0, 20.0, [700., 800., 900., 1000., 1100., 1200.])
    print(f"Interpolated heat capacity at T={test_temp} using interpolate_equidistant: {result.expr}")
    print(f"Assignments: {result.assignments}")

    result_density = interpolate_equidistant(
        test_temp, 1878, 8, [4236.3, 4234.9, 4233.6, 4232.3, 4230.9, 4229.7, 4227.0])
    print(f"Interpolated density at T={test_temp} using interpolate_equidistant: {result_density}")

    # Test interpolate_lookup
    print("\nTesting interpolate_lookup:")
    test_T_array = [1878.0, 1884.2, 1894.7, 1905.3, 1915.8, 1926.3, 1928.0]
    test_v_array = [4236.3, 4234.9, 4233.6, 4232.3, 4230.9, 4229.7, 4227.0]
    result_lookup = interpolate_lookup(
        Temp, test_T_array, test_v_array)
    print(f"Interpolated value using interpolate_lookup: {result_lookup}")

    # Test thermal diffusivity calculation
    heat_conductivity = 500.  # Example value for testing
    density = 5000.  # Example value for testing
    heat_capacity = 600.  # Example value for testing
    thermal_diffusivity = thermal_diffusivity_by_heat_conductivity(
        heat_conductivity, density, heat_capacity)
    print(f"Calculated thermal diffusivity: {thermal_diffusivity}")
