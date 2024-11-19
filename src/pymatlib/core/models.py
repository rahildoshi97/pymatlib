import numpy as np
import sympy as sp
from typing import Union, List
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes


def Wrapper(value: Union[sp.Expr, float, np.float32, np.float64, ArrayTypes]) -> Union[sp.Expr, List[sp.Expr]]:
    if isinstance(value, sp.Expr):
        return sp.simplify(value)
    if isinstance(value, (float, np.float32, np.float64)):
        return sp.Float(float(value))
    if isinstance(value, ArrayTypes):  # Handles lists, tuples, or arrays
        return [sp.Float(float(v)) for v in value]
    raise ValueError(f"Unsupported type for value in Wrapper: {type(value)}")


def MaterialPropertyWrapper(value: Union[sp.Expr, float, np.float32, np.float64, ArrayTypes]) -> MaterialProperty:
    wrapped_value = Wrapper(value)
    return MaterialProperty(wrapped_value)


def density_by_thermal_expansion(
        temperature: Union[float, ArrayTypes, sp.Expr],  # changed temperature from np.ndarray to ArrayType
        temperature_base: float,
        density_base: float,
        thermal_expansion_coefficient: Union[float, MaterialProperty]) \
        -> MaterialProperty:  # removed np.ndarray for tec, added MaterialProperty for tec, since thermal_expansion_coefficient: PropertyTypes
        # the return type is no longer Union[float, np.ndarray, sp.Expr] but just a MaterialProperty
    from pymatlib.core.interpolators import interpolate_property
    """
    Calculate density based on thermal expansion using the formula:
    rho(T) = rho_0 / (1 + tec * (T - T_0))^3
    """
    if isinstance(temperature, ArrayTypes):
        temperature = np.asarray(temperature)

    if isinstance(temperature, ArrayTypes) and isinstance(thermal_expansion_coefficient, MaterialProperty):
        raise TypeError(f"Incompatible combination of temperature (type:{type(temperature)}) and thermal expansion coefficient ({type(thermal_expansion_coefficient)})")

    if isinstance(thermal_expansion_coefficient, float):
        density = density_base * (1 + thermal_expansion_coefficient * (temperature - temperature_base)) ** (-3)
        if isinstance(density, np.ndarray):
            T_dbte = sp.Symbol('T_dbte')
            density = interpolate_property(T_dbte, temperature, density)
        else:
            isinstance(density, (float, sp.Expr))
            density = MaterialPropertyWrapper(density)
        return density

    else:  # isinstance(thermal_expansion_coefficient, MaterialProperty)
        tec_expr = thermal_expansion_coefficient.expr  # extract the expr (sp.Expr) for tec
        density = density_base * (1 + tec_expr * (temperature - temperature_base)) ** (-3)
        return MaterialPropertyWrapper(density)


def thermal_diffusivity_by_heat_conductivity(
        heat_conductivity: Union[float, MaterialProperty],
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty]) \
        -> MaterialProperty:
    """
    Calculate thermal diffusivity using the formula:
    alpha(T) = k(T) / (rho(T) * c_p(T))
    """
    # Input validation to check for incompatible data types
    incompatible_inputs = []
    if isinstance(heat_conductivity, np.ndarray):
        incompatible_inputs.append(f"heat_conductivity (type: {type(heat_conductivity)})")
    if isinstance(density, np.ndarray):
        incompatible_inputs.append(f"density (type: {type(density)})")
    if isinstance(heat_capacity, np.ndarray):
        incompatible_inputs.append(f"heat_capacity (type: {type(heat_capacity)})")
    if incompatible_inputs:
        raise TypeError(f"Incompatible input data type(s): {', '.join(incompatible_inputs)}")

    sub_assignments = [
        assignment for prop in [heat_conductivity, density, heat_capacity]
        if isinstance(prop, MaterialProperty)
        for assignment in prop.assignments
    ]
    # Handle the symbolic expression, using `.expr` only for MaterialProperty objects
    k_expr = heat_conductivity.expr if isinstance(heat_conductivity, MaterialProperty) else Wrapper(heat_conductivity)
    rho_expr = density.expr if isinstance(density, MaterialProperty) else Wrapper(density)
    cp_expr = heat_capacity.expr if isinstance(heat_capacity, MaterialProperty) else Wrapper(heat_capacity)

    result = k_expr / (rho_expr * cp_expr)
    result = sp.simplify(result)  # computationally expensive
    return MaterialProperty(result, sub_assignments)


if __name__ == '__main__':
    # Test 1: Single temperature value for density calculation
    T_0 = 800.
    T_1 = 1000.
    tec = 1e-6
    rho = 8000.
    print(f"Test 1 - Single temperature value: {T_1}, Density: {density_by_thermal_expansion(T_1, T_0, rho, tec)}")
    # Expected output: Density value at T_1 = 1000

    # Test 2: Temperature array for density calculation with constant TEC
    T_a = np.linspace(1000, 2000, 3)
    tec_a = np.ones(T_a.shape) * tec
    print("-----", type(Wrapper(tec_a)))
    print(f"Test 2a - Temperature array with constant TEC: {T_a}, Density: {density_by_thermal_expansion(T_a, T_0, rho, tec)}")
    # Expected output: Array of densities for temperatures in T_a

    # Test 3: Temperature array for density calculation with array TEC
    print(f"Test 2b - Temperature array with array TEC: {T_a}, Density: {density_by_thermal_expansion(T_a, T_0, rho, tec)}")
    # Expected output: Array of densities with temperature-dependent TEC

    # Test 4: Thermal diffusivity calculation with scalar values
    k = 30.
    c_p = 600.
    print(f"Test 3 - Thermal diffusivity with scalar values: {thermal_diffusivity_by_heat_conductivity(k, rho, c_p)}")
    # Expected output: Scalar value of thermal diffusivity

    # Test 5: Thermal diffusivity calculation with array values for heat conductivity
    k_a = np.linspace(30, 40, 3)
    print(f"Test 4a - Thermal diffusivity with array of heat conductivity: {thermal_diffusivity_by_heat_conductivity(k, rho, c_p)}")
    # Expected output: Array of thermal diffusivity

    # Test 6: Thermal diffusivity with density calculated by thermal expansion
    calculated_densities = density_by_thermal_expansion(T_a, T_0, rho, tec)
    print(f"Test 4b - Thermal diffusivity with calculated densities: {thermal_diffusivity_by_heat_conductivity(k, calculated_densities, c_p)}")
    # Expected output: Array of thermal diffusivity considering temperature-dependent density

    # Test 7: Symbolic computation for density with sympy Symbol temperature
    T_symbolic = sp.Symbol('T')
    print(f"Test 5 - Symbolic density computation: {density_by_thermal_expansion(T_symbolic, T_0, rho, tec)}")
    # Expected output: Sympy expression for density as a function of temperature

    # Test 8: Symbolic computation for thermal diffusivity
    k_symbolic = sp.Symbol('k')
    c_p_symbolic = sp.Symbol('c_p')
    rho_symbolic = sp.Symbol('rho')
    print(f"Test 6 - Symbolic thermal diffusivity computation: {thermal_diffusivity_by_heat_conductivity(k_symbolic, rho_symbolic, c_p_symbolic)}")

    # New test case for mixed input types
    print("\nTest 9: Mixed input types")

    # For density_by_thermal_expansion
    T_mixed = np.linspace(800, 1000, 3)  # numpy array
    T_base = 293.15  # float
    rho_base = 8000.0  # float
    tec_symbolic = sp.Symbol('alpha')  # symbolic

    try:
        result_density = density_by_thermal_expansion(T_mixed, T_base, rho_base, tec_symbolic)
        print(f"Density with mixed inputs: {result_density}")
    except Exception as e:
        print(f"Error in density calculation with mixed inputs: {str(e)}")

    # For thermal_diffusivity_by_heat_conductivity
    k_mixed = np.linspace(20, 30, 3)  # numpy array
    rho_symbolic = sp.Symbol('rho')  # symbolic
    cp_float = 500.0  # float

    try:
        result_diffusivity = thermal_diffusivity_by_heat_conductivity(k_mixed, rho_symbolic, cp_float)
        print(f"Thermal diffusivity with mixed inputs: {result_diffusivity}")
    except Exception as e:
        print(f"Error in thermal diffusivity calculation with mixed inputs: {str(e)}")

    # Test 11: Edge cases with zero and negative values
    print("\nTest 11: Edge cases with zero and negative values")
    try:
        print(density_by_thermal_expansion(-273.15, T_0, rho, tec))  # Absolute zero
    except ValueError as e:
        print(f"Handled error for negative temperature: {e}")

    try:
        print(density_by_thermal_expansion(T_1, T_0, rho, -1e-6))  # Negative TEC
    except ValueError as e:
        print(f"Handled error for negative TEC: {e}")

    # Test 12: Large arrays
    print("\nTest 12: Large arrays")
    large_T = np.linspace(1000, 2000, 10000)
    large_k = np.linspace(30, 40, 10000)
    try:
        large_density = density_by_thermal_expansion(large_T, T_0, rho, tec)
        print(f"Large density array: {large_density}")
    except Exception as e:
        print(f"Error with large density array: {e}")

    try:
        large_diffusivity = thermal_diffusivity_by_heat_conductivity(large_k, rho, c_p)
        print(f"Large diffusivity array: {large_diffusivity}")
    except Exception as e:
        print(f"Error with large diffusivity array: {e}")

    # Test 13: Complex numbers (if applicable)
    print("\nTest 13: Complex numbers")
    complex_T = sp.Symbol('T') + sp.I
    try:
        complex_density = density_by_thermal_expansion(complex_T, T_0, rho, tec)
        print(f"Complex density: {complex_density}")
    except Exception as e:
        print(f"Error with complex density: {e}")

    try:
        complex_diffusivity = thermal_diffusivity_by_heat_conductivity(k_symbolic, rho_symbolic + sp.I, c_p_symbolic)
        print(f"Complex diffusivity: {complex_diffusivity}")
    except Exception as e:
        print(f"Error with complex diffusivity: {e}")
