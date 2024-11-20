import numpy as np
import sympy as sp
from typing import Union, List, Tuple, get_args, TypeVar
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes

# Type variables for more specific type hints
NumericType = TypeVar('NumericType', float, np.float32, np.float64)

# Constants
ABSOLUTE_ZERO = -273.15  # Celsius
DEFAULT_TOLERANCE = 1e-10


def validate_density_parameters(
        temperature: Union[float, ArrayTypes, sp.Expr],
        density_base: float,
        thermal_expansion_coefficient: Union[float, MaterialProperty]) -> None:
    """
    Validate physical quantities for density calculation.
    """
    if isinstance(temperature, float):
        if temperature < ABSOLUTE_ZERO:
            raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}°C)")
    elif isinstance(temperature, ArrayTypes):
        temp_array = np.asarray(temperature)
        if np.any(temp_array < ABSOLUTE_ZERO):
            raise ValueError("Temperature array contains values below absolute zero")

    if density_base <= 0:
        raise ValueError("Base density must be positive")

    if isinstance(thermal_expansion_coefficient, float) and thermal_expansion_coefficient <= -1:
        raise ValueError("Thermal expansion coefficient must be greater than -1")

def validate_thermal_diffusivity_parameters(
        heat_conductivity: Union[float, MaterialProperty],
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty]) -> None:
    """
    Validate physical quantities for thermal diffusivity calculation.
    """
    for param_name, param_value in [
        ("heat_conductivity", heat_conductivity),
        ("density", density),
        ("heat_capacity", heat_capacity)
    ]:
        if isinstance(param_value, np.ndarray):
            raise TypeError(f"Incompatible input data type for {param_name}: {type(param_value)}")
        if isinstance(param_value, float) and param_value <= 0:
            raise ValueError(f"{param_name} must be positive")


def wrapper(value: Union[sp.Expr, NumericType, ArrayTypes]) \
        -> Union[sp.Expr, List[sp.Expr]]:
    """
    Wraps various input types into sympy expressions.

    Args:
        value: Input value of various types.

    Returns:
        Wrapped sympy expression or list of expressions.

    Raises:
        ValueError: If the input type is unsupported.
    """
    if isinstance(value, sp.Expr):
        return sp.simplify(value)
    if isinstance(value, (float, np.float32, np.float64)):  # np.floating
        if abs(value) < DEFAULT_TOLERANCE:
            return sp.Float(0.0)
        return sp.Float(float(value))
    if isinstance(value, ArrayTypes):  # Handles lists, tuples, and arrays
        try:
            return [sp.Float(float(v)) for v in value]
        except (ValueError, TypeError) as _e:
            raise ValueError(f"Array contains non-numeric values: {_e}")
    raise ValueError(f"Unsupported type for value in Wrapper: {type(value)}")


def material_property_wrapper(value: Union[sp.Expr, NumericType, ArrayTypes]) \
        -> MaterialProperty:
    """
    Wraps a value into a MaterialProperty object.

    Args:
        value: Input value to be wrapped.

    Returns:
        MaterialProperty object containing the wrapped value.
    """
    wrapped_value = wrapper(value)
    return MaterialProperty(wrapped_value)


def density_by_thermal_expansion(
        temperature: Union[float, ArrayTypes, sp.Expr],
        temperature_base: float,
        density_base: float,
        thermal_expansion_coefficient: Union[float, MaterialProperty],
        validate: bool = True) \
        -> MaterialProperty:
    """
    Calculate density based on thermal expansion.
    rho(T) = rho_0 / (1 + tec * (T - T_0))^3

    Args:
        temperature: Current temperature(s).
        temperature_base: Base temperature for the reference density.
        density_base: Reference density at the base temperature.
        thermal_expansion_coefficient: Thermal expansion coefficient.
        validate: Whether to perform physical validation

    Returns:
        MaterialProperty representing the calculated density.

    Raises:
        TypeError: If incompatible types are provided.
        ValueError: If physically impossible values are used.
    """
    from pymatlib.core.interpolators import interpolate_property

    if validate:
        validate_density_parameters(temperature, density_base, thermal_expansion_coefficient)

    if isinstance(temperature, ArrayTypes):
            temperature = np.asarray(temperature)

    if isinstance(temperature, ArrayTypes) and isinstance(thermal_expansion_coefficient, MaterialProperty):
        raise TypeError(
            f"Incompatible combination of temperature (type:{type(temperature)}) "
            f"and thermal expansion coefficient ({type(thermal_expansion_coefficient)})")

    try:
        if isinstance(thermal_expansion_coefficient, float):
            density = density_base * (1 + thermal_expansion_coefficient * (temperature - temperature_base)) ** (-3)
            if isinstance(density, np.ndarray):
                T_dbte = sp.Symbol('T_dbte')
                density = interpolate_property(T_dbte, temperature, density)
            else:
                if isinstance(density, (float, sp.Expr)):
                    density = material_property_wrapper(density)
            return density

        else:  # isinstance(thermal_expansion_coefficient, MaterialProperty)
            tec_expr = thermal_expansion_coefficient.expr
            density = density_base * (1 + tec_expr * (temperature - temperature_base)) ** (-3)
            return material_property_wrapper(density)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in density calculation")


def thermal_diffusivity_by_heat_conductivity(
        heat_conductivity: Union[float, MaterialProperty],
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        validate: bool = True) \
        -> MaterialProperty:
    """
    Calculate thermal diffusivity using heat conductivity, density, and heat capacity.
    alpha(T) = k(T) / (rho(T) * c_p(T))

    Args:
        heat_conductivity: Thermal conductivity of the material.
        density: Density of the material.
        heat_capacity: Specific heat capacity of the material.
        validate: Whether to perform physical validation

    Returns:
        MaterialProperty representing the calculated thermal diffusivity.

    Raises:
        TypeError: If incompatible input data types are provided.
        ValueError: If physically impossible values are used.
    """
    if validate:
        validate_thermal_diffusivity_parameters(heat_conductivity, density, heat_capacity)

    # Input validation to check for incompatible data types
    for param_name, param_value in [
        ("heat_conductivity", heat_conductivity),
        ("density", density),
        ("heat_capacity", heat_capacity)
    ]:
        if isinstance(param_value, np.ndarray):
            raise TypeError(f"Incompatible input data type for {param_name}: {type(param_value)}")
        if isinstance(param_value, float) and param_value <= 0:
            raise ValueError(f"{param_name} must be positive")

    sub_assignments = [
        assignment for prop in [heat_conductivity, density, heat_capacity]
        if isinstance(prop, MaterialProperty)
        for assignment in prop.assignments
    ]
    '''sub_assignments = []
    for prop in [heat_conductivity, density, heat_capacity]:
        if isinstance(prop, MaterialProperty):
            sub_assignments.extend(prop.assignments)'''

    # Handle the symbolic expression, using `.expr` only for MaterialProperty objects
    k_expr = heat_conductivity.expr if isinstance(heat_conductivity, MaterialProperty) else wrapper(heat_conductivity)
    rho_expr = density.expr if isinstance(density, MaterialProperty) else wrapper(density)
    cp_expr = heat_capacity.expr if isinstance(heat_capacity, MaterialProperty) else wrapper(heat_capacity)

    try:
        _result = k_expr / (rho_expr * cp_expr)
        return MaterialProperty(_result, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in thermal diffusivity calculation")


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
    print("-----", type(wrapper(tec_a)))
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
    large_T = np.linspace(1000, 2000, 1000)
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

    # Additional test for division by zero
    print("\nTest 14: Division by zero")
    try:
        zero_density = material_property_wrapper(0)
        result = thermal_diffusivity_by_heat_conductivity(1.0, zero_density, 1.0)
        print(f"Result with zero density: {result}")
    except ValueError as e:
        print(f"Handled error for zero density: {e}")

    # Test for negative values
    print("\nTest 15: Negative values")
    try:
        result = density_by_thermal_expansion(1000, 20, -1000, 1e-6)
        print(f"Result with negative density: {result}")
    except ValueError as e:
        print(f"Handled error for negative density: {e}")

    try:
        result = thermal_diffusivity_by_heat_conductivity(-1.0, 1000, 500)
        print(f"Result with negative heat conductivity: {result}")
    except ValueError as e:
        print(f"Handled error for negative heat conductivity: {e}")