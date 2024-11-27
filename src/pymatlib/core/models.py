import numpy as np
import sympy as sp
from typing import Union, List, TypeVar
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes

# Type variables for more specific type hints
NumericType = TypeVar('NumericType', float, np.float32, np.float64)

# Constants
ABSOLUTE_ZERO = 0.0  # Kelvin
DEFAULT_TOLERANCE = 1e-10


def validate_density_parameters(
        temperature: Union[float, ArrayTypes, sp.Expr],
        temperature_base: float,
        density_base: float,
        thermal_expansion_coefficient: Union[float, MaterialProperty]) -> None:
    """
    Validate physical quantities for density calculation.
    """
    if temperature_base < ABSOLUTE_ZERO:
        raise ValueError(f"Base temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if isinstance(temperature, float):
        if temperature < ABSOLUTE_ZERO:
            raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    elif isinstance(temperature, ArrayTypes):
        temp_array = np.asarray(temperature)
        if np.any(temp_array < ABSOLUTE_ZERO):
            raise ValueError("Temperature array contains values below absolute zero")
    if density_base <= 0:
        raise ValueError("Base density must be positive")
    if isinstance(thermal_expansion_coefficient, float) and (thermal_expansion_coefficient < -3e-5 or thermal_expansion_coefficient > 0.001):
        raise ValueError("Thermal expansion coefficient must be between -3e-5 and 0.001")

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
        validate_density_parameters(temperature, temperature_base, density_base, thermal_expansion_coefficient)

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
