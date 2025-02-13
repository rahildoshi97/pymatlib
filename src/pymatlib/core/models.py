import numpy as np
import sympy as sp
from typing import Union, List, TypeVar
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes

# Type variables for more specific type hints
NumericType = TypeVar('NumericType', float, np.float32, np.float64)

# Constants
ABSOLUTE_ZERO = 0.0  # Kelvin
DEFAULT_TOLERANCE = 1e-10


def wrapper(value: Union[sp.Expr, NumericType, ArrayTypes, MaterialProperty]) \
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
    if isinstance(value, MaterialProperty):
        return value.expr
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
        temperature: Union[float, sp.Symbol],
        temperature_base: float,
        density_base: float,
        thermal_expansion_coefficient: Union[float, MaterialProperty]) \
        -> MaterialProperty:
    """
    Calculate density based on thermal expansion.
    rho(T) = rho_0 / (1 + tec * (T - T_0))^3
    Args:
        temperature: Current temperature(s).
        temperature_base: Base temperature for the reference density.
        density_base: Reference density at the base temperature.
        thermal_expansion_coefficient: Thermal expansion coefficient.
    Returns:
        MaterialProperty representing the calculated density.
    Raises:
        TypeError: If incompatible types are provided.
        ValueError: If physically impossible values are used.
    """
    if isinstance(temperature, float):
        if temperature < ABSOLUTE_ZERO:
            raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if isinstance(temperature, ArrayTypes):
        raise TypeError(f"Incompatible input type for temperature. Expected float or sp.Expr, got {type(temperature)}")
    if temperature_base < ABSOLUTE_ZERO:
        raise ValueError(f"Base temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if density_base <= 0:
        raise ValueError("Base density must be positive")
    if isinstance(thermal_expansion_coefficient, float) and (thermal_expansion_coefficient < -3e-5 or thermal_expansion_coefficient > 0.001):
        raise ValueError(f"Thermal expansion coefficient must be between -3e-5 and 0.001, got {thermal_expansion_coefficient}")

    try:
        tec_expr = thermal_expansion_coefficient.expr if isinstance(thermal_expansion_coefficient, MaterialProperty) else wrapper(thermal_expansion_coefficient)
        sub_assignments = thermal_expansion_coefficient.assignments if isinstance(thermal_expansion_coefficient, MaterialProperty) else []
        density = density_base * (1 + tec_expr * (temperature - temperature_base)) ** (-3)
        return MaterialProperty(density, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in density calculation")


def thermal_diffusivity_by_heat_conductivity(
        heat_conductivity: Union[float, MaterialProperty],
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty]) \
        -> MaterialProperty:
    """
    Calculate thermal diffusivity using heat conductivity, density, and heat capacity.
    alpha(T) = k(T) / (rho(T) * c_p(T))
    Args:
        heat_conductivity: Thermal conductivity of the material.
        density: Density of the material.
        heat_capacity: Specific heat capacity of the material.
    Returns:
        MaterialProperty representing the calculated thermal diffusivity.
    Raises:
        TypeError: If incompatible input data types are provided.
        ValueError: If physically impossible values are used.
    """
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

    # Handle the symbolic expression, using `.expr` only for MaterialProperty objects
    k_expr = heat_conductivity.expr if isinstance(heat_conductivity, MaterialProperty) else wrapper(heat_conductivity)
    rho_expr = density.expr if isinstance(density, MaterialProperty) else wrapper(density)
    cp_expr = heat_capacity.expr if isinstance(heat_capacity, MaterialProperty) else wrapper(heat_capacity)

    try:
        thermal_diffusivity = k_expr / (rho_expr * cp_expr)
        return MaterialProperty(thermal_diffusivity, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in thermal diffusivity calculation")


def energy_density(
        temperature: Union[float, sp.Symbol],
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    # Input validation to check for incompatible data types
    if isinstance(temperature, float) and temperature < ABSOLUTE_ZERO:
        raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if isinstance(density, float) and density <= 0:
        raise ValueError(f"Density must be positive, got {density}")
    if isinstance(heat_capacity, float) and heat_capacity <= 0:
        raise ValueError(f"Heat capacity must be positive, got {heat_capacity}")
    if isinstance(latent_heat, float) and latent_heat < 0:
        raise ValueError(f"Latent heat cannot be negative (should be zero or positive), got {latent_heat}")

    sub_assignments = [
        assignment for prop in [density, heat_capacity, latent_heat]
        if isinstance(prop, MaterialProperty)
        for assignment in prop.assignments
    ]

    density_expr = density.expr if isinstance(density, MaterialProperty) else wrapper(density)
    heat_capacity_expr = heat_capacity.expr if isinstance(heat_capacity, MaterialProperty) else wrapper(heat_capacity)
    latent_heat_expr = latent_heat.expr if isinstance(latent_heat, MaterialProperty) else wrapper(latent_heat)

    _energy_density = density_expr * (temperature * heat_capacity_expr + latent_heat_expr)
    return MaterialProperty(_energy_density, sub_assignments)
