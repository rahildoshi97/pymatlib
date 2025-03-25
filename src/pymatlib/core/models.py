import numpy as np
import sympy as sp
from typing import Union, List, TypeVar
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes

# Type variables for more specific type hints
NumericType = TypeVar('NumericType', float, np.float32, np.float64)

# Constants
ABSOLUTE_ZERO = 0.0  # Kelvin
# DEFAULT_TOLERANCE = 1e-10


def sympy_wrapper(value: Union[sp.Expr, NumericType, ArrayTypes, MaterialProperty]) \
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
    if isinstance(value, (float, np.int32, np.int64, np.float32, np.float64)):  # np.floating
        # if abs(value) < DEFAULT_TOLERANCE:
            # return sp.Float(0.0)
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
    wrapped_value = sympy_wrapper(value)
    return MaterialProperty(wrapped_value)


'''def _validate_positive(*values, names=None):
    """Validate that float values are positive."""
    if names is None:
        names = [f"Value {i+1}" for i in range(len(values))]

    for value, name in zip(values, names):
        if isinstance(value, float) and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")'''


'''def _validate_non_negative(*values, names=None):
    """Validate that float values are non-negative."""
    if names is None:
        names = [f"Value {i+1}" for i in range(len(values))]

    for value, name in zip(values, names):
        if isinstance(value, float) and value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")'''


def _prepare_material_expressions(*properties):
    """Prepare expressions and collect assignments from material properties."""
    sub_assignments = []
    expressions = []
    for prop in properties:
        if isinstance(prop, MaterialProperty):
            sub_assignments.extend(prop.assignments)
            expressions.append(prop.expr)
        else:
            expressions.append(sympy_wrapper(prop))
    # If there's only one expression, return it directly instead of in a list
    if len(expressions) == 1:
        return expressions[0], sub_assignments
    else:
        return expressions, sub_assignments


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
    # if temperature_base < ABSOLUTE_ZERO:
        # raise ValueError(f"Base temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    # if density_base <= 0:
        # raise ValueError("Base density must be positive")
    # if isinstance(thermal_expansion_coefficient, float) and (thermal_expansion_coefficient < -3e-5 or thermal_expansion_coefficient > 0.001):
        # raise ValueError(f"Thermal expansion coefficient must be between -3e-5 and 0.001, got {thermal_expansion_coefficient}")

    try:
        tec_expr, sub_assignments \
            = _prepare_material_expressions(thermal_expansion_coefficient)
        # tec_expr = thermal_expansion_coefficient.expr if isinstance(thermal_expansion_coefficient, MaterialProperty) else sympy_wrapper(thermal_expansion_coefficient)
        # sub_assignments = thermal_expansion_coefficient.assignments if isinstance(thermal_expansion_coefficient, MaterialProperty) else []
        density_expr = density_base * (1 + tec_expr * (temperature - temperature_base)) ** (-3)
        return MaterialProperty(density_expr, sub_assignments)
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
    # _validate_positive(heat_conductivity, density, heat_capacity, names=["Heat conductivity", "Density", "Heat capacity"])

    (k_expr, rho_expr, cp_expr), sub_assignments \
        = _prepare_material_expressions(heat_conductivity, density, heat_capacity)

    try:
        thermal_diffusivity = k_expr / (rho_expr * cp_expr)
        return MaterialProperty(thermal_diffusivity, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in thermal diffusivity calculation")


# TODO: Add models for specific_enthalpy (h)
def specific_enthalpy_sensible(
        temperature: Union[float, sp.Symbol],
        heat_capacity: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    heat_capacity_expr, sub_assignments \
        = _prepare_material_expressions(heat_capacity)

    specific_enthalpy_expr = temperature * heat_capacity_expr
    return MaterialProperty(specific_enthalpy_expr, sub_assignments)


def specific_enthalpy_with_latent_heat(
        temperature: Union[float, sp.Symbol],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    (heat_capacity_expr, latent_heat_expr), sub_assignments \
        = _prepare_material_expressions(heat_capacity, latent_heat)

    specific_enthalpy_expr = temperature * heat_capacity_expr + latent_heat_expr
    return MaterialProperty(specific_enthalpy_expr, sub_assignments)


def energy_density(
        density: Union[float, MaterialProperty],
        specific_enthalpy: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    # _validate_positive(density, specific_enthalpy, names=["Density", "Specific enthalpy"])

    (density_expr, specific_enthalpy_expr), sub_assignments \
        = _prepare_material_expressions(density, specific_enthalpy)

    #print(density_expr)
    #print(specific_enthalpy_expr)
    energy_density_expr = density_expr * specific_enthalpy_expr
    return MaterialProperty(energy_density_expr, sub_assignments)
