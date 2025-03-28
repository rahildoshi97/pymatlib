import numpy as np
import sympy as sp
from typing import Union, List, TypeVar
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes

# Type variables for more specific type hints
NumericType = TypeVar('NumericType', float, np.float32, np.float64)

# Constants
ABSOLUTE_ZERO = 0.0  # Kelvin


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
    try:
        tec_expr, sub_assignments \
            = _prepare_material_expressions(thermal_expansion_coefficient)
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
    (k_expr, rho_expr, cp_expr), sub_assignments \
        = _prepare_material_expressions(heat_conductivity, density, heat_capacity)
    try:
        thermal_diffusivity = k_expr / (rho_expr * cp_expr)
        return MaterialProperty(thermal_diffusivity, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in thermal diffusivity calculation")


def specific_enthalpy_sensible(
        T: Union[float, sp.Symbol],
        temperature_array: np.ndarray,
        heat_capacity: Union[float, MaterialProperty]) \
        -> MaterialProperty:
    """
    Calculate specific enthalpy array using heat capacity only.
    h(T_i) = h(T_(i-1)) + c_p(T_i)*(T_i - T_(i-1))
    """
    print("Entering specific_enthalpy_sensible")
    # Validate temperature array is sorted
    if not np.all(temperature_array[:-1] <= temperature_array[1:]):
        raise ValueError("Temperature array must be sorted in ascending order for correct enthalpy calculation.")
    # print(temperature_array)
    print(temperature_array.shape)
    # Initialize enthalpy array
    enthalpy_array = np.zeros_like(temperature_array)
    # Calculate initial enthalpy at the first temperature
    T_0 = temperature_array[0]
    cp_0 = heat_capacity.evalf(T, T_0)
    print(T_0, cp_0)

    # Calculate enthalpy at first temperature point
    enthalpy_array[0] = cp_0 * T_0

    # Iterate through temperature points to calculate enthalpy incrementally
    for i in range(1, len(temperature_array)):
        T_i = temperature_array[i]
        T_prev = temperature_array[i-1]

        # Evaluate material properties at current temperature
        cp_i = heat_capacity.evalf(T, T_i)

        # Calculate enthalpy increment
        sensible_heat = cp_i * (T_i - T_prev)

        # Update enthalpy
        enthalpy_array[i] = enthalpy_array[i-1] + sensible_heat

    from pymatlib.core.interpolators import interpolate_property
    return interpolate_property(T, temperature_array, enthalpy_array)


def specific_enthalpy_with_latent_heat(
        T: Union[float, sp.Symbol],
        temperature_array: np.ndarray,
        heat_capacity: MaterialProperty,
        latent_heat: MaterialProperty) -> MaterialProperty:
    """
    Calculate specific enthalpy array using heat capacity and latent heat.
    h(T_i) = h(T_(i-1)) + c_p(T_i)*(T_i - T_(i-1)) + [L(T_i) - L(T_(i-1))]
    """
    print("Entering specific_enthalpy_with_latent_heat")
    # Validate temperature array is sorted
    if not np.all(temperature_array[:-1] <= temperature_array[1:]):
        raise ValueError("Temperature array must be sorted in ascending order for correct enthalpy calculation.")
    # print(temperature_array)
    print(temperature_array.shape)
    # Initialize enthalpy array
    enthalpy_array = np.zeros_like(temperature_array)
    # Calculate initial enthalpy at the first temperature
    T_0 = temperature_array[0]
    cp_0 = heat_capacity.evalf(T, T_0)
    L_0 = latent_heat.evalf(T, T_0)
    print(T_0, cp_0, L_0)

    # Calculate enthalpy at first temperature point
    enthalpy_array[0] = cp_0 * T_0 + L_0

    # Iterate through temperature points to calculate enthalpy incrementally
    for i in range(1, len(temperature_array)):
        T_i = temperature_array[i]
        T_prev = temperature_array[i-1]

        # Evaluate material properties at current temperature
        cp_i = heat_capacity.evalf(T, T_i)
        L_i = latent_heat.evalf(T, T_i)
        L_prev = latent_heat.evalf(T, T_prev)

        # Calculate enthalpy increment
        sensible_heat = cp_i * (T_i - T_prev)
        latent_heat_change = L_i - L_prev

        # Update enthalpy
        enthalpy_array[i] = enthalpy_array[i-1] + sensible_heat + latent_heat_change

    from pymatlib.core.interpolators import interpolate_property
    return interpolate_property(T, temperature_array, enthalpy_array)


def energy_density(
        density: Union[float, MaterialProperty],
        specific_enthalpy: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    (density_expr, specific_enthalpy_expr), sub_assignments \
        = _prepare_material_expressions(density, specific_enthalpy)

    energy_density_expr = density_expr * specific_enthalpy_expr
    return MaterialProperty(energy_density_expr, sub_assignments)
