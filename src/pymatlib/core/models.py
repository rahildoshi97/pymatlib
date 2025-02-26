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
    if isinstance(value, (float, np.int32, np.int64, np.float32, np.float64)):  # np.floating
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


def _validate_positive(*values, names=None):
    """Validate that float values are positive."""
    if names is None:
        names = [f"Value {i+1}" for i in range(len(values))]

    for value, name in zip(values, names):
        if isinstance(value, float) and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(*values, names=None):
    """Validate that float values are non-negative."""
    if names is None:
        names = [f"Value {i+1}" for i in range(len(values))]

    for value, name in zip(values, names):
        if isinstance(value, float) and value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")


def _prepare_material_expressions(*properties):
    """Prepare expressions and collect assignments from material properties."""
    sub_assignments = []
    expressions = []

    for prop in properties:
        if isinstance(prop, MaterialProperty):
            sub_assignments.extend(prop.assignments)
            expressions.append(prop.expr)
        else:
            expressions.append(wrapper(prop))

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
    _validate_positive(heat_conductivity, density, heat_capacity, names=["Heat conductivity", "Density", "Heat capacity"])

    (k_expr, rho_expr, cp_expr), sub_assignments \
        = _prepare_material_expressions(heat_conductivity, density, heat_capacity)

    try:
        thermal_diffusivity = k_expr / (rho_expr * cp_expr)
        return MaterialProperty(thermal_diffusivity, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in thermal diffusivity calculation")


def energy_density_standard(
        temperature: Union[float, sp.Symbol],
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    # Input validation to check for incompatible data types
    if isinstance(temperature, float) and temperature < ABSOLUTE_ZERO:
        raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    '''if isinstance(density, float) and density <= 0:
        raise ValueError(f"Density must be positive, got {density}")
    if isinstance(heat_capacity, float) and heat_capacity <= 0:
        raise ValueError(f"Heat capacity must be positive, got {heat_capacity}")'''
    _validate_positive(density, heat_capacity, names=['Density', 'Heat capacity'])
    '''if isinstance(latent_heat, float) and latent_heat < 0:
        raise ValueError(f"Latent heat cannot be negative (should be zero or positive), got {latent_heat}")'''
    _validate_non_negative(latent_heat, names=['Latent heat'])

    '''sub_assignments = [
        assignment for prop in [density, heat_capacity, latent_heat]
        if isinstance(prop, MaterialProperty)
        for assignment in prop.assignments
    ]'''
    (density_expr, heat_capacity_expr, latent_heat_expr), sub_assignments \
        = _prepare_material_expressions(density, heat_capacity, latent_heat)

    density_expr = density.expr if isinstance(density, MaterialProperty) else wrapper(density)
    heat_capacity_expr = heat_capacity.expr if isinstance(heat_capacity, MaterialProperty) else wrapper(heat_capacity)
    latent_heat_expr = latent_heat.expr if isinstance(latent_heat, MaterialProperty) else wrapper(latent_heat)

    energy_density_expr = density_expr * (temperature * heat_capacity_expr + latent_heat_expr)
    return MaterialProperty(energy_density_expr, sub_assignments)
    # Just FYI: ps.Assignment(T.center, (s.h / s.density_mat - s.latent_heat_mat) / s.heat_capacity_mat)


def energy_density_enthalpy_based(
        density: Union[float, MaterialProperty],
        specific_enthalpy: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    _validate_positive(density, specific_enthalpy, names=["Density", "Specific enthalpy"])
    _validate_non_negative(latent_heat, names=["Latent heat"])

    (density_expr, specific_enthalpy_expr, latent_heat_expr), sub_assignments \
        = _prepare_material_expressions(density, specific_enthalpy, latent_heat)

    energy_density_expr = density_expr * (specific_enthalpy_expr + latent_heat_expr)
    return MaterialProperty(energy_density_expr, sub_assignments)


def energy_density_total_enthalpy(
        density: Union[float, MaterialProperty],
        specific_enthalpy: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    _validate_positive(density, specific_enthalpy, names=["Density", "Specific enthalpy"])

    (density_expr, specific_enthalpy_expr), sub_assignments \
        = _prepare_material_expressions(density, specific_enthalpy)

    #print(density_expr)
    #print(specific_enthalpy_expr)
    energy_density_expr = density_expr * specific_enthalpy_expr
    return MaterialProperty(energy_density_expr, sub_assignments)


'''
def compute_energy_density_MaterialProperty(
        temperature: Union[float, sp.Expr],
        _density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) -> Tuple[MaterialProperty, float, float, float]:
    """
    Compute energy density for MaterialProperty objects dynamically and return as a MaterialProperty.
    """
    properties = {
        "density": _density,
        "heat_capacity": heat_capacity,
        "latent_heat": latent_heat
    }

    evaluated_properties = {}
    assignments = []

    for name, prop in properties.items():
        print(f"Processing {name}: {prop}, type: {type(prop)}")

        if isinstance(prop, MaterialProperty):
            print(f"{name} is a MaterialProperty")
            if len(prop.assignments) == 0:
                if isinstance(prop.expr, sp.Float):
                    evaluated_properties[name] = float(prop.expr)
                elif isinstance(prop.expr, (sp.Piecewise, sp.Add, sp.Mul)):
                    free_symbols = prop.expr.free_symbols
                    if free_symbols:
                        substitutions = {symbol: temperature for symbol in free_symbols}
                        evaluated_properties[name] = prop.expr.subs(substitutions).evalf() if isinstance(temperature, float) else prop.expr.subs(substitutions)
                    else:
                        evaluated_properties[name] = prop.expr.evalf() if isinstance(temperature, float) else prop.expr
                print(f"Evaluated {name}: {evaluated_properties[name]}")
            elif len(prop.assignments) >= 1:
                expr = prop.expr
                subs_dict = {}
                for assignment in prop.assignments:
                    ass_lhs = assignment.lhs
                    ass_rhs = assignment.rhs
                    print(f"{name} assignment - LHS: {ass_lhs}, RHS: {ass_rhs}")

                    if isinstance(ass_rhs, (sp.Piecewise, sp.Add)):
                        subs_dict_rhs = {symbol: temperature for symbol in ass_rhs.free_symbols}
                        ass_rhs_val = ass_rhs.subs(subs_dict_rhs).evalf() if isinstance(temperature, float) else ass_rhs.subs(subs_dict_rhs)
                        print(f"ass_rhs_val: {ass_rhs_val}")
                        ass_rhs_val = int(ass_rhs_val) if assignment.lhs_type == "int" and isinstance(temperature, float) else ass_rhs_val
                        subs_dict_lhs = {symbol: ass_rhs_val for symbol in ass_lhs.free_symbols}
                        subs_dict.update(subs_dict_rhs)
                        subs_dict.update(subs_dict_lhs)
                        print(f"updated subs_dict: {subs_dict}")
                    elif isinstance(ass_rhs, Tuple):
                        tuple_values = [
                            rhs_part.subs({symbol: temperature for symbol in rhs_part.free_symbols}).evalf()
                            if isinstance(temperature, float)
                            else rhs_part.subs({symbol: temperature for symbol in rhs_part.free_symbols})
                            for rhs_part in ass_rhs
                        ]
                        subs_dict[ass_lhs] = ass_rhs
                        print(f"Updated subs_dict with tuple: {subs_dict}")
                    else:
                        raise TypeError(f"Unknown assignment type: {type(ass_rhs)}")
                print(f"Substitution dictionary for {name}: {subs_dict}")
                evaluated_properties[name] = expr.subs(subs_dict).evalf() if isinstance(temperature, float) else expr.subs(subs_dict)
                print(f"Final evaluated {name}: {evaluated_properties[name]}")
                assignments.extend(prop.assignments)
            else:
                raise NotImplementedError(f"Unsupported number of assignments for {name}: {len(prop.assignments)}")
        else:
            # Wrap the non-MaterialProperty inputs directly
            evaluated_properties[name] = wrapper(prop)

    # Compute the final energy density expression
    energy_density_expr = evaluated_properties["density"] * (temperature * evaluated_properties["heat_capacity"] + evaluated_properties["latent_heat"])

    # Combine all assignments into a MaterialProperty
    energy_density_property = MaterialProperty(
        expr=energy_density_expr,
        assignments=assignments
    )

    print(f"Returning MaterialProperty with expr: {energy_density_property.expr} "
          f"and assignments: {energy_density_property.assignments}")

    return (
        energy_density_property,
        evaluated_properties["density"],
        evaluated_properties["heat_capacity"],
        evaluated_properties["latent_heat"],
    )


def compute_energy_density_array(
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty],
        file_path: Optional[str] = None,
        temperature_array: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute energy density for temperatures from a file or a NumPy array.
    Args:
        density (float or MaterialProperty): Density of the material.
        heat_capacity (float or MaterialProperty): Specific heat capacity.
        latent_heat (float or MaterialProperty): Latent heat of the material.
        file_path (Optional[str]): Path to the temperature data file (optional).
        temperature_array (Optional[np.ndarray]): Array of temperature values (optional).
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of temperatures.
            - Array of corresponding energy densities.
    Raises:
        ValueError: If both `file_path` and `temperature_array` are provided, or neither is provided.
    """
    print(f"Inside function\nfile_path: {file_path}, temperature_array: {temperature_array}")
    # Validate input
    if file_path is None and temperature_array is not None:
        if isinstance(temperature_array, np.ndarray):
            if temperature_array.size == 0:
                raise ValueError(f"Temperature array must not be empty")
        else:
            raise TypeError(f"Temperature array must be a numpy array, got {type(temperature_array)}")

    if file_path is not None and temperature_array is not None:
        raise ValueError("Provide only one of `file_path` or `temperature_array`, not both.")
    if file_path is None and temperature_array is None:
        raise ValueError("You must provide either `file_path` or `temperature_array`.")

    _temperatures = []
    _densities = []
    _heat_capacities = []
    _latent_heats = []
    _energy_densities = []

    # Handle temperatures from file
    if isinstance(file_path, str) and file_path:
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Skip non-data lines
                    if not line.strip() or not line[0].isdigit():
                        continue

                    # Extract the temperature from the first column
                    temperature = float(line.split()[0])
                    _temperatures.append(temperature)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        except Exception as e:
            raise ValueError(f"An error occurred while processing the file: {e}")

    # Handle temperatures from NumPy array
    if temperature_array is None:
        raise ValueError("temperature_array is None")
    if temperature_array is not None:
        _temperatures = temperature_array.tolist()
        print(f"_temperatures: {_temperatures}")

    # Compute energy densities
    for temperature in _temperatures:
        print(f"temperature: {temperature}")
        energy_density, density_val, specific_heat_val, latent_heat_val = compute_energy_density_MaterialProperty(
            temperature=temperature,
            _density=density,
            heat_capacity=heat_capacity,
            latent_heat=latent_heat,
        )
        _densities.append(density_val)
        _heat_capacities.append(specific_heat_val)
        _latent_heats.append(latent_heat_val)
        _energy_densities.append(float(energy_density.expr))

    # Convert lists to NumPy arrays
    return (np.array(_temperatures),
            np.array(_densities),
            np.array(_heat_capacities),
            np.array(_latent_heats),
            np.array(_energy_densities))
'''
