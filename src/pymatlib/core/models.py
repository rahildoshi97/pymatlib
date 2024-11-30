import numpy as np
import sympy as sp
from typing import Union, List, TypeVar, Tuple, Optional
from pymatlib.core.typedefs import MaterialProperty, ArrayTypes

# Type variables for more specific type hints
NumericType = TypeVar('NumericType', float, np.float32, np.float64)

# Constants
ABSOLUTE_ZERO = 0.0  # Kelvin
DEFAULT_TOLERANCE = 1e-10


'''
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
'''

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

    if isinstance(temperature, float):
        if temperature < ABSOLUTE_ZERO:
            raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if isinstance(temperature, ArrayTypes):
        temperature = np.asarray(temperature)
        if np.any(temperature < ABSOLUTE_ZERO):
            raise ValueError("Temperature array contains values below absolute zero")
    if temperature_base < ABSOLUTE_ZERO:
        raise ValueError(f"Base temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if density_base <= 0:
        raise ValueError("Base density must be positive")
    if isinstance(thermal_expansion_coefficient, float) and (thermal_expansion_coefficient < -3e-5 or thermal_expansion_coefficient > 0.001):
        raise ValueError("Thermal expansion coefficient must be between -3e-5 and 0.001")
    if isinstance(temperature, ArrayTypes) and isinstance(thermal_expansion_coefficient, MaterialProperty):
        raise TypeError(
            f"Incompatible combination of temperature (type:{type(temperature)}) "
            f"and thermal expansion coefficient ({type(thermal_expansion_coefficient)})")

    try:
        if isinstance(thermal_expansion_coefficient, float):
            density = density_base * (1 + thermal_expansion_coefficient * (temperature - temperature_base)) ** (-3)
            print(f"density: {density}")
            if isinstance(density, np.ndarray):  # isinstance(temperature, ArrayTypes)
                T_dbte = sp.Symbol('T_dbte')
                density = interpolate_property(T_dbte, temperature, density)
            else:
                if isinstance(density, (float, sp.Expr)):
                    density = material_property_wrapper(density)
            return density

        else:  # isinstance(thermal_expansion_coefficient, MaterialProperty)
            sub_assignments = thermal_expansion_coefficient.assignments if isinstance(thermal_expansion_coefficient, MaterialProperty) else []
            tec_expr = thermal_expansion_coefficient.expr
            density = density_base * (1 + tec_expr * (temperature - temperature_base)) ** (-3)
            return MaterialProperty(density, sub_assignments)
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
        thermal_diffusivity = k_expr / (rho_expr * cp_expr)
        return MaterialProperty(thermal_diffusivity, sub_assignments)
    except ZeroDivisionError:
        raise ValueError("Division by zero encountered in thermal diffusivity calculation")


def compute_energy_density_MaterialProperty1(
        temperature: Union[float, sp.Expr],
        _density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) \
        -> MaterialProperty:

    print(f"_density: {_density}, type: {type(_density)}")
    print(f"heat_capacity: {heat_capacity}, type: {type(heat_capacity)}")
    print(f"latent_heat: {latent_heat}, type: {type(latent_heat)}")

    # Input validation to check for incompatible data types
    if isinstance(temperature, float) and temperature < ABSOLUTE_ZERO:
        raise ValueError(f"Temperature cannot be below absolute zero ({ABSOLUTE_ZERO}K)")
    if isinstance(_density, float) and _density <= 0:
        raise ValueError(f"Density must be positive, got {_density}")
    if isinstance(heat_capacity, float) and heat_capacity <= 0:
        raise ValueError(f"Heat capacity must be positive, got {heat_capacity}")
    if isinstance(latent_heat, float) and latent_heat < 0:
        raise ValueError(f"Latent heat cannot be negative (should be zero or positive), got {latent_heat}")

    sub_assignments = [
        assignment for prop in [_density, heat_capacity, latent_heat]
        if isinstance(prop, MaterialProperty)
        for assignment in prop.assignments
    ]
    # debugging
    if len(sub_assignments) > 0:
        for i in range(len(sub_assignments)):
            print(f"sub_assignments[{i}]: {sub_assignments[i]}\n")

    if isinstance(_density, MaterialProperty):
        print(f"_density: {_density}")
        if len(_density.assignments) == 0:
            if isinstance(_density.expr, sp.Float):
                _density = float(_density.expr)
            elif isinstance(_density.expr, (sp.Piecewise, sp.Add, sp.Mul)):
                free_symbols = _density.expr.free_symbols
                if free_symbols:
                    _density = _density.expr.subs({symbol: temperature for symbol in free_symbols}).evalf()
                else:
                    _density = _density.expr.evalf()
            print(f"density: {_density}")

        elif len(_density.assignments) >= 1:
            expr = _density.expr
            print(f"expr: {expr}")
            subs_dict = {}
            for assignment in _density.assignments:
                ass_lhs = assignment.lhs
                print(f"ass_lhs: {ass_lhs}")
                ass_rhs = assignment.rhs
                print(f"ass_rhs: {ass_rhs}")

                if isinstance(ass_rhs, (sp.Piecewise, sp.Add)):
                    expr_symbols = expr.free_symbols
                    print(f"expr_symbols: {expr_symbols}")
                    ass_lhs_symbols = ass_lhs.free_symbols
                    print(f"ass_lhs_symbols: {ass_lhs_symbols}")
                    ass_rhs_symbols = ass_rhs.free_symbols
                    print(f"ass_rhs_symbols: {ass_rhs_symbols}")
                    all_symbols = ass_lhs_symbols.union(ass_rhs_symbols)  # expr_symbols.union(ass_lhs_symbols).union(ass_rhs_symbols)
                    print(f"all_symbols: {all_symbols}")

                    subs_dict_rhs = {symbol: temperature for symbol in ass_rhs_symbols}
                    print(f"subs_dict_rhs: {subs_dict_rhs}")

                    # ass_rhs_val = ass_rhs.subs({next(iter(ass_rhs_symbols)): temperature}).evalf()
                    ass_rhs_val = ass_rhs.subs(subs_dict_rhs).evalf()
                    print(f"ass_rhs_val: {ass_rhs_val}")
                    ass_rhs_val = int(ass_rhs_val) if assignment.lhs_type == "int" else ass_rhs_val
                    print(f"new ass_rhs_val: {ass_rhs_val}")

                    subs_dict_lhs = {symbol: ass_rhs_val for symbol in ass_lhs_symbols}
                    print(f"subs_dict_lhs: {subs_dict_lhs}")

                    # subs_dict = {**subs_dict_rhs, **subs_dict_lhs}
                    subs_dict.update(subs_dict_rhs)
                    subs_dict.update(subs_dict_lhs)
                    print(f"updated subs_dict: {subs_dict}")

                elif isinstance(ass_rhs, Tuple):
                    tuple_values = [
                        rhs_part.subs({symbol: temperature for symbol in rhs_part.free_symbols}).evalf()
                        for rhs_part in ass_rhs
                    ]
                    print(f"tuple_values: {tuple_values}")

                    # Use the first value in the tuple or apply custom logic for your use case
                    subs_dict[ass_lhs] = ass_rhs  # or handle tuple_values differently
                    print(f"Updated subs_dict with tuple: {subs_dict}")
                else:
                    raise TypeError(f"Unknown assignment type: {type(ass_rhs)}")

            print("calculating _density")
            _density = expr.subs(subs_dict).evalf()
            print(f"Final evaluated _density: {_density}")
        else:
            # TODO: Handle unsupported cases or provide a fallback
            raise NotImplementedError(f"Unsupported number of assignments: {len(_density.assignments)}")
    else:
        _density = wrapper(_density)
    print(f"_density: {_density}")

    heat_capacity_expr = heat_capacity.expr if isinstance(heat_capacity, MaterialProperty) else wrapper(heat_capacity)
    latent_heat_expr = latent_heat.expr if isinstance(latent_heat, MaterialProperty) else wrapper(latent_heat)

    energy_density = _density * (temperature * heat_capacity_expr + latent_heat_expr)
    print(f"energy_density: {energy_density}")

    return MaterialProperty(energy_density, sub_assignments)
    # Just FYI: ps.Assignment(T.center, (s.h / s.density_mat - s.latent_heat_mat) / s.heat_capacity_mat)


def compute_energy_density_MaterialProperty(
        temperature: Union[float, sp.Expr],
        _density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty]) -> MaterialProperty:
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

    return energy_density_property


def compute_energy_density_array(
        density: Union[float, MaterialProperty],
        heat_capacity: Union[float, MaterialProperty],
        latent_heat: Union[float, MaterialProperty],
        file_path: Optional[str] = None,
        temperature_array: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
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
    # Validate input
    if file_path and temperature_array is not None:
        raise ValueError("Provide only one of `file_path` or `temperature_array`, not both.")
    if not file_path and temperature_array is None:
        raise ValueError("You must provide either `file_path` or `temperature_array`.")

    _temperatures = []
    _energy_densities = []

    # Handle temperatures from file
    if file_path:
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
    if temperature_array is not None:
        _temperatures = temperature_array.tolist()

    # Compute energy densities
    for temperature in _temperatures:
        print(f"temperature: {temperature}")
        energy_density = compute_energy_density_MaterialProperty(
            temperature=temperature,
            _density=density,
            heat_capacity=heat_capacity,
            latent_heat=latent_heat,
        )
        _energy_densities.append(float(energy_density.expr))

    # Convert lists to NumPy arrays
    return np.array(_temperatures), np.array(_energy_densities)


'''
# Example usage
file_path = "/local/ca00xebo/repos/pymatlib/src/pymatlib/data/alloys/SS316L/density_temperature.txt"
density = 8000.0  # Example density (in kg/m³)
heat_capacity = 500.0  # Example heat capacity (in J/(kg·K))
latent_heat = 250000.0  # Example latent heat (in J/kg)

temperatures, energy_densities = compute_energy_density_from_file(
    file_path, density, heat_capacity, latent_heat
)

# Print the results
print("Temperatures (K):")
print(temperatures)
print("\nEnergy Densities (J/m³):")
print(energy_densities)
'''
