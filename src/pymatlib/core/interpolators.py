import numpy as np
import sympy as sp
from typing import Union, List, Tuple
from pymatlib.core.models import wrapper, material_property_wrapper
from pymatlib.core.typedefs import Assignment, ArrayTypes, MaterialProperty
from pymatlib.core.data_handler import check_equidistant, check_strictly_increasing
from pystencils.types import PsCustomType
from pystencilssfg import SfgComposer
from pystencilssfg.composer.custom import CustomGenerator


COUNT = 0


class InterpolationArrayContainer(CustomGenerator):
    def __init__(self, name: str, temperature_array: np.ndarray, energy_density_array: np.ndarray):
        super().__init__()
        self.name = name
        self.T_array = temperature_array
        self.E_array = energy_density_array

        # Prepare arrays and determine best method
        self.data = prepare_interpolation_arrays(self.T_array, self.E_array)
        self.method = self.data["method"]

        # Store arrays for binary search (always available)
        self.T_bs = self.data["T_bs"]
        self.E_bs = self.data["E_bs"]

        # Store arrays for double lookup if available
        if self.method == "double_lookup":
            self.T_eq = self.data["T_eq"]
            self.E_neq = self.data["E_neq"]
            self.E_eq = self.data["E_eq"]
            self.inv_delta_E_eq = self.data["inv_delta_E_eq"]
            self.idx_map = self.data["idx_map"]
            self.has_double_lookup = True
        else:
            self.has_double_lookup = False

    @classmethod
    def from_material(cls, name: str, material):
        return cls(name, material.energy_density_temperature_array, material.energy_density_array)

    def generate(self, sfg: SfgComposer):
        sfg.include("<array>")
        sfg.include("interpolate_binary_search_cpp.h")

        # Binary search arrays (always included)
        T_bs_arr_values = ", ".join(str(v) for v in self.T_bs)
        E_bs_arr_values = ", ".join(str(v) for v in self.E_bs)

        E_target = sfg.var("E_target", "double")

        public_members = [
            # Binary search arrays
            f"static constexpr std::array< double, {self.T_bs.shape[0]} > T_bs {{ {T_bs_arr_values} }}; \n"
            f"static constexpr std::array< double, {self.E_bs.shape[0]} > E_bs {{ {E_bs_arr_values} }}; \n",

            # Binary search method
            sfg.method("interpolateBS", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                sfg.expr("return interpolate_binary_search_cpp({}, *this);", E_target)
            )
        ]

        # Add double lookup if available
        if self.has_double_lookup:
            sfg.include("interpolate_double_lookup_cpp.h")

            T_eq_arr_values = ", ".join(str(v) for v in self.T_eq)
            E_neq_arr_values = ", ".join(str(v) for v in self.E_neq)
            E_eq_arr_values = ", ".join(str(v) for v in self.E_eq)
            idx_mapping_arr_values = ", ".join(str(v) for v in self.idx_map)

            public_members.extend([
                # Double lookup arrays
                f"static constexpr std::array< double, {self.T_eq.shape[0]} > T_eq {{ {T_eq_arr_values} }}; \n"
                f"static constexpr std::array< double, {self.E_neq.shape[0]} > E_neq {{ {E_neq_arr_values} }}; \n"
                f"static constexpr std::array< double, {self.E_eq.shape[0]} > E_eq {{ {E_eq_arr_values} }}; \n"
                f"static constexpr double inv_delta_E_eq = {self.inv_delta_E_eq}; \n"
                f"static constexpr std::array< int, {self.idx_map.shape[0]} > idx_map {{ {idx_mapping_arr_values} }}; \n",

                # Double lookup method
                sfg.method("interpolateDL", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_double_lookup_cpp({}, *this);", E_target)
                )
            ])

        # Add interpolate method that uses recommended approach
        if self.has_double_lookup:
            public_members.append(
                sfg.method("interpolate", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_double_lookup_cpp({}, *this);", E_target)
                )
            )
        else:
            public_members.append(
                sfg.method("interpolate", returns=PsCustomType("[[nodiscard]] double"), inline=True, const=True)(
                    sfg.expr("return interpolate_binary_search_cpp({}, *this);", E_target)
                )
            )

        # Generate the class
        sfg.klass(self.name)(
            sfg.public(*public_members)
        )


def interpolate_equidistant(
        T: Union[float, sp.Symbol],
        T_base: float,
        T_incr: float,
        v_array: ArrayTypes) -> MaterialProperty:
    """
    Perform equidistant interpolation for symbolic or numeric temperature values.

    :param T: Temperature, either symbolic (sp.Symbol) or numeric (float).
    :param T_base: Base temperature for the interpolation.
    :param T_incr: Increment in temperature for each step in the interpolation array.
    :param v_array: Array of values corresponding to the temperatures.
    :return: Interpolated value as a MaterialProperty object.
    """
    if T_incr < 0:
        T_incr *= -1
        T_base -= T_incr * (len(v_array) - 1)
        v_array = np.flip(v_array)

    if isinstance(T, sp.Symbol):
        global COUNT
        label = '_' + str(COUNT).zfill(3)
        COUNT += 1

        sym_data = sp.IndexedBase(label + "_var", shape=len(v_array))
        sym_idx = sp.symbols(label + '_idx')  # , cls=sp.Idx)
        sym_dec = sp.Symbol(label + "_dec")

        T_base = wrapper(T_base)  # T_base = sp.Float(T_base)
        T_incr = wrapper(T_incr)  # T_incr = sp.Float(T_incr)

        pos = (T - T_base) / T_incr
        pos_dec = pos - sp.floor(pos)

        result = sp.Piecewise(
            (wrapper(v_array[0]), T < T_base),
            (wrapper(v_array[-1]), T >= T_base + (len(v_array) - 1) * T_incr),
            ((1 - sym_dec) * sym_data[sym_idx] + sym_dec * sym_data[sym_idx + 1], True)
        )
        sub_expressions = [
            Assignment(sym_data, tuple(wrapper(v_array)), "double[]"),
            Assignment(sym_idx, pos, "int"),
            Assignment(sym_dec, pos_dec, "double")
        ]
        return MaterialProperty(result, sub_expressions)

    elif isinstance(T, float):
        n = len(v_array)

        min_temp = T_base
        max_temp = T_base + (n - 1) * T_incr
        if T <= min_temp:
            return material_property_wrapper(v_array[0])
        elif T >= max_temp:
            return material_property_wrapper(v_array[-1])

        pos = (T - T_base) / T_incr
        pos_int = int(pos)
        pos_dec = pos - pos_int

        value = (1 - pos_dec) * v_array[pos_int] + pos_dec * v_array[pos_int + 1]
        return material_property_wrapper(value)

    else:
        raise ValueError(f"Unsupported type for T: {type(T)}")


def interpolate_lookup(
        T: Union[float, sp.Symbol],
        T_array: ArrayTypes,
        v_array: ArrayTypes) -> MaterialProperty:
    """
    Perform lookup-based interpolation for symbolic or numeric temperature values.

    :param T: Temperature, either symbolic (sp.Symbol) or numeric (float).
    :param T_array: Array of temperatures for lookup.
    :param v_array: Array of values corresponding to the temperatures.
    :return: Interpolated value as a float or MaterialProperty object.
    :raises ValueError: If T_array and v_array lengths do not match or if T is of unsupported type.
    """
    if len(T_array) != len(v_array):
        raise ValueError("T_array and v_array must have the same length")

    if T_array[0] > T_array[-1]:
        T_array = np.flip(T_array)
        v_array = np.flip(v_array)

    if isinstance(T, sp.Symbol):
        global COUNT
        label = '_' + str(COUNT).zfill(3)
        COUNT += 1

        sym_expr = sp.Symbol(label + "_expr")

        T_array = wrapper(T_array)
        v_array = wrapper(v_array)

        conditions = [
            (v_array[0], T < T_array[0]),
            (v_array[-1], T >= T_array[-1])]
        for i in range(len(T_array) - 1):
            interp_expr = (
                    v_array[i] + (v_array[i + 1] - v_array[i]) / (T_array[i + 1] - T_array[i]) * (T - T_array[i]))
            conditions.append(
                (interp_expr, sp.And(T >= T_array[i], T < T_array[i + 1]) if i + 1 < len(T_array) - 1 else True))
        sub_expressions = [
            Assignment(sym_expr, sp.Piecewise(*conditions), "double"),
        ]
        return MaterialProperty(sym_expr, sub_expressions)

    elif isinstance(T, float):
        if T <= T_array[0]:
            return material_property_wrapper(v_array[0])
        elif T >= T_array[-1]:
            return material_property_wrapper(v_array[-1])
        else:
            v = np.interp(T, T_array, v_array)
            return material_property_wrapper(float(v))
    else:
        raise ValueError(f"Invalid input for T: {type(T)}")


def interpolate_property(
        T: Union[float, sp.Symbol],
        temp_array: ArrayTypes,
        prop_array: ArrayTypes,
        temp_array_limit: int = 6,  # if len(temp_array) < 6, force_lookup
        force_lookup: bool = False) -> MaterialProperty:
    """
    Perform interpolation based on the type of temperature array (equidistant or not).

    :param T: Temperature, either symbolic (sp.Symbol) or numeric (float).
    :param temp_array: Array of temperatures.
    :param prop_array: Array of property values corresponding to the temperatures.
    :param temp_array_limit: Minimum length of the temperature array to force equidistant interpolation.
    :param force_lookup: Boolean to force lookup interpolation regardless of temperature array type.
    :return: Interpolated value.
    :raises ValueError:
    :raises ValueError:
        - If T (numeric) is not in range.
        - If temp_array or prop_array length < 2.
        - If temp_array and prop_array lengths mismatch.
    :raises TypeError: If temp_array or prop_array is not a list, tuple, or ndarray.
    """
    if isinstance(T, float):
        if T < 20.0 or T > 20000.0:
            raise ValueError(f"Temperature must be between 20K and 20000K")

    if len(temp_array) <= 1 or len(prop_array) <= 1:
        raise ValueError(f"Length of temp_array {len(temp_array)} or length of prop_array {len(prop_array)} <= 1")

    if not isinstance(temp_array, (Tuple, List, np.ndarray)):
        raise TypeError(f"Expected temp_array to be a list or array or tuple, got {type(temp_array)}")

    if not isinstance(prop_array, (Tuple, List, np.ndarray)):
        raise TypeError(f"Expected prop_array to be a list or array or tuple, got {type(prop_array)}")

    if len(temp_array) != len(prop_array):
        raise ValueError("T_array and v_array must have the same length")

    if temp_array[0] > temp_array[-1]:
        temp_array = np.flip(temp_array)
        prop_array = np.flip(prop_array)

    incr = check_equidistant(np.asarray(temp_array))

    if force_lookup or incr == 0 or len(temp_array) < temp_array_limit:
        print('interpolate_lookup')
        return interpolate_lookup(T, temp_array, prop_array)
    else:
        print('interpolate_equidistant')
        return interpolate_equidistant(T, float(temp_array[0]), incr, prop_array)


# Moved from models.py to interpolators.py
def temperature_from_energy_density(
        T: sp.Expr,
        temperature_array: np.ndarray,
        h_in: float,
        energy_density: MaterialProperty) -> float:
    """
    Compute the temperature for energy density using linear interpolation.
    Args:
        T: Symbol for temperature in material property.
        temperature_array: Array of temperature values.
        h_in: Target property value.
        energy_density: Material property for energy_density.
    Returns:
        Corresponding temperature(s) for the input energy density value(s).
    """
    T_min, T_max = np.min(temperature_array), np.max(temperature_array)
    # print(f"T_min, T_max: {T_min, T_max}")
    h_min, h_max = energy_density.evalf(T, T_min), energy_density.evalf(T, T_max)
    # print(f"h_min, h_max: {h_min, h_max}")

    if h_in < h_min or h_in > h_max:
        raise ValueError(f"The input energy density value of {h_in} is outside the computed range {h_min, h_max}.")

    tolerance: float = 5e-6
    max_iterations: int = 5000

    iteration_count = 0

    for _ in range(max_iterations):
        iteration_count += 1
        # Linear interpolation to find T_1
        T_1 = T_min + (h_in - h_min) * (T_max - T_min) / (h_max - h_min)
        # print(f"T_1: {T_1}, type(T_1): {type(T_1)}")
        # Evaluate h_1 at T_1
        # h_1 = energy_density.evalf(T, np.array([T_1]))[0]
        h_1 = energy_density.evalf(T, T_1)
        # print(f"h_1: {h_1}")
        # print(f"h_min, h_max: {h_min, h_max}")

        if abs(h_1 - h_in) < tolerance:
            # print(f"Linear interpolation converged in {iteration_count} iterations.")
            return T_1

        if h_1 < h_in:
            T_min, h_min = T_1, h_1
            # print(f"T_min, h_min: {T_min, h_min}")
        else:
            T_max, h_max = T_1, h_1
            # print(f"T_max, h_max: {T_max, h_max}")

    raise RuntimeError(f"Linear interpolation did not converge within {max_iterations} iterations.")


# Moved from models.py to interpolators.py
def interpolate_binary_search(
        temperature_array: np.ndarray,
        h_in: float,
        energy_density_array: np.ndarray,
        epsilon: float = 1e-6) -> float:

    # Input validation
    if len(temperature_array) != len(energy_density_array):
        raise ValueError("temperature_array and energy_density_array must have the same length.")

    n = len(temperature_array)
    is_ascending = temperature_array[0] < temperature_array[-1]
    print(f"Python - is_ascending: {is_ascending}")
    print(f"Python - h_in: {h_in}")
    print(f"Python - First few T values: {temperature_array[:5]}")
    print(f"Python - First few E values: {energy_density_array[:5]}")

    # Critical boundary indices
    start_idx = 0 if is_ascending else n - 1
    end_idx = n - 1 if is_ascending else 0

    # Boundary checks
    if h_in <= energy_density_array[start_idx]:
        return float(temperature_array[start_idx])
    if h_in >= energy_density_array[end_idx]:
        return float(temperature_array[end_idx])

    # Binary search
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        mid_val = energy_density_array[mid]
        print(f"Python - mid: {mid}, mid_val: {mid_val}")

        if abs(mid_val - h_in) < epsilon:
            print(f"Python - Exact match found at {mid}")
            return float(temperature_array[mid])

        if (mid_val > h_in) == is_ascending:
            right = mid - 1
        else:
            left = mid + 1
        print(f"Python - left: {left}, right: {right}")

    print(f"Python - Final indices - left: {left}, right: {right}")
    print(f"Python - Interpolating between T[{right}]={temperature_array[right]} and T[{left}]={temperature_array[left]}")
    print(f"Python - Interpolating between E[{right}]={energy_density_array[right]} and E[{left}]={energy_density_array[left]}")

    # Linear interpolation
    x0 = energy_density_array[right]
    x1 = energy_density_array[left]
    y0 = temperature_array[right]
    y1 = temperature_array[left]

    result = float(y0 + (y1 - y0) * (h_in - x0) / (x1 - x0))
    print(f"Python - Result: {result}")
    return result


def E_eq_from_E_neq(E_neq: np.ndarray) -> Tuple[np.ndarray, float]:
    delta_min: float = np.min(np.abs(np.diff(E_neq)))
    if delta_min < 1.:
        raise ValueError(f"Energy density array points are very closely spaced, delta = {delta_min}")
    print(f"delta_min:", delta_min)
    delta_E_eq = max(np.floor(delta_min * 0.95), 1.)
    print(f"delta_E_eq:", delta_E_eq)
    E_eq = np.arange(E_neq[0], E_neq[-1] + delta_E_eq, delta_E_eq, dtype=np.float64)
    print(f"np.size(E_eq):", np.size(E_eq))
    inv_delta_E_eq: float = float(1. / (E_eq[1] - E_eq[0]))
    return E_eq, inv_delta_E_eq


def create_idx_mapping(E_neq: np.ndarray, E_eq: np.ndarray) -> np.ndarray:
    """idx_map = np.zeros(len(E_eq), dtype=int)
    for i, e in enumerate(E_eq):
        idx = int(np.searchsorted(E_neq, e) - 1)
        idx = max(0, min(idx, len(E_neq) - 2))  # Bound check
        idx_map[i] = idx"""
    # idx_map = np.searchsorted(E_neq, E_eq) - 1
    idx_map = np.searchsorted(E_neq, E_eq, side='right') - 1
    idx_map = np.clip(idx_map, 0, len(E_neq) - 2)
    print(f"idx_map: {idx_map}")
    return idx_map.astype(np.int32)


def prepare_interpolation_arrays(T_array: np.ndarray, E_array: np.ndarray) -> dict:
    """
    Validates input arrays and prepares data for interpolation.
    Returns a dictionary with arrays and metadata for the appropriate method.
    """
    # Convert to numpy arrays if not already
    T_array = np.asarray(T_array)
    E_array = np.asarray(E_array)

    # Basic validation
    if len(T_array) != len(E_array):
        raise ValueError("Temperature and energy density arrays must have the same length")

    if len(T_array) < 2:
        raise ValueError("Arrays must contain at least 2 elements")

    # Check if arrays are monotonic
    T_increasing = np.all(np.diff(T_array) > 0)
    T_decreasing = np.all(np.diff(T_array) < 0)
    E_increasing = np.all(np.diff(E_array) > 0)
    E_decreasing = np.all(np.diff(E_array) < 0)

    if not (T_increasing or T_decreasing):
        raise ValueError("Temperature array must be monotonically increasing or decreasing")

    if not (E_increasing or E_decreasing):
        raise ValueError("Energy density array must be monotonically increasing or decreasing")

    # Check if energy increases with temperature (both increasing or both decreasing)
    if (T_increasing and E_decreasing) or (T_decreasing and E_increasing):
        raise ValueError("Energy density must increase with temperature")

    # Create copies to avoid modifying original arrays
    T_bs = np.copy(T_array)
    E_bs = np.copy(E_array)

    # Flip arrays if temperature is in descending order
    if T_decreasing:
        print("Temperature array is descending, flipping arrays for processing")
        T_bs = np.flip(T_bs)
        E_bs = np.flip(E_bs)

    try:
        check_strictly_increasing(T_bs, "Temperature array")
        check_strictly_increasing(E_bs, "Energy density array")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Continuing with interpolation, but results may be less accurate")

    # Use your existing check_equidistant function to determine if suitable for double lookup
    T_incr = check_equidistant(T_bs)
    is_equidistant = T_incr != 0.0

    result = {
        "T_bs": T_bs,
        "E_bs": E_bs,
        "method": "binary_search"
    }

    # If temperature is equidistant, prepare for double lookup
    if is_equidistant:
        print(f"Temperature array is equidistant with increment {T_incr}, using double lookup")
        # Create equidistant energy array and mapping
        E_eq, inv_delta_E_eq = E_eq_from_E_neq(E_bs)
        idx_mapping = create_idx_mapping(E_bs, E_eq)

        result.update({
            "T_eq": T_bs,
            "E_neq": E_bs,
            "E_eq": E_eq,
            "inv_delta_E_eq": inv_delta_E_eq,
            "idx_map": idx_mapping,
            "method": "double_lookup"
        })
    else:
        print("Temperature array is not equidistant, using binary search")

    return result
'''
    # Determine if suitable for double lookup (temperature must be equidistant)
    T_diffs = np.diff(T_bs)
    is_equidistant = np.allclose(T_diffs, T_diffs[0], rtol=1e-5)

    result = {
        "T_bs": T_bs,
        "E_bs": E_bs,
        "method": "binary_search"
    }

    # If temperature is equidistant, prepare for double lookup
    if is_equidistant:
        # Create equidistant energy array and mapping
        E_eq, inv_delta_E_eq = E_eq_from_E_neq(E_bs)
        idx_mapping = create_idx_mapping(E_bs, E_eq)

        result.update({
            "T_eq": T_bs,
            "E_neq": E_bs,
            "E_eq": E_eq,
            "inv_delta_E_eq": inv_delta_E_eq,
            "idx_map": idx_mapping,
            "method": "double_lookup"
        })

    return result
'''

def interpolate_double_lookup(E_target: float, T_eq: np.ndarray, E_neq: np.ndarray, E_eq: np.ndarray, inv_delta_E_eq: float, idx_map: np.ndarray) -> float:
    print(f"\nPython Debug:")
    print(f"E_target: {E_target}")
    print(f"First few T_eq values: {T_eq[:5]}")
    print(f"First few E_neq values: {E_neq[:5]}")
    print(f"First few E_eq values: {E_eq[:5]}")
    print(f"First few idx_map values: {idx_map[:5]}")

    if E_target <= E_neq[0]:
        print(f"Lower boundary case: returning {T_eq[0]}")
        return float(T_eq[0])
    if E_target >= E_neq[-1]:
        print(f"Upper boundary case: returning {T_eq[-1]}")
        return float(T_eq[-1])

    idx_E_eq = int((E_target - E_eq[0]) * inv_delta_E_eq)
    print(f"idx_E_eq initial: {idx_E_eq}")
    idx_E_eq = min(idx_E_eq, len(idx_map) - 1)
    print(f"idx_E_eq after bound check: {idx_E_eq}")

    idx_E_neq = idx_map[idx_E_eq]
    print(f"idx_E_neq initial: {idx_E_neq}")
    if E_neq[idx_E_neq + 1] < E_target and idx_E_neq + 1 < len(E_neq):
        idx_E_neq += 1
        print(f"idx_E_neq adjusted: {idx_E_neq}")

    E1, E2 = E_neq[idx_E_neq], E_neq[idx_E_neq + 1]
    T1, T2 = T_eq[idx_E_neq], T_eq[idx_E_neq + 1]
    print(f"Interpolation points:")
    print(f"E1, E2: {E1}, {E2}")
    print(f"T1, T2: {T1}, {T2}")

    result = float(T1 + (T2 - T1) * (E_target - E1) / (E2 - E1))
    print(f"Final result: {result}")
    return result
