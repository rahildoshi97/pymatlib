import numpy as np
import sympy as sp
from typing import Union, List, Tuple
from pymatlib.core.models import wrapper, material_property_wrapper
from pymatlib.core.typedefs import Assignment, ArrayTypes, MaterialProperty

COUNT = 0


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


def check_equidistant(temp: np.ndarray) -> float:
    """
    Tests if the temperature values are equidistant.

    :param temp: Array of temperature values.
    :return: The common difference if equidistant, otherwise 0.
    """
    if len(temp) <= 1:
        return 0.0

    temperature_diffs = np.diff(temp)
    if np.allclose(temperature_diffs, temperature_diffs[0], atol=1e-10):
        return float(temperature_diffs[0])
    return 0


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
        return interpolate_lookup(T, temp_array, prop_array)
    else:
        return interpolate_equidistant(
            T, float(temp_array[0]), incr, prop_array)
