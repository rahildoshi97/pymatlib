import pytest
import numpy as np
import sympy as sp
from pymatlib.core.interpolators import (
    interpolate_equidistant,
    interpolate_lookup,
    check_equidistant,
    interpolate_property
)
from pymatlib.core.typedefs import MaterialProperty

def test_check_equidistant():
    """Test the check_equidistant function."""
    # Test equidistant array
    temp_array = np.array([100.0, 200.0, 300.0, 400.0])
    assert check_equidistant(temp_array) == 100.0

    # Test non-equidistant array
    temp_array = np.array([100.0, 250.0, 300.0, 400.0])
    assert check_equidistant(temp_array) == 0

    # Test edge cases
    assert check_equidistant(np.array([])) == 0  # Empty array
    assert check_equidistant(np.array([100.0])) == 0  # Single value
    assert check_equidistant(np.array([100.0, 200.0])) == 100.0  # Two values

def test_interpolate_equidistant():
    """Test the interpolate_equidistant function."""
    T_base = 200.0
    T_incr = 100.0
    v_array = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Test boundary cases
    result = interpolate_equidistant(200.0, T_base, T_incr, v_array)  # Exactly at base
    assert isinstance(result, MaterialProperty)

    result = interpolate_equidistant(600.0, T_base, T_incr, v_array)  # At max
    assert isinstance(result, MaterialProperty)

    # Test out of bounds cases
    result = interpolate_equidistant(150.0, T_base, T_incr, v_array)  # Below range
    assert isinstance(result, MaterialProperty)

    result = interpolate_equidistant(650.0, T_base, T_incr, v_array)  # Above range
    assert isinstance(result, MaterialProperty)

def test_interpolate_equidistant_symbolic():
    """Test symbolic interpolation with equidistant values."""
    T = sp.Symbol('T')
    T_base = 200.0
    T_incr = 100.0
    v_array = np.array([10.0, 20.0, 30.0])

    result = interpolate_equidistant(T, T_base, T_incr, v_array)
    assert isinstance(result, MaterialProperty)
    assert hasattr(result, 'expr')
    assert hasattr(result, 'assignments')

def test_interpolate_lookup_edge_cases():
    """Test edge cases for lookup interpolation."""
    temp_array = np.array([200.0, 300.0, 400.0])
    v_array = np.array([10.0, 20.0, 30.0])

    # Test exact matches
    result = interpolate_lookup(200.0, temp_array, v_array)
    assert isinstance(result, MaterialProperty)

    # Test reversed arrays
    result = interpolate_lookup(300.0, np.flip(temp_array), np.flip(v_array))
    assert isinstance(result, MaterialProperty)

    # Test error cases
    with pytest.raises(ValueError):
        interpolate_lookup(300.0, temp_array, v_array[:-1])  # Mismatched lengths

def test_interpolate_property_validation():
    """Test input validation in interpolate_property."""
    temp_array = np.array([200.0, 300.0, 400.0])
    v_array = np.array([10.0, 20.0, 30.0])

    # Test type validation
    with pytest.raises(TypeError):
        interpolate_property(300.0, "invalid", v_array)

    with pytest.raises(TypeError):
        interpolate_property(300.0, temp_array, "invalid")

    # Test length validation
    with pytest.raises(ValueError):
        interpolate_property(300.0, temp_array, v_array[:-1])

    with pytest.raises(ValueError):
        interpolate_property(300.0, np.array([]), np.array([]))  # Empty arrays

def test_interpolate_lookup():
    """Test the interpolate_lookup function."""
    temp_array = np.array([200.0, 300.0, 400.0, 500.0, 600.0])
    v_array = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Test numeric interpolation
    result = interpolate_lookup(350.0, temp_array, v_array)
    assert isinstance(result, MaterialProperty)

    # Test below range
    result = interpolate_lookup(150.0, temp_array, v_array)
    assert isinstance(result, MaterialProperty)

    # Test above range
    result = interpolate_lookup(650.0, temp_array, v_array)
    assert isinstance(result, MaterialProperty)

    # Test symbolic interpolation
    T = sp.Symbol('T')
    result = interpolate_lookup(T, temp_array, v_array)
    assert isinstance(result, MaterialProperty)

    # Test error cases
    with pytest.raises(ValueError):
        interpolate_lookup(350.0, temp_array, v_array[:-1])  # Mismatched lengths

def test_interpolate_property():
    """Test the interpolate_property function."""
    temp_array = np.array([200.0, 300.0, 400.0, 500.0, 600.0])
    v_array = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Test with equidistant temperatures
    result = interpolate_property(350.0, temp_array, v_array)
    assert isinstance(result, MaterialProperty)

    # Test with non-equidistant temperatures
    temp_array_non_equi = np.array([200.0, 250.0, 400.0, 500.0, 600.0])
    result = interpolate_property(350.0, temp_array_non_equi, v_array)
    assert isinstance(result, MaterialProperty)

    # Test with force_lookup=True
    result = interpolate_property(350.0, temp_array, v_array, force_lookup=True)
    assert isinstance(result, MaterialProperty)

    # Test error cases
    with pytest.raises(TypeError):
        interpolate_property(350.0, 123, v_array)  # Invalid temp_array type

    with pytest.raises(ValueError):
        interpolate_property(350.0, temp_array, v_array[:-1])  # Mismatched lengths

def test_interpolate_property1():
    # Test interpolating properties with float temperature inputs
    T_symbol = sp.Symbol('T')  # Define a symbolic variable
    T_value = 1400.0
    temp_array = np.array([1300.0, 1400.0, 1500.0])
    prop_array = np.array([100.0, 200.0, 300.0])
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test interpolating properties with symbolic temperature inputs
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test interpolating properties with different combinations of temperature and property arrays
    temp_array = [1300.0, 1400.0, 1500.0]
    prop_array = [100.0, 200.0, 300.0]
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    temp_array = tuple([1300.0, 1400.0, 1500.0])
    prop_array = tuple([100.0, 200.0, 300.0])
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test with ascending and descending order arrays
    temp_array_asc = np.array([1300.0, 1400.0, 1500.0])
    prop_array_asc = np.array([100.0, 200.0, 300.0])
    result_asc = interpolate_property(T_symbol, temp_array_asc, prop_array_asc)
    assert isinstance(result_asc, MaterialProperty)

    temp_array_desc = np.array([1500.0, 1400.0, 1300.0])
    prop_array_desc = np.array([300.0, 200.0, 100.0])
    result_desc = interpolate_property(T_symbol, temp_array_desc, prop_array_desc)
    assert isinstance(result_desc, MaterialProperty)

    # Test with different temp_array_limit values
    result_limit_6 = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, temp_array_limit=6)
    assert isinstance(result_limit_6, MaterialProperty)

    result_limit_3 = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, temp_array_limit=3)
    assert isinstance(result_limit_3, MaterialProperty)

    # Test with force_lookup True and False
    result_force_lookup_true = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, force_lookup=True)
    assert isinstance(result_force_lookup_true, MaterialProperty)

    result_force_lookup_false = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, force_lookup=False)
    assert isinstance(result_force_lookup_false, MaterialProperty)
    # assert np.isclose(result_force_lookup_false.evalf(T_symbol, T_value), 200.0)

def test_interpolate_lookup1():
    # Define a symbolic variable
    T_symbol = sp.Symbol('T')

    # Test lookup interpolation with float temperature inputs
    T_value = 1400.0
    temp_array = np.array([1300.0, 1400.0, 1500.0])
    prop_array = np.array([100.0, 200.0, 300.0])
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test lookup interpolation with symbolic temperature inputs
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test with different combinations of temperature and property arrays
    temp_array = [1300.0, 1400.0, 1500.0]
    prop_array = [100.0, 200.0, 300.0]
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    temp_array = tuple([1300.0, 1400.0, 1500.0])
    prop_array = tuple([100.0, 200.0, 300.0])
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test with ascending and descending order arrays
    temp_array_asc = np.array([1300.0, 1400.0, 1500.0])
    prop_array_asc = np.array([100.0, 200.0, 300.0])
    result_asc = interpolate_lookup(T_symbol, temp_array_asc, prop_array_asc)
    assert isinstance(result_asc, MaterialProperty)

    temp_array_desc = np.array([1500.0, 1400.0, 1300.0])
    prop_array_desc = np.array([300.0, 200.0, 100.0])
    result_desc = interpolate_lookup(T_symbol, temp_array_desc, prop_array_desc)
    assert isinstance(result_desc, MaterialProperty)
    # assert np.isclose(result_desc.evalf(T_symbol, T_value), 200.0)

def test_interpolate_equidistant1():
    # Define a symbolic variable
    T_symbol = sp.Symbol('T')

    # Test equidistant interpolation with float temperature inputs
    T_value = 1400.0
    temp_base = 1300.0
    temp_incr = 100.0
    prop_array = np.array([100.0, 200.0, 300.0])
    result = interpolate_equidistant(T_value, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)

    # Test equidistant interpolation with symbolic temperature inputs
    result = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with different combinations of temperature and property arrays
    temp_base = 1300.0
    temp_incr = 100.0
    prop_array = [100.0, 200.0, 300.0]
    result = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    temp_base = 1300.0
    temp_incr = 100.0
    prop_array = tuple([100.0, 200.0, 300.0])
    result = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with ascending and descending order arrays
    temp_base = 1300.0
    temp_incr = 100.0
    prop_array_asc = np.array([100.0, 200.0, 300.0])
    result_asc = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array_asc)
    assert isinstance(result_asc, MaterialProperty)
    assert np.isclose(result_asc.evalf(T_symbol, T_value), 200.0)

    temp_base = 1300.0
    temp_incr = 100.0
    prop_array_desc = np.array([300.0, 200.0, 100.0])
    result_desc = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array_desc)
    assert isinstance(result_desc, MaterialProperty)
    assert np.isclose(result_desc.evalf(T_symbol, T_value), 200.0)

# Define the temp fixture
@pytest.fixture
def temp():
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])

def test_test_equidistant(temp):
    # Test with equidistant temperature array
    assert check_equidistant(temp) == 100.0

    # Test with non-equidistant temperature array
    temp_non_equidistant = np.array([100.0, 150.0, 300.0, 450.0, 500.0])
    assert check_equidistant(temp_non_equidistant) == 0.0
