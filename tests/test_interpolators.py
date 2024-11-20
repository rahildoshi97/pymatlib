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