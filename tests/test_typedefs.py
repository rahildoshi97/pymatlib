import pytest
import numpy as np
import sympy as sp
from pymatlib.core.typedefs import Assignment, MaterialProperty, ArrayTypes, PropertyTypes

def test_assignment():
    """Test Assignment dataclass functionality."""
    # Test basic assignment
    x = sp.Symbol('x')
    assignment = Assignment(x, 100, 'double')
    assert assignment.lhs == x
    assert assignment.rhs == 100
    assert assignment.lhs_type == 'double'

    # Test with tuple
    v = sp.IndexedBase('v')
    assignment = Assignment(v, (1, 2, 3), 'double[]')
    assert assignment.lhs == v
    assert assignment.rhs == (1, 2, 3)
    assert assignment.lhs_type == 'double[]'

def test_material_property_constant():
    """Test MaterialProperty with constant values."""
    # Test constant property
    mp = MaterialProperty(sp.Float(405.))
    assert mp.evalf(sp.Symbol('T'), 100.) == 405.0

    # Test with assignments
    mp.assignments.append(Assignment(sp.Symbol('A'), (100, 200), 'int'))
    assert isinstance(mp.assignments, list)
    assert len(mp.assignments) == 1

def test_material_property_temperature_dependent():
    """Test MaterialProperty with temperature-dependent expressions."""
    T = sp.Symbol('T')

    # Test linear dependency
    mp = MaterialProperty(T * 100.)
    assert mp.evalf(T, 2.0) == 200.0

    # Test polynomial
    mp = MaterialProperty(T**2 + T + 1)
    assert mp.evalf(T, 2.0) == 7.0

def test_material_property_indexed():
    """Test MaterialProperty with indexed base expressions."""
    T = sp.Symbol('T')
    v = sp.IndexedBase('v')
    i = sp.Symbol('i', integer=True)

    mp = MaterialProperty(v[i])
    mp.assignments.extend([
        Assignment(v, (3, 6, 9), 'float'),
        Assignment(i, T / 100, 'int')
    ])

    assert mp.evalf(T, 97) == 3  # i=0
    assert mp.evalf(T, 150) == 6  # i=1

def test_material_property_errors():
    """Test error handling in MaterialProperty."""
    T = sp.Symbol('T')
    X = sp.Symbol('X')  # Different symbol

    # Test evaluation with wrong symbol
    mp = MaterialProperty(X * 100)
    with pytest.raises(TypeError, match="Symbol T not found in expression or assignments"):
        mp.evalf(T, 100.0)

    # Test invalid assignments
    with pytest.raises(ValueError, match="None assignments are not allowed"):
        MaterialProperty(T * 100, assignments=[None])

def test_material_property_array_evaluation():
    """Test MaterialProperty evaluation with arrays."""
    T = sp.Symbol('T')
    mp = MaterialProperty(T * 100)

    # Test with numpy array
    temps = np.array([1.0, 2.0, 3.0])
    result = mp.evalf(T, temps)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, temps * 100.0)

    # Test with invalid array values
    with pytest.raises(TypeError):
        mp.evalf(T, np.array(['a', 'b', 'c']))

def test_material_property_piecewise():
    """Test MaterialProperty with piecewise functions."""
    T = sp.Symbol('T')
    mp = MaterialProperty(sp.Piecewise(
        (100, T < 0),
        (200, T >= 100),
        (T, True)
    ))

    assert mp.evalf(T, -10) == 100
    assert mp.evalf(T, 50) == 50
    assert mp.evalf(T, 150) == 200

def test_material_property_numpy_types():
    """Test MaterialProperty with numpy numeric types."""
    T = sp.Symbol('T')
    mp = MaterialProperty(T * 100)

    assert mp.evalf(T, np.float32(1.0)) == 100.0
    assert mp.evalf(T, np.float64(1.0)) == 100.0
