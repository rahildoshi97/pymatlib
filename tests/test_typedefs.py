import pytest
import numpy as np
import sympy as sp
from pymatlib.core.typedefs import Assignment, MaterialProperty

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
    mp = MaterialProperty(sp.Float(405.))
    assert mp.evalf(sp.Symbol('T'), 100.) == 405.0  # Expected output: 405.0

def test_material_property_linear():
    """Test MaterialProperty with linear temperature-dependent property."""
    T = sp.Symbol('T')
    mp = MaterialProperty(T * 100.)
    assert mp.evalf(T, 100.) == 10000.0  # Expected output: 10000.0

def test_material_property_indexed():
    """Test MaterialProperty with indexed base expressions."""
    T = sp.Symbol('T')
    v = sp.IndexedBase('v')
    i = sp.Symbol('i', integer=True)

    mp = MaterialProperty(v[i])

    # Assignments should include how i is determined
    mp.assignments.extend([
        Assignment(v, (3, 6, 9), 'float'),
        Assignment(i, T / 100, 'int')  # i is determined by T
    ])

    # Evaluate at a temperature value that results in i=0
    assert mp.evalf(T, 97) == 3.0  # Should evaluate correctly based on index logic

def test_material_property_errors():
    """Test error handling in MaterialProperty."""
    T = sp.Symbol('T')
    X = sp.Symbol('X')  # Different symbol

    # Test evaluation with wrong symbol
    mp = MaterialProperty(X * 100)
    with pytest.raises(TypeError):
        mp.evalf(T, 100.0)  # Should raise TypeError because X != T

    # Test invalid assignments
    with pytest.raises(ValueError):
        MaterialProperty(T * 100, assignments=[None])  # None assignment should raise ValueError

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

def test_material_property_array_evaluation():
    """Test MaterialProperty evaluation with arrays."""
    T = sp.Symbol('T')

    mp = MaterialProperty(T * 100)

    # Test with numpy array
    temps = np.array([1.0, 2.0, 3.0])

    result = mp.evalf(T, temps)

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, temps * 100.0)
