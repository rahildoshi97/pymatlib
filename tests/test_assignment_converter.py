import pytest
import numpy as np
import sympy as sp
import pystencils as ps
from pystencils.types import PsArrayType
from pystencils.types.quick import Arr, Fp
from pymatlib.core.typedefs import Assignment
from pymatlib.core.assignment_converter import type_mapping, assignment_converter


def test_type_mapping():
    """Test the type_mapping function for different input types and lengths."""
    # Test array types
    array_type = type_mapping("double[]", 5)
    assert isinstance(array_type, PsArrayType)
    assert array_type.base_type.width == 64  # Check if it's 64-bit (double)

    array_type = type_mapping("float[]", 3)
    assert isinstance(array_type, PsArrayType)
    assert array_type.base_type.width == 32  # Check if it's 32-bit (double)

    # Test scalar types
    assert type_mapping("double", 1) == np.dtype('float64')
    assert type_mapping("float", 1) == np.dtype('float32')
    assert type_mapping("int", 1) == np.dtype('int32')
    assert type_mapping("bool", 1) == np.dtype('bool')

    # Test invalid type
    with pytest.raises(ValueError):
        type_mapping("invalid_type", 1)

def test_assignment_converter():
    """Test the assignment_converter function with various assignments."""
    # Define test symbols
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    z = sp.Symbol('z')

    # Create test assignments
    assignments = [
        Assignment(lhs=x, rhs=sp.Symbol('value_x'), lhs_type='double[]'),
        Assignment(lhs=y, rhs=sp.Symbol('value_y'), lhs_type='float[]'),
        Assignment(lhs=z, rhs=sp.Symbol('value_z'), lhs_type='int')
    ]

    # Convert assignments
    assignments_converted, symbol_map = assignment_converter(assignments)

    # Test converted assignments
    assert len(assignments_converted) == 3
    assert all(isinstance(conv, ps.Assignment) for conv in assignments_converted)

    # Test symbol mapping
    assert len(symbol_map) == 3
    assert all(isinstance(sym, ps.TypedSymbol) for sym in symbol_map.values())
    assert x in symbol_map
    assert y in symbol_map
    assert z in symbol_map

def test_invalid_assignments():
    """Test assignment_converter with invalid assignments."""
    # Test with None values
    with pytest.raises(ValueError):
        assignment_converter([Assignment(None, None, None)])

    # Test with missing type
    x = sp.Symbol('x')
    with pytest.raises(ValueError):
        assignment_converter([Assignment(x, sp.Symbol('value_x'), None)])