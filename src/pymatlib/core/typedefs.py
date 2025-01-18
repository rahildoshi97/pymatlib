"""
typedefs.py

This module defines custom types, type aliases, and utility classes used throughout the pymatlib module,
enhancing code readability and consistency. It includes data structures for assignments and the evaluation
of material properties as functions of symbolic variables and numerical values.

Classes:
    Assignment: A dataclass representing an assignment operation with a left-hand side (lhs),
                right-hand side (rhs), and the type of the lhs.
    MaterialProperty: A dataclass that represents a material property, which can be evaluated as a function
                      of a symbolic variable (e.g., temperature) and potentially includes symbolic assignments.

Type Aliases:
    ArrayTypes: A type alias for numerical data types, which can be either numpy arrays, lists, or tuples.
    PropertyTypes: A type alias for properties that can be either constant floats or MaterialProperty instances.

Functions:
    type_mapping(type_str: str) -> Union[np.dtype, Arr]: Maps a string representation of a type to a corresponding
                                                         numpy or pystencils data type.
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from typing import List, Union, Tuple, get_args


@dataclass
class Assignment:
    """
    Represents an assignment operation within the simulation.

    Attributes:
        lhs (sp.Symbol): The left-hand side of the assignment, typically a symbolic variable.
        rhs (Union[tuple, sp.Expr]): The right-hand side of the assignment, which can be a tuple or sympy expression.
        lhs_type (str): The type of the left-hand side, indicating the data type.
    """
    lhs: sp.Symbol
    rhs: Union[tuple, sp.Expr]
    lhs_type: str


# Type aliases for various data types used in the project
ArrayTypes = Union[np.ndarray, List, Tuple]  # Numerical values can be represented as numpy arrays, lists, or tuples

@dataclass
class MaterialProperty:
    """
    Represents a material property that can be evaluated as a function of a symbolic variable (e.g., temperature).
    It can include symbolic assignments that are resolved during the evaluation.

    Attributes:
        expr (sp.Expr): The symbolic expression representing the material property.
        assignments (List[Assignment]): A list of Assignment instances representing any symbolic assignments that
                                        need to be resolved during evaluation.

    Methods:
        evalf(symbol: sp.Symbol, temperature: Union[float, ArrayTypes]) -> Union[float, np.ndarray]:
            Evaluates the material property for a given temperature, resolving any symbolic assignments.
    """
    expr: sp.Expr
    assignments: List[Assignment] = field(default_factory=list)

    def __post_init__(self):
        """Validate assignments after initialization."""
        if any(assignment is None for assignment in self.assignments):
            raise ValueError("None assignments are not allowed")

        # Validate each assignment
        for assignment in self.assignments:
            if not isinstance(assignment, Assignment):
                raise ValueError(f"Invalid assignment type: {type(assignment)}")
            if assignment.lhs is None or assignment.rhs is None or assignment.lhs_type is None:
                raise ValueError("Assignment fields cannot be None")

    def evalf(self, symbol: sp.Symbol, temperature: Union[float, ArrayTypes]) -> Union[float, np.ndarray]:
        """
        Evaluates the material property at specific temperature values.

        Args:
            symbol (sp.Symbol): The symbolic variable (e.g., temperature) to substitute in the expression.
            temperature (Union[float, ArrayTypes]): The temperature(s) at which to evaluate the property.

        Returns:
            Union[float, np.ndarray]: The evaluated property value(s) at the given temperature(s).

        Raises:
            TypeError: If:
                - symbol is not found in expression or assignments
                - temperature contains non-numeric values
                - invalid type for temperature
        """
        # If the expression has no symbolic variables, return it as a constant float
        if not self.expr.free_symbols:
            return float(self.expr)

        # Collect all relevant symbols from expressions and assignments
        all_symbols = self.expr.free_symbols.union(
            *(assignment.rhs.free_symbols
              for assignment in self.assignments
              if isinstance(assignment.rhs, sp.Expr))
        )
        # If we have symbols but the provided one isn't among them, raise TypeError
        if all_symbols and symbol not in all_symbols:
            raise TypeError(f"Symbol {symbol} not found in expression or assignments")

        # Handle array inputs
        # If temperature is a numpy array, list, or tuple (ArrayTypes), evaluate the property for each temperature
        if isinstance(temperature, get_args(ArrayTypes)):
            return np.array([self.evalf(symbol, float(t)) for t in temperature])

        # Convert numeric types to float
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError):
            raise TypeError(f"Temperature must be numeric, got {type(temperature)}")

        # Prepare substitutions for symbolic assignments
        substitutions = [(symbol, temperature_value)]
        for assignment in self.assignments:
            if isinstance(assignment.rhs, sp.Expr):
                # Evaluate the right-hand side expression at the given temperature
                value = sp.N(assignment.rhs.subs(symbol, temperature_value))
                value = int(value) if assignment.lhs_type == "int" else value
                substitutions.append((assignment.lhs, value))
            else:
                substitutions.append((assignment.lhs, assignment.rhs))

        # Evaluate the material property with the substitutions
        result = sp.N(self.expr.subs(substitutions))
        # Try to convert the result to float if possible
        try:
            return float(result)
        except TypeError:
            return result  # Return the symbolic expression if it cannot be converted to a float

# Type alias for properties that can be either constant floats or MaterialProperty instances
PropertyTypes = Union[float, MaterialProperty]
