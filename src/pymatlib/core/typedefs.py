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
# def evalf(self, symbol: sp.Symbol, temperature: Union[sp.Symbol, float, ArrayTypes]) -> Union[float, np.ndarray]:

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

    def evalf(self, symbol: sp.Symbol, temperature: Union[float, ArrayTypes]) -> Union[float, np.ndarray]:
        """
        Evaluates the material property at specific temperature values.

        Args:
            symbol (sp.Symbol): The symbolic variable (e.g., temperature) to substitute in the expression.
            temperature (Union[float, ArrayTypes]): The temperature(s) at which to evaluate the property.

        Returns:
            Union[float, np.ndarray]: The evaluated property value(s) at the given temperature(s).
        """
        # If the expression has no symbolic variables, return it as a constant float
        if not self.expr.free_symbols:
            return float(self.expr)

        # If temperature is a numpy array, list, or tuple (ArrayTypes), evaluate the property for each temperature
        if isinstance(temperature, get_args(ArrayTypes)):
            return np.array([self.evalf(symbol, t) for t in temperature])

        # Convert any numpy scalar to Python float
        elif isinstance(temperature, np.generic):
            temperature = float(temperature)

        # Prepare substitutions for symbolic assignments
        substitutions = [(symbol, temperature)]
        for assignment in self.assignments:
            if isinstance(assignment.rhs, sp.Expr):
                # Evaluate the right-hand side expression at the given temperature
                value = sp.N(assignment.rhs.subs(symbol, temperature))
                value = int(value) if assignment.lhs_type == "int" else value
                substitutions.append((assignment.lhs, value))
            else:
                substitutions.append((assignment.lhs, assignment.rhs))

        # Evaluate the material property with the substitutions
        result = sp.N(self.expr.subs(substitutions))
        # return float(result)
        # Try to convert the result to float if possible
        try:
            return float(result)
        except TypeError:
            return result  # Return the symbolic expression if it cannot be converted to a float

# Type alias for properties that can be either constant floats or MaterialProperty instances
PropertyTypes = Union[float, MaterialProperty]

if __name__ == '__main__':
    # Example usage and tests for MaterialProperty and Assignment classes

    T = sp.Symbol('T')
    v = sp.IndexedBase('v')
    i = sp.Symbol('i', type=sp.Integer)

    # Example 1: Constant material property
    mp0 = MaterialProperty(sp.Float(405.))
    mp0.assignments.append(Assignment(sp.Symbol('A'), (100, 200), 'int'))
    print(mp0)
    print(mp0.evalf(T, 100.))

    # Example 2: Linear temperature-dependent property
    mp1 = MaterialProperty(T * 100.)
    print(mp1)
    print(mp1.evalf(T, 100.))

    # Example 3: Indexed base with symbolic assignments
    mp2 = MaterialProperty(v[i])
    mp2.assignments.append(Assignment(v, (3, 6, 9), 'float'))
    mp2.assignments.append(Assignment(i, T / 100, 'int'))
    print(mp2)
    print(mp2.evalf(T, 97))  # Should evaluate with i=0 (since 97/100 is 0 when converted to int)
    print(mp2.evalf(T, 107))  # Should evaluate with i=1 (since 107/100 is 1 when converted to int)

    # Example 4: Evaluate an expression with an array of temperatures
    rho = 1e-6 * T ** 3 * 4000
    print(rho * np.array([10, 20, 30]))

if __name__ == '__main__':
    # Example usage and tests for MaterialProperty and Assignment classes

    T = sp.Symbol('T')
    v = sp.IndexedBase('v')
    i = sp.Symbol('i', type=sp.Integer)

    # Test 1: Constant material property
    mp0 = MaterialProperty(sp.Float(405.))
    mp0.assignments.append(Assignment(sp.Symbol('A'), (100, 200), 'int'))
    print(f"Test 1 - Constant property, no symbolic variables: {mp0.evalf(T, 100.)}")  # Expected output: 405.0

    # Test 2: Linear temperature-dependent property
    mp1 = MaterialProperty(T * 100.)
    print(f"Test 2 - Linear temperature-dependent property: {mp1.evalf(T, 100.)}")  # Expected output: 10000.0

    # Test 3: Indexed base with symbolic assignments
    mp2 = MaterialProperty(v[i])
    mp2.assignments.append(Assignment(v, (3, 6, 9), 'float'))
    mp2.assignments.append(Assignment(i, T / 100, 'int'))
    print(f"Test 3a - Indexed base with i=0: {mp2.evalf(T, 97)}")  # Expected output: 3 (i=0)
    print(f"Test 3b - Indexed base with i=1: {mp2.evalf(T, 107)}")  # Expected output: 6 (i=1)

    # Test 4: Evaluate an expression with an array of temperatures
    rho = 1e-6 * T ** 3 * 4000
    temps = np.array([10, 20, 30])
    print(f"Test 4 - Expression evaluated with array: {rho * temps}")  # Expected output: array of evaluated values

    # Additional Tests:

    # Test 5: Non-linear temperature-dependent property
    mp3 = MaterialProperty(T ** 2 + T * 50 + 25)
    print(f"Test 5 - Non-linear temperature-dependent property: {mp3.evalf(T, 10)}")  # Expected output: 625.0

    # Test 6: Temperature array evaluation for non-linear expression
    print(f"Test 6 - Non-linear property with temperature array: {mp3.evalf(T, temps)}")
    # Expected output: array of evaluated values for T in [10, 20, 30]

    # Test 7: Property with multiple symbolic assignments
    mp4 = MaterialProperty(v[i] * T)
    mp4.assignments.append(Assignment(v, (3, 6, 9), 'float'))
    mp4.assignments.append(Assignment(i, T / 100, 'int'))
    print(f"Test 7a - Property with multiple symbolic assignments at T=95: {mp4.evalf(T, 95)}")  # Expected: 285.0
    print(f"Test 7b - Property with multiple symbolic assignments at T=205: {mp4.evalf(T, 205)}")  # Expected: 1230.0

    # Test 8: Constant property evaluated with temperature array
    mp5 = MaterialProperty(sp.Float(500.))
    print(f"Test 8 - Constant property with temperature array: {mp5.evalf(T, temps)}")
    # Expected output: array of 500s for each temperature

    # Test 9: Handling numpy scalar input
    scalar_temp = np.float64(150.0)
    mp6 = MaterialProperty(T + 50)
    print(f"Test 9 - Handling numpy scalar input: {mp6.evalf(T, scalar_temp)}")  # Expected output: 200.0

    # Test 10: Property with no symbolic dependencies
    mp7 = MaterialProperty(sp.Float(1500.))
    print(f"Test 10 - Property with no symbolic dependencies: {mp7.evalf(T, 200.)}")  # Expected output: 1500.0

    # Test 11: Piecewise function
    mp8 = MaterialProperty(sp.Piecewise((100, T < 0), (200, T >= 100), (T, True)))
    print(f"Test 11a - Piecewise function at T=-10: {mp8.evalf(T, -10)}")  # Expected: 100
    print(f"Test 11b - Piecewise function at T=50: {mp8.evalf(T, 50)}")   # Expected: 50
    print(f"Test 11c - Piecewise function at T=150: {mp8.evalf(T, 150)}") # Expected: 200

    # Test 12: Complex expression with trigonometric functions
    mp9 = MaterialProperty(100 * sp.sin(T) + 50 * sp.cos(T))
    print(f"Test 12a - Complex expression at T=0: {mp9.evalf(T, 0)}")       # Expected: 50
    print(f"Test 12b - Complex expression at T=pi/2: {mp9.evalf(T, np.pi/2)}") # Expected: 100

    # Test 13: Expression with logarithmic and exponential functions
    mp10 = MaterialProperty(10 * sp.log(T + 1) + 5 * sp.exp(T/100))
    print(f"Test 13 - Log and exp expression at T=10: {mp10.evalf(T, 10)}") # Expected: ~28.19

    # Test 14: Property with multiple variables
    T2 = sp.Symbol('T2')
    mp11 = MaterialProperty(T * T2)
    res = mp11.evalf(T, 10)
    print(f"Test 14 - Multi-variable property: {res}")  # This should raise an error or return a symbolic expression
    assert isinstance(res, sp.Expr)
    assert sp.Eq(res, 10 * T2)

    # Test 15: Handling very large and very small numbers
    mp12 = MaterialProperty(1e20 * T + 1e-20)
    print(f"Test 15a - Large numbers: {mp12.evalf(T, 1e5)}")
    print(f"Test 15b - Small numbers: {mp12.evalf(T, 1e-5)}")

    # Test 16: Property with rational functions
    mp13 = MaterialProperty((T**2 + 1) / (T + 2))
    print(f"Test 16 - Rational function at T=3: {mp13.evalf(T, 3)}")

    # Test 17: Handling undefined values (division by zero)
    mp14 = MaterialProperty(1 / (T - 1))
    try:
        res = mp14.evalf(T, 1)
        print(f"Test 17 - Division by zero at T=1: {res}")  # "zoo" in SymPy represents complex infinity
    except ZeroDivisionError:
        print("Test 17 - Division by zero at T=1: ZeroDivisionError raised as expected")

    # Test 18: Property with absolute value
    mp15 = MaterialProperty(sp.Abs(T - 50))
    print(f"Test 18a - Absolute value at T=30: {mp15.evalf(T, 30)}")
    print(f"Test 18b - Absolute value at T=70: {mp15.evalf(T, 70)}")

    # Test 19: Property with floor and ceiling functions
    mp16 = MaterialProperty(sp.floor(T/10) + sp.ceiling(T/10))
    print(f"Test 19 - Floor and ceiling at T=25: {mp16.evalf(T, 25)}")

    # Test 20: Handling complex numbers
    mp17 = MaterialProperty(sp.sqrt(T))
    print(f"Test 20a - Square root at T=4: {mp17.evalf(T, 4)}")
    res = mp17.evalf(T, -1)
    print(f"Test 20b - Square root at T=-1: {res}")
    assert isinstance(res, sp.Expr)
