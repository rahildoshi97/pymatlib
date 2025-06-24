import sympy as sp
from typing import Union
import logging

logger = logging.getLogger(__name__)


class PiecewiseInverter:
    """
    Creates inverse functions for linear piecewise functions only.
    Simplified version that supports only degree 1 polynomial.
    """

    def __init__(self, tolerance: float = 1e-12):
        """Initialize the inverter with numerical tolerance."""
        self.tolerance = tolerance

    @staticmethod
    def create_inverse(piecewise_func: Union[sp.Piecewise, sp.Expr],
                       input_symbol: Union[sp.Symbol, sp.Basic],
                       output_symbol: Union[sp.Symbol, sp.Basic],
                       tolerance: float = 1e-12) -> sp.Piecewise:
        """
        Create the inverse of a linear piecewise function.

        This static method provides a convenient interface for creating inverse
        functions without requiring explicit instantiation of PiecewiseInverter.
        Args:
            piecewise_func: The original piecewise function E = f(T)
            input_symbol: Original input symbol (e.g., T)
            output_symbol: Output symbol for inverse function (e.g., E)
            tolerance: Numerical tolerance for inversion stability (default: 1e-12)
        Returns:
            Inverse piecewise function T = f_inv(E)
        Raises:
            ValueError: If any piece has degree > 1
        Examples:
            >>> T = sp.Symbol('T')
            >>> E = sp.Symbol('E')
            >>> piecewise_function = sp.Piecewise((2*T + 100, T < 500), (3*T - 400, True))
            >>> inverse = PiecewiseInverter.create_inverse(piecewise_function, T, E)
        """
        inverter = PiecewiseInverter(tolerance)
        return inverter._create_inverse_impl(piecewise_func, input_symbol, output_symbol)

    def _create_inverse_impl(self, piecewise_func: Union[sp.Piecewise, sp.Expr],
                             input_symbol: Union[sp.Symbol, sp.Basic],
                             output_symbol: Union[sp.Symbol, sp.Basic]) -> sp.Piecewise:
        """
        Internal implementation for creating inverse functions.

        This method contains the core logic for inverting piecewise functions,
        separated from the public static interface for better maintainability.
        Args:
            piecewise_func: The original piecewise function
            input_symbol: Original input symbol
            output_symbol: Output symbol for inverse function
        Returns:
            Inverse piecewise function
        """
        logger.info(f"Creating inverse for linear piecewise function with {len(piecewise_func.args)} pieces")
        # Validate that all pieces are linear
        self._validate_linear_only(piecewise_func, input_symbol)
        # Process each piece
        inverse_conditions = []
        for i, (expr, condition) in enumerate(piecewise_func.args):
            if condition == True:  # Final piece
                inverse_expr = self._invert_linear_expression(expr, input_symbol, output_symbol)
                inverse_conditions.append((inverse_expr, True))
            else:
                # Extract boundary and create inverse
                boundary = self._extract_boundary(condition, input_symbol)
                inverse_expr = self._invert_linear_expression(expr, input_symbol, output_symbol)
                # Calculate energy at boundary for domain condition
                boundary_energy = float(expr.subs(input_symbol, boundary))
                inverse_conditions.append((inverse_expr, output_symbol < boundary_energy))
        return sp.Piecewise(*inverse_conditions)

    @staticmethod
    def _validate_linear_only(piecewise_func: sp.Piecewise, input_symbol: sp.Symbol) -> None:
        """Validate that all pieces are linear (degree <= 1)."""
        for i, (expr, condition) in enumerate(piecewise_func.args):
            degree = sp.degree(expr, input_symbol)
            if degree > 1:
                raise ValueError(f"Piece {i} has degree {degree}. Only linear functions (degree = 1) are supported.")

    @staticmethod
    def _extract_boundary(condition: sp.Basic, symbol: sp.Symbol) -> float:
        """Extract the boundary value from a condition like 'T < 300.0'."""
        if hasattr(condition, 'rhs'):
            return float(condition.rhs)
        elif hasattr(condition, 'args') and len(condition.args) == 2:
            if condition.args[0] == symbol:
                return float(condition.args[1])
            elif condition.args[1] == symbol:
                return float(condition.args[0])
        raise ValueError(f"Cannot extract boundary from condition: {condition}")

    def _invert_linear_expression(self, expr: sp.Expr, input_symbol: sp.Symbol,
                                  output_symbol: sp.Symbol) -> Union[float, sp.Expr]:
        """
        Invert a linear expression: ax + b = y → x = (y - b) / a
        Args:
            expr: Linear expression to invert
            input_symbol: Input variable (x)
            output_symbol: Output variable (y)
        Returns:
            Inverted expression
        """
        degree = sp.degree(expr, input_symbol)
        if degree == 0:
            # Constant function - return the constant as temperature
            return float(expr)
        elif degree == 1:
            # Linear function: ax + b = y → x = (y - b) / a
            coeffs = sp.Poly(expr, input_symbol).all_coeffs()
            a, b = float(coeffs[0]), float(coeffs[1])
            if abs(a) < self.tolerance:
                raise ValueError("Linear coefficient is too small for stable inversion")
            return (output_symbol - b) / a
        else:
            raise ValueError(f"Expression has degree {degree}, only linear expressions are supported")

    @staticmethod
    def create_energy_density_inverse(material, output_symbol_name: str = 'E') -> sp.Piecewise:
        """
        Create inverse function for energy density: T = f_inv(E)
        Args:
            material: Material object with energy_density property
            output_symbol_name: Symbol name for energy density (default: 'E')
        Returns:
            Inverse piecewise function
        Raises:
            ValueError: If energy density is not linear piecewise
        """
        if not hasattr(material, 'energy_density'):
            raise ValueError("Material does not have energy_density property")
        energy_density_func = material.energy_density
        if not isinstance(energy_density_func, sp.Piecewise):
            raise ValueError(f"Energy density must be a piecewise function, found: {type(energy_density_func)}")
        # Extract the temperature symbol
        symbols = energy_density_func.free_symbols
        if len(symbols) != 1:
            raise ValueError(f"Energy density function must have exactly one symbol, found: {symbols}")
        temp_symbol = list(symbols)[0]
        print(f"Using temperature symbol: {temp_symbol}, type: {type(temp_symbol)}")
        energy_symbol = sp.Symbol(output_symbol_name)
        # Create the inverter and generate inverse
        inverter = PiecewiseInverter()
        inverse_func = inverter.create_inverse(energy_density_func, temp_symbol, energy_symbol)
        logger.info(f"Created inverse function: T = f_inv({output_symbol_name})")
        return inverse_func
