import sympy as sp
from typing import Union
import logging
from pathlib import Path
from pymatlib.core.yaml_parser.api import create_material_from_yaml

logger = logging.getLogger(__name__)

class PiecewiseInverter:
    """
    Creates inverse functions for linear piecewise functions only.
    Simplified version that supports only degree 1 polynomial.
    """
    def __init__(self, tolerance: float = 1e-12):
        """Initialize the inverter with numerical tolerance."""
        self.tolerance = tolerance

    def create_inverse(self, piecewise_func: Union[sp.Piecewise, sp.Expr],
                       input_symbol: Union[sp.Symbol, sp.Basic],
                       output_symbol: Union[sp.Symbol, sp.Basic]) ->  sp.Piecewise:
        """
        Create the inverse of a linear piecewise function.
        Args:
            piecewise_func: The original piecewise function E = f(T)
            input_symbol: Original input symbol (e.g., T)
            output_symbol: Output symbol for inverse function (e.g., E)
        Returns:
            Inverse piecewise function T = f_inv(E)
        Raises:
            ValueError: If any piece has degree > 1
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
        raise ValueError("Energy density must be a piecewise function")

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

def test_with_real_material():
    """Test with actual SS316L material properties (linear case only)."""
    print("Testing Linear Piecewise Inverse with Real SS316L Material")
    print("=" * 60)

    T = sp.Symbol('T_K')
    E = sp.Symbol('E')

    # Load material
    yaml_path = Path(__file__).parent.parent.parent.parent / "src" / "pymatlib" / "data" / "alloys" / "SS304L" / "SS304L.yaml"
    ss316l = create_material_from_yaml(yaml_path=yaml_path, T=T, enable_plotting=True)

    print(f"Energy Density Function: {ss316l.energy_density}")

    try:
        # Create inverse
        inverse_energy_density = create_energy_density_inverse(ss316l, 'E')
        print(f"Inverse function: {inverse_energy_density}")

        # Test temperatures
        test_temperatures = [
            # Low temperature region
            0, 50, 150, 250, 299,
            # Boundary transitions
            300, 301, 350, 500,
            # Heating phase
            800, 1000, 1200, 1500, 1602.147, 1605, 1606.5, 1667, 1667.5,
            # Phase transition region
            1668, 1668.5, 1669, 1670, 1685, 1700, 1725, 1734, 1734.5, 1735,
            # High temperature liquid phase
            1800, 2000, 2500, 2999,
            # Ultra-high temperature
            3000, 3500, 4000
        ]

        print("\nRound-trip accuracy test:")
        max_error = 0.0

        for temp in test_temperatures:
            try:
                # Forward: T -> E
                energy = float(ss316l.energy_density.subs(T, temp))

                # Backward: E -> T
                recovered_temp = float(inverse_energy_density.subs(E, energy))

                error = abs(temp - recovered_temp)
                max_error = max(max_error, error)

                status = "✓" if error < 1e-10 else "!" if error < 1e-6 else "✗"
                print(f"{status} T={temp:6.1f}K -> E={energy:12.2e} -> T={recovered_temp:6.1f}K, Error={error:.2e}")

            except Exception as e:
                print(f"✗ Error at T={temp}K: {e}")

        print(f"\nMaximum error: {max_error:.2e}")

        if max_error < 1e-10:
            print("✓ EXCELLENT: All tests passed with machine precision")
        elif max_error < 1e-6:
            print("✓ GOOD: All tests passed with high precision")
        else:
            print("⚠ WARNING: Some tests have larger errors")

    except ValueError as e:
        if "degree" in str(e).lower():
            print(f"✗ EXPECTED ERROR: {e}")
            print("This is expected if the material has quadratic energy density.")
            print("The simplified inverter only supports linear piecewise functions.")
        else:
            print(f"✗ UNEXPECTED ERROR: {e}")
    except Exception as e:
        print(f"✗ FAILED: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Run test
    test_with_real_material()
