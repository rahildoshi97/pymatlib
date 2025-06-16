"""Test script for piecewise inversion functionality."""

import sympy as sp
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymatlib.parsing.api import create_material
from pymatlib.algorithms.inversion import create_energy_density_inverse

def test_with_real_material():
    """Test with actual SS304L material properties (linear case only)."""
    print("Testing Linear Piecewise Inverse with Real SS304L Material")
    print("=" * 60)

    T = sp.Symbol('T_K')
    E = sp.Symbol('E')

    # Load material
    current_file = Path(__file__)
    yaml_path = current_file.parent.parent.parent / "src" / "pymatlib" / "data" / "materials" / "alloys" / "SS304L" / "SS304L.yaml"
    ss304l = create_material(yaml_path=yaml_path, T=T, enable_plotting=True)

    print(f"Energy Density Function: {ss304l.energy_density}")

    try:
        # Create inverse
        inverse_energy_density = create_energy_density_inverse(ss304l, 'E')
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
                energy = float(ss304l.energy_density.subs(T, temp))
                # Backward: E -> T
                recovered_temp = float(inverse_energy_density.subs(E, energy))
                error = abs(temp - recovered_temp)
                max_error = max(max_error, error)
                status = "✓" if error < 1e-10 else "!" if error < 1e-6 else "✗"
                print(f"{status} T={temp:6.1f}K -> E={energy:12.2e} -> T={recovered_temp:6.1f}K, Error={error:.2e}")
            except Exception as e:
                print(f"✗ Error at T={temp}K: {e}")

        print(f"\nMaximum error: {max_error:.2e}")

    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    test_with_real_material()
