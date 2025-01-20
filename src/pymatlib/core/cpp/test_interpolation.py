import numpy as np
import pytest
from pymatlib.core.cpp.fast_interpolation import temperature_from_energy_density_array

def test_numpy_implementation():
    # Test data
    temperatures = np.array([3273.15, 3263.15, 3253.15, 3243.15])
    energy_densities = np.array([1.71e10, 1.70e10, 1.69e10, 1.68e10])
    h_in = 1.695e10

    # Test normal interpolation
    result = temperature_from_energy_density_array(temperatures, h_in, energy_densities)
    assert 3243.15 < result < 3273.15
    print(f"NumPy test temperature: {result}")

    # Test boundary values
    result_min = temperature_from_energy_density_array(temperatures, 1.67e10, energy_densities)
    assert abs(result_min - 3243.15) < 1e-6

    result_max = temperature_from_energy_density_array(temperatures, 1.72e10, energy_densities)
    assert abs(result_max - 3273.15) < 1e-6

    # Test error cases
    with pytest.raises(RuntimeError):
        wrong_size = np.array([1.0, 2.0])
        temperature_from_energy_density_array(temperatures, h_in, wrong_size)

if __name__ == "__main__":
    test_numpy_implementation()
    print("All Python tests completed successfully")
