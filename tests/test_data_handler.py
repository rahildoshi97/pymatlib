import pytest
import numpy as np
from pymatlib.core.data_handler import (
    read_data, celsius_to_kelvin, fahrenheit_to_kelvin,
    thousand_times, check_equidistant
)

# Test temperature conversion functions
def test_celsius_to_kelvin():
    """Test celsius to kelvin conversion."""
    assert celsius_to_kelvin(0) == 273.15
    assert celsius_to_kelvin(100) == 373.15
    assert np.allclose(celsius_to_kelvin(np.array([0, 100])), np.array([273.15, 373.15]))

def test_fahrenheit_to_kelvin():
    """Test fahrenheit to kelvin conversion."""
    assert np.isclose(fahrenheit_to_kelvin(32), 273.15)
    assert np.isclose(fahrenheit_to_kelvin(212), 373.15)
    assert np.allclose(fahrenheit_to_kelvin(np.array([32, 212])), np.array([273.15, 373.15]))

def test_thousand_times():
    """Test multiplication by 1000."""
    assert thousand_times(1) == 1000
    assert thousand_times(0.001) == 1
    assert np.allclose(thousand_times(np.array([1, 0.001])), np.array([1000, 1]))

def test_check_equidistant():
    """Test equidistant temperature checking."""
    # Test equidistant values
    assert check_equidistant(np.array([1, 2, 3, 4])) == 1.0
    # Test non-equidistant values
    assert check_equidistant(np.array([1, 2, 4, 8])) is False
    # Test single value
    assert check_equidistant(np.array([1])) is False
    # Test empty array
    assert check_equidistant(np.array([])) is False
    # Test with tolerance
    assert np.isclose(check_equidistant(np.array([1, 2.001, 3, 4]), tolerance=0.002), 1.0, atol=0.002)

def test_read_data(tmp_path):
    """Test data reading functionality."""
    # Test successful read
    test_file = tmp_path / "test.txt"
    test_file.write_text("""Temperature Property
1.0 10.0
2.0 20.0
3.0 30.0
""")

    temp, prop = read_data(str(test_file))
    assert np.allclose(temp, [1.0, 2.0, 3.0])
    assert np.allclose(prop, [10.0, 20.0, 30.0])

    # Test file with invalid data - should raise ValueError
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("""Temperature Property
1.0 10.0
NaN 20.0
3.0 30.0
""")

    with pytest.raises(ValueError, match="Data contains NaN values in rows: 2"):
        read_data(str(invalid_file))

def test_read_data_errors(tmp_path):
    """Test error handling in read_data function."""
    # Test file with wrong number of columns
    wrong_columns = tmp_path / "wrong_columns.txt"
    wrong_columns.write_text("""Temperature Property Extra
1.0 10.0 100.0
2.0 20.0 200.0
""")

    # Test file with NaN values
    nan_file = tmp_path / "nan_values.txt"
    nan_file.write_text("""Temperature Property
1.0 10.0
NaN 20.0
3.0 30.0
""")

    # Test file with duplicate temperatures
    duplicate_file = tmp_path / "duplicates.txt"
    duplicate_file.write_text("""Temperature Property
1.0 10.0
1.0 20.0
3.0 30.0
""")

    # Test each error case
    with pytest.raises(ValueError, match="Data should have exactly two columns"):
        temp, prop = read_data(str(wrong_columns))

    with pytest.raises(ValueError, match="Data contains NaN values"):
        temp, prop = read_data(str(nan_file))

    with pytest.raises(ValueError, match="Duplicate temperature entries found"):
        temp, prop = read_data(str(duplicate_file))
