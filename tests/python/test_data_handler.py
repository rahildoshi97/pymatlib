import pytest
import numpy as np
from pymatlib.core.data_handler import read_data_from_txt

def test_read_data(tmp_path):
    """Test data reading functionality."""
    # Test successful read
    test_file = tmp_path / "test.txt"
    test_file.write_text("""Temperature Property
1.0 10.0
2.0 20.0
3.0 30.0
""")

    temp, prop = read_data_from_txt(str(test_file))
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
        read_data_from_txt(str(invalid_file))

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
        temp, prop = read_data_from_txt(str(wrong_columns))

    with pytest.raises(ValueError, match="Data contains NaN values"):
        temp, prop = read_data_from_txt(str(nan_file))

    with pytest.raises(ValueError, match="Duplicate temperature entries found"):
        temp, prop = read_data_from_txt(str(duplicate_file))
