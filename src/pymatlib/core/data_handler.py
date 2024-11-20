import numpy as np
from typing import Union, Tuple


def print_results(file_path: str, temperatures: np.ndarray, material_property: np.ndarray) -> None:
    """
    Prints the results of the test.

    Parameters:
        file_path (str): The path to the data file.
        temperatures (np.ndarray): Array of temperatures.
        material_property (np.ndarray): Array of material properties.
    """
    print(f"File Path: {file_path}")
    print(f"Temperatures: {temperatures}")
    print(f"Material Properties: {material_property}")
    print("-" * 40)


def read_data(file_path: str, header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from a file.

    Args:
        file_path (str): The path to the data file.
        header (bool): Indicates if the file contains a header row.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and property arrays.

    Raises:
        ValueError: If:
            - Data has incorrect number of columns
            - Data contains NaN values
            - Data contains duplicate temperature entries
    """
    print(f"Reading data from file: {file_path}")
    data = np.loadtxt(file_path, dtype=float, skiprows=1 if header else 0)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Data should have exactly two columns")

    temp = data[:, 0]
    prop = data[:, 1]

    # Check for NaN values
    if np.any(np.isnan(temp)) or np.any(np.isnan(prop)):
        nan_rows = np.where(np.isnan(temp) | np.isnan(prop))[0] + 1
        raise ValueError(f"Data contains NaN values in rows: {', '.join(map(str, nan_rows))}")

    # Check for duplicate temperatures
    unique_temp, counts = np.unique(temp, return_counts=True)
    duplicates = unique_temp[counts > 1]
    if len(duplicates) > 0:
        duplicate_rows = [str(idx + 1) for idx, value in enumerate(temp) if value in duplicates]
        raise ValueError(f"Duplicate temperature entries found in rows: {', '.join(duplicate_rows)}")

    return temp, prop

def celsius_to_kelvin(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts Celsius temperatures to Kelvin.

    Parameters:
        temp (Union[float, np.ndarray]): Temperature(s) in Celsius.

    Returns:
        Union[float, np.ndarray]: Temperature(s) in Kelvin.
    """
    return temp + 273.15

def fahrenheit_to_kelvin(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts Fahrenheit temperatures to Kelvin.

    Parameters:
        temp (Union[float, np.ndarray]): Temperature(s) in Fahrenheit.

    Returns:
        Union[float, np.ndarray]: Temperature(s) in Kelvin.
    """
    return celsius_to_kelvin((temp - 32.0) * (5.0 / 9.0))


def thousand_times(q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Multiplies the input value by 1000.
    Example: J/g-K -> J/kg-K
    Example: g/cm³ -> kg/m³

    Parameters:
        q (Union[float, np.ndarray]): Input value(s).

    Returns:
        Union[float, np.ndarray]: Scaled value(s).
    """
    return q * 1000

def check_equidistant(temp: np.ndarray, tolerance: float = 1.0e-3) -> Union[float, bool]:
    """
    Tests if the temperature values are equidistant.

    Parameters:
        temp (np.ndarray): Array of temperature values.
        tolerance (float): Tolerance for checking equidistant spacing.

    Returns:
        Union[float, bool]: The common difference if equidistant, otherwise False.
    """
    if len(temp) < 2:
        return False

    temperature_diffs = np.diff(temp)
    unique_diffs = np.unique(temperature_diffs)

    if len(unique_diffs) == 1:
        return float(unique_diffs[0])

    # Check if the differences are approximately the same within the tolerance
    if len(unique_diffs) > 1:
        diffs_within_tolerance = np.all(np.abs(unique_diffs - unique_diffs[0]) <= tolerance)
        if diffs_within_tolerance:
            return float(unique_diffs[0])

    return False

def run_tests() -> None:
    """
    Runs predefined tests to validate data reading and processing functions.
    """
    test_cases = [
        {"name": "Normal Case", "file_path": "test_files/test_data_normal.txt"},
        {"name": "Normal Case 1", "file_path": "test_files/test_data_normal1.txt"},
        {"name": "Case with Variable Spacing", "file_path": "test_files/test_data_variable_spacing.txt"},
        {"name": "Case with Sparse Data", "file_path": "test_files/test_data_sparse.txt"},
        {"name": "Case with Missing Values", "file_path": "test_files/test_data_missing_values.txt"},
        {"name": "Case with Descending Data", "file_path": "test_files/test_data_descending.txt"},
        {"name": "Case with Duplicates", "file_path": "test_files/test_data_duplicates.txt"},
        {"name": "Empty File", "file_path": "test_files/test_data_empty.txt"},
        {"name": "File with Only Header", "file_path": "test_files/test_data_header.txt"},
        {"name": "Invalid Data Entries", "file_path": "test_files/test_data_invalid_data.txt"},
        {"name": "Mismatched Columns", "file_path": "test_files/test_data_mismatched_columns.txt"},
        {"name": "Missing Data Entries", "file_path": "test_files/test_data_missing_data.txt"},
        {"name": "Ascending density_temperature values", "file_path": "test_files/test_data_density_temperature_ascending.txt"},
        {"name": "Descending density_temperature values", "file_path": "test_files/test_data_density_temperature_descending.txt"}
    ]

    for case in test_cases:
        print(f"Running test: {case['name']}")

        # Read cleaned data
        cleaned_temp, cleaned_prop = read_data(case['file_path'], header=case.get("header", True))

        if cleaned_temp.size == 0 or cleaned_prop.size == 0:
            print(f"Skipping test: {case['name']} due to data read error.")
            continue

        print(f"Cleaned Temperatures: {cleaned_temp}")

        # Convert temperatures to Kelvin
        temp_kelvin = celsius_to_kelvin(cleaned_temp)
        print(f"Converted Temperatures to Kelvin: {temp_kelvin}")

        # Check if temperatures are equidistant
        dtemp = check_equidistant(temp_kelvin)
        if not dtemp:
            print("Temperatures are not equidistant")
        else:
            print(f"Temperatures are equidistant with increment {dtemp}")

        # Print results
        print_results(case['file_path'], cleaned_temp, cleaned_prop)


if __name__ == "__main__":
    run_tests()
