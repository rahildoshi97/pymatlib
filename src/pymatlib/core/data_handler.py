import os
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Any
from matplotlib import pyplot as plt
import pandas as pd


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


def read_data_from_txt(file_path: str, header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from a txt file.

    Args:
        file_path (str): The path to the txt file.
        header (bool): Indicates if the file contains a header row.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and property arrays.

    Raises:
        ValueError: If:
            - Data has incorrect number of columns
            - Data contains NaN values
            - Data contains duplicate temperature entries
    """
    print(f"Reading data from txt file: {file_path}")
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

def read_data_from_excel(file_path: str, temp_col: str, prop_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from specific columns in an Excel file.

    Args:
        file_path (str): Path to the Excel file
        temp_col (str): Column name/letter for temperature data
        prop_col (str): Column name/letter for property data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and property arrays

    Raises:
        ValueError: If:
            - Required columns are not found
            - Data contains NaN values
            - Data contains duplicate temperature entries
    """
    print(f"Reading data from Excel file: {file_path}")

    # Read specific columns from Excel
    df = pd.read_excel(file_path)

    # Convert to numpy arrays
    temp = df[temp_col].to_numpy()
    prop = df[prop_col].to_numpy()

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


def read_data_from_file1(file_config: Union[str, Dict], header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from a file based on configuration.

    Args:
        file_config: Either a path string or a dictionary containing file configuration
            If dictionary, required keys:
                - file: Path to data file
                - temp_col: Temperature column name/index
                - prop_col: Property column name/index
            If string, treated as direct file path
        header (bool): Indicates if the file contains a header row.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and property arrays

    Raises:
        ValueError: If:
            - For txt files: Data has incorrect number of columns (must be exactly 2)
            - Data contains NaN values
            - Data contains duplicate temperature entries
    """
    # Handle string (direct path) or dictionary configuration
    if isinstance(file_config, str):
        #print('string')
        file_path = file_config
        # For direct file paths, assume first two columns are temperature and property
        temp_col = 0
        prop_col = 1
    else:
        #print('dict')
        file_path = file_config['file']
        temp_col = file_config['temp_col']
        prop_col = file_config['prop_col']
        #temp_col = file_config.get('temp_col', 0)
        #prop_col = file_config.get('prop_col', 1)

    print(f"Reading data from file: {file_path}")

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=0 if header else None)
    elif file_path.endswith('.csv'):
        # Use pandas read_csv for CSV files
        df = pd.read_csv(file_path, header=0 if header else None)
    else:
        # For txt files, assume columns are space/tab separated
        data = np.loadtxt(file_path, dtype=float, skiprows=1 if header else 0)

        # Check for correct dimensions - only for txt files when using direct path
        if isinstance(file_config, str) and (data.ndim != 2 or data.shape[1] != 2):
            raise ValueError("Data should have exactly two columns")
        elif data.ndim != 2:
            raise ValueError("Data should be two-dimensional")

        # Handle both column name (which would be an index for txt files) and column index
        if isinstance(temp_col, int):
            if temp_col >= data.shape[1]:
                raise ValueError(f"Temperature column index {temp_col} out of bounds (file has {data.shape[1]} columns)")
            temp = data[:, temp_col]
        else:
            temp = data[:, 0]  # Default to first column

        if isinstance(prop_col, int):
            if prop_col >= data.shape[1]:
                raise ValueError(f"Property column index {prop_col} out of bounds (file has {data.shape[1]} columns)")
            prop = data[:, prop_col]
        else:
            prop = data[:, 1]  # Default to second column

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

    # Process pandas DataFrame (for both Excel and CSV)
    # Handle both column name (string) and column index (integer)
    if isinstance(temp_col, str):
        if temp_col not in df.columns:
            raise ValueError(f"Temperature column '{temp_col}' not found in file")
        temp = df[temp_col].to_numpy(dtype=np.float64)
    else:
        if temp_col >= len(df.columns):
            raise ValueError(f"Temperature column index {temp_col} out of bounds (file has {len(df.columns)} columns)")
        temp = df.iloc[:, temp_col].to_numpy(dtype=np.float64)

    if isinstance(prop_col, str):
        if prop_col not in df.columns:
            raise ValueError(f"Property column '{prop_col}' not found in file")
        prop = df[prop_col].to_numpy(dtype=np.float64)
    else:
        if prop_col >= len(df.columns):
            raise ValueError(f"Property column index {prop_col} out of bounds (file has {len(df.columns)} columns)")
        prop = df.iloc[:, prop_col].to_numpy(dtype=np.float64)

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


def read_data_from_file(file_config: Union[str, Dict], header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from a file based on configuration.

    Args:
        file_config: Either a path string or a dictionary containing file configuration
            If string (direct path):
                - File must have exactly 2 columns
                - First column is temperature, second is property
            If dictionary:
                - file: Path to data file
                - temp_col: Temperature column name/index
                - prop_col: Property column name/index
        header (bool): Indicates if the file contains a header row.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and property arrays

    Raises:
        ValueError: If:
            - For direct path: Data doesn't have exactly two columns
            - For dictionary config: Specified column names don't match headers
            - Data contains NaN values
            - Data contains duplicate temperature entries
    """
    # Handle string (direct path) or dictionary configuration
    if isinstance(file_config, str):
        file_path = file_config
        direct_path = True
        # For direct file paths, assume first two columns are temperature and property
        temp_col = 0
        prop_col = 1
    else:
        file_path = file_config['file']
        direct_path = False
        temp_col = file_config['temp_col']
        prop_col = file_config['prop_col']

    print(f"Reading data from file: {file_path}")

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=0 if header else None)
    elif file_path.endswith('.csv'):
        # Use pandas read_csv for CSV files
        df = pd.read_csv(file_path, header=0 if header else None)
    else:
        # For txt files
        if header:
            # Read the header line to get column names
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                column_names = header_line.split()

            # Now read the data
            data = np.loadtxt(file_path, dtype=float, skiprows=1)

            # Direct path case - check for exactly 2 columns
            if direct_path:
                if data.shape[1] != 2:
                    raise ValueError(f"Data should have exactly two columns, but found {data.shape[1]} columns")
                temp = data[:, 0]
                prop = data[:, 1]
            # Dictionary case - match column names
            else:
                # Handle temperature column
                if isinstance(temp_col, str):
                    if temp_col in column_names:
                        temp_idx = column_names.index(temp_col)
                    else:
                        raise ValueError(f"Temperature column '{temp_col}' not found in file. "
                                         f"Available columns: {', '.join(column_names)}")
                else:
                    if temp_col >= data.shape[1]:
                        raise ValueError(f"Temperature column index {temp_col} out of bounds (file has {data.shape[1]} columns)")
                    temp_idx = temp_col

                # Handle property column
                if isinstance(prop_col, str):
                    if prop_col in column_names:
                        prop_idx = column_names.index(prop_col)
                    else:
                        raise ValueError(f"Property column '{prop_col}' not found in file. "
                                         f"Available columns: {', '.join(column_names)}")
                else:
                    if prop_col >= data.shape[1]:
                        raise ValueError(f"Property column index {prop_col} out of bounds (file has {data.shape[1]} columns)")
                    prop_idx = prop_col

                temp = data[:, temp_idx]
                prop = data[:, prop_idx]
        else:
            # No header
            data = np.loadtxt(file_path, dtype=float, skiprows=0)

            # Direct path case - check for exactly 2 columns
            if direct_path:
                if data.shape[1] != 2:
                    raise ValueError(f"Data should have exactly two columns, but found {data.shape[1]} columns")
                temp = data[:, 0]
                prop = data[:, 1]
            # Dictionary case - use column indices
            else:
                if isinstance(temp_col, str):
                    raise ValueError(f"Column name '{temp_col}' specified, but file has no header row")
                if isinstance(prop_col, str):
                    raise ValueError(f"Column name '{prop_col}' specified, but file has no header row")

                if temp_col >= data.shape[1]:
                    raise ValueError(f"Temperature column index {temp_col} out of bounds (file has {data.shape[1]} columns)")
                if prop_col >= data.shape[1]:
                    raise ValueError(f"Property column index {prop_col} out of bounds (file has {data.shape[1]} columns)")

                temp = data[:, temp_col]
                prop = data[:, prop_col]

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

    # Process pandas DataFrame (for both Excel and CSV)
    # Handle both column name (string) and column index (integer)
    if isinstance(temp_col, str):
        if temp_col in df.columns:
            temp = df[temp_col].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Temperature column '{temp_col}' not found in file. "
                             f"Available columns: {', '.join(df.columns)}")
    else:
        if temp_col >= len(df.columns):
            raise ValueError(f"Temperature column index {temp_col} out of bounds (file has {len(df.columns)} columns)")
        temp = df.iloc[:, temp_col].to_numpy(dtype=np.float64)

    if isinstance(prop_col, str):
        if prop_col in df.columns:
            prop = df[prop_col].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Property column '{prop_col}' not found in file. "
                             f"Available columns: {', '.join(df.columns)}")
    else:
        if prop_col >= len(df.columns):
            raise ValueError(f"Property column index {prop_col} out of bounds (file has {len(df.columns)} columns)")
        prop = df.iloc[:, prop_col].to_numpy(dtype=np.float64)

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

'''def check_equidistant(temp: np.ndarray, tolerance: float = 1.0e-3) -> Union[float, bool]:
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

    return False'''


# Moved from interpolators.py to data_handler.py
def check_equidistant(temp: np.ndarray, rtol: float = 1e-5, atol: float = 1e-10) -> float:
    """
    Tests if the temperature values are equidistant.

    :param temp: Array of temperature values.
    :param rtol: Relative tolerance for comparison.
    :param atol: Absolute tolerance for comparison.
    :return: The common difference if equidistant, otherwise 0.
    """
    if len(temp) < 2:
        raise ValueError(f"Array has length < 2")

    temperature_diffs = np.diff(temp)
    if np.allclose(temperature_diffs, temperature_diffs[0], rtol=rtol, atol=atol):
        return float(temperature_diffs[0])
    return 0.0


def check_strictly_increasing(arr, name="Array", threshold=1e-10, raise_error=True):
    """
    Check if array is strictly monotonically increasing.

    Args:
        arr: numpy array to check
        name: name of array for reporting
        threshold: minimum required difference between consecutive elements
        raise_error: if True, raises ValueError; if False, returns False on failure

    Returns:
        bool: True if array is strictly increasing, False otherwise (if raise_error=False)

    Raises:
        ValueError: If array is not strictly increasing and raise_error=True
    """
    for i in range(1, len(arr)):
        diff = arr[i] - arr[i-1]
        if diff <= threshold:
            # Prepare error message with context
            start_idx = max(0, i-2)
            end_idx = min(len(arr), i+3)
            context = "\nSurrounding values:\n"
            for j in range(start_idx, end_idx):
                context += f"Index {j}: {arr[j]:.10e}\n"

            error_msg = (
                f"{name} is not strictly increasing at index {i}:\n"
                f"Previous value ({i-1}): {arr[i-1]:.10e}\n"
                f"Current value  ({i}): {arr[i]:.10e}\n"
                f"Difference: {diff:.10e}\n"
                f"{context}"
            )

            if raise_error:
                raise ValueError(error_msg)
            else:
                print(f"Warning: {error_msg}")
                return False

    print(f"{name} is strictly monotonically increasing")
    return True


def find_min_max_temperature(temperatures_input) -> tuple:
    """
    Find the minimum and maximum temperature from either a text file or a NumPy array.

    Args:
        temperatures_input (str or np.ndarray):
            - If str, it is the path to the text file.
            - If np.ndarray, it is the array of temperatures.

    Returns:
        tuple: A tuple containing (min_temperature, max_temperature).
    """
    try:
        # Case 1: Input is a file path
        if isinstance(temperatures_input, str):
            temperatures = []
            with open(temperatures_input, 'r') as file:
                for line in file:
                    # Skip empty lines or non-data lines
                    if not line.strip() or not line[0].isdigit():
                        continue

                    # Split the line and extract the first column (temperature)
                    parts = line.split()
                    temperature = float(parts[0])
                    temperatures.append(temperature)

        # Case 2: Input is a NumPy array
        elif isinstance(temperatures_input, np.ndarray):
            temperatures = temperatures_input.tolist()

        else:
            raise TypeError("Input must be either a file path (str) or a numpy array.")

        # Get min and max temperatures
        min_temperature = min(temperatures)
        max_temperature = max(temperatures)

        return min_temperature, max_temperature

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{temperatures_input}' does not exist.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the input: {e}")


def plot_arrays(x_arr: np.ndarray, y_arr: np.ndarray, x_label: str = None, y_label: str = None) -> None:
    # Set labels and titles
    x_label = x_label or "x-axis"
    y_label = y_label or "y-axis"

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_arr, y_arr, 'b-', linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')
    plt.grid(True)

    # Define filename and directory
    filename = f"{y_label.replace('/', '_')}_vs_{x_label.replace('/', '_')}.png"
    directory = "plots"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    filepath = os.path.join(directory, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot saved as {filepath}")


if __name__ == '__main__':
    # Example usage:
    # 1. Using a file path
    base_dir = Path(__file__).parent  # Directory of the current file
    _file_path = str( base_dir / '..' / 'data' / 'alloys' / 'SS316L' / 'density_temperature.txt' )
    min_temp, max_temp = find_min_max_temperature(_file_path)
    print(f"Minimum Temperature from file: {min_temp}")
    print(f"Maximum Temperature from file: {max_temp}")

    # 2. Using a numpy array
    temperature_array = np.array([3300, 500, 800, 1000, 1500])
    min_temp, max_temp = find_min_max_temperature(temperature_array)
    print(f"Minimum Temperature from array: {min_temp}")
    print(f"Maximum Temperature from array: {max_temp}")
