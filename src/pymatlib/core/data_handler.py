import os
import logging
import numpy as np
from typing import Union, Tuple, Dict
from matplotlib import pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def read_data_from_file(file_config: Dict, header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from a file with robust missing value handling.
    Args:
        file_config (Dict): Dictionary containing file configuration with keys:
            - file_path: Path to data file
            - temperature_header: Temperature column name/index
            - value_header: Property column name/index
        header (bool): Indicates if the file contains a header row
    Returns:
        Tuple[np.ndarray, np.ndarray]: Temperature and property arrays
    Raises:
        ValueError: If data validation fails or missing values cannot be handled
        FileNotFoundError: If the specified file doesn't exist
    """
    # Extract configuration
    file_path = file_config['file_path']
    temp_col = file_config['temperature_header']
    prop_col = file_config['value_header']
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    logger.info(f"Reading data from file: {file_path}")
    NA_VALUES = ('', ' ', '  ', '   ', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', 'NA')
    try:
        # Read file based on extension with comprehensive missing value handling
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=0 if header else None, na_values=NA_VALUES)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=0 if header else None, na_values=NA_VALUES)
        else:
            # Handle text files
            return _handle_text_file(file_path, header, temp_col, prop_col)
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")
    # Extract data from DataFrame with missing value handling
    temp, prop = _extract_dataframe_columns(df, temp_col, prop_col, file_path)
    # Validate and clean the data
    temp, prop = _validate_and_clean_data(temp, prop, file_path)
    return temp, prop


def _handle_text_file(file_path: str, header: bool, temp_col: Union[str, int], prop_col: Union[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Handle text file reading with missing value support."""
    NA_VALUES = ('', ' ', '  ', '   ', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', 'NA')
    try:
        if header:
            # Read header to get column names
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                column_names = header_line.split()
            # Read data, treating various strings as NaN
            data = pd.read_csv(file_path, sep=r'\s+', skiprows=1, header=None, na_values=NA_VALUES).values
            # Handle column name/index mapping
            temp_idx = _get_column_index(temp_col, column_names, data.shape[1], "temperature")
            prop_idx = _get_column_index(prop_col, column_names, data.shape[1], "property")
            print(f"Using temperature column: {temp_col} (index {temp_idx}), ")
            print(f"Using property column: {prop_col} (index {prop_idx})")
            # quit()
        else:
            # No header case
            data = pd.read_csv(file_path, sep=r'\s+', header=None, na_values=NA_VALUES).values
            if isinstance(temp_col, str) or isinstance(prop_col, str):
                raise ValueError("Column names specified, but file has no header row")
            temp_idx, prop_idx = temp_col, prop_col
        temp = data[:, temp_idx]
        prop = data[:, prop_idx]
        return temp, prop
    except Exception as e:
        raise ValueError(f"Error processing text file: {str(e)}")


def _extract_dataframe_columns(df: pd.DataFrame, temp_col: Union[str, int], prop_col: Union[str, int],
                               file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract temperature and property columns from DataFrame with robust error handling."""
    # Handle temperature column
    if isinstance(temp_col, str):
        if temp_col not in df.columns:
            available_cols = ', '.join(df.columns.astype(str))
            raise ValueError(f"Temperature column '{temp_col}' not found in file {file_path}. "
                             f"Available columns: {available_cols}")
        temp_series = df[temp_col]
    else:
        if temp_col >= len(df.columns):
            raise ValueError(f"Temperature column index {temp_col} out of bounds "
                             f"(file has {len(df.columns)} columns)")
        temp_series = df.iloc[:, temp_col]
    # Handle property column
    if isinstance(prop_col, str):
        if prop_col not in df.columns:
            available_cols = ', '.join(df.columns.astype(str))
            raise ValueError(f"Property column '{prop_col}' not found in file {file_path}. "
                             f"Available columns: {available_cols}")
        prop_series = df[prop_col]
    else:
        if prop_col >= len(df.columns):
            raise ValueError(f"Property column index {prop_col} out of bounds "
                             f"(file has {len(df.columns)} columns)")
        prop_series = df.iloc[:, prop_col]
    # Convert to numeric, coercing errors to NaN
    temp = pd.to_numeric(temp_series, errors='coerce').to_numpy(dtype=np.float64)
    prop = pd.to_numeric(prop_series, errors='coerce').to_numpy(dtype=np.float64)
    return temp, prop


def _get_column_index(col_identifier: Union[str, int], column_names: list,
                      num_cols: int, col_type: str) -> int:
    """Get column index from name or validate numeric index."""
    if isinstance(col_identifier, str):
        if col_identifier in column_names:
            return column_names.index(col_identifier)
        else:
            raise ValueError(f"{col_type.capitalize()} column '{col_identifier}' not found. "
                             f"Available columns: {', '.join(column_names)}")
    else:
        if col_identifier >= num_cols:
            raise ValueError(f"{col_type.capitalize()} column index {col_identifier} "
                             f"out of bounds (file has {num_cols} columns)")
        return col_identifier


def _validate_and_clean_data(temp: np.ndarray, prop: np.ndarray, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Validate data quality and handle missing values appropriately."""
    # Check for completely empty arrays
    if len(temp) == 0 or len(prop) == 0:
        raise ValueError(f"No valid data found in file: {file_path}")
    # Identify rows with missing values
    temp_nan_mask = np.isnan(temp)
    prop_nan_mask = np.isnan(prop)
    any_nan_mask = temp_nan_mask | prop_nan_mask
    # Report missing values if found
    if np.any(any_nan_mask):
        nan_count = np.sum(any_nan_mask)
        total_count = len(temp)
        nan_percentage = (nan_count / total_count) * 100
        logger.warning(f"Found {nan_count} rows ({nan_percentage:.1f}%) with missing values in {file_path}")
        # If too many missing values, raise an error
        if nan_percentage > 50:
            raise ValueError(f"Too many missing values ({nan_percentage:.1f}%) in file: {file_path}. "
                             "Please clean the data or check file format.")
        # Remove rows with missing values
        valid_mask = ~any_nan_mask
        temp = temp[valid_mask]
        prop = prop[valid_mask]
        logger.info(f"Removed {nan_count} rows with missing values. "
                    f"Remaining data points: {len(temp)}")
    # Final validation
    if len(temp) < 2:
        raise ValueError(f"Insufficient valid data points ({len(temp)}) after cleaning missing values")
    # Check for duplicate temperatures
    unique_temp, counts = np.unique(temp, return_counts=True)
    duplicates = unique_temp[counts > 1]
    if len(duplicates) > 0:
        duplicate_indices = []
        for dup_temp in duplicates:
            indices = np.where(temp == dup_temp)[0]
            duplicate_indices.extend(indices[1:])  # Keep first occurrence
        logger.warning(f"Found {len(duplicate_indices)} duplicate temperature entries. "
                       f"Removing duplicates.")
        # Remove duplicates
        keep_mask = np.ones(len(temp), dtype=bool)
        keep_mask[duplicate_indices] = False
        temp = temp[keep_mask]
        prop = prop[keep_mask]
    # Ensure strictly increasing temperature order
    if not np.all(np.diff(temp) > 0):
        # Sort by temperature
        sort_indices = np.argsort(temp)
        temp = temp[sort_indices]
        prop = prop[sort_indices]
        logger.info("Data sorted by temperature for consistency")
    return temp, prop


def check_monotonicity(arr: np.ndarray, name: str = "Array",
                       mode: str = "strictly_increasing",
                       threshold: float = 1e-10,
                       raise_error: bool = True) -> bool:
    """
    Universal monotonicity checker supporting multiple modes.
    Args:
        arr: numpy array to check
        name: name of array for reporting
        mode: 'strictly_increasing', 'non_decreasing', 'strictly_decreasing', 'non_increasing'
        threshold: minimum required difference between consecutive elements
        raise_error: if True, raises ValueError; if False, returns False on failure
    """
    for i in range(1, len(arr)):
        diff = arr[i] - arr[i-1]
        # Check condition based on mode
        violation = False
        if mode == "strictly_increasing" and diff <= threshold:
            violation = True
        elif mode == "non_decreasing" and diff < -threshold:
            violation = True
        elif mode == "strictly_decreasing" and diff >= -threshold:
            violation = True
        elif mode == "non_increasing" and diff > threshold:
            violation = True
        if violation:
            # Create detailed error message
            start_idx = max(0, i-2)
            end_idx = min(len(arr), i+3)
            context = "\nSurrounding values:\n"
            for j in range(start_idx, end_idx):
                context += f"Index {j}: {arr[j]:.10e}\n"
            error_msg = (
                f"{name} is not {mode.replace('_', ' ')} at index {i}:\n"
                f"Previous value ({i-1}): {arr[i-1]:.10e}\n"
                f"Current value ({i}): {arr[i]:.10e}\n"
                f"Difference: {diff:.10e}\n"
                f"{context}"
            )
            if raise_error:
                raise ValueError(error_msg)
            else:
                print(f"Warning: {error_msg}")
                return False
    logger.info(f"{name} is {mode.replace('_', ' ')}")
    return True
