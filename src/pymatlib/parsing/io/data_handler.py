import os
import logging
import numpy as np
from typing import Union, Tuple, Dict
import pandas as pd

from pymatlib.parsing.config.yaml_keys import FILE_PATH_KEY, TEMPERATURE_HEADER_KEY, VALUE_HEADER_KEY
from pymatlib.data.constants import ProcessingConstants, FileConstants

logger = logging.getLogger(__name__)

def read_data_from_file(file_config: Dict, header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads temperature and property data from a file with missing value handling.
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
    file_path: str = file_config[FILE_PATH_KEY]
    temp_col: str = file_config[TEMPERATURE_HEADER_KEY]
    prop_col: str = file_config[VALUE_HEADER_KEY]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    _, file_extension = os.path.splitext(file_path)
    if file_extension not in FileConstants.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: '{file_extension}'. "
                         f"Supported types are: {FileConstants.SUPPORTED_EXTENSIONS}")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > FileConstants.MAX_FILE_SIZE_MB:
        raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds the maximum limit "
                         f"of {FileConstants.MAX_FILE_SIZE_MB} MB.")
    NA_VALUES = FileConstants.NA_VALUES
    try:
        # Read file based on extension and handle missing values
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(
                file_path,
                header=0 if header else None,
                na_values=NA_VALUES,
                converters={
                    temp_col: lambda x: pd.to_numeric(x, errors='coerce'),
                    prop_col: lambda x: pd.to_numeric(x, errors='coerce')
                } if header else None
            )
        elif file_path.endswith('.csv'):
            df = pd.read_csv(
                file_path,
                header=0 if header else None,
                na_values=NA_VALUES,
                encoding=FileConstants.DEFAULT_ENCODING,
                converters={
                    temp_col: lambda x: pd.to_numeric(x, errors='coerce'),
                    prop_col: lambda x: pd.to_numeric(x, errors='coerce')
                } if header else None
            )
        else:
            return _read_text_file(file_path, header, temp_col, prop_col)
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")
    # Extract data from DataFrame with missing value handling
    temp, prop = _extract_dataframe_columns(df, temp_col, prop_col, file_path)
    temp, prop = _validate_and_clean_data(temp, prop, file_path)
    return temp, prop

def _read_text_file(file_path: str, header: bool, temp_col: Union[str, int], prop_col: Union[str, int]) \
                    -> Tuple[np.ndarray, np.ndarray]:
    """Read text file with missing value support."""
    NA_VALUES = FileConstants.NA_VALUES
    try:
        if header:
            # Read header to get column names
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                column_names = header_line.split()
            # Read data, treating various strings as NaN
            data = pd.read_csv(file_path, sep=r'\\s+', skiprows=1, header=None,
                               na_values=NA_VALUES, encoding=FileConstants.DEFAULT_ENCODING).values
            # Handle column name/index mapping
            temp_idx = _get_column_index(temp_col, column_names, data.shape[1], "temperature")
            prop_idx = _get_column_index(prop_col, column_names, data.shape[1], "property")
        else:
            # No header case
            data = pd.read_csv(file_path, sep=r'\s+', header=None, na_values=NA_VALUES,
                               encoding=FileConstants.DEFAULT_ENCODING).values
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
    try:
        # First, ensure we're working with the actual values, not already-converted data
        if hasattr(temp_series, 'values'):
            temp_values = temp_series.values
        else:
            temp_values = temp_series
        if hasattr(prop_series, 'values'):
            prop_values = prop_series.values
        else:
            prop_values = prop_series
        # Convert to numeric, handling any remaining non-numeric values
        temp_numeric = pd.to_numeric(temp_values, errors='coerce')
        prop_numeric = pd.to_numeric(prop_values, errors='coerce')
        # Explicitly convert to float64 numpy arrays
        temp = np.asarray(temp_numeric, dtype=np.float64)
        prop = np.asarray(prop_numeric, dtype=np.float64)
        # Validate that conversion was successful
        if temp.dtype.kind not in ['f', 'i']:  # not float or int
            raise ValueError(f"Temperature column could not be converted to numeric type. Got dtype: {temp.dtype}")
        if prop.dtype.kind not in ['f', 'i']:  # not float or int
            raise ValueError(f"Property column could not be converted to numeric type. Got dtype: {prop.dtype}")
        # Additional validation: check for successful conversion
        temp_nan_count = np.sum(np.isnan(temp))
        prop_nan_count = np.sum(np.isnan(prop))
        if temp_nan_count > 0:
            logger.warning(f"Temperature column has {temp_nan_count} NaN values after conversion")
        if prop_nan_count > 0:
            logger.warning(f"Property column has {prop_nan_count} NaN values after conversion")
    except Exception as e:
        raise ValueError(f"Failed to convert data to numeric format in {file_path}: {str(e)}") from e
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
        if nan_percentage > ProcessingConstants.MAX_MISSING_VALUE_PERCENTAGE:
            raise ValueError(f"Too many missing values ({nan_percentage:.1f}%) in file: {file_path}. "
                             "Please clean the data or check file format.")
        if nan_percentage > 0:
            logger.info(f"Cleaning data by removing rows with missing values ({nan_percentage:.1f}%)")
        # Remove rows with missing values
        valid_mask = ~any_nan_mask
        temp = temp[valid_mask]
        prop = prop[valid_mask]
        logger.info(f"Removed {nan_count} rows with missing values. "
                    f"Remaining data points: {len(temp)}")
    # Final validation
    if len(temp) < ProcessingConstants.MIN_DATA_POINTS:
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
    return temp, prop
