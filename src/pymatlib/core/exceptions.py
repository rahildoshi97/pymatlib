"""Custom exceptions for pymatlib core functionality."""

class MaterialError(Exception):
    """Base exception for all material-related errors."""
    pass

class MaterialCompositionError(MaterialError):
    """Exception raised when material composition validation fails."""
    pass

class MaterialTemperatureError(MaterialError):
    """Exception raised when material temperature validation fails."""
    pass
