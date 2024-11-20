import pytest
import numpy as np
import sympy as sp
from pymatlib.core.models import (
    validate_density_parameters,
    validate_thermal_diffusivity_parameters,
    wrapper,
    material_property_wrapper,
    density_by_thermal_expansion,
    thermal_diffusivity_by_heat_conductivity
)
from pymatlib.core.typedefs import MaterialProperty

def test_validate_density_parameters():
    """Test density parameter validation."""
    # Valid parameters
    validate_density_parameters(300.0, 8000.0, 1e-6)

    # Test temperature validation
    with pytest.raises(ValueError, match="Temperature cannot be below absolute zero"):
        validate_density_parameters(-274.0, 8000.0, 1e-6)

    # Test density validation
    with pytest.raises(ValueError, match="Base density must be positive"):
        validate_density_parameters(300.0, -8000.0, 1e-6)

    # Test thermal expansion coefficient validation
    with pytest.raises(ValueError, match="Thermal expansion coefficient must be greater than -1"):
        validate_density_parameters(300.0, 8000.0, -1.5)

def test_validate_thermal_diffusivity_parameters():
    """Test thermal diffusivity parameter validation."""
    # Valid parameters
    validate_thermal_diffusivity_parameters(30.0, 8000.0, 500.0)

    # Test positive value validation
    with pytest.raises(ValueError, match="heat_conductivity must be positive"):
        validate_thermal_diffusivity_parameters(-30.0, 8000.0, 500.0)

    with pytest.raises(ValueError, match="density must be positive"):
        validate_thermal_diffusivity_parameters(30.0, -8000.0, 500.0)

    with pytest.raises(ValueError, match="heat_capacity must be positive"):
        validate_thermal_diffusivity_parameters(30.0, 8000.0, -500.0)

def test_wrapper():
    """Test wrapper function."""
    # Test numeric values
    assert isinstance(wrapper(1.0), sp.Float)
    assert isinstance(wrapper(0.0), sp.Float)

    # Test symbolic expressions
    T = sp.Symbol('T')
    assert wrapper(T) == T

    # Test arrays
    assert isinstance(wrapper([1.0, 2.0]), list)
    assert all(isinstance(x, sp.Float) for x in wrapper([1.0, 2.0]))

    # Test invalid input
    with pytest.raises(ValueError):
        wrapper("invalid")

def test_material_property_wrapper():
    """Test material property wrapper function."""
    # Test numeric value
    result = material_property_wrapper(1.0)
    assert isinstance(result, MaterialProperty)

    # Test symbolic expression
    T = sp.Symbol('T')
    result = material_property_wrapper(T)
    assert isinstance(result, MaterialProperty)

    # Test array
    result = material_property_wrapper([1.0, 2.0])
    assert isinstance(result, MaterialProperty)

def test_density_by_thermal_expansion():
    """Test density calculation by thermal expansion."""
    # Test numeric calculation
    result = density_by_thermal_expansion(1000.0, 293.15, 8000.0, 1e-6)
    assert isinstance(result, MaterialProperty)

    # Test symbolic calculation
    T = sp.Symbol('T')
    result = density_by_thermal_expansion(T, 293.15, 8000.0, 1e-6)
    assert isinstance(result, MaterialProperty)

    # Test error cases
    with pytest.raises(ValueError):
        density_by_thermal_expansion(-274.0, 293.15, 8000.0, 1e-6)

def test_thermal_diffusivity_by_heat_conductivity():
    """Test thermal diffusivity calculation."""
    # Test numeric calculation
    result = thermal_diffusivity_by_heat_conductivity(30.0, 8000.0, 500.0)
    assert isinstance(result, MaterialProperty)

    # Test symbolic calculation
    k = sp.Symbol('k')
    result = thermal_diffusivity_by_heat_conductivity(k, 8000.0, 500.0)
    assert isinstance(result, MaterialProperty)

    # Test error cases
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(-30.0, 8000.0, 500.0)

def test_models():
    """Test all model functions comprehensively."""
    # Test wrapper function edge cases
    def test_wrapper_edge_cases():
        # Test very small numbers (close to tolerance)
        assert wrapper(1e-11) == sp.Float(0.0)
        # Test array with mixed types
        with pytest.raises(ValueError):
            wrapper([1.0, "invalid"])
        # Test unsupported type
        with pytest.raises(ValueError):
            wrapper(complex(1, 1))

    # Test material_property_wrapper validation
    def test_material_property_wrapper_validation():
        # Test with invalid input types
        with pytest.raises(ValueError):
            material_property_wrapper("invalid")
        # Test with None
        with pytest.raises(ValueError):
            material_property_wrapper(None)

    # Test density parameter validation
    def test_density_validation():
        # Test temperature validation at exactly absolute zero
        with pytest.raises(ValueError):
            validate_density_parameters(-273.15, 8000.0, 1e-6)
        # Test array with mixed valid/invalid temperatures
        with pytest.raises(ValueError):
            validate_density_parameters(np.array([100.0, -274.0]), 8000.0, 1e-6)
        # Test zero density
        with pytest.raises(ValueError):
            validate_density_parameters(300.0, 0.0, 1e-6)
        # Test thermal expansion coefficient edge cases
        with pytest.raises(ValueError):
            validate_density_parameters(300.0, 8000.0, -1.0)

    # Test thermal diffusivity parameter validation
    def test_thermal_diffusivity_validation():
        # Test zero values
        with pytest.raises(ValueError):
            validate_thermal_diffusivity_parameters(0.0, 8000.0, 500.0)
        # Test incompatible types
        with pytest.raises(TypeError):
            validate_thermal_diffusivity_parameters(np.array([30.0]), 8000.0, 500.0)

    # Test density calculation with various input combinations
    def test_density_calculations():
        # Test with MaterialProperty as thermal expansion coefficient
        tec = material_property_wrapper(1e-6)
        result = density_by_thermal_expansion(300.0, 293.15, 8000.0, tec)
        assert isinstance(result, MaterialProperty)

        # Test with array temperatures and MaterialProperty TEC
        temps = np.linspace(300, 400, 5)
        with pytest.raises(TypeError):
            density_by_thermal_expansion(temps, 293.15, 8000.0, tec)

    # Test thermal diffusivity calculations
    def test_thermal_diffusivity_calculations():
        # Test with all MaterialProperty inputs
        k = material_property_wrapper(30.0)
        rho = material_property_wrapper(8000.0)
        cp = material_property_wrapper(500.0)
        result = thermal_diffusivity_by_heat_conductivity(k, rho, cp)
        assert isinstance(result, MaterialProperty)

        # Test assignment combination
        assert hasattr(result, 'assignments')
