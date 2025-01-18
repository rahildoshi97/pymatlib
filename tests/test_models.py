import pytest
import numpy as np
import sympy as sp
from pymatlib.core.models import (
    wrapper,
    material_property_wrapper,
    density_by_thermal_expansion,
    thermal_diffusivity_by_heat_conductivity
)
from pymatlib.core.typedefs import MaterialProperty


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

def test_density_by_thermal_expansion1():
    """Test calculating density with various inputs."""

    # Test with float inputs
    T = 1400.0
    T_ref = 1000.0
    rho_ref = 8000.0
    alpha = 1e-5

    result = density_by_thermal_expansion(T, T_ref, rho_ref, alpha)
    assert np.isclose(result.expr, 7904.762911)  # Ensure this expected value is correct

    # Test with symbolic temperature input
    T_symbolic = sp.Symbol('T')
    result_sym = density_by_thermal_expansion(T_symbolic, T_ref, rho_ref, alpha)
    assert isinstance(result_sym, MaterialProperty)
    assert np.isclose(result_sym.evalf(T_symbolic, T), 7904.76)

    # Test with numpy array for temperature input
    T_array = np.array([1300.0, 1400.0, 1500.0])
    with pytest.raises(TypeError):
        density_by_thermal_expansion(T_array, T_ref, rho_ref, alpha)  # Invalid temperature type

    # Test edge cases with negative values
    with pytest.raises(ValueError):
        density_by_thermal_expansion(-100.0, T_ref, rho_ref, alpha)  # Invalid temperature

    with pytest.raises(ValueError):
        density_by_thermal_expansion(T, T_ref, -8000.0, alpha)  # Invalid density

    with pytest.raises(ValueError):
        density_by_thermal_expansion(T, T_ref, rho_ref, -1e-4)  # Invalid thermal expansion coefficient

def test_thermal_diffusivity_by_heat_conductivity1():
    # Test calculating thermal diffusivity with float heat conductivity, density, and heat capacity inputs
    k = 50.0
    rho = 8000.0
    c_p = 500.0
    result = thermal_diffusivity_by_heat_conductivity(k, rho, c_p)
    assert np.isclose(result.expr, 1.25e-5)

    # Test calculating thermal diffusivity with symbolic heat conductivity, density, and heat capacity inputs
    k = sp.Symbol('k')
    rho = sp.Symbol('rho')
    c_p = sp.Symbol('c_p')
    result = thermal_diffusivity_by_heat_conductivity(k, rho, c_p)
    assert isinstance(result, MaterialProperty)
    assert result.expr == k / (rho * c_p)

def test_density_by_thermal_expansion_invalid_inputs():
    # Test calculating density with invalid temperature or thermal expansion coefficient inputs
    with pytest.raises(ValueError):
        density_by_thermal_expansion(-100.0, 1000.0, 8000.0, 1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(1000.0, -1000.0, 8000.0, 1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(1000.0, 1000.0, -8000.0, 1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(1000.0, 1000.0, 8000.0, -1e-4)
    with pytest.raises(TypeError):
        density_by_thermal_expansion(np.array([-100.0, 1000.0]), 1000.0, 8000.0, 1e-5)

def test_thermal_diffusivity_invalid_inputs():
    # Test calculating thermal diffusivity with invalid heat conductivity, density, or heat capacity inputs
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(-10.0, 8000.0, 500.0)  # Negative heat conductivity
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(50.0, -8000.0, 500.0)  # Negative density
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(50.0, 8000.0, -500.0)  # Negative heat capacity
