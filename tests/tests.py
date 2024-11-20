import pytest
import numpy as np
import sympy as sp
from pystencils.types.quick import Arr

from src.pymatlib.core.alloy import Alloy
from src.pymatlib.data.alloys.SS316L.SS316L import create_SS316L
from src.pymatlib.core.assignment_converter import type_mapping, assignment_converter
from src.pymatlib.data.element_data import Fe, Cr, Mn, Ni
from src.pymatlib.core.interpolators import interpolate_property, interpolate_lookup, interpolate_equidistant, \
    check_equidistant
from src.pymatlib.core.models import density_by_thermal_expansion, thermal_diffusivity_by_heat_conductivity
from src.pymatlib.core.typedefs import MaterialProperty, Assignment


def test_alloy_creation():
    # Test creating an alloy with valid elemental composition and phase transition temperatures
    alloy = Alloy(elements=[Fe, Cr, Mn, Ni], composition=[0.7, 0.2, 0.05, 0.05], temperature_solidus=1700, temperature_liquidus=1800)
    assert np.allclose(alloy.composition, [0.7, 0.2, 0.05, 0.05])
    assert alloy.temperature_solidus == 1700.
    assert alloy.temperature_liquidus == 1800.

    # Test creating an alloy with invalid elemental composition
    with pytest.raises(ValueError):
        Alloy(elements=[Fe, Cr], composition=[0.6, 0.5], temperature_solidus=1700., temperature_liquidus=1800.)

    # Test creating an alloy with invalid phase transition temperatures
    with pytest.raises(ValueError):
        Alloy(elements=[Fe, Cr, Mn, Ni], composition=[0.7, 0.2, 0.05, 0.05], temperature_solidus=1900., temperature_liquidus=1800.)

def test_create_SS316L():
    # Test creating SS316L alloy with a float temperature input
    alloy = create_SS316L(1400.0)
    assert isinstance(alloy.density, MaterialProperty)
    assert isinstance(alloy.heat_capacity, MaterialProperty)
    assert isinstance(alloy.heat_conductivity, MaterialProperty)
    assert isinstance(alloy.thermal_diffusivity, MaterialProperty)

    # Test creating SS316L alloy with a symbolic temperature input
    T = sp.Symbol('T')
    alloy = create_SS316L(T)
    assert isinstance(alloy.density, MaterialProperty)
    assert isinstance(alloy.heat_capacity, MaterialProperty)
    assert isinstance(alloy.heat_conductivity, MaterialProperty)
    assert isinstance(alloy.thermal_diffusivity, MaterialProperty)

    # Test creating SS316L alloy with different combinations of property calculation methods
    alloy = create_SS316L(1400.0)
    alloy.density = alloy.density.expr
    alloy.heat_capacity = alloy.heat_capacity.evalf(T, 1400.0)
    assert isinstance(alloy.density, sp.Expr)
    assert isinstance(alloy.heat_capacity, float)

def test_interpolate_property():
    # Test interpolating properties with float temperature inputs
    T_symbol = sp.Symbol('T')  # Define a symbolic variable
    T_value = 1400.0
    temp_array = np.array([1300.0, 1400.0, 1500.0])
    prop_array = np.array([100.0, 200.0, 300.0])
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test interpolating properties with symbolic temperature inputs
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test interpolating properties with different combinations of temperature and property arrays
    temp_array = [1300.0, 1400.0, 1500.0]
    prop_array = [100.0, 200.0, 300.0]
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    temp_array = tuple([1300.0, 1400.0, 1500.0])
    prop_array = tuple([100.0, 200.0, 300.0])
    result = interpolate_property(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with ascending and descending order arrays
    temp_array_asc = np.array([1300.0, 1400.0, 1500.0])
    prop_array_asc = np.array([100.0, 200.0, 300.0])
    result_asc = interpolate_property(T_symbol, temp_array_asc, prop_array_asc)
    assert isinstance(result_asc, MaterialProperty)
    assert np.isclose(result_asc.evalf(T_symbol, T_value), 200.0)

    temp_array_desc = np.array([1500.0, 1400.0, 1300.0])
    prop_array_desc = np.array([300.0, 200.0, 100.0])
    result_desc = interpolate_property(T_symbol, temp_array_desc, prop_array_desc)
    assert isinstance(result_desc, MaterialProperty)
    assert np.isclose(result_desc.evalf(T_symbol, T_value), 200.0)

    # Test with different temp_array_limit values
    result_limit_6 = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, temp_array_limit=6)
    assert isinstance(result_limit_6, MaterialProperty)
    assert np.isclose(result_limit_6.evalf(T_symbol, T_value), 200.0)

    result_limit_3 = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, temp_array_limit=3)
    assert isinstance(result_limit_3, MaterialProperty)
    assert np.isclose(result_limit_3.evalf(T_symbol, T_value), 200.0)

    # Test with force_lookup True and False
    result_force_lookup_true = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, force_lookup=True)
    assert isinstance(result_force_lookup_true, MaterialProperty)
    assert np.isclose(result_force_lookup_true.evalf(T_symbol, T_value), 200.0)

    result_force_lookup_false = interpolate_property(T_symbol, temp_array_asc, prop_array_asc, force_lookup=False)
    assert isinstance(result_force_lookup_false, MaterialProperty)
    assert np.isclose(result_force_lookup_false.evalf(T_symbol, T_value), 200.0)

def test_interpolate_lookup():
    # Define a symbolic variable
    T_symbol = sp.Symbol('T')

    # Test lookup interpolation with float temperature inputs
    T_value = 1400.0
    temp_array = np.array([1300.0, 1400.0, 1500.0])
    prop_array = np.array([100.0, 200.0, 300.0])
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test lookup interpolation with symbolic temperature inputs
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with different combinations of temperature and property arrays
    temp_array = [1300.0, 1400.0, 1500.0]
    prop_array = [100.0, 200.0, 300.0]
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    temp_array = tuple([1300.0, 1400.0, 1500.0])
    prop_array = tuple([100.0, 200.0, 300.0])
    result = interpolate_lookup(T_symbol, temp_array, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with ascending and descending order arrays
    temp_array_asc = np.array([1300.0, 1400.0, 1500.0])
    prop_array_asc = np.array([100.0, 200.0, 300.0])
    result_asc = interpolate_lookup(T_symbol, temp_array_asc, prop_array_asc)
    assert isinstance(result_asc, MaterialProperty)
    assert np.isclose(result_asc.evalf(T_symbol, T_value), 200.0)

    temp_array_desc = np.array([1500.0, 1400.0, 1300.0])
    prop_array_desc = np.array([300.0, 200.0, 100.0])
    result_desc = interpolate_lookup(T_symbol, temp_array_desc, prop_array_desc)
    assert isinstance(result_desc, MaterialProperty)
    assert np.isclose(result_desc.evalf(T_symbol, T_value), 200.0)

def test_interpolate_equidistant():
    # Define a symbolic variable
    T_symbol = sp.Symbol('T')

    # Test equidistant interpolation with float temperature inputs
    T_value = 1400.0
    temp_base = 1300.0
    temp_incr = 100.0
    prop_array = np.array([100.0, 200.0, 300.0])
    result = interpolate_equidistant(T_value, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test equidistant interpolation with symbolic temperature inputs
    result = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with different combinations of temperature and property arrays
    temp_base = 1300.0
    temp_incr = 100.0
    prop_array = [100.0, 200.0, 300.0]
    result = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    temp_base = 1300.0
    temp_incr = 100.0
    prop_array = tuple([100.0, 200.0, 300.0])
    result = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array)
    assert isinstance(result, MaterialProperty)
    assert np.isclose(result.evalf(T_symbol, T_value), 200.0)

    # Test with ascending and descending order arrays
    temp_base = 1300.0
    temp_incr = 100.0
    prop_array_asc = np.array([100.0, 200.0, 300.0])
    result_asc = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array_asc)
    assert isinstance(result_asc, MaterialProperty)
    assert np.isclose(result_asc.evalf(T_symbol, T_value), 200.0)

    temp_base = 1300.0
    temp_incr = 100.0
    prop_array_desc = np.array([300.0, 200.0, 100.0])
    result_desc = interpolate_equidistant(T_symbol, temp_base, temp_incr, prop_array_desc)
    assert isinstance(result_desc, MaterialProperty)
    assert np.isclose(result_desc.evalf(T_symbol, T_value), 200.0)

# Define the temp fixture
@pytest.fixture
def temp():
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])

def test_test_equidistant(temp):
    # Test with equidistant temperature array
    assert check_equidistant(temp) == 100.0

    # Test with non-equidistant temperature array
    temp_non_equidistant = np.array([100.0, 150.0, 300.0, 450.0, 500.0])
    assert check_equidistant(temp_non_equidistant) == 0.0

def test_density_by_thermal_expansion():
    # Test calculating density with float temperature and thermal expansion coefficient inputs
    T = 1400.0
    T_ref = 1000.0
    rho_ref = 8000.0
    alpha = 1e-5
    result = density_by_thermal_expansion(T, T_ref, rho_ref, alpha)
    assert np.isclose(result, 7904.76)

    # Test calculating density with symbolic temperature and thermal expansion coefficient inputs
    T = sp.Symbol('T')
    result = density_by_thermal_expansion(T, T_ref, rho_ref, alpha)
    assert isinstance(result, sp.Expr)
    assert np.isclose(result.subs(T, 1400.0), 7904.76)

    # Test calculating density with numpy array temperature and thermal expansion coefficient inputs
    T = np.array([1300.0, 1400.0, 1500.0])
    result = density_by_thermal_expansion(T, T_ref, rho_ref, alpha)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [7928.43, 7904.76, 7881.19])

def test_thermal_diffusivity_by_heat_conductivity():
    # Test calculating thermal diffusivity with float heat conductivity, density, and heat capacity inputs
    k = 50.0
    rho = 8000.0
    c_p = 500.0
    result = thermal_diffusivity_by_heat_conductivity(k, rho, c_p)
    assert np.isclose(result, 1.25e-5)

    # Test calculating thermal diffusivity with symbolic heat conductivity, density, and heat capacity inputs
    k = sp.Symbol('k')
    rho = sp.Symbol('rho')
    c_p = sp.Symbol('c_p')
    result = thermal_diffusivity_by_heat_conductivity(k, rho, c_p)
    assert isinstance(result, sp.Expr)
    assert result == k / (rho * c_p)

    # Test calculating thermal diffusivity with numpy array heat conductivity, density, and heat capacity inputs
    k = np.array([50.0, 60.0, 70.0])
    rho = np.array([8000.0, 8100.0, 8200.0])
    c_p = np.array([500.0, 510.0, 520.0])
    result = thermal_diffusivity_by_heat_conductivity(k, rho, c_p)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [1.25e-5, 1.45243282e-5, 1.64165103e-5])

def test_alloy_single_element():
    # Test creating an alloy with a single element
    alloy = Alloy(elements=[Fe], composition=[1.0], temperature_solidus=1800, temperature_liquidus=1900)
    assert len(alloy.elements) == 1
    assert alloy.composition == [1.0]

def test_alloy_property_modification():
    # Test accessing and modifying individual alloy properties
    alloy = create_SS316L(1400.0)
    assert isinstance(alloy.density, MaterialProperty)
    # Set the density to a MaterialProperty instance with a constant expression
    alloy.density = MaterialProperty(expr=sp.Float(8000.0))
    assert alloy.density.expr == sp.Float(8000.0)

def test_create_SS316L_invalid_temperature():
    # Test creating SS316L alloy with invalid temperature inputs
    with pytest.raises(ValueError):
        create_SS316L(-100.0)  # Negative temperature
    with pytest.raises(ValueError):
        create_SS316L(3500.0)  # Temperature outside valid range

def test_density_by_thermal_expansion_invalid_inputs():
    # Test calculating density with invalid temperature or thermal expansion coefficient inputs
    with pytest.raises(ValueError):
        density_by_thermal_expansion(-100.0, 1000.0, 8000.0, 1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(1000.0, -1000.0, 8000.0, 1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(1000.0, 1000.0, -8000.0, 1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(1000.0, 1000.0, 8000.0, -1e-5)
    with pytest.raises(ValueError):
        density_by_thermal_expansion(np.array([-100.0, 1000.0]), 1000.0, 8000.0, 1e-5)

def test_thermal_diffusivity_invalid_inputs():
    # Test calculating thermal diffusivity with invalid heat conductivity, density, or heat capacity inputs
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(-10.0, 8000.0, 500.0)  # Negative heat conductivity
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(50.0, -8000.0, 500.0)  # Negative density
    with pytest.raises(ValueError):
        thermal_diffusivity_by_heat_conductivity(50.0, 8000.0, -500.0)  # Negative heat capacity

def test_type_mapping_unsupported():
    # Test mapping unsupported type strings
    with pytest.raises(ValueError):
        type_mapping("unsupported_type", 1)

# Additional test cases
def test_type_mapping():
    assert isinstance(type_mapping("double[]", 5), Arr)
    assert isinstance(type_mapping("float[]", 3), Arr)
    assert type_mapping("double", 1) == np.dtype('float64')
    assert type_mapping("float", 1) == np.dtype('float32')
    assert type_mapping("int", 1) == np.dtype('int32')
    assert type_mapping("bool", 1) == np.dtype('bool')

    with pytest.raises(ValueError):
        type_mapping("unsupported_type", 1)

def test_assignment_converter_invalid():
    # Test converting assignments with invalid or missing attributes
    invalid_assignment = Assignment(lhs=None, rhs=None, lhs_type=None)
    with pytest.raises(ValueError, match="Invalid assignment: lhs, rhs, and lhs_type must not be None"):
        assignment_converter([invalid_assignment])
