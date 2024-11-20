import pytest
import numpy as np
from pymatlib.core.alloy import Alloy, AlloyCompositionError, AlloyTemperatureError
from pymatlib.data.element_data import Ti, Al, V

def test_valid_alloy():
    Ti64 = Alloy([Ti, Al, V], [0.90, 0.06, 0.04], 1878, 1928)
    assert np.isclose(Ti64.atomic_number, 0.90 * Ti.atomic_number + 0.06 * Al.atomic_number + 0.04 * V.atomic_number)
    assert Ti64.heat_conductivity is None
    Ti64.heat_conductivity = 34.
    assert Ti64.heat_conductivity == 34.

def test_invalid_composition():
    with pytest.raises(ValueError):
        Alloy([Ti, Al, V], [0.5, 0.5], 1878., 1928.)

def test_empty_composition():
    with pytest.raises(ValueError):
        Alloy([Ti], [], 1878., 1928.)

def test_single_element_alloy():
    single_element_alloy = Alloy([Ti], [1.0], 1878., 1928.)
    assert single_element_alloy.atomic_number == Ti.atomic_number

def test_boundary_temperatures():
    boundary_alloy = Alloy([Ti, Al, V], [0.33, 0.33, 0.34], -273.15, 10000)
    assert boundary_alloy.temperature_solidus == -273.15
    assert boundary_alloy.temperature_liquidus == 10000.

def test_default_properties():
    default_alloy = Alloy([Ti], [1.0], 1878., 1928.)
    assert default_alloy.density is None

def test_composition_sum_error():
    """Test if AlloyCompositionError is raised when composition sum is not 1.0"""
    with pytest.raises(AlloyCompositionError):
        Alloy([Ti, Al, V], [0.5, 0.4, 0.4], 1878., 1928.)

def test_temperature_order_error():
    """Test if AlloyTemperatureError is raised when solidus > liquidus"""
    with pytest.raises(AlloyTemperatureError):
        Alloy([Ti], [1.0], 1928., 1878.)

def test_empty_elements_error():
    """Test if ValueError is raised when elements list is empty"""
    with pytest.raises(ValueError):
        Alloy([], [1.0], 1878., 1928.)

def test_elements_composition_length_mismatch():
    """Test if ValueError is raised when elements and composition lengths don't match"""
    with pytest.raises(ValueError):
        Alloy([Ti, Al], [0.5], 1878., 1928.)

def test_solidification_interval():
    """Test if solidification_interval returns correct temperatures"""
    alloy = Alloy([Ti], [1.0], 1878., 1928.)
    assert alloy.solidification_interval() == (1878., 1928.)

def test_calculated_properties():
    """Test if all calculated properties are set correctly"""
    alloy = Alloy([Ti], [1.0], 1878., 1928.)
    assert hasattr(alloy, 'atomic_number')
    assert hasattr(alloy, 'atomic_mass')
    assert hasattr(alloy, 'temperature_boil')
    assert isinstance(alloy.atomic_number, float)
    assert isinstance(alloy.atomic_mass, float)
    assert isinstance(alloy.temperature_boil, float)

def test_optional_properties_initialization():
    """Test if all optional properties are initialized to None"""
    alloy = Alloy([Ti], [1.0], 1878., 1928.)
    assert all(getattr(alloy, prop) is None for prop in [
        'density', 'dynamic_viscosity', 'heat_capacity', 'heat_conductivity',
        'kinematic_viscosity', 'latent_heat_of_fusion', 'latent_heat_of_vaporization',
        'surface_tension', 'thermal_diffusivity', 'thermal_expansion_coefficient'
    ])

def test_composition_values_range():
    """Test if ValueError is raised when composition values are not between 0 and 1"""
    with pytest.raises(ValueError):
        Alloy([Ti], [-0.5], 1878., 1928.)
    with pytest.raises(ValueError):
        Alloy([Ti], [1.5], 1878., 1928.)

# works only with property descriptors and private field definition in alloy.py
'''def test_invalid_property_assignment():
    Ti64 = Alloy([Ti, Al, V], [0.90, 0.06, 0.04], 1878., 1928.)
    with pytest.raises(TypeError):
        Ti64.heat_conductivity = "invalid_value"'''
