"""Tests for property processor base functionality."""

import pytest
import numpy as np
import sympy as sp
from pathlib import Path
from unittest.mock import Mock
from pymatlib.parsing.processors.property_processor_base import PropertyProcessorBase
from pymatlib.core.materials import Material
from pymatlib.core.elements import ChemicalElement


class TestPropertyProcessorBase:
    """Test the base property processor functionality."""

    @pytest.fixture
    def processor(self):
        """Create a property processor instance."""
        return PropertyProcessorBase()

    @pytest.fixture
    def sample_material(self):
        """Create a sample material for testing."""
        element = ChemicalElement(
            name="Iron", atomic_number=26, atomic_mass=55.845,
            melting_temperature=1811, boiling_temperature=3134,
            latent_heat_of_fusion=13800, latent_heat_of_vaporization=340000
        )
        return Material(
            name="TestMaterial",
            material_type="pure_metal",
            elements=[element],
            composition=[1.0],
            melting_temperature=sp.Float(1811),
            boiling_temperature=sp.Float(3134)
        )

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.processed_properties == set()
        assert processor.base_dir is None
        assert processor.visualizer is None

    def test_set_processing_context(self, processor):
        """Test setting processing context."""
        base_dir = Path("/test/path")
        visualizer = Mock()
        processed_props = {'prop1', 'prop2'}
        processor.set_processing_context(base_dir, visualizer, processed_props)
        assert processor.base_dir == base_dir
        assert processor.visualizer == visualizer
        assert processor.processed_properties == processed_props

    def test_finalize_with_data_arrays_valid(self, processor, sample_material):
        """Test finalization with valid data arrays."""
        temp_array = np.array([300, 400, 500])
        prop_array = np.array([100, 150, 200])
        T = sp.Symbol('T')
        config = {'bounds': ['constant', 'constant']}

        result = processor.finalize_with_data_arrays(
            material=sample_material,
            prop_name='test_prop',
            temp_array=temp_array,
            prop_array=prop_array,
            T=T,
            config=config,
            prop_type='KEY_VAL'
        )
        assert result is False  # Symbolic case
        assert hasattr(sample_material, 'test_prop')
        assert 'test_prop' in processor.processed_properties

    def test_finalize_with_data_arrays_numeric_temperature(self, processor, sample_material):
        """Test finalization with numeric temperature."""
        temp_array = np.array([300, 400, 500])
        prop_array = np.array([100, 150, 200])
        config = {'bounds': ['constant', 'constant']}

        result = processor.finalize_with_data_arrays(
            material=sample_material,
            prop_name='test_prop',
            temp_array=temp_array,
            prop_array=prop_array,
            T=400.0,  # Numeric temperature
            config=config,
            prop_type='KEY_VAL'
        )
        assert result is True  # Numeric case completed
        assert hasattr(sample_material, 'test_prop')
        assert isinstance(getattr(sample_material, 'test_prop'), sp.Float)

    def test_finalize_with_data_arrays_invalid_input(self, processor, sample_material):
        """Test error handling for invalid input arrays."""
        T = sp.Symbol('T')
        config = {'bounds': ['constant', 'constant']}
        # Test None arrays
        with pytest.raises(ValueError, match="cannot be None"):
            processor.finalize_with_data_arrays(
                material=sample_material,
                prop_name='test_prop',
                temp_array=None,
                prop_array=np.array([100, 150, 200]),
                T=T,
                config=config,
                prop_type='KEY_VAL'
            )

    def test_finalize_with_data_arrays_mismatched_lengths(self, processor, sample_material):
        """Test error handling for mismatched array lengths."""
        temp_array = np.array([300, 400])
        prop_array = np.array([100, 150, 200])  # Different length
        T = sp.Symbol('T')
        config = {'bounds': ['constant', 'constant']}
        with pytest.raises(ValueError, match="same length"):
            processor.finalize_with_data_arrays(
                material=sample_material,
                prop_name='test_prop',
                temp_array=temp_array,
                prop_array=prop_array,
                T=T,
                config=config,
                prop_type='KEY_VAL'
            )

    def test_finalize_with_piecewise_function(self, processor, sample_material):
        """Test finalization with piecewise function."""
        T = sp.Symbol('T')
        piecewise_func = sp.Piecewise((100, T < 400), (200, True))
        config = {'bounds': ['constant', 'constant']}
        result = processor.finalize_with_piecewise_function(
            material=sample_material,
            prop_name='test_prop',
            piecewise_func=piecewise_func,
            T=T,
            config=config,
            prop_type='PIECEWISE_EQUATION'
        )
        assert result is False  # Symbolic case
        assert hasattr(sample_material, 'test_prop')
        assert getattr(sample_material, 'test_prop') == piecewise_func

    def test_reset_processing_state(self, processor):
        """Test resetting processing state."""
        processor.processed_properties.add('test_prop')
        assert len(processor.processed_properties) == 1
        processor.reset_processing_state()
        assert len(processor.processed_properties) == 0

    def test_get_processed_properties(self, processor):
        """Test getting processed properties."""
        processor.processed_properties.add('prop1')
        processor.processed_properties.add('prop2')
        props = processor.get_processed_properties()
        assert props == {'prop1', 'prop2'}
        # Should return a copy, not the original set
        props.add('prop3')
        assert 'prop3' not in processor.processed_properties

    def test_is_property_processed(self, processor):
        """Test checking if property is processed."""
        assert not processor.is_property_processed('test_prop')
        processor.processed_properties.add('test_prop')
        assert processor.is_property_processed('test_prop')

    def test_set_visualizer(self, processor):
        """Test setting visualizer."""
        visualizer = Mock()
        processor.set_visualizer(visualizer)
        assert processor.visualizer == visualizer
