# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the compiled MaterialEvaluator fast-evaluation path."""

import numpy as np
import pytest
import sympy as sp

from materforge import MaterialEvaluator, create_material
from materforge.core.materials import Material


@pytest.fixture
def aluminum(aluminum_yaml_path, temp_symbol):
    """A fully built aluminium material on the shared T symbol."""
    return create_material(aluminum_yaml_path, temp_symbol, enable_plotting=False)


# --- construction / introspection ---------------------------------------

def test_compile_returns_evaluator(aluminum):
    ev = aluminum.compile()
    assert isinstance(ev, MaterialEvaluator)


def test_evaluator_infers_dependency_symbol(aluminum, temp_symbol):
    ev = aluminum.compile()
    assert ev.symbol == temp_symbol


def test_property_names_match_material(aluminum):
    ev = aluminum.compile()
    assert ev.property_names() == aluminum.property_names()
    assert len(ev) == len(aluminum.property_names())


def test_repr_mentions_name_and_count(aluminum):
    text = repr(aluminum.compile())
    assert "Aluminum" in text
    assert "MaterialEvaluator" in text


# --- scalar evaluation agrees with Material.evaluate ---------------------

@pytest.mark.parametrize("value", [400.0, 500.0, 933.47, 1200.0])
def test_scalar_matches_symbolic_evaluate(aluminum, temp_symbol, value):
    ev = aluminum.compile()
    reference = aluminum.evaluate(temp_symbol, value)
    fast = ev(value)
    for name, result in fast.items():
        assert isinstance(result, float)
        expected = float(reference.properties[name])
        assert result == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_scalar_returns_plain_floats(aluminum):
    ev = aluminum.compile()
    for result in ev(500.0).values():
        assert isinstance(result, float)


# --- array evaluation ---------------------------------------------------

def test_array_returns_arrays_shaped_like_input(aluminum):
    ev = aluminum.compile()
    grid = np.linspace(400.0, 800.0, 7)
    out = ev(grid)
    for name, result in out.items():
        assert isinstance(result, np.ndarray)
        assert result.shape == grid.shape


def test_array_values_match_pointwise_scalar(aluminum):
    ev = aluminum.compile()
    grid = np.linspace(400.0, 800.0, 5)
    out = ev(grid)
    for i, value in enumerate(grid):
        scalar = ev(float(value))
        for name in out:
            assert out[name][i] == pytest.approx(scalar[name], rel=1e-9, abs=1e-9)


def test_array_accepts_python_list(aluminum):
    ev = aluminum.compile()
    out = ev([400.0, 600.0, 800.0])
    assert out["density"].shape == (3,)


def test_constant_property_broadcasts_over_array(aluminum):
    ev = aluminum.compile()
    grid = np.linspace(400.0, 800.0, 6)
    out = ev(grid)
    # melting_temperature is a scalar constant - it must fill the whole array.
    melting = out["melting_temperature"]
    assert melting.shape == grid.shape
    assert np.allclose(melting, melting[0])


def test_zero_dim_array_treated_as_scalar(aluminum):
    ev = aluminum.compile()
    result = ev(np.asarray(500.0))
    assert isinstance(result["density"], float)


# --- reuse / caching behaviour ------------------------------------------

def test_evaluator_is_reusable(aluminum):
    ev = aluminum.compile()
    first = ev(500.0)
    second = ev(500.0)
    assert first == second


def test_function_returns_callable_for_one_property(aluminum):
    ev = aluminum.compile()
    density = ev.function("density")
    assert float(np.asarray(density(500.0)).reshape(-1)[0]) == pytest.approx(
        ev(500.0)["density"], rel=1e-9)


def test_function_unknown_name_raises(aluminum):
    ev = aluminum.compile()
    with pytest.raises(KeyError):
        ev.function("not_a_property")


# --- symbol handling ----------------------------------------------------

def test_custom_dependency_symbol_is_inferred(aluminum_yaml_path):
    u_c = sp.Symbol("u_C")
    material = create_material(aluminum_yaml_path, u_c, enable_plotting=False)
    ev = material.compile()
    assert ev.symbol == u_c
    assert ev(500.0)["density"] > 0


def test_explicit_symbol_override_accepted(aluminum, temp_symbol):
    ev = aluminum.compile(symbol=temp_symbol)
    assert ev.symbol == temp_symbol


def test_non_symbol_argument_raises_type_error(aluminum):
    with pytest.raises(TypeError):
        aluminum.compile(symbol="T")


def test_wrong_symbol_raises_value_error(aluminum):
    with pytest.raises(ValueError):
        aluminum.compile(symbol=sp.Symbol("Z"))


def test_multiple_symbols_raise_value_error():
    material = Material(name="Multivariate")
    material.p = sp.Symbol("X") ** 2
    material.q = sp.Symbol("Y") + 1
    with pytest.raises(ValueError):
        material.compile()


# --- all-constant material ----------------------------------------------

def test_all_constant_material_has_no_symbol():
    material = Material(name="Constants")
    material.a = 1.5
    material.b = 7.0
    ev = material.compile()
    assert ev.symbol is None
    assert ev(300.0) == {"a": 1.5, "b": 7.0}


def test_all_constant_material_broadcasts_over_array():
    material = Material(name="Constants")
    material.a = 1.5
    ev = material.compile()
    out = ev(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(out["a"], 1.5)
    assert out["a"].shape == (3,)


# --- bad input ----------------------------------------------------------

def test_none_value_raises(aluminum):
    ev = aluminum.compile()
    with pytest.raises(ValueError):
        ev(None)


def test_non_numeric_value_raises(aluminum):
    ev = aluminum.compile()
    with pytest.raises(ValueError):
        ev("not a number")


# --- steel (FILE_IMPORT + computed properties) --------------------------

def test_steel_scalar_matches_symbolic_evaluate(steel_yaml_path, temp_symbol):
    material = create_material(steel_yaml_path, temp_symbol, enable_plotting=False)
    ev = material.compile()
    reference = material.evaluate(temp_symbol, 800.0)
    fast = ev(800.0)
    for name in reference.property_names() & ev.property_names():
        assert fast[name] == pytest.approx(float(reference.properties[name]),
                                           rel=1e-7, abs=1e-7)
