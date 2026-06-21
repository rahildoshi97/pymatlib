"""Unit tests for the build cache (materforge.parsing.cache)."""

import pytest
import sympy as sp

from materforge import clear_cache, create_material
from materforge.core.materials import Material
from materforge.parsing import cache
from materforge.parsing.config.material_yaml_parser import MaterialYAMLParser


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Point the cache at a throwaway dir and ensure it is enabled."""
    target = tmp_path / "mfcache"
    monkeypatch.setenv("MATERFORGE_CACHE_DIR", str(target))
    monkeypatch.delenv("MATERFORGE_DISABLE_CACHE", raising=False)
    return target


# --- location & enable/disable resolution ---

def test_cache_dir_env_override(cache_env):
    assert cache.cache_dir() == cache_env


def test_cache_dir_falls_back_to_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("MATERFORGE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache.cache_dir() == tmp_path / "materforge"


def test_is_disabled_reads_env(monkeypatch):
    monkeypatch.setenv("MATERFORGE_DISABLE_CACHE", "1")
    assert cache.is_disabled() is True
    monkeypatch.setenv("MATERFORGE_DISABLE_CACHE", "off")
    assert cache.is_disabled() is False


# --- key computation ---

def test_compute_key_is_stable_and_dependency_sensitive(tmp_path):
    yaml_file = tmp_path / "m.yaml"
    yaml_file.write_bytes(b"name: X\n")
    cfg = {"properties": {}}
    key = cache.compute_key(yaml_file, sp.Symbol("T"), tmp_path, cfg)
    assert key == cache.compute_key(yaml_file, sp.Symbol("T"), tmp_path, cfg)
    assert cache.compute_key(yaml_file, sp.Symbol("u_C"), tmp_path, cfg) != key


def test_compute_key_tracks_yaml_and_data_file_changes(tmp_path):
    yaml_file = tmp_path / "m.yaml"
    yaml_file.write_bytes(b"name: X\n")
    data_file = tmp_path / "table.csv"
    data_file.write_text("1,2\n")
    cfg = {"properties": {"heat_capacity": {"file_path": "table.csv"}}}
    sym = sp.Symbol("T")
    base = cache.compute_key(yaml_file, sym, tmp_path, cfg)
    # editing the referenced data file changes the key
    data_file.write_text("3,4\n")
    assert cache.compute_key(yaml_file, sym, tmp_path, cfg) != base
    # editing the YAML changes the key
    data_file.write_text("1,2\n")
    yaml_file.write_bytes(b"name: Y\n")
    assert cache.compute_key(yaml_file, sym, tmp_path, cfg) != base


# --- store / load round-trip ---

def test_store_load_roundtrip_preserves_expressions(cache_env):
    t = sp.Symbol("T")
    material = Material(name="demo", properties={"density": sp.Float(2700.0), "k": 3 * t + 1})
    cache.store("abc123", material)
    loaded = cache.load("abc123")
    assert loaded is not None
    assert loaded.name == "demo"
    assert loaded.properties["density"] == sp.Float(2700.0)
    assert loaded.properties["k"] == 3 * t + 1


def test_store_load_roundtrip_preserves_sample_data(cache_env):
    import numpy as np
    from materforge.core.materials import PropertySamples

    material = Material(name="demo", properties={"k": sp.Symbol("T")})
    material.sample_data["k"] = PropertySamples(
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), "TABULAR_DATA")
    cache.store("withsamples", material)
    loaded = cache.load("withsamples")
    assert loaded is not None
    assert "k" in loaded.sample_data
    assert list(loaded.sample_data["k"].x) == [1.0, 2.0]
    assert list(loaded.sample_data["k"].y) == [3.0, 4.0]
    assert loaded.sample_data["k"].prop_type == "TABULAR_DATA"


def test_load_miss_returns_none(cache_env):
    assert cache.load("no-such-key") is None


def test_corrupt_entry_falls_back_to_none(cache_env):
    cache_env.mkdir(parents=True, exist_ok=True)
    (cache_env / "broken.mfcache").write_bytes(b"definitely not a pickle")
    assert cache.load("broken") is None


def test_disabled_cache_neither_stores_nor_loads(cache_env, monkeypatch):
    cache.store("k", Material(name="x", properties={}))
    monkeypatch.setenv("MATERFORGE_DISABLE_CACHE", "1")
    assert cache.load("k") is None
    cache.store("k2", Material(name="y", properties={}))
    monkeypatch.delenv("MATERFORGE_DISABLE_CACHE", raising=False)
    assert cache.load("k2") is None  # store was skipped while disabled


def test_clear_removes_all_entries(cache_env):
    cache.store("k1", Material(name="a", properties={}))
    cache.store("k2", Material(name="b", properties={}))
    assert cache.clear() == 2
    assert cache.load("k1") is None


# --- end-to-end through create_material ---

def test_create_material_writes_one_cache_entry(cache_env, aluminum_yaml_path, temp_symbol):
    clear_cache()
    create_material(aluminum_yaml_path, temp_symbol, enable_plotting=False)
    assert len(list(cache_env.glob("*.mfcache"))) == 1


def test_default_build_uses_the_cache(cache_env, aluminum_yaml_path, temp_symbol):
    # The default call (plotting off) must populate the cache: a cache entry is
    # written only on a non-plotting build, so its presence also proves the
    # default does not silently run a plotting build.
    clear_cache()
    create_material(aluminum_yaml_path, temp_symbol)
    assert len(list(cache_env.glob("*.mfcache"))) == 1


def test_second_build_hits_cache_and_skips_regression(
    cache_env, aluminum_yaml_path, temp_symbol, monkeypatch
):
    clear_cache()
    first = create_material(aluminum_yaml_path, temp_symbol, enable_plotting=False)
    # A genuine cache hit must return before any rebuild work happens.
    def fail_if_rebuilt(self, *args, **kwargs):
        raise AssertionError("regression ran again instead of using the cache")
    monkeypatch.setattr(MaterialYAMLParser, "create_material", fail_if_rebuilt)
    second = create_material(aluminum_yaml_path, temp_symbol, enable_plotting=False)
    assert second.property_names() == first.property_names()
    for name in first.property_names():
        assert second.properties[name] == first.properties[name]


def test_use_cache_false_bypasses_cache(cache_env, aluminum_yaml_path, temp_symbol):
    clear_cache()
    create_material(aluminum_yaml_path, temp_symbol, enable_plotting=False, use_cache=False)
    assert list(cache_env.glob("*.mfcache")) == []


def test_plotting_run_does_not_consult_cache(
    cache_env, aluminum_yaml_path, temp_symbol, monkeypatch
):
    clear_cache()
    called = {"load": False}
    monkeypatch.setattr(cache, "load", lambda key: called.__setitem__("load", True) or None)
    # Stub the build so the test exercises the cache guard, not matplotlib.
    monkeypatch.setattr(
        MaterialYAMLParser, "create_material",
        lambda self, **kwargs: Material(name="stub", properties={}),
    )
    # enable_plotting=True must skip the cache entirely (no load, no store)
    create_material(aluminum_yaml_path, temp_symbol, enable_plotting=True)
    assert called["load"] is False
    assert list(cache_env.glob("*.mfcache")) == []
