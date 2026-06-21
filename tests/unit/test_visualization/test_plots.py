"""Unit tests for materforge.visualization.plots (post-build plotting helpers)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sympy as sp  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from materforge import compare_materials, plot_property, plot_residuals  # noqa: E402
from materforge.core.materials import Material, PropertySamples  # noqa: E402

T = sp.Symbol("T")


def _material(name="m"):
    """Material with one linear, data-backed property."""
    mat = Material(name=name)
    setattr(mat, "p", sp.Float(2.0) * T + sp.Float(1.0))
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mat.sample_data["p"] = PropertySamples(x, 2.0 * x + 1.0, "TABULAR_DATA")
    return mat


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotProperty:
    def test_returns_axes(self):
        assert isinstance(plot_property(_material(), "p"), Axes)

    def test_draws_curve_and_data(self):
        ax = plot_property(_material(), "p")
        assert len(ax.lines) >= 1        # fitted curve
        assert len(ax.collections) >= 1  # scatter of source data

    def test_show_data_false_omits_scatter(self):
        ax = plot_property(_material(), "p", show_data=False)
        assert len(ax.collections) == 0

    def test_uses_provided_axes(self):
        _, ax = plt.subplots()
        assert plot_property(_material(), "p", ax=ax) is ax

    def test_no_range_without_data_raises(self):
        mat = Material(name="bare")
        setattr(mat, "p", sp.Float(5.0))  # constant, no samples
        with pytest.raises(ValueError):
            plot_property(mat, "p")

    def test_dep_range_enables_plotting_without_data(self):
        mat = Material(name="bare")
        setattr(mat, "p", sp.Float(2.0) * T)
        assert isinstance(plot_property(mat, "p", dep_range=(0, 10)), Axes)

    def test_writes_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plot_property(_material(), "p")
        assert list(tmp_path.iterdir()) == []


class TestPlotResiduals:
    def test_returns_axes(self):
        assert isinstance(plot_residuals(_material(), "p"), Axes)

    def test_missing_property_raises(self):
        with pytest.raises(KeyError):
            plot_residuals(_material(), "nope")


class TestCompareMaterials:
    def test_overlays_each_material(self):
        ax = compare_materials([_material("a"), _material("b")], "p")
        assert isinstance(ax, Axes)
        assert len(ax.lines) >= 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compare_materials([], "p")

    def test_label_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            compare_materials([_material()], "p", labels=["a", "b"])

    def test_custom_labels_used(self):
        ax = compare_materials([_material("a"), _material("b")], "p", labels=["x", "y"])
        legend = ax.get_legend()
        texts = {t.get_text() for t in legend.get_texts()}
        assert {"x", "y"} <= texts
