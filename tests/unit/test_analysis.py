"""Unit tests for materforge.analysis (fit-quality metrics)."""

import numpy as np
import pytest
import sympy as sp

from materforge import (
    FitQuality,
    create_material,
    fit_quality,
    fit_report,
    mae,
    max_abs_error,
    r_squared,
    residuals,
    rmse,
)
from materforge.core.materials import Material, PropertySamples

T = sp.Symbol("T")


def _linear_material(slope=2.0, intercept=1.0, noise=None, prop_type="TABULAR_DATA"):
    """Material with one property ``p = slope*T + intercept`` and matching samples."""
    mat = Material(name="linear")
    mat.p = sp.Float(slope) * T + sp.Float(intercept)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = slope * x + intercept
    if noise is not None:
        y = y + np.asarray(noise, dtype=float)
    mat.sample_data["p"] = PropertySamples(x, y, prop_type)
    return mat


# --- pure array metrics ---

class TestPureMetrics:
    def test_r_squared_perfect(self):
        assert r_squared([1, 2, 3], [1, 2, 3]) == 1.0

    def test_rmse_mae_max_perfect(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0
        assert max_abs_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_rmse_known_value(self):
        assert rmse([1, 2, 3], [1, 2, 3.5]) == pytest.approx((0.25 / 3) ** 0.5)

    def test_mae_and_max_known(self):
        assert mae([0, 0, 0], [1, -2, 3]) == pytest.approx(2.0)
        assert max_abs_error([0, 0, 0], [1, -2, 3]) == 3.0

    def test_r_squared_constant_obs_perfect(self):
        assert r_squared([5, 5, 5], [5, 5, 5]) == 1.0

    def test_r_squared_constant_obs_imperfect(self):
        assert r_squared([5, 5, 5], [5, 5, 6]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rmse([1, 2, 3], [1, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            r_squared([], [])


# --- material-aware fit quality ---

class TestFitQuality:
    def test_perfect_fit(self):
        fq = fit_quality(_linear_material(), "p")
        assert isinstance(fq, FitQuality)
        assert fq.property == "p"
        assert fq.n_points == 5
        assert fq.r_squared == pytest.approx(1.0)
        assert fq.rmse == pytest.approx(0.0, abs=1e-9)

    def test_imperfect_fit_matches_known_rmse(self):
        noise = np.array([0.0, 0.5, -0.5, 0.0, 0.25])
        fq = fit_quality(_linear_material(noise=noise), "p")
        # observed = predicted + noise, so rmse is sqrt(mean(noise**2))
        assert fq.rmse == pytest.approx(float(np.sqrt(np.mean(noise ** 2))))
        assert fq.r_squared < 1.0

    def test_missing_property_raises_keyerror(self):
        with pytest.raises(KeyError):
            fit_quality(_linear_material(), "does_not_exist")

    def test_non_material_raises_typeerror(self):
        with pytest.raises(TypeError):
            fit_quality("not a material", "p")

    def test_symbol_passthrough(self):
        fq = fit_quality(_linear_material(), "p", symbol=T)
        assert fq.r_squared == pytest.approx(1.0)


class TestResiduals:
    def test_perfect_fit_zero_residuals(self):
        x, res = residuals(_linear_material(), "p")
        assert x.shape == (5,)
        assert np.allclose(res, 0.0, atol=1e-9)

    def test_residual_sign_is_predicted_minus_observed(self):
        # observed = predicted + 1 everywhere -> residual = -1
        x, res = residuals(_linear_material(noise=np.ones(5)), "p")
        assert np.allclose(res, -1.0)


class TestFitReport:
    def test_includes_only_data_backed_properties(self):
        mat = _linear_material()
        mat.c = sp.Float(7.0)  # constant -> no samples
        report = fit_report(mat)
        assert set(report) == {"p"}
        assert isinstance(report["p"], FitQuality)

    def test_empty_when_no_samples(self):
        mat = Material(name="bare")
        mat.c = sp.Float(1.0)
        assert fit_report(mat) == {}

    def test_non_material_raises_typeerror(self):
        with pytest.raises(TypeError):
            fit_report(object())


class TestFitQualityStr:
    def test_str_is_readable(self):
        fq = FitQuality("density", 0.99, 1.5, 1.0, 3.0, 10)
        text = str(fq)
        assert "density" in text and "R²" in text and "n=10" in text


# --- end-to-end: which property types retain source data ---

class TestSampleDataCapture:
    def test_only_data_backed_properties_are_captured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATERFORGE_DISABLE_CACHE", "1")
        yaml = tmp_path / "m.yaml"
        yaml.write_text(
            "name: M\n"
            "properties:\n"
            "  c: 5\n"
            "  ref: 400\n"
            "  tab:\n"
            "    dependency: [300, 400, 500]\n"
            "    value: [10, 20, 30]\n"
            "    bounds: [constant, constant]\n"
            "  step:\n"
            "    dependency: ref + 10\n"
            "    value: [0, 1]\n"
            "    bounds: [constant, constant]\n"
        )
        mat = create_material(str(yaml), T, enable_plotting=False)
        # Genuinely data-backed -> captured; exact definitions -> not captured.
        assert "tab" in mat.sample_data
        assert "c" not in mat.sample_data
        assert "ref" not in mat.sample_data
        assert "step" not in mat.sample_data
        samples = mat.sample_data["tab"]
        assert list(samples.x) == [300.0, 400.0, 500.0]
        assert list(samples.y) == [10.0, 20.0, 30.0]
        assert samples.prop_type == "TABULAR_DATA"
        # Plain tabular interpolation reproduces its points exactly.
        assert fit_quality(mat, "tab").r_squared == pytest.approx(1.0)

    def test_regression_property_has_genuine_fit_error(self, tmp_path, monkeypatch):
        # A single linear segment cannot pass through a quadratic's points, so the
        # stored regression has a real, non-zero error against its source data -
        # the headline use of fit_quality (vs. ~0 for plain interpolation above).
        monkeypatch.setenv("MATERFORGE_DISABLE_CACHE", "1")
        yaml = tmp_path / "m.yaml"
        yaml.write_text(
            "name: M\n"
            "properties:\n"
            "  quad:\n"
            "    dependency: [0, 1, 2, 3, 4]\n"
            "    value: [0, 1, 4, 9, 16]\n"
            "    bounds: [constant, constant]\n"
            "    regression:\n"
            "      simplify: pre\n"
            "      degree: 1\n"
            "      segments: 1\n"
        )
        mat = create_material(str(yaml), T, enable_plotting=False)
        fq = fit_quality(mat, "quad")
        assert fq.n_points == 5
        assert fq.max_abs_error > 0.0
        assert fq.r_squared < 1.0
        # residuals agree with the summary metric
        _, res = residuals(mat, "quad")
        assert float(np.max(np.abs(res))) == pytest.approx(fq.max_abs_error)
