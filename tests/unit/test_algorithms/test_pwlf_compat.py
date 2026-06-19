"""Regression tests for the pwlf macOS/Accelerate 0-D residual shim.

pwlf 2.5.x crashes with ``IndexError: ... array is 0-dimensional`` when SciPy's
LAPACK backend returns the lstsq residual as a 0-D ``np.ndarray`` (observed on
macOS/Accelerate; Linux/OpenBLAS returns a NumPy scalar and is unaffected).
``materforge.algorithms._pwlf_compat`` routes pwlf through a 0-D-safe proxy.
"""

import numpy as np
import pwlf
import scipy.linalg as scipy_linalg
import sympy as sp

from materforge.algorithms import _pwlf_compat
from materforge.algorithms.regression_processor import RegressionProcessor


def _force_zero_dim_residual(real_lstsq):
    """Wrap a real lstsq so its residual mimics the macOS 0-D ndarray return."""
    def fake(*args, **kwargs):
        beta, residual, rank, singular_values = real_lstsq(*args, **kwargs)
        forced = np.asarray(residual, dtype=float).sum()  # always a 0-D ndarray
        return beta, np.asarray(forced), rank, singular_values
    return fake


def test_proxy_coerces_zero_dim_residual():
    """The proxy turns a 0-D ndarray residual into a non-ndarray scalar."""
    proxy = _pwlf_compat._ScipyLinalgProxy()
    a = np.vander(np.linspace(0.0, 1.0, 10), 3)
    y = np.linspace(0.0, 1.0, 10)
    _, residual, _, _ = proxy.lstsq(a, y)
    # 0-D ndarray would break pwlf's ``residual[0]``; a scalar does not.
    assert not (isinstance(residual, np.ndarray) and residual.ndim == 0)


def test_proxy_forwards_other_attributes():
    """Anything that is not lstsq is delegated to scipy.linalg unchanged."""
    proxy = _pwlf_compat._ScipyLinalgProxy()
    assert proxy.solve is scipy_linalg.solve
    assert proxy.norm is scipy_linalg.norm


def test_install_is_idempotent():
    """Calling install repeatedly leaves a single proxy in place."""
    _pwlf_compat.install()
    first = pwlf.pwlf.linalg
    _pwlf_compat.install()
    assert pwlf.pwlf.linalg is first
    assert isinstance(pwlf.pwlf.linalg, _pwlf_compat._ScipyLinalgProxy)


def test_pwlf_fit_survives_zero_dim_residual(monkeypatch):
    """A pwlf fit must not crash when the backend yields a 0-D residual."""
    _pwlf_compat.install()
    monkeypatch.setattr(
        scipy_linalg, "lstsq", _force_zero_dim_residual(scipy_linalg.lstsq)
    )
    x = np.linspace(0.0, 10.0, 60)
    y = 3.0 * x + 2.0
    model = pwlf.PiecewiseLinFit(x, y, degree=1, seed=13579)
    model.fit(n_segments=2)  # raised IndexError before the shim
    assert np.isfinite(model.ssr)


def test_regression_processor_survives_zero_dim_residual(monkeypatch):
    """End-to-end: RegressionProcessor builds a Piecewise under a 0-D residual."""
    monkeypatch.setattr(
        scipy_linalg, "lstsq", _force_zero_dim_residual(scipy_linalg.lstsq)
    )
    t = sp.Symbol("T")
    x = np.linspace(300.0, 600.0, 80)
    y = 100.0 + 0.5 * x
    result = RegressionProcessor.process_regression(
        x, y, t, lower_bound_type="constant", upper_bound_type="constant",
        degree=1, segments=2,
    )
    assert isinstance(result, sp.Piecewise)
