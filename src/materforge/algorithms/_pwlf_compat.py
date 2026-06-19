# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility shim for a pwlf 2.5.x bug on the macOS (Accelerate) LAPACK backend.

``pwlf.PiecewiseLinFit.lstsq`` reads the residual returned by
``scipy.linalg.lstsq`` and does ``ssr = ssr[0]``. The residual's *type* depends
on the LAPACK backend SciPy was built against:

* Linux / OpenBLAS returns it as a NumPy scalar (``np.float64``), so pwlf's
  ``isinstance(ssr, np.ndarray)`` check is ``False`` and the indexing is skipped.
* macOS / Accelerate returns it as a **0-D** ``np.ndarray``, so the check is
  ``True``, ``ssr.size == 0`` is ``False``, and ``ssr[0]`` raises
  ``IndexError: too many indices for array: array is 0-dimensional``.

This makes any regression-backed material fail to build on macOS while passing
on Linux. We route pwlf's *module-local* ``linalg`` reference through a thin
proxy that normalises a 0-D ndarray residual to a NumPy scalar. Only the
residual's *type* changes — every numeric value (and therefore every fit
result) is identical, so stored regression baselines are unaffected. The global
``scipy.linalg`` module is left untouched for every other caller.
"""

from __future__ import annotations

import numpy as np
import pwlf.pwlf as _pwlf_module
import scipy.linalg as _scipy_linalg


class _ScipyLinalgProxy:
    """Forwards to ``scipy.linalg`` but makes ``lstsq``'s residual 0-D-safe."""

    def __getattr__(self, name: str):
        return getattr(_scipy_linalg, name)

    @staticmethod
    def lstsq(*args, **kwargs):
        beta, residual, rank, singular_values = _scipy_linalg.lstsq(*args, **kwargs)
        if isinstance(residual, np.ndarray) and residual.ndim == 0:
            residual = residual.item()
        return beta, residual, rank, singular_values


def install() -> None:
    """Idempotently route pwlf's ``lstsq`` calls through the 0-D-safe proxy."""
    if not isinstance(_pwlf_module.linalg, _ScipyLinalgProxy):
        _pwlf_module.linalg = _ScipyLinalgProxy()
