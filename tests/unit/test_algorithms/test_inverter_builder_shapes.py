"""Inverting Piecewise functions as produced by PiecewiseBuilder.

These guard the inverter against the condition/expression shapes SymPy emits when
it simplifies a built Piecewise: a merged ``Or`` condition (when a linear lower
bound makes the extrapolation piece equal segment 0) and a constant piece equal
to exactly ``0.0`` (SymPy degree ``-oo``). Both used to abort inversion.
"""

import numpy as np
import sympy as sp

from materforge.algorithms.piecewise_builder import PiecewiseBuilder
from materforge.algorithms.piecewise_inverter import PiecewiseInverter

T = sp.Symbol("T")
E = sp.Symbol("E")


def _build(bounds):
    dep = np.array([300.0, 400.0, 500.0, 600.0])
    val = np.array([0.0, 100.0, 250.0, 450.0])  # strictly increasing, starts at 0.0
    return PiecewiseBuilder.build_from_data(dep, val, T, {"bounds": bounds}, "energy")


def _roundtrip_max_error(pw, inv, points):
    f = sp.lambdify(T, pw, "numpy")
    finv = sp.lambdify(E, inv, "numpy")
    return max(abs(float(finv(float(f(t)))) - t) for t in points)


class TestInverterOnBuilderOutput:
    def test_linear_bounds_merged_or_condition(self):
        # Linear bounds make the lower extrapolation piece equal segment 0, so
        # SymPy merges them into an `Or` condition the inverter must parse.
        pw = _build(["linear", "linear"])
        inv = PiecewiseInverter.create_inverse(pw, T, E)
        assert _roundtrip_max_error(pw, inv, [350, 420, 480, 550, 590]) < 1e-6

    def test_constant_bounds_with_zero_valued_piece(self):
        # The lower constant piece is exactly 0.0 (SymPy reports degree -oo).
        pw = _build(["constant", "constant"])
        inv = PiecewiseInverter.create_inverse(pw, T, E)
        # Within the data range the inverse recovers the dependency value.
        assert _roundtrip_max_error(pw, inv, [320, 450, 520, 580]) < 1e-6
