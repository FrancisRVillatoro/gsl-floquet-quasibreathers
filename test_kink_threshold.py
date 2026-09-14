"""
Test-suite for kink_threshold.py.  Run with:  python -m pytest -q test_kink_threshold.py  (about 40 s).
"""
from fractions import Fraction
import numpy as np
import pytest
from scipy import special

import gsl_floquet as gf
from kink_threshold import edge_slope, a_internal_mode_vanishes, canonical_slope_derivative


def test_sine_gordon_threshold_state():
    """For sine-Gordon the odd edge solution is tanh x: beta = 0 to 1e-9."""
    assert abs(edge_slope(gf.DressedGSL(0.0, 0.0))) < 1e-9


def test_gsl_kink_has_an_internal_mode_at_a_zero():
    """beta < 0 at a = 0 for b = 0.5 and b = 1 (internal mode present, as in the 2021 paper)."""
    assert edge_slope(gf.DressedGSL(0.0, 0.5)) < -0.1
    assert edge_slope(gf.DressedGSL(0.0, 1.0)) < -0.3


def test_threshold_slopes_are_exact_rationals():
    """d beta / d r for V_2, V_3, V_4 equal -8/3, -256/45, -944/105 (Richardson from r = 2e-3, 1e-3)."""
    for m, cand in [(2, Fraction(-8, 3)), (3, Fraction(-256, 45)), (4, Fraction(-944, 105))]:
        w2 = canonical_slope_derivative(m, 2e-3)
        w1 = canonical_slope_derivative(m, 1e-3)
        rich = (4 * w1 - w2) / 3
        assert abs(rich - float(cand)) < 2e-6


def test_internal_mode_vanishes_near_j01_over_2():
    """a_k(0.125) = 1.19776, within 1e-4, and a_k(b) -> j_01/2 with an O(b^2) shift of -0.30 b^2."""
    j = special.jn_zeros(0, 1)[0] / 2
    ak = a_internal_mode_vanishes(0.125)
    assert abs(ak - 1.19776) < 1e-4
    assert abs((ak - j) / 0.125 ** 2 + 0.298) < 0.01


def test_no_internal_mode_beyond_a_k():
    """beta > 0 (no internal mode) between a_k and a_* for b = 0.5."""
    ak = a_internal_mode_vanishes(0.5)
    for a in [ak + 0.1, ak + 0.5, ak + 1.0]:
        assert edge_slope(gf.DressedGSL(a, 0.5)) > 0.0


def test_first_order_prediction_of_a_k():
    """The V_3 + V_4 first-order formula reproduces a_k(0.25) to 4e-4."""
    j = special.jn_zeros(0, 1)[0] / 2
    w = {m: float(c) for m, c in [(2, Fraction(-8, 3)), (3, Fraction(-256, 45)), (4, Fraction(-944, 105))]}
    b = 0.25
    c = gf.fourier_coefficients(b)
    den = 2 * c[1] * special.j1(2 * j)
    pred = j + (c[2] * special.j0(3 * j) * w[3] / w[2] + c[3] * special.j0(4 * j) * w[4] / w[2]) / den
    assert abs(pred - a_internal_mode_vanishes(b)) < 4e-4
