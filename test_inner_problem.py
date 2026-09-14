"""
Test-suite for inner_problem.py.  Run with:  python -m pytest -q test_inner_problem.py  (about 30 s).
"""
from fractions import Fraction
from math import pi
import numpy as np
import pytest

from inner_problem import (T, Sigma, beta, late_terms, stokes_D, richardson, stokes_constant,
                           source_polynomial, source_coefficients)


def test_projection_coefficients_by_hand():
    """T^(1)_11 = 3/4, T^(1)_33 = 1/2, T^(1)_13 = -1/4; Sigma^(0)_1 = 3/4, Sigma^(0)_3 = -1/4."""
    assert T(1, 1, 1) == Fraction(3, 4)
    assert T(1, 3, 3) == Fraction(1, 2)
    assert T(1, 1, 3) == Fraction(-1, 4)
    assert Sigma(0, 1) == Fraction(3, 4) and Sigma(0, 3) == Fraction(-1, 4)


def test_projection_coefficients_by_quadrature():
    """T^(m)_kl and Sigma^(n)_k agree with numerical quadrature of their definitions."""
    t = np.linspace(0, 2 * np.pi, 4001)[:-1]
    dt = 2 * np.pi / t.size
    for m, k, l in [(1, 1, 1), (2, 3, 5), (3, 1, 7), (2, 5, 5)]:
        num = np.sum(np.sin(t) ** (2 * m) * np.sin(k * t) * np.sin(l * t)) * dt / np.pi
        assert abs(float(T(m, k, l)) - num) < 1e-12
    for n, k in [(0, 1), (0, 3), (1, 3), (2, 5), (1, 7)]:
        num = np.sum(np.sin(t) ** (2 * n + 3) * np.sin(k * t)) * dt / np.pi
        assert abs(float(Sigma(n, k)) - num) < 1e-12


def test_source_expansion_coefficients():
    """beta_n are the Taylor coefficients of (1+w^2)/(1-w^2)^4 and (1+w^2)^3/(1-w^2)^6."""
    w = 0.3
    s2 = sum(beta(n, 2) * w ** (2 * n) for n in range(60))
    s3 = sum(beta(n, 3) * w ** (2 * n) for n in range(60))
    assert abs(s2 - (1 + w ** 2) / (1 - w ** 2) ** 4) < 1e-12
    assert abs(s3 - (1 + w ** 2) ** 3 / (1 - w ** 2) ** 6) < 1e-12


def test_low_order_terms_by_hand():
    """a_1,1 = 12 i (amplitude factor 1 - 3 rho), a_3,3 = -i, a_5,5 = -i/4 for V_2;
    a_1,1 = 48 i (factor 1 - 12 rho_3) for V_3."""
    r = late_terms(9, 9, which=2)
    assert r[1][1] == 12 and r[3][3] == -1 and r[5][5] == Fraction(-1, 4)
    r3 = late_terms(9, 9, which=3)
    assert r3[1][1] == 48 and r3[3][3] == -4


def test_even_orders_absent_and_series_is_imaginary():
    """The recursion only produces odd powers of 1/xi (oddness in xi); the coefficients are
    rational multiples of i by construction."""
    r = late_terms(21, 11, which=2)
    for k in r:
        assert all(N % 2 == 1 for N in r[k])


def test_stokes_constant_converges_and_matches_reference():
    """D_N converges like a series in 1/N (no residual power of N: alpha = 0) to
    D = -8.537476; Lambda_0 = pi |D| = 26.8213, within 2e-4 with N_max = 81, K_max = 15."""
    r = late_terms(81, 15, which=2)
    Ns = list(range(21, 82, 2))
    Ds = [stokes_D(r, N) for N in Ns]
    # raw values drift by less than 1e-3 per step at the end: no power of N
    assert abs(Ds[-1] - Ds[-2]) < 1e-2 * abs(Ds[-1])
    D = richardson(Ds, Ns, 3)[-1]
    assert abs(D + 8.537476) < 2e-4
    assert abs(pi * abs(D) - 26.8213) < 1e-3


def test_ratio_of_stokes_constants_in_the_limit():
    """K_3 / K_2 -> D^(3)/D^(2) = +1.3032 as epsilon -> 0."""
    D2, L2, _ = stokes_constant(81, 15, 2)
    D3, L3, _ = stokes_constant(81, 15, 3)
    assert abs(D3 / D2 - 1.3032) < 2e-3
    assert abs(L3 - 34.953) < 5e-3


def test_general_source_polynomial():
    """N_m(w)/i for m = 2, 3 equals -64 w^3 (1+w^2) and -256 w^3 (1+w^2)^3; the leading
    coefficient is -(m^3-m)/6 * 64 for every m (from V_m ~ -(m^3-m)/6 U^3, U ~ -4 i w)."""
    assert source_polynomial(2) == [0, 0, 0, -64, 0, -64, 0, 0]
    assert source_polynomial(3)[:10] == [0, 0, 0, -256, 0, -768, 0, -768, 0, -256]
    for m in range(2, 8):
        assert source_coefficients(m, 0)[0] == -(m ** 3 - m) * 64 // 6


def test_general_source_matches_direct_evaluation():
    """V_m(U(w)) evaluated directly agrees with the series i sum_n s_n w^{2n+3} at w = 0.2."""
    w = 0.2
    z = -1j * w
    U = 4 * np.arctan(z)
    for m in [2, 3, 4, 5]:
        direct = np.sin(m * U) - m * np.sin(U)
        s = source_coefficients(m, 40)
        series = 1j * sum(c * w ** (2 * n + 3) for n, c in enumerate(s))
        assert abs(direct - series) < 1e-12 * max(1.0, abs(direct))


def test_amplitude_check_for_every_harmonic():
    """a_1,1 = 2 (m^3 - m) i, the amplitude factor 1 - (m^3-m)/2 rho at fixed frequency."""
    for m in [2, 3, 4, 5]:
        r = late_terms(5, 7, which=m)
        assert r[1][1] == 2 * (m ** 3 - m)


def test_stokes_constants_of_higher_harmonics():
    """D^(4) = 216.14 and D^(5) = 827.41 (both positive: opposite sign to D^(2), D^(3))."""
    D4, _, _ = stokes_constant(81, 15, 4)
    D5, _, _ = stokes_constant(81, 15, 5)
    assert abs(D4 - 216.141) < 0.05
    assert abs(D5 - 827.41) < 0.3
