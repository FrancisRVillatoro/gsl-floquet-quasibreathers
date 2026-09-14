"""
Test-suite for qb_newton.py.  Run with:  python -m pytest -q test_qb_newton.py  (about 60 s).
"""

import numpy as np
import pytest

import gsl_floquet as gf
from qb_newton import (HarmonicBalance, sg_breather_harmonics, small_amplitude_coefficients,
                       born_radiation, emitted_harmonics)
from kg_spectral import SpectralKG, sg_breather, sg_breather_t


# --------------------------------------------------------------------------
# 1. Sine-Gordon: the exact breather is a fixed point of the Newton iteration
# --------------------------------------------------------------------------

def test_sine_gordon_breather_is_a_solution():
    """Seeded with the exact breather (omega = 0.6, K = 10), Newton converges in at most
    two iterations to a residual below 1e-11 and does not move the solution by more than
    the K-truncation error."""
    L, N, om, K = 120.0, 1024, 0.6, 10
    hb = HarmonicBalance(L, N, np.sin, np.cos, K=K, kappa=1.0)
    U0 = sg_breather_harmonics(hb.x, om, K)
    U, info = hb.solve(U0, om, tol=1e-11, maxit=6)
    assert info["converged"] and info["iterations"] <= 2
    assert np.max(np.abs(U - U0)) < 1e-8


@pytest.mark.parametrize("K,tol", [(4, 3e-4), (6, 6e-6), (8, 6e-8)])
def test_truncation_error_decays_exponentially(K, tol):
    """The distance to the exact breather falls by roughly two orders of magnitude per
    pair of harmonics (1.6e-4, 3.4e-6, 2.6e-8 for K = 4, 6, 8)."""
    L, N, om = 120.0, 1024, 0.6
    hb = HarmonicBalance(L, N, np.sin, np.cos, K=K, kappa=1.0)
    U, info = hb.solve(sg_breather_harmonics(hb.x, om, K), om, tol=1e-11, maxit=8)
    assert info["converged"]
    assert np.max(np.abs(U - sg_breather_harmonics(hb.x, om, K))) < tol


def test_converges_from_a_perturbed_seed():
    """Newton converges from a seed 20% off in amplitude."""
    L, N, om, K = 120.0, 1024, 0.6, 8
    hb = HarmonicBalance(L, N, np.sin, np.cos, K=K, kappa=1.0)
    Uex = sg_breather_harmonics(hb.x, om, K)
    U, info = hb.solve(1.2 * Uex, om, tol=1e-11, maxit=25)
    assert info["converged"]
    assert np.max(np.abs(U - Uex)) < 1e-7


def test_solution_is_even_in_x():
    L, N, om, K = 120.0, 1024, 0.6, 8
    hb = HarmonicBalance(L, N, np.sin, np.cos, K=K, kappa=1.0)
    U, _ = hb.solve(sg_breather_harmonics(hb.x, om, K), om, tol=1e-11, maxit=8)
    assert np.max(np.abs(U - hb.symmetrise(U))) < 1e-14


def test_initial_data_matches_the_exact_breather():
    """(u, u_t) reconstructed from the harmonics agree with the exact breather.  The
    velocity is the less accurate of the two at t = 0, where every neglected harmonic
    contributes with weight k omega and in phase; K = 12 gives 2e-7 there and 1e-7 at
    t = T/8, against 8e-9 for the field itself."""
    L, N, om, K = 120.0, 1024, 0.6, 12
    hb = HarmonicBalance(L, N, np.sin, np.cos, K=K, kappa=1.0)
    U, _ = hb.solve(sg_breather_harmonics(hb.x, om, K), om, tol=1e-11, maxit=8)
    for t in [0.0, 2 * np.pi / om / 8]:
        u, v = hb.initial_data(U, om, t=t)
        assert np.max(np.abs(u - sg_breather(hb.x, t, om))) < 1e-8
        assert np.max(np.abs(v - sg_breather_t(hb.x, t, om))) < 5e-7


# --------------------------------------------------------------------------
# 2. Small-amplitude coefficients
# --------------------------------------------------------------------------

def test_small_amplitude_coefficients_sine_gordon():
    """For sine-Gordon (a = b = 0) kappa = kappa_3 = kappa_5 = 1, hence eta_5 = 0."""
    k1, k3, k5 = small_amplitude_coefficients(gf.DressedGSL(0.0, 0.0))
    assert abs(k1 - 1) < 1e-13 and abs(k3 - 1) < 1e-13 and abs(k5 - 1) < 1e-13


@pytest.mark.parametrize("a,b", [(0.0, 0.5), (1.2, 0.5), (1.5, 1.0)])
def test_small_amplitude_coefficients_match_the_taylor_expansion(a, b):
    """kappa, kappa_3, kappa_5 reproduce F(u) = kappa u - kappa_3 u^3/6 + kappa_5 u^5/120
    to O(u^7): the residual falls by 2^7 when u is halved."""
    d = gf.DressedGSL(a, b)
    k1, k3, k5 = small_amplitude_coefficients(d)

    def res(u):
        return abs(float(d.F(u)) - (k1 * u - k3 * u ** 3 / 6 + k5 * u ** 5 / 120))

    r1, r2 = res(0.2), res(0.1)
    assert 100.0 < r1 / r2 < 160.0


# --------------------------------------------------------------------------
# 3. Dressed GSL: the Newton solution is time-periodic under the integrator
# --------------------------------------------------------------------------

def test_dressed_quasi_breather_is_periodic_in_time():
    """(a, b) = (1.25, 0.5), Omega = 0.8: on a periodic box (no absorbing layers) the
    field returns to itself to 1e-6 after one period and to 1e-4 after ten.  The step
    must divide the period exactly, otherwise the comparison measures the sampling
    offset (u passes through zero at t = 0, where |u_t| is largest)."""
    a, b, Om, K = 1.25, 0.5, 0.8, 10
    L, N = 140.0, 1024
    d = gf.DressedGSL(a, b)
    k1, k3, _ = small_amplitude_coefficients(d)
    om = Om * np.sqrt(k1)
    hb = HarmonicBalance(L, N, d.F, d.dF, K=K, kappa=k1)
    U, info = hb.solve(np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, Om, K),
                       om, tol=1e-11, maxit=25)
    assert info["converged"]
    u0, v0 = hb.initial_data(U, om)
    kg = SpectralKG(L, N, d.F, d.G, kappa=k1, G_ref=0.0)
    kg.set_initial(u0, v0)
    T = 2 * np.pi / om
    h = T / 200.0                      # an exact divisor of the period
    kg.run(h, 200)
    assert np.max(np.abs(kg.u - u0)) < 1e-6
    kg.run(h, 1800)
    assert np.max(np.abs(kg.u - u0)) < 1e-4


def test_radiated_power_is_independent_of_the_box():
    """The radiated power measured with absorbing layers agrees to three significant
    figures for L = 140 and L = 180 (a = 0.6, b = 0.5, Omega = 0.8), which is what makes
    it a physical rate rather than a box artefact."""
    a, b, Om, K = 0.6, 0.5, 0.8, 10
    out = []
    for L, N, xs, W in [(140.0, 1024, 45.0, 22.0), (180.0, 1280, 60.0, 28.0)]:
        d = gf.DressedGSL(a, b)
        k1, k3, _ = small_amplitude_coefficients(d)
        om = Om * np.sqrt(k1)
        hb = HarmonicBalance(L, N, d.F, d.dF, K=K, kappa=k1)
        U, _ = hb.solve(np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, Om, K),
                        om, tol=1e-11, maxit=25)
        kg = SpectralKG(L, N, d.F, d.G, kappa=k1,
                        sponge=dict(x_s=xs, width=W, sigma0=0.8), G_ref=0.0)
        u0, v0 = hb.initial_data(U, om)
        kg.set_initial(u0, v0)
        E0 = kg.energy()
        T = 2 * np.pi / om
        rec = []
        kg.run_until(min(0.1, 0.9 * kg.h_max), 60 * T,
                     observer=lambda kg, u: rec.append((kg.t, kg.absorbed)), every=100)
        rec = np.array(rec)
        sel = rec[:, 0] > 30 * T
        out.append(np.polyfit(rec[sel, 0] / T, rec[sel, 1] / E0, 1)[0])
    assert abs(out[0] / out[1] - 1.0) < 5e-3


# --------------------------------------------------------------------------
# 4. Energy and radiation diagnostics
# --------------------------------------------------------------------------

def test_breather_energy_equals_16_beta():
    """The sine-Gordon breather has E = 16 sqrt(1 - omega^2); the harmonic-balance energy
    reproduces it to 1e-5 with K = 10 (the residual is the harmonic truncation)."""
    L, N, om, K = 120.0, 1024, 0.6, 10
    hb = HarmonicBalance(L, N, np.sin, np.cos, K=K, kappa=1.0)
    U, _ = hb.solve(sg_breather_harmonics(hb.x, om, K), om, tol=1e-11, maxit=8)
    E = hb.energy(U, om, lambda u: 1.0 - np.cos(u))
    assert abs(E - 16.0 * np.sqrt(1.0 - om ** 2)) < 1e-5


def test_first_born_source_is_box_independent():
    """Built from the bound fundamental alone, Nhat_3(q_3) agrees to 10% between
    L = 140 and L = 200; built from the full box solution it does not, because the
    standing tail of U_3 enters resonantly (that is why first_born=False must not be
    used on a periodic box)."""
    out_b, out_f = [], []
    for L, N in [(140.0, 1024), (200.0, 1536)]:
        d = gf.DressedGSL(1.0, 0.5)
        k1, k3, _ = small_amplitude_coefficients(d)
        om = 0.8 * np.sqrt(k1)
        hb = HarmonicBalance(L, N, d.F, d.dF, K=10, kappa=k1)
        U, _ = hb.solve(np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, 0.8, 10),
                        om, tol=1e-11, maxit=25)
        E0 = hb.energy(U, om, d.G)
        out_b.append(born_radiation(hb, U, om, k1, E0, first_born=True)[3]["Nhat"])
        out_f.append(born_radiation(hb, U, om, k1, E0, first_born=False)[3]["Nhat"])
    assert abs(out_b[0] / out_b[1] - 1.0) < 0.10
    assert abs(out_f[0] / out_f[1] - 1.0) > 1.0


def test_first_born_overestimates_sine_gordon():
    """Documented limitation: for sine-Gordon the exact radiation is zero, yet the first
    Born predicts p_3 of order 1e-2 per period.  The estimate is a structural diagnostic,
    not a quantitative prediction."""
    hb = HarmonicBalance(120.0, 1024, np.sin, np.cos, K=10, kappa=1.0)
    U, _ = hb.solve(sg_breather_harmonics(hb.x, 0.6, 10), 0.6, tol=1e-11, maxit=8)
    E0 = hb.energy(U, 0.6, lambda u: 1.0 - np.cos(u))
    assert born_radiation(hb, U, 0.6, 1.0, E0)[3]["p"] > 1e-3


def test_emitted_harmonics_on_the_exact_breather():
    """At x = 20 the exact sine-Gordon breather (omega = 0.6) shows only its evanescent
    fundamental, (8 beta / omega) exp(-beta x) = 1.2e-6, to within 2%, and no third
    harmonic above 1e-10."""
    om, L, N = 0.6, 160.0, 2048
    kg = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=1.0)
    kg.set_initial(sg_breather(kg.x, 0.0, om), sg_breather_t(kg.x, 0.0, om))
    A = emitted_harmonics(kg, om, 20.0, n_periods=60, n_fit=30)
    beta = np.sqrt(1.0 - om ** 2)
    assert abs(A[1] / (8.0 * beta / om * np.exp(-beta * 20.0)) - 1.0) < 0.02
    assert A[3] < 1e-10
