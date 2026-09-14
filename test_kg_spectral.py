"""
Test-suite for kg_spectral.py.  Run with:  python -m pytest -q test_kg_spectral.py
(about 40 s).  Every test states the property checked and the tolerance.
"""

import numpy as np
import pytest
from scipy import interpolate

import gsl_floquet as gf
from kg_spectral import (SPLITTING_COEFFICIENTS, SpectralKG, sg_breather, sg_breather_t,
                         zero_crossing_frequency)


# --------------------------------------------------------------------------
# 0. Coefficient tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["strang", "BM4", "BM6"])
def test_coefficients_consistent(method):
    """sum a = sum b = 1, len(b) = len(a) + 1, symmetric (time-reversible) compositions."""
    a, b = SPLITTING_COEFFICIENTS[method]["coeffs"]()
    assert abs(a.sum() - 1.0) < 1e-14 and abs(b.sum() - 1.0) < 1e-14
    assert b.size == a.size + 1
    assert np.allclose(a, a[::-1]) and np.allclose(b, b[::-1])


def test_bm6_has_eleven_evaluations():
    a, b = SPLITTING_COEFFICIENTS["BM6"]["coeffs"]()
    assert a.size == 11 and b.size == 12


# --------------------------------------------------------------------------
# 1. Order of convergence on the exact sine-Gordon breather
# --------------------------------------------------------------------------

def _breather_error(method, h, omega=0.6, L=60.0, N=768, periods=4):
    kg = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=1.0, method=method)
    x = kg.x
    kg.set_initial(sg_breather(x, 0.0, omega), sg_breather_t(x, 0.0, omega))
    t_end = periods * 2.0 * np.pi / omega
    n = int(round(t_end / h))
    kg.run(t_end / n, n)
    return np.max(np.abs(kg.u - sg_breather(x, kg.t, omega)))


@pytest.mark.parametrize("method,ratio_lo,ratio_hi", [("strang", 3.0, 5.0), ("BM4", 12.0, 20.0), ("BM6", 45.0, 90.0)])
def test_order_of_convergence(method, ratio_lo, ratio_hi):
    """Error ratios under step halving: 4 (order 2), 16 (order 4), 64 (order 6).
    This validates the Blanes-Moan coefficient tables and the composition loop."""
    steps = [0.2, 0.1, 0.05]
    errs = [_breather_error(method, h) for h in steps]
    for i in range(2):
        r = errs[i] / errs[i + 1]
        assert ratio_lo < r < ratio_hi, (method, errs)


def test_bm6_accuracy_level():
    """BM6 with h = 0.1 reproduces the omega = 0.6 breather to 1e-8 after four periods."""
    assert _breather_error("BM6", 0.1) < 1e-8


# --------------------------------------------------------------------------
# 2. Long-time behaviour: bounded energy error, small phase error
# --------------------------------------------------------------------------

def test_long_run_energy_and_phase():
    """100 periods of the omega = 0.6 breather with BM6, h = 0.1: relative energy error
    stays below 1e-10 (no drift) and the solution error stays below 5e-7."""
    omega, L, N = 0.6, 60.0, 768
    kg = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=1.0)
    x = kg.x
    kg.set_initial(sg_breather(x, 0.0, omega), sg_breather_t(x, 0.0, omega))
    E0 = kg.energy()
    T = 2.0 * np.pi / omega
    dE, err = [], []
    h = 0.1
    for _ in range(10):
        kg.run_until(h, kg.t + 10.0 * T)
        dE.append(abs(kg.energy() / E0 - 1.0))
        err.append(np.max(np.abs(kg.u - sg_breather(x, kg.t, omega))))
    assert max(dE) < 1e-10
    assert max(err) < 5e-7
    # measured breather frequency from zero crossings of u(0, t)
    samples_t, samples_u = [], []
    kg2 = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=1.0)
    kg2.set_initial(sg_breather(x, 0.0, omega), sg_breather_t(x, 0.0, omega))
    i0 = int(np.argmin(np.abs(x)))
    kg2.run_until(h, 20 * T, observer=lambda kg, u: (samples_t.append(kg.t), samples_u.append(u[i0])))
    assert abs(zero_crossing_frequency(samples_t, samples_u) - omega) < 1e-5


def test_dealias_flag_irrelevant_for_smooth_data():
    """Zeroing the top third of the spectrum does not change a resolved solution (1e-10)."""
    omega, L, N = 0.6, 60.0, 768
    res = []
    for dealias in (True, False):
        kg = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=1.0, dealias=dealias)
        kg.set_initial(sg_breather(kg.x, 0.0, omega), sg_breather_t(kg.x, 0.0, omega))
        kg.run_until(0.1, 4 * 2 * np.pi / omega)
        res.append(kg.u)
    assert np.max(np.abs(res[0] - res[1])) < 1e-10


# --------------------------------------------------------------------------
# 3. Absorbing layer
# --------------------------------------------------------------------------

def test_sponge_absorbs_linear_packet_without_reflection():
    """A right-moving linear wave packet (k0 = 2) is absorbed in the layer:
    interior energy < 1e-8 E0 afterwards, absorbed energy = E0 to 1e-8, and
    E_total + absorbed is conserved to 1e-10."""
    L, N, kappa = 240.0, 2048, 1.0
    kg = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=kappa,
                    sponge=dict(x_s=60.0, width=40.0, sigma0=1.0))
    x = kg.x
    A, k0, x0, sx = 1e-3, 2.0, -40.0, 8.0
    z = A * np.exp(-(x - x0) ** 2 / (2 * sx ** 2)) * np.exp(1j * k0 * (x - x0))
    kk = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    omk = np.sqrt(kappa + kk * kk)
    u0 = np.real(z)
    v0 = np.real(np.fft.ifft(-1j * omk * np.fft.fft(z)))     # each mode ~ exp(i(kx - w t))
    kg.set_initial(u0, v0)
    E0 = kg.energy()
    kg.run_until(0.1, 300.0)
    assert kg.energy(interior=(-60.0, 60.0)) < 1e-8 * E0
    assert abs(kg.absorbed / E0 - 1.0) < 1e-8
    assert abs((kg.energy() + kg.absorbed) / E0 - 1.0) < 1e-10


def test_sponge_sees_nothing_from_exact_breather():
    """The exact sine-Gordon breather does not radiate: absorbed energy < 1e-12 after 20 periods."""
    omega, L, N = 0.6, 160.0, 2048
    kg = SpectralKG(L, N, np.sin, lambda u: 1.0 - np.cos(u), kappa=1.0,
                    sponge=dict(x_s=40.0, width=30.0, sigma0=1.0))
    kg.set_initial(sg_breather(kg.x, 0.0, omega), sg_breather_t(kg.x, 0.0, omega))
    E0 = kg.energy()
    kg.run_until(0.1, 20 * 2 * np.pi / omega)
    assert kg.absorbed < 1e-12 * E0
    assert abs(kg.energy() / E0 - 1.0) < 1e-10


# --------------------------------------------------------------------------
# 4. Kinks of the dressed GSL with twisted boundary conditions
# --------------------------------------------------------------------------

def _static_run(d, u1, u0_builder, L, N, t_end=100.0, h=0.1):
    kappa = float(d.dF(u1))
    kg = SpectralKG(L, N, d.F, d.G, kappa=kappa, twist=1, G_ref=u1)
    u0 = u0_builder(kg.x)
    kg.set_initial(u0, np.zeros_like(kg.x))
    E0 = kg.energy()
    kg.run_until(h, t_end)
    return np.max(np.abs(kg.u - u0)), E0, kg.energy()


def test_static_kink_regime_I():
    """(a, b) = (1.5, 1): the 2pi-kink stays static to 1e-6 over t = 100; energy equals the
    Bogomolny quadrature to 1e-9 and is conserved to 1e-12."""
    d = gf.DressedGSL(1.5, 1.0)
    err, E0, E1 = _static_run(d, 0.0, lambda x: d.kink_on_grid(0.0, 2 * np.pi, x), 100.0, 1024)
    assert err < 1e-6
    assert abs(E0 - d.kink_energy(0.0, 2 * np.pi)) < 1e-9 * E0
    assert abs(E1 / E0 - 1.0) < 1e-12


def test_static_kink_regime_III():
    """b = 1, a beyond a_**: the inverted 2pi-kink pi -> 3pi is static (1e-6) with the right energy."""
    b = 1.0
    a = gf.first_zero_mass2(b, at=np.pi) + 0.3
    d = gf.DressedGSL(a, b)
    assert d.regime() == "III"
    err, E0, E1 = _static_run(d, np.pi, lambda x: d.kink_on_grid(np.pi, 3 * np.pi, x), 100.0, 1024)
    assert err < 1e-6
    assert abs(E0 - d.kink_energy(np.pi, 3 * np.pi)) < 1e-9 * E0
    assert abs(E1 / E0 - 1.0) < 1e-12


def test_static_kink_pair_regime_II():
    """b = 1, a in (a_*, a_**): the alternating array (kink through 0 at -L/4, kink through pi
    at +L/4) has total twist 2pi, stays static to 1e-6 over t = 100, and its energy is the sum
    of the two Bogomolny energies (1e-9)."""
    b = 1.0
    a = 0.5 * (gf.first_zero_mass2(b, at=0.0) + gf.first_zero_mass2(b, at=np.pi))
    d = gf.DressedGSL(a, b)
    assert d.regime() == "II"
    us = d.vacua()[0]
    L = 300.0

    def pair(x):
        return d.kink_on_grid(-us, us, x + L / 4) + d.kink_on_grid(us, 2 * np.pi - us, x - L / 4) - us

    err, E0, E1 = _static_run(d, -us, pair, L, 3072)
    assert err < 1e-6
    Ek = d.kink_energy(-us, us) + d.kink_energy(us, 2 * np.pi - us)
    assert abs(E0 - Ek) < 1e-9 * E0
    assert abs(E1 / E0 - 1.0) < 1e-12


def test_boosted_kink_travels_at_the_right_speed():
    """Lorentz-boosted 2pi-kink (v = 0.4, regime I): the centre follows x0 + v t to 1e-4 over
    t = 100 on the twisted ring, and the energy equals gamma E_kink to 1e-7."""
    d = gf.DressedGSL(1.5, 1.0)
    v = 0.4
    gam = 1.0 / np.sqrt(1.0 - v * v)
    L, N = 200.0, 2048
    kg = SpectralKG(L, N, d.F, d.G, kappa=float(d.dF(0.0)), twist=1, G_ref=0.0)
    xk, uk = d.static_kink(0.0, 2 * np.pi)
    spl = interpolate.CubicSpline(xk, uk)

    def prof(xx):
        z = gam * xx
        out = spl(np.clip(z, xk[0], xk[-1]))
        return np.where(z < xk[0], 0.0, np.where(z > xk[-1], 2 * np.pi, out))

    x0 = -50.0
    u0 = prof(kg.x - x0)
    v0 = -gam * v * np.sqrt(np.maximum(2.0 * d.dG_from(u0, 0.0), 0.0))
    kg.set_initial(u0, v0)
    E0 = kg.energy()
    centres, times = [], []

    def obs(kg, u):
        i = np.nonzero((u[:-1] < np.pi) & (u[1:] >= np.pi))[0][0]
        centres.append(kg.x[i] + (np.pi - u[i]) * (kg.x[i + 1] - kg.x[i]) / (u[i + 1] - u[i]))
        times.append(kg.t)

    kg.run_until(0.1, 100.0, observer=obs, every=50)
    assert np.max(np.abs(np.array(centres) - (x0 + v * np.array(times)))) < 1e-4
    assert abs(E0 - gam * d.kink_energy(0.0, 2 * np.pi)) < 1e-7 * E0
    assert abs(kg.energy() / E0 - 1.0) < 1e-12


# --------------------------------------------------------------------------
# 5. Clenshaw evaluation of the series (used by the integrator)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("b", [1.0, 5.0, 20.0])
def test_clenshaw_matches_direct_evaluation(b):
    d = gf.DressedGSL(2.0, b)
    u = np.linspace(-7.0, 7.0, 2001)
    assert np.max(np.abs(d.F(u) - d.F_direct(u))) < 1e-13
    assert np.max(np.abs(d.G(u) - d.G_direct(u))) < 1e-13
    assert np.max(np.abs(d.dF(u) - d.dF_direct(u))) < 1e-13
