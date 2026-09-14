"""
Test-suite for gsl_floquet.py.  Run with:  python -m pytest -q test_gsl_floquet.py

Every test states the identity it checks and the tolerance used.
"""

import numpy as np
import pytest
from scipy import integrate, interpolate, special
from scipy.integrate import solve_ivp

import gsl_floquet as gf

B_VALUES = [0.0, 0.45, 1.0, 2.5, 5.0, 20.0]
A_VALUES = [0.0, 0.7, 1.5, 2.2933, 3.1, 4.0]
U_VALUES = np.array([-2.9, -1.3, -0.4, 0.0, 0.25, 0.9, 1.57, 2.4, 3.0, 4.7, 6.0])


# --------------------------------------------------------------------------
# 0. Undressed GSL: closed forms are consistent with each other
# --------------------------------------------------------------------------

@pytest.mark.parametrize("b", B_VALUES)
def test_G_prime_is_F(b):
    """G_b'(u) = F_b(u) (central differences, h = 1e-5, tolerance 1e-9)."""
    h = 1e-5
    u = U_VALUES
    dG = (gf.G_gsl(u + h, b) - gf.G_gsl(u - h, b)) / (2 * h)
    assert np.max(np.abs(dG - gf.F_gsl(u, b))) < 1e-9


@pytest.mark.parametrize("b", B_VALUES)
def test_F_prime_closed_form(b):
    """F_b'(u) closed form (= V_Sch of the 2021 paper) vs central differences."""
    h = 1e-5
    u = U_VALUES
    dF = (gf.F_gsl(u + h, b) - gf.F_gsl(u - h, b)) / (2 * h)
    assert np.max(np.abs(dF - gf.dF_gsl(u, b))) < 1e-8 * (1 + b)


def test_G_sine_gordon_limit():
    """G_0(u) = 1 - cos u."""
    u = U_VALUES
    assert np.max(np.abs(gf.G_gsl(u, 0.0) - (1 - np.cos(u)))) < 1e-15


# --------------------------------------------------------------------------
# 1. Fourier coefficients of F_b
# --------------------------------------------------------------------------

@pytest.mark.parametrize("b", B_VALUES)
def test_fourier_coefficients_vs_quad(b):
    """c_m(b) from the FFT equals (2/pi) int_0^pi F_b sin(m u) du (quad) for m <= 6."""
    c = gf.fourier_coefficients(b)
    for m in range(1, 7):
        ref, _ = integrate.quad(lambda u: gf.F_gsl(u, b) * np.sin(m * u), 0, np.pi,
                                limit=400, epsabs=1e-13, epsrel=1e-12)
        ref *= 2 / np.pi
        cm = c[m - 1] if m <= c.size else 0.0     # truncated coefficients are < tol
        assert abs(cm - ref) < 1e-11, (b, m, cm, ref)


@pytest.mark.parametrize("b", [0.45, 1.0, 5.0, 20.0])
def test_fourier_coefficients_converged(b):
    """Doubling the number of FFT samples does not change the retained c_m."""
    c1 = gf.fourier_coefficients(b)
    n = gf._next_pow2(max(1024, 128 * (1.0 + b)))
    c2 = gf.fourier_coefficients(b, n_samples=2 * n)
    M = min(c1.size, c2.size)
    assert np.max(np.abs(c1[:M] - c2[:M])) < 1e-13


def test_fourier_coefficients_small_b_expansion():
    """c_1 = 1 - b^2/2 + O(b^4), c_2 = b^2/4 + O(b^4) (double sine-Gordon reduction)."""
    for b in [0.05, 0.1]:
        c = gf.fourier_coefficients(b)
        assert abs(c[0] - (1 - b**2 / 2)) < 2 * b**4
        assert abs(c[1] - b**2 / 4) < 2 * b**4


# --------------------------------------------------------------------------
# 2. Dressed nonlinearity: series vs direct quadrature, limits, symmetries
# --------------------------------------------------------------------------

@pytest.mark.parametrize("b", B_VALUES)
@pytest.mark.parametrize("a", A_VALUES)
def test_series_vs_quadrature_F(a, b):
    """F_{b,a}(u): Fourier-Bessel series vs trapezoidal phase average (tol 1e-11)."""
    d = gf.DressedGSL(a, b)
    ref = gf.F_dressed_quad(U_VALUES, a, b)
    assert np.max(np.abs(d.F(U_VALUES) - ref)) < 1e-11


@pytest.mark.parametrize("b", [0.45, 1.0, 5.0])
@pytest.mark.parametrize("a", [1.5, 2.2933, 4.0])
def test_series_vs_quadrature_G_and_dF(a, b):
    """G_{b,a}(u) - G_{b,a}(0) and F_{b,a}'(u): series vs quadrature (tol 1e-11)."""
    d = gf.DressedGSL(a, b)
    assert np.max(np.abs(d.G(U_VALUES) - gf.G_dressed_quad(U_VALUES, a, b))) < 1e-11
    assert np.max(np.abs(d.dF(U_VALUES) - gf.dF_dressed_quad(U_VALUES, a, b))) < 1e-11


@pytest.mark.parametrize("b", [1.0, 5.0])
@pytest.mark.parametrize("a", [1.5, 4.0])
def test_quadrature_vs_scipy_quad(a, b):
    """Trapezoidal phase average agrees with scipy.integrate.quad at a few points."""
    for u in [0.3, 1.2, 2.9]:
        ref, _ = integrate.quad(lambda th: gf.F_gsl(u + a * np.sin(th), b), 0, 2 * np.pi,
                                limit=400, epsabs=1e-13, epsrel=1e-12)
        ref /= 2 * np.pi
        assert abs(gf.F_dressed_quad(u, a, b) - ref) < 1e-11


@pytest.mark.parametrize("b", B_VALUES)
def test_a_zero_recovers_undressed(b):
    """F_{b,0} = F_b and G_{b,0} = G_b."""
    d = gf.DressedGSL(0.0, b)
    assert np.max(np.abs(d.F(U_VALUES) - gf.F_gsl(U_VALUES, b))) < 1e-12
    assert np.max(np.abs(d.G(U_VALUES) - gf.G_gsl(U_VALUES, b))) < 1e-12


@pytest.mark.parametrize("a", [0.7, 2.4048, 4.0])
def test_b_zero_is_dressed_sine_gordon(a):
    """F_{0,a}(u) = J_0(a) sin u exactly (dynamic localisation of the sine-Gordon term)."""
    d = gf.DressedGSL(a, 0.0)
    assert np.max(np.abs(d.F(U_VALUES) - special.j0(a) * np.sin(U_VALUES))) < 1e-13


@pytest.mark.parametrize("a", [1.0, 2.4048])
def test_small_b_expansion_dressed(a):
    """F_{b,a} = (1 - b^2/2) J_0(a) sin u + (b^2/4) J_0(2a) sin 2u + O(b^4):
    the residual decreases by ~16 when b is halved."""
    def residual(b):
        d = gf.DressedGSL(a, b)
        approx = ((1 - b**2 / 2) * special.j0(a) * np.sin(U_VALUES)
                  + (b**2 / 4) * special.j0(2 * a) * np.sin(2 * U_VALUES))
        return np.max(np.abs(d.F(U_VALUES) - approx))
    r1, r2 = residual(0.2), residual(0.1)
    assert r1 / r2 > 12.0 and r1 / r2 < 20.0


@pytest.mark.parametrize("b", [0.45, 2.5])
@pytest.mark.parametrize("a", [1.5, 3.1])
def test_symmetries(a, b):
    """F_{b,a} is odd, 2pi-periodic, even in a; G_{b,a} is even and 2pi-periodic."""
    d = gf.DressedGSL(a, b)
    dm = gf.DressedGSL(-a, b)
    u = U_VALUES
    assert np.max(np.abs(d.F(-u) + d.F(u))) < 1e-13
    assert np.max(np.abs(d.F(u + 2 * np.pi) - d.F(u))) < 1e-12
    assert np.max(np.abs(dm.F(u) - d.F(u))) < 1e-13
    assert np.max(np.abs(d.G(-u) - d.G(u))) < 1e-13
    assert np.max(np.abs(d.G(u + 2 * np.pi) - d.G(u))) < 1e-12


@pytest.mark.parametrize("b", [0.45, 1.0, 5.0])
@pytest.mark.parametrize("a", [1.5, 2.6])
def test_G_prime_is_F_dressed(a, b):
    """G_{b,a}'(u) = F_{b,a}(u) by central differences (h = 1e-5, tol 1e-9)."""
    d = gf.DressedGSL(a, b)
    h = 1e-5
    dG = (d.G(U_VALUES + h) - d.G(U_VALUES - h)) / (2 * h)
    assert np.max(np.abs(dG - d.F(U_VALUES))) < 1e-9


# --------------------------------------------------------------------------
# 3. Effective mass and the 2013 paper's Eq. (22)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("b", [0.45, 1.0, 2.5, 5.0])
@pytest.mark.parametrize("a", [0.0, 0.7, 1.7, 2.2933, 3.1, 4.0, 6.0])
def test_mass2_three_ways(a, b):
    """m_eff^2 = sum_m m c_m J_0(m a) = <F_b'(a sin theta)> = 2 delta / b^2 (Eq. 22)."""
    ms = gf.effective_mass2(a, b, "series")
    mq = gf.effective_mass2(a, b, "quad")
    mp = gf.effective_mass2(a, b, "paper")
    assert abs(ms - mq) < 1e-11
    assert abs(ms - mp) < 1e-9


@pytest.mark.parametrize("b", B_VALUES)
def test_mass2_at_a_zero_is_one(b):
    """m_eff^2(0, b) = F_b'(0) = 1 for every b (hence delta(0, b) = b^2 / 2)."""
    assert abs(gf.effective_mass2(0.0, b, "series") - 1.0) < 1e-12
    if b > 0:
        assert abs(gf.delta_paper(0.0, b) - b**2 / 2) < 1e-12


def test_first_zero_matches_paper_and_bessel():
    """a_*(1) = 2.293 (the '2.3' of the 2013 paper) and a_*(b) -> 2.404826 as b -> 0."""
    assert abs(gf.first_zero_mass2(1.0) - 2.2933) < 5e-4
    assert abs(gf.first_zero_mass2(0.45) - 2.3671) < 5e-4
    assert abs(gf.first_zero_mass2(1e-3) - special.jn_zeros(0, 1)[0]) < 1e-6


def test_mass2_equals_class_property():
    d = gf.DressedGSL(1.7, 1.0)
    assert abs(d.mass2 - gf.effective_mass2(1.7, 1.0, "series")) < 1e-14


# --------------------------------------------------------------------------
# 4. Vacuum structure
# --------------------------------------------------------------------------

def test_vacua_regime_I():
    """For m_eff^2 > 0 the only vacuum in [0, 2pi) is u = 0 and the barrier is at pi."""
    d = gf.DressedGSL(1.5, 1.0)
    assert d.mass2 > 0
    minima, maxima = d.critical_points()
    assert minima.size == 1 and abs(minima[0]) < 1e-12
    assert maxima.size == 1 and abs(maxima[0] - np.pi) < 1e-12


def _regime_II_window(b):
    """(a_*, a_**): zeros of F'(0) and F'(pi); between them the vacua are +-u*."""
    a1 = gf.first_zero_mass2(b, at=0.0)
    a2 = gf.first_zero_mass2(b, at=np.pi)
    assert a1 < a2
    return a1, a2


def test_regime_window_small_b():
    """The window (a_*, a_**) shrinks with b: width ~ b^2 |J_0(2 a_1)| / |J_0'(a_1)|."""
    a1 = special.jn_zeros(0, 1)[0]
    for b in [0.2, 0.1]:
        lo, hi = _regime_II_window(b)
        width = hi - lo
        # leading-order estimate: |c_1 J_0(a)| < 2 c_2 |J_0(2a)| with J_0(a) ~ J_0'(a1)(a - a1)
        est = 2 * (b**2 / 4) * abs(special.j0(2 * a1)) / abs(special.j1(a1)) * 2
        assert abs(width - est) < 0.25 * est
        assert lo < a1 < hi


def test_vacua_regime_II():
    """Inside (a_*, a_**) the vacua are +-u* (mod 2pi), degenerate, with 0 and pi barrier tops;
    at the zero of J_0 the vacua sit at +-pi/2 up to O(b^2)."""
    b = 0.45
    lo, hi = _regime_II_window(b)
    a = 0.5 * (lo + hi)
    d = gf.DressedGSL(a, b)
    assert d.regime() == "II"
    minima, maxima = d.critical_points()
    assert minima.size == 2
    assert abs(minima[0] + minima[1] - 2 * np.pi) < 1e-10      # u* and 2pi - u*
    assert maxima.size == 2 and abs(maxima[0]) < 1e-12 and abs(maxima[1] - np.pi) < 1e-12
    assert abs(d.G(minima[0]) - d.G(minima[1])) < 1e-13
    d0 = gf.DressedGSL(special.jn_zeros(0, 1)[0], 0.1)
    assert d0.regime() == "II"
    assert abs(d0.vacua()[0] - np.pi / 2) < 0.02


def test_vacua_regime_III():
    """Beyond a_** the vacuum is u = pi (inverted miniband) and u = 0 is a barrier top."""
    b = 0.45
    _, hi = _regime_II_window(b)
    d = gf.DressedGSL(hi + 0.3, b)
    assert d.regime() == "III"
    minima, maxima = d.critical_points()
    assert minima.size == 1 and abs(minima[0] - np.pi) < 1e-12
    assert maxima.size == 1 and abs(maxima[0]) < 1e-12


def test_regime_I_default():
    assert gf.DressedGSL(1.5, 1.0).regime() == "I"
    assert gf.DressedGSL(0.0, 5.0).regime() == "I"


# --------------------------------------------------------------------------
# 5. Static kinks
# --------------------------------------------------------------------------

def _check_ode_residual(d, x, u, tol):
    """u_xx = F(u) on a fine uniform grid from a cubic spline of the profile (weak check)."""
    spl = interpolate.CubicSpline(x, u)
    xf = np.linspace(x[0] + 0.5, x[-1] - 0.5, 4001)
    res = spl(xf, 2) - d.F(spl(xf))
    return np.max(np.abs(res)) < tol


def _check_against_ivp(d, u1, u2, x, u, tol):
    """Independent check: integrate u'' = F(u) from the centre with DOP853 and compare."""
    uc = u[np.argmin(np.abs(x))]
    p0 = np.sqrt(2.0 * float(d.dG_from(uc, u1)))
    mu = np.sqrt(float(d.dF(u1)))
    L = 6.0 / mu
    sol = solve_ivp(lambda t, y: [y[1], float(d.F(y[0]))], (0.0, L), [uc, p0],
                    method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    xs = np.linspace(0.0, L, 200)
    spl = interpolate.CubicSpline(x, u)
    return np.max(np.abs(sol.sol(xs)[0] - spl(xs))) < tol


def test_sine_gordon_kink_recovered():
    """a = b = 0: the static kink is 4 arctan(exp x) (1e-11 on the quadrature grid,
    1e-7 through the spline interpolation of kink_on_grid)."""
    d = gf.DressedGSL(0.0, 0.0)
    x, u = d.static_kink(0.0, 2 * np.pi)
    assert np.max(np.abs(u - 4 * np.arctan(np.exp(x)))) < 1e-11
    xg = np.linspace(-12, 12, 481)
    ug = d.kink_on_grid(0.0, 2 * np.pi, xg)
    assert np.max(np.abs(ug - 4 * np.arctan(np.exp(xg)))) < 1e-7


@pytest.mark.parametrize("b", [1.5, 10.0])
def test_undressed_kink_vs_arctan_quadrature(b):
    """a = 0: the kink agrees with the arctan-ansatz quadrature of the 2021 paper,
    (1/sqrt2) int_0^w sqrt(1 + sqrt(1 + 2 b^2 sech^2 t)) dt = x, u = 4 arctan(e^w)  (tol 1e-9)."""
    d = gf.DressedGSL(0.0, b)
    x, u = d.static_kink(0.0, 2 * np.pi)
    spl = interpolate.CubicSpline(x, u)
    for w in [-3.0, -1.0, 0.4, 2.0, 5.0]:
        xref, _ = integrate.quad(lambda t: np.sqrt(1 + np.sqrt(1 + 2 * b**2 / np.cosh(t)**2)), 0, w,
                                 epsabs=1e-13, epsrel=1e-12)
        xref /= np.sqrt(2.0)
        assert abs(spl(xref) - 4 * np.arctan(np.exp(w))) < 1e-7


def test_2pi_kink_regime_I():
    """2pi-kink for (a, b) = (1.5, 1.0): limits, centre, ODE residual, IVP check, energy."""
    d = gf.DressedGSL(1.5, 1.0)
    x, u = d.static_kink(0.0, 2 * np.pi)
    assert u[0] < 1e-7 and 2 * np.pi - u[-1] < 1e-7
    assert abs(u[np.argmin(np.abs(x))] - np.pi) < 1e-10
    assert np.all(np.diff(x) > 0) and np.all(np.diff(u) > 0)
    assert _check_ode_residual(d, x, u, 1e-3)
    assert _check_against_ivp(d, 0.0, 2 * np.pi, x, u, 1e-8)
    E_quad = d.kink_energy(0.0, 2 * np.pi)
    spl = interpolate.CubicSpline(x, u)
    xf = np.linspace(x[0], x[-1], 20001)
    E_grid = np.trapezoid(0.5 * spl(xf, 1)**2 + d.G(spl(xf)), xf)
    assert abs(E_quad - E_grid) < 1e-6 * E_quad
    assert 0 < E_quad < 8.0        # shallower than the sine-Gordon kink (E = 8) for this (a, b)


def test_two_kinks_regime_II():
    """In regime II there are two inequivalent kinks between adjacent vacua:
    through u = 0 (from -u* to u*) and through u = pi (from u* to 2pi - u*)."""
    b = 0.45
    lo, hi = _regime_II_window(b)
    a = lo + 0.25 * (hi - lo)          # off the symmetric point so that E1 != E2
    d = gf.DressedGSL(a, b)
    us = d.vacua()[0]
    x1, u1 = d.static_kink(-us, us)              # through 0
    x2, u2 = d.static_kink(us, 2 * np.pi - us)   # through pi
    assert _check_ode_residual(d, x1, u1, 1e-3) and _check_ode_residual(d, x2, u2, 1e-3)
    assert _check_against_ivp(d, -us, us, x1, u1, 1e-8)
    assert _check_against_ivp(d, us, 2 * np.pi - us, x2, u2, 1e-8)
    E1, E2 = d.kink_energy(-us, us), d.kink_energy(us, 2 * np.pi - us)
    assert E1 > 0 and E2 > 0 and abs(E1 - E2) > 1e-6 * (E1 + E2)


def test_inverted_2pi_kink_regime_III():
    """Regime III: the 2pi-kink from pi to 3pi (through 2pi) exists and satisfies the first integral."""
    b = 0.45
    _, hi = _regime_II_window(b)
    d = gf.DressedGSL(hi + 0.3, b)
    x, u = d.static_kink(np.pi, 3 * np.pi)
    assert abs(u[0] - np.pi) < 1e-7 and abs(u[-1] - 3 * np.pi) < 1e-7
    assert _check_ode_residual(d, x, u, 1e-3)
    assert _check_against_ivp(d, np.pi, 3 * np.pi, x, u, 1e-8)


def test_kink_rejects_non_degenerate_vacua():
    """A static kink between non-degenerate points must be refused."""
    d = gf.DressedGSL(1.5, 1.0)
    with pytest.raises(ValueError):
        d.static_kink(0.0, 1.0)


# --------------------------------------------------------------------------
# 6. Expansion about the vacuum u = pi (regime III)
# --------------------------------------------------------------------------

def test_shifted_to_pi_matches_the_original():
    """F_shift(w) = F(pi + w) and G_shift(w) = G(pi + w) - G(pi) to round-off."""
    d = gf.DressedGSL(2.76, 0.5)
    s = d.shifted(np.pi)
    w = np.linspace(-3.0, 3.0, 501)
    assert np.max(np.abs(s.F(w) - d.F(np.pi + w))) < 1e-13
    assert np.max(np.abs(s.dF(w) - d.dF(np.pi + w))) < 1e-13
    assert np.max(np.abs(s.G(w) - (d.G(np.pi + w) - d.G(np.pi)))) < 1e-12


def test_shifted_vacuum_is_a_minimum_in_regime_III():
    """In regime III the shifted nonlinearity has its vacuum at w = 0 with positive mass,
    equal to F'(pi) of the unshifted one."""
    b = 0.5
    a = gf.first_zero_mass2(b, at=np.pi) + 0.3
    d = gf.DressedGSL(a, b)
    assert d.regime() == "III"
    s = d.shifted(np.pi)
    assert s.mass2 > 0
    assert abs(s.mass2 - float(d.dF(np.pi))) < 1e-13
    assert s.regime() == "I"          # about its own vacuum it looks like regime I


def test_shifted_zero_is_the_identity():
    d = gf.DressedGSL(1.5, 1.0)
    s = d.shifted(0.0)
    w = np.linspace(-3.0, 3.0, 101)
    assert np.max(np.abs(s.F(w) - d.F(w))) < 1e-15


def test_shifted_rejects_other_vacua():
    d = gf.DressedGSL(2.42, 0.5)
    with pytest.raises(ValueError):
        d.shifted(1.5)
