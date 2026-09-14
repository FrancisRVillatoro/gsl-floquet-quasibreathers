"""
experiment_ratio.py -- the signed ratio K_3 / K_2 of Stokes constants, and the O(b^4) shift of a_c.

The canonical problem carrying both integrability-breaking harmonics is

    u_tt - u_xx + sin u + r_2 V_2(u) + r_3 V_3(u) = 0,     V_m(u) = sin(m u) - m sin(u),

whose linear mass is exactly 1 for any (r_2, r_3).  At first order the third-harmonic far
field is |A_3| = |r_2 K_2 + r_3 K_3|, so fixing r_3 and scanning r_2 locates a cancellation
at r_2* = -r_3 K_3 / K_2, which gives the ratio WITH ITS SIGN.  The ratio is extracted at
two amplitudes and Richardson-extrapolated to zero, because the O(rho) correction to
linearity is about -31 rho and is not negligible at r ~ 0.01.

Section 9 of the paper predicts that the same ratio fixes the O(b^4) shift of the
cancellation point of the GSL equation,

    a_c(b) - j_01/2 = [c_3(b) J_0(3 a_0) / (2 c_2(b) J_1(2 a_0))] (K_3 / K_2),   a_0 = j_01/2,

so the two determinations are independent and must agree.

Run:  python experiment_ratio.py     (about 2 minutes)
"""

import time
import numpy as np
from scipy import special

import gsl_floquet as gf
from qb_newton import HarmonicBalance, sg_breather_harmonics, emitted_harmonics
from kg_spectral import SpectralKG

K_HARM, L, N = 10, 160.0, 1024
SPONGE = dict(x_s=50.0, width=25.0, sigma0=0.8)


def canonical_pair(r2, r3, Omega, periods=120, n_fit=60, m=3):
    """|A_3| of the quasi-breather of the two-harmonic canonical problem
    sin u + r2 V_2 + r3 V_m, with V_m = sin(m u) - m sin u (default m = 3)."""
    F = lambda u: np.sin(u) + r2 * (np.sin(2 * u) - 2 * np.sin(u)) + r3 * (np.sin(m * u) - m * np.sin(u))
    dF = lambda u: np.cos(u) + r2 * (2 * np.cos(2 * u) - 2 * np.cos(u)) + r3 * (m * np.cos(m * u) - m * np.cos(u))
    G = lambda u: ((1 - np.cos(u)) + r2 * (0.5 * (1 - np.cos(2 * u)) - 2 * (1 - np.cos(u)))
                   + r3 * ((1 - np.cos(m * u)) / m - m * (1 - np.cos(u))))
    hb = HarmonicBalance(L, N, F, dF, K=K_HARM, kappa=1.0)
    U, info = hb.solve(sg_breather_harmonics(hb.x, Omega, K_HARM), Omega, tol=1e-11, maxit=30)
    kg = SpectralKG(L, N, F, G, kappa=1.0, sponge=SPONGE, G_ref=0.0)
    u0, v0 = hb.initial_data(U, Omega)
    kg.set_initial(u0, v0)
    return emitted_harmonics(kg, Omega, 34.0, n_periods=periods, n_fit=n_fit)[3], info["converged"]


def zero_of_the_pair(r3, scan, Omega):
    """r_2* where |A_3| vanishes, from a signed linear fit through the V."""
    A = np.array([canonical_pair(r2, r3, Omega)[0] for r2 in scan])
    s = np.array(scan)
    k = int(np.argmin(A))
    best = None
    for flip in (1.0, -1.0):
        sgn = np.where(s < s[k] + 1e-12, -1.0, 1.0)
        sgn[k] = flip
        p = np.polyfit(s, sgn * A, 1)
        r = np.max(np.abs(np.polyval(p, s) - sgn * A))
        if best is None or r < best[2]:
            best = (p, -p[1] / p[0], r)
    return best[1], best[2], A


if __name__ == "__main__":
    Omega = 0.80
    print(f"K_3/K_2 from the cancellation of the combined canonical perturbation, Omega = {Omega}")
    ratios = {}
    for r3, scan in [(0.005, [0.0055, 0.0063, 0.0071, 0.0079]),
                     (0.0025, [0.0028, 0.0032, 0.0036, 0.0040])]:
        t0 = time.perf_counter()
        z, res, A = zero_of_the_pair(r3, scan, Omega)
        ratios[r3] = z / r3
        print(f"   r_3 = {r3:.4f}   |A_3| = {np.array2string(A, precision=3)}")
        print(f"                r_2* = {z:.6f}   r_2*/r_3 = {z/r3:.4f}   residual {res:.2e}   "
              f"({time.perf_counter()-t0:.0f} s)")
    lo, hi = 0.0025, 0.005
    ratio = -(2 * ratios[lo] - ratios[hi])
    print(f"   Richardson to zero amplitude:  K_3/K_2 = {ratio:.4f}")

    print(f"\nK_4/K_2 from the cancellation of sin u + r_2 V_2 + r_4 V_4, Omega = {Omega}")
    ratios4 = {}
    for r4, scan in [(0.0004, [0.0015, 0.0022, 0.0029, 0.0036]),
                     (0.0002, [0.00075, 0.0011, 0.00145, 0.0018])]:
        t0 = time.perf_counter()
        A = np.array([canonical_pair(r2, r4, Omega, m=4)[0] for r2 in scan])
        s = np.array(scan); k = int(np.argmin(A)); best = None
        for flip in (1.0, -1.0):
            sgn = np.where(s < s[k] + 1e-12, -1.0, 1.0); sgn[k] = flip
            p = np.polyfit(s, sgn * A, 1); res = np.max(np.abs(np.polyval(p, s) - sgn * A))
            if best is None or res < best[2]:
                best = (p, -p[1] / p[0], res)
        ratios4[r4] = best[1] / r4
        print(f"   r_4 = {r4:.4f}   r_2*/r_4 = {best[1]/r4:.4f}   ({time.perf_counter()-t0:.0f} s)")
    ratio4 = -(2 * ratios4[0.0002] - ratios4[0.0004])
    print(f"   Richardson to zero amplitude:  K_4/K_2 = {ratio4:.4f}   (inner problem, eps -> 0: -25.32)")

    a0 = special.jn_zeros(0, 1)[0] / 2
    print(f"\ncross-check against the measured a_c(b)  (a_0 = j01/2 = {a0:.7f})")
    print("     b      term from V_3    term from V_4    predicted a_c    measured a_c")
    for b, ac_meas in [(0.125, 1.2054), (0.25, 1.2148), (0.5, 1.2535)]:
        c = gf.fourier_coefficients(b)
        den = 2 * c[1] * special.j1(2 * a0)
        t3 = c[2] * special.j0(3 * a0) * ratio / den
        t4 = c[3] * special.j0(4 * a0) * ratio4 / den
        print(f"   {b:.3f}   {t3:+.5f}         {t4:+.5f}         {a0 + t3 + t4:.4f}           {ac_meas:.4f}")
    print("   (V_3 gives the b^2 term, V_4 the b^4 term; what is left at b = 0.5 is O(b^6))")
