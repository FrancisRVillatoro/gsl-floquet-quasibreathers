"""
experiment_second_zero.py -- the second cancellation, at the second zero of J_0(2a).

The paper predicts that the radiation of the quasi-breather is switched off whenever
the second miniband harmonic is dynamically localised, J_0(2a) = 0.  The first zero,
a = j_01/2 = 1.20241, lies in regime I and has been verified.  The second, a = j_02/2 =
2.76004, lies in regime III, where J_0(a) < 0, the vacuum is u = pi and the miniband is
inverted.  Expanding about that vacuum, u = pi + w,

    F(pi + w) = sum_m (-1)^m c_m(b) J_0(m a) sin(m w),

so the perturbation parameters change sign relative to regime I,

    rho_2' = -c_2 J_0(2a) / (c_1 J_0(a)),     rho_3' = +c_3 J_0(3a) / (c_1 J_0(a)),

while the leading cancellation is still exactly at J_0(2a) = 0.  Using the same ratio
K_3/K_2 = -1.372 measured in experiment_ratio.py, the O(b^4) shift becomes

    a_c - j_02/2 = -rho_3' (K_3/K_2) / (d rho_2'/da),   d rho_2'/da = 2 c_2 J_1(2a)/(c_1 J_0(a)),

which is negative here and about 2.6 times smaller in magnitude than in regime I: the
predicted coefficients are -0.075 b^2, -0.072 b^2 and -0.062 b^2 for b = 0.125, 0.25, 0.5,
against +0.194 b^2 at the first zero.  A sign flip and a change of magnitude, with no free
parameter, is a sharp test of the mechanism.

Run:  python experiment_second_zero.py     (about 6 minutes)
"""

import time
import numpy as np
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from qb_newton import (HarmonicBalance, sg_breather_harmonics, small_amplitude_coefficients,
                       emitted_harmonics)
from kg_spectral import SpectralKG

K_HARM, L, N = 10, 200.0, 1024
SPONGE = dict(x_s=65.0, width=30.0, sigma0=0.5)
X_PROBE = 45.0
OMEGA = 0.8
K3K2 = -1.372
A0 = special.jn_zeros(0, 2)[1] / 2.0


def third_harmonic_amplitude(a, b, u_v, Omega=OMEGA, periods=120, n_fit=60):
    """|A_3| emitted by the quasi-breather sitting on the vacuum u_v (0 or pi)."""
    d = gf.DressedGSL(a, b).shifted(u_v)
    k1, k3, _ = small_amplitude_coefficients(d)
    if k3 <= 0:
        return np.nan, False, np.nan
    omega = Omega * np.sqrt(k1)
    hb = HarmonicBalance(L, N, d.F, d.dF, K=K_HARM, kappa=k1)
    U, info = hb.solve(np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, Omega, K_HARM),
                       omega, tol=1e-11, maxit=30)
    kg = SpectralKG(L, N, d.F, d.G, kappa=k1, sponge=SPONGE, G_ref=0.0)
    u0, v0 = hb.initial_data(U, omega)
    kg.set_initial(u0, v0)
    A = emitted_harmonics(kg, omega, X_PROBE, n_periods=periods,
                          samples_per_period=128, n_fit=n_fit)
    return A[3], info["converged"], hb.core_amplitude(U)


def predicted_shift(b):
    c = gf.fourier_coefficients(b)
    drho2 = 2 * c[1] * special.j1(2 * A0) / (c[0] * special.j0(A0))
    rho3 = c[2] * special.j0(3 * A0) / (c[0] * special.j0(A0))
    return -rho3 * K3K2 / drho2


if __name__ == "__main__":
    print(f"second zero of J_0(2a):  a_0' = j_02/2 = {A0:.7f}")
    print("regime and breather size at a_0':")
    for b in [0.25, 0.5]:
        d = gf.DressedGSL(A0, b)
        s = d.shifted(np.pi)
        k1, k3, _ = small_amplitude_coefficients(s)
        print(f"   b = {b:.3f}   regime {d.regime()}   kappa = {k1:.5f}   kappa_3 = {k3:.5f}   "
              f"lambda = {np.sqrt(k1/k3):.4f}   predicted a_c = {A0 + predicted_shift(b):.4f}")

    results = {}
    for b, scan in [(0.5, [2.712, 2.720, 2.728, 2.736, 2.744]),
                    (0.25, [2.750, 2.754, 2.757, 2.760, 2.764]),
                    (0.125, [2.7565, 2.7580, 2.7590, 2.7600, 2.7615])]:
        print(f"\nscan at b = {b} (predicted a_c = {A0 + predicted_shift(b):.4f})")
        As, cores = [], []
        for a in scan:
            t0 = time.perf_counter()
            A, conv, core = third_harmonic_amplitude(a, b, np.pi)
            As.append(A); cores.append(core)
            print(f"   a = {a:.4f}   |A_3| = {A:.5e}   core = {core:.4f} ({core/np.pi:.2f} pi)   "
                  f"conv = {conv}   ({time.perf_counter()-t0:.0f} s)")
        As = np.array(As); s = np.array(scan)
        k = int(np.argmin(As))
        best = None
        for flip in (1.0, -1.0):
            sgn = np.where(s < s[k] + 1e-12, -1.0, 1.0); sgn[k] = flip
            p = np.polyfit(s, sgn * As, 1)
            r = np.max(np.abs(np.polyval(p, s) - sgn * As))
            if best is None or r < best[2]:
                best = (p, -p[1] / p[0], r)
        ac = best[1]
        results[b] = (s, As, ac)
        print(f"   -> measured a_c = {ac:.4f}   predicted {A0 + predicted_shift(b):.4f}   "
              f"(shift measured {ac-A0:+.4f}, predicted {predicted_shift(b):+.4f}; "
              f"coefficient {(ac-A0)/b**2:+.4f} b^2 vs {predicted_shift(b)/b**2:+.4f} b^2)")

    bs = np.array(sorted(results))
    co = np.array([(results[b][2] - A0) / b ** 2 for b in bs])
    ex = np.polyfit(bs[:2] ** 2, co[:2], 1)[1]
    lim = -(3 / 8) * special.j0(3 * A0) / (2 * special.j1(2 * A0)) * K3K2
    print(f"\ncoefficient of the b^2 shift: measured {co} , extrapolated to b -> 0 = {ex:+.4f}")
    print(f"predicted b -> 0 limit (c_3/c_2 -> 3 b^2/8) = {lim:+.5f}; "
          f"the same formula at the first zero gives {-(3/8)*special.j0(3*(special.jn_zeros(0,1)[0]/2))/(2*special.j1(2*(special.jn_zeros(0,1)[0]/2)))*K3K2*-1:+.5f}")

    fig, ax = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.2))
    for i, (b, (s, As, ac)) in enumerate(results.items()):
        ax[i].plot(s, As, "o-")
        ax[i].axvline(A0, color="gray", ls="--", lw=0.8)
        ax[i].axvline(A0 + predicted_shift(b), color="C1", ls=":", lw=1.2)
        ax[i].axvline(ac, color="C3", ls="-", lw=0.8)
        ax[i].set_xlabel("a"); ax[i].set_ylabel("$|A_3|$")
        ax[i].set_title(f"b = {b}: measured {ac:.4f}, predicted {A0+predicted_shift(b):.4f}",
                        fontsize=10)
    fig.tight_layout(); fig.savefig("experiment_second_zero.png", dpi=150)
    print("\nsaved experiment_second_zero.png")
