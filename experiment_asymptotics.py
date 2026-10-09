"""
experiment_asymptotics.py -- exponential asymptotics of the quasi-breather radiation.

The analysis is written up in the paper.  In the variables rescaled by the linear
mass (x, t -> sqrt(kappa) x, sqrt(kappa) t; u = lambda w with lambda = sqrt(kappa/kappa_3))
the equation is sine-Gordon plus corrections, the breather has amplitude parameter
epsilon = sqrt(1 - Omega^2), and the third-harmonic radiation is beyond all orders:

    |A_3| ~ K(a, b) epsilon^gamma exp(-pi q_3 / (2 epsilon)),    q_3 = sqrt(8 - 9 epsilon^2),

where the exponent comes from the singularity of sech(epsilon x) at x = i pi / (2 epsilon)
and is therefore the SAME for every (a, b), while the Stokes constant K carries all the
dependence on the nonlinearity.  Two predictions follow.

(P1)  a_c, defined by K(a_c, b) = 0, does not depend on Omega, whereas the depth of the
      minimum is governed by exp(-pi q_3 / epsilon) and changes by orders of magnitude.
      Test: fit ln sqrt(P) = alpha - beta pi q_3 / (2 epsilon) + gamma ln epsilon at fixed
      a and check beta = 1.

(P2)  At b = 0 the equation is sine-Gordon with mass J_0(a), which is integrable for every
      a, so K(a, 0) = 0 identically and K = b^2 K_2(a) + O(b^4).  The only term that breaks
      integrability at O(b^2) is (b^2/4) J_0(2a) sin(2u) (the sin u term only renormalises
      the mass), so K_2(a) is proportional to J_0(2a) and

          a_c(b) -> j_{0,1} / 2 = 1.2024136  as b -> 0,   P ~ b^4 at fixed a.

      Tests: P(b) at fixed a, and a_c(b) from a fit of P = C (a - a_c)^2 + F.

Run:  python experiment_asymptotics.py     (about 12 minutes)
"""

import time
import numpy as np
from scipy import optimize, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from qb_newton import HarmonicBalance, sg_breather_harmonics, small_amplitude_coefficients
from kg_spectral import SpectralKG
from results_io import save_csv

K_HARM, L, N = 10, 140.0, 1024
SPONGE = dict(x_s=45.0, width=22.0, sigma0=0.8)


def radiated_power(a, b, Omega, periods=120, fit=60, L=L, N=N, sponge=SPONGE):
    d = gf.DressedGSL(a, b)
    k1, k3, _ = small_amplitude_coefficients(d)
    if k3 <= 0:
        return np.nan
    omega = Omega * np.sqrt(k1)
    hb = HarmonicBalance(L, N, d.F, d.dF, K=K_HARM, kappa=k1)
    U, info = hb.solve(np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, Omega, K_HARM),
                       omega, tol=1e-11, maxit=25)
    if not info["converged"]:
        return np.nan
    kg = SpectralKG(L, N, d.F, d.G, kappa=k1, sponge=sponge, G_ref=0.0)
    u0, v0 = hb.initial_data(U, omega)
    kg.set_initial(u0, v0)
    E0 = kg.energy()
    T = 2.0 * np.pi / omega
    rec = []
    kg.run_until(min(0.1, 0.9 * kg.h_max), periods * T,
                 observer=lambda kg, u: rec.append((kg.t, kg.absorbed)), every=100)
    rec = np.array(rec)
    sel = rec[:, 0] > (periods - fit) * T
    return np.polyfit(rec[sel, 0] / T, rec[sel, 1] / E0, 1)[0]


if __name__ == "__main__":
    # ================================================================ (P1) the exponent
    print("(P1)  frequency dependence at fixed a = 0.6, b = 0.5")
    print("   Omega   epsilon    q_3     pi q_3/(2 eps)     P            ln sqrt(P)")
    Omegas = np.array([0.72, 0.76, 0.80, 0.84, 0.88, 0.92])
    P1 = []
    for Om in Omegas:
        t0 = time.perf_counter()
        p = radiated_power(0.6, 0.5, Om)
        P1.append(p)
        eps = np.sqrt(1 - Om ** 2); q3 = np.sqrt(8 - 9 * eps ** 2)
        print(f"   {Om:.2f}   {eps:.4f}   {q3:.4f}   {np.pi*q3/(2*eps):8.4f}    {p:.4e}   "
              f"{0.5*np.log(p):8.4f}   ({time.perf_counter()-t0:.0f} s)")
    P1 = np.array(P1)
    eps = np.sqrt(1 - Omegas ** 2)
    save_csv("power_vs_Omega_a0.6_b0.5.csv",
             ["Omega", "epsilon", "P_per_period_over_E0"], zip(Omegas, eps, P1))
    q3 = np.sqrt(8 - 9 * eps ** 2)
    E = np.pi * q3 / (2 * eps)
    M = np.column_stack([np.ones_like(E), -E, np.log(eps)])
    coef, *_ = np.linalg.lstsq(M, 0.5 * np.log(P1), rcond=None)
    res = 0.5 * np.log(P1) - M @ coef
    print(f"   fit ln sqrt(P) = alpha - beta pi q_3/(2 eps) + gamma ln eps:")
    print(f"     alpha = {coef[0]:.4f}   beta = {coef[1]:.4f}   gamma = {coef[2]:.4f}   "
          f"max residual = {np.max(np.abs(res)):.4f}")
    print(f"   beta = 1 is the prediction; a fit without the algebraic term gives "
          f"beta = {np.linalg.lstsq(M[:, :2], 0.5*np.log(P1), rcond=None)[0][1]:.4f}")

    # ================================================================ (P2a) the b^4 law
    print("\n(P2a) b dependence at fixed a = 0.6, Omega = 0.8   (prediction P ~ b^4)")
    bs = np.array([0.125, 0.25, 0.5, 1.0])
    P2 = []
    for b in bs:
        t0 = time.perf_counter()
        p = radiated_power(0.6, b, 0.8)
        P2.append(p)
        print(f"   b = {b:.3f}   P = {p:.4e}   P/b^4 = {p/b**4:.4e}   ({time.perf_counter()-t0:.0f} s)")
    P2 = np.array(P2)
    rho_exact = []
    for b in bs:
        c = gf.fourier_coefficients(b)
        rho_exact.append(c[1] * special.j0(1.2) / (c[0] * special.j0(0.6)))
    save_csv("power_vs_b_a0.6_Om0.8.csv",
             ["b", "rho_exact", "P_per_period_over_E0"], zip(bs, rho_exact, P2))
    sl = np.polyfit(np.log(bs), np.log(P2), 1)[0]
    print(f"   measured exponent d ln P / d ln b = {sl:.3f}   (prediction 4)")

    # ================================================================ (P2b) a_c(b)
    print("\n(P2b) the cancellation point vs b   (prediction a_c -> j_01/2 = 1.2024136)")
    windows = {0.125: (1.19, 1.225), 0.25: (1.19, 1.245), 0.5: (1.225, 1.285), 1.0: (1.28, 1.42)}
    acs, acs_err = [], []
    for b in bs:
        lo, hi = windows[b]
        aa = np.linspace(lo, hi, 5)
        pp = []
        for a in aa:
            t0 = time.perf_counter()
            pp.append(radiated_power(a, b, 0.8))
        pp = np.array(pp)

        def model(par, aa=aa):
            C, ac, F = par
            return C * (aa - ac) ** 2 + abs(F)

        ok = np.isfinite(pp)
        aa, pp = aa[ok], pp[ok]
        guess = [np.nanmax(pp) / (0.5 * (hi - lo)) ** 2, aa[int(np.argmin(pp))], np.nanmin(pp)]
        sol = optimize.least_squares(lambda par: (model(par, aa) - pp) / pp, guess)
        ac = sol.x[1]
        J = sol.jac
        cov = np.linalg.inv(J.T @ J) * np.sum(sol.fun ** 2) / max(len(aa) - 3, 1)
        err = np.sqrt(abs(cov[1, 1]))
        acs.append(ac); acs_err.append(err)
        print(f"   b = {b:.3f}   a values {np.round(aa,4)}")
        print(f"              P values {np.array2string(pp, precision=3)}")
        print(f"              a_c = {ac:.4f} +- {err:.4f}   floor = {abs(sol.x[2]):.2e}")
    acs = np.array(acs); acs_err = np.array(acs_err)
    j01_2 = special.jn_zeros(0, 1)[0] / 2
    sl2 = np.polyfit(bs ** 2, acs, 1)
    print(f"   linear fit a_c = A + B b^2:  A = {sl2[1]:.4f}  (prediction {j01_2:.4f}),  B = {sl2[0]:.4f}")
    print(f"   using only b <= 0.5:  A = {np.polyfit(bs[:3]**2, acs[:3], 1)[1]:.4f}")
    # Prediction column is filled by experiment_ratio.py, which computes K_3/K_2 and K_4/K_2.
    save_csv("ac_first_zero_vs_b_Om0.8.csv",
             ["b", "a_c_measured", "a_c_predicted_V3_V4"],
             [(b, ac, np.nan) for b, ac in zip(bs[:3], acs[:3])])

    # ================================================================ figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogy(E, P1, "o")
    xx = np.linspace(E.min(), E.max(), 50)
    ax[0].semilogy(xx, np.exp(2 * (coef[0] - coef[1] * xx)) * np.interp(xx, E, eps ** (2 * coef[2])), "-", lw=0.8)
    ax[0].set_xlabel(r"$\pi q_3/(2\epsilon)$"); ax[0].set_ylabel("$P/E_0$ per period")
    ax[0].set_title(f"(P1) exponent: $\\beta$ = {coef[1]:.3f} (predicted 1)")
    ax[1].loglog(bs, P2, "o-")
    ax[1].loglog(bs, P2[2] * (bs / 0.5) ** 4, "--", lw=0.8, label="$b^4$")
    ax[1].set_xlabel("b"); ax[1].set_ylabel("$P/E_0$ per period"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"(P2a) slope = {sl:.2f} (predicted 4)")
    ax[2].errorbar(bs ** 2, acs, yerr=acs_err, fmt="o")
    bb = np.linspace(0, 1.05, 20)
    ax[2].plot(bb, np.polyval(sl2, bb), "-", lw=0.8)
    ax[2].axhline(j01_2, color="gray", ls="--", lw=0.8)
    ax[2].annotate("$j_{0,1}/2$", (0.02, j01_2), fontsize=8, va="bottom")
    ax[2].set_xlabel("$b^2$"); ax[2].set_ylabel("$a_c$")
    ax[2].set_title(f"(P2b) $a_c(0^+)$ = {sl2[1]:.4f}")
    fig.tight_layout(); fig.savefig("experiment_asymptotics.png", dpi=150)
    print("\nsaved experiment_asymptotics.png")
