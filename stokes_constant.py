"""
stokes_constant.py -- numerical determination of the Stokes constant K_2.

The paper reduces the b -> 0 limit of the Floquet-dressed GSL equation to a single
canonical problem.  After rescaling by the linear mass, the dressed nonlinearity is

    Ftilde(u) = (sin u + rho sin 2u) / (1 + 2 rho) = sin u + rho V(u) + O(rho^2),
    V(u) = sin(2u) - 2 sin(u) = -u^3 + u^5/4 - ...,
    rho  = c_2(b) J_0(2a) / (c_1(b) J_0(a)),

so that the mass is exactly 1 for every rho and V'(0) = V''(0) = 0.  At rho = 0 the
equation is sine-Gordon and the breather does not radiate at all; the radiation is
therefore linear in rho at leading order, and the object to compute is

    Kcal(epsilon) = lim_{rho -> 0} |A_3| / rho,     epsilon = sqrt(1 - Omega^2),

the third-harmonic far-field amplitude per unit rho of the breather of reduced frequency
Omega.  The asymptotic prediction of Section 3 of the paper is

    Kcal(epsilon) ~ C epsilon^gamma exp(-pi sqrt2 / epsilon),

and C is the Stokes constant.  Both C and gamma are obtained here by fitting Kcal over a
range of epsilon, exactly as Segur and Kruskal had to determine the analogous constant for
the phi^4 breather numerically.

The script does four things.

  (1) checks that |A_3| / rho is independent of rho (linearity in the perturbation);
  (2) computes Kcal(epsilon) over a range of Omega;
  (3) fits ln Kcal + pi sqrt2 / epsilon = ln C + gamma ln epsilon;
  (4) tests the resulting prediction |A_3| = rho Kcal(epsilon) against a direct measurement
      on the GSL equation itself at small b, where no fitted quantity intervenes.

Run:  python stokes_constant.py     (about 8 minutes)
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

K_HARM, L, N = 10, 160.0, 1024
SPONGE = dict(x_s=50.0, width=25.0, sigma0=0.8)
X_PROBE = 34.0
PI_SQRT2 = np.pi * np.sqrt(2.0)


def dsg(rho):
    """Nonlinearity and potential of the canonical problem (mass exactly 1)."""
    F = lambda u: np.sin(u) + rho * (np.sin(2 * u) - 2 * np.sin(u))
    dF = lambda u: np.cos(u) + rho * (2 * np.cos(2 * u) - 2 * np.cos(u))
    G = lambda u: (1 - np.cos(u)) + rho * (0.5 * (1 - np.cos(2 * u)) - 2 * (1 - np.cos(u)))
    return F, dF, G


def third_harmonic(F, dF, G, kappa, Omega, seed, periods=120, n_fit=60, L=L, N=N):
    """Stationary quasi-breather at reduced frequency Omega, then |A_k| at the probe."""
    omega = Omega * np.sqrt(kappa)
    hb = HarmonicBalance(L, N, F, dF, K=K_HARM, kappa=kappa)
    U, info = hb.solve(seed(hb.x), omega, tol=1e-11, maxit=30)
    if not info["converged"]:
        return None, info
    kg = SpectralKG(L, N, F, G, kappa=kappa, sponge=SPONGE, G_ref=0.0)
    u0, v0 = hb.initial_data(U, omega)
    kg.set_initial(u0, v0)
    A = emitted_harmonics(kg, omega, X_PROBE, n_periods=periods, n_fit=n_fit)
    return A, info


def canonical(rho, Omega, **kw):
    F, dF, G = dsg(rho)
    seed = lambda x: sg_breather_harmonics(x, Omega, K_HARM)
    return third_harmonic(F, dF, G, 1.0, Omega, seed, **kw)


if __name__ == "__main__":
    # ================================================================ (1) linearity in rho
    print("(1) linearity of the emission in rho, at Omega = 0.8")
    print("    rho        |A_3|          |A_3|/rho      |A_5|")
    for rho in [0.005, 0.01, 0.02, 0.04]:
        t0 = time.perf_counter()
        A, info = canonical(rho, 0.8)
        print(f"   {rho:.4f}   {A[3]:.5e}   {A[3]/rho:.5e}   {A[5]:.3e}   ({time.perf_counter()-t0:.0f} s)")

    # ================================================================ (2) Kcal(epsilon)
    print("\n(2) Kcal(epsilon) = |A_3|/rho, Richardson-extrapolated to rho = 0 from rho = 0.005, 0.01")
    print("    Omega   epsilon   |A3|/rho (0.005)  |A3|/rho (0.01)   Kcal(0)     ln Kcal + pi sqrt2/eps")
    Omegas = np.array([0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.93])
    Kc, eps = [], []
    for Om in Omegas:
        t0 = time.perf_counter()
        k1_ = canonical(0.005, Om)[0][3] / 0.005
        k2_ = canonical(0.010, Om)[0][3] / 0.010
        k = 2.0 * k1_ - k2_                       # removes the O(rho) bias
        e = np.sqrt(1 - Om ** 2)
        Kc.append(k); eps.append(e)
        print(f"   {Om:.2f}   {e:.4f}   {k1_:.5e}      {k2_:.5e}     {k:.5e}   "
              f"{np.log(k)+PI_SQRT2/e:8.4f}   ({time.perf_counter()-t0:.0f} s)")
    Kc = np.array(Kc); eps = np.array(eps)

    # ================================================================ (3) the fit
    y = np.log(Kc) + PI_SQRT2 / eps
    p = np.polyfit(np.log(eps), y, 1)
    gamma, lnC = p[0], p[1]
    resid = y - np.polyval(p, np.log(eps))
    print(f"\n(3) fit  ln Kcal = ln C + gamma ln eps - pi sqrt2 / eps")
    print(f"    gamma = {gamma:.4f}    C = {np.exp(lnC):.4f}    max residual = {np.max(np.abs(resid)):.4f}")
    for g_fix in [0.0, 0.5, 1.0]:
        c = np.mean(y - g_fix * np.log(eps))
        r = np.max(np.abs(y - g_fix * np.log(eps) - c))
        print(f"    gamma fixed to {g_fix:.1f}:  C = {np.exp(c):.4f}   max residual = {r:.4f}")

    # ================================================================ (4) prediction vs the GSL
    print("\n(4) prediction for the GSL equation at small b (no fitted quantity beyond C, gamma)")
    print("     b      a     rho          Kcal(eps)     predicted |A_3|   measured |A_3|   ratio")
    Omega = 0.8
    e0 = np.sqrt(1 - Omega ** 2)
    Kc0 = np.exp(lnC) * e0 ** gamma * np.exp(-PI_SQRT2 / e0)
    rows = []
    for b, a in [(0.125, 0.6), (0.25, 0.6), (0.125, 1.0), (0.5, 0.6)]:
        d = gf.DressedGSL(a, b)
        k1, k3, _ = small_amplitude_coefficients(d)
        c = gf.fourier_coefficients(b)
        rho_g = c[1] * special.j0(2 * a) / (c[0] * special.j0(a))
        seed = lambda x, k1=k1, k3=k3: np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * x, Omega, K_HARM)
        t0 = time.perf_counter()
        A, info = third_harmonic(d.F, d.dF, d.G, k1, Omega, seed)
        pred = rho_g * Kc0
        rows.append((b, a, rho_g, pred, A[3]))
        print(f"   {b:.3f}  {a:.1f}   {rho_g:.5f}    {Kc0:.4e}    {pred:.4e}       {A[3]:.4e}     "
              f"{A[3]/pred:6.3f}   ({time.perf_counter()-t0:.0f} s)")

    # ================================================================ figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(1.0 / eps, Kc, "o", label="measured")
    xx = np.linspace((1 / eps).min(), (1 / eps).max(), 100)
    ax[0].semilogy(xx, np.exp(lnC) * (1 / xx) ** gamma * np.exp(-PI_SQRT2 * xx), "-", lw=0.8,
                   label=f"$C\\,\\epsilon^{{{gamma:.2f}}}\\,e^{{-\\pi\\sqrt{{2}}/\\epsilon}}$, $C$ = {np.exp(lnC):.1f}")
    ax[0].set_xlabel(r"$1/\epsilon$"); ax[0].set_ylabel(r"$\mathcal{K}=|A_3|/\rho$")
    ax[0].set_title("Stokes constant of the double sine-Gordon perturbation"); ax[0].legend(fontsize=8)
    ax[1].plot(np.log(eps), y, "o")
    ax[1].plot(np.log(eps), np.polyval(p, np.log(eps)), "-", lw=0.8)
    ax[1].set_xlabel(r"$\ln\epsilon$"); ax[1].set_ylabel(r"$\ln\mathcal{K}+\pi\sqrt{2}/\epsilon$")
    ax[1].set_title(f"$\\gamma$ = {gamma:.3f}, residual < {np.max(np.abs(resid)):.3f}")
    fig.tight_layout(); fig.savefig("stokes_constant.png", dpi=150)
    print("\nsaved stokes_constant.png")
