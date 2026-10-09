"""
experiment_ac_map.py -- map a_c(b, Omega), plus the b=1 continuation at Omega=0.92.

Every number written to the two published CSV files is recomputed here.  No published
cancellation point is hard-coded.  For b=0.25 and b=0.5 the third-harmonic zero is found
from a five-point scan of |A_3|.  For b=1, where direct solves from the sine-Gordon seed
cease to be reliable at large a, the harmonic-balance solution is continued in a and the
zero is refined on a denser continuation grid.
"""

import time
import numpy as np
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from experiment_second_zero import third_harmonic_amplitude
from qb_newton import (HarmonicBalance, sg_breather_harmonics,
                       small_amplitude_coefficients, emitted_harmonics)
from kg_spectral import SpectralKG
from results_io import save_csv

A0 = special.jn_zeros(0, 1)[0] / 2


def alpha2_model(Omega):
    """Only a scan-centering model; no published value is taken from it."""
    x = np.array([0.600 ** 2, 0.341 ** 2, 0.0])
    y = np.array([0.194, 0.000, -0.184])
    p = np.polyfit(x, y, 2)
    return float(np.polyval(p, 1 - Omega ** 2))


def signed_zero(s, A):
    """Zero from the signed-linear V fit used throughout the paper."""
    s = np.asarray(s, float); A = np.asarray(A, float)
    k = int(np.argmin(A)); best = None
    for flip in (1.0, -1.0):
        sgn = np.where(s < s[k] + 1e-12, -1.0, 1.0)
        sgn[k] = flip
        p = np.polyfit(s, sgn * A, 1)
        residual = float(np.max(np.abs(np.polyval(p, s) - sgn * A)))
        cand = (-p[1] / p[0], residual)
        if best is None or cand[1] < best[1]:
            best = cand
    return float(best[0]), float(best[1])


def locate(b, Omega, half_width, n=5, verbose=True):
    """a_c from a fresh scan of |A_3| centred on the asymptotic estimate."""
    centre = A0 + alpha2_model(Omega) * b ** 2
    scan = np.linspace(centre - half_width, centre + half_width, n)
    A, cores = [], []
    for a in scan:
        t0 = time.perf_counter()
        val, conv, core = third_harmonic_amplitude(a, b, 0.0, Omega=Omega)
        if not conv:
            raise RuntimeError(f"harmonic balance failed at b={b}, Omega={Omega}, a={a}")
        A.append(val); cores.append(core)
        if verbose:
            print(f"      a = {a:.5f}   |A_3| = {val:.5e}   core = {core/np.pi:.3f} pi   "
                  f"({time.perf_counter()-t0:.0f} s)", flush=True)
    ac, residual = signed_zero(scan, A)
    return ac, float(np.max(cores)), centre, residual


def continuation_point(a, Omega, previous_U=None, L=200.0, N=1024, K=10):
    """One b=1 continuation point; returns U and the quantities archived in the CSV."""
    b = 1.0
    d = gf.DressedGSL(a, b)
    k1, k3, _ = small_amplitude_coefficients(d)
    omega = Omega * np.sqrt(k1)
    hb = HarmonicBalance(L, N, d.F, d.dF, K=K, kappa=k1)
    if previous_U is None:
        seed = np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, Omega, K)
    else:
        seed = previous_U
    U, info = hb.solve(seed, omega, tol=1e-11, maxit=35)
    if not info["converged"]:
        raise RuntimeError(f"b=1 continuation failed at a={a}: residual={info['residual']:.3e}")
    sponge = dict(x_s=65.0, width=30.0, sigma0=0.5)
    kg = SpectralKG(L, N, d.F, d.G, kappa=k1, sponge=sponge, G_ref=0.0)
    u0, v0 = hb.initial_data(U, omega)
    kg.set_initial(u0, v0)
    A = emitted_harmonics(kg, omega, 45.0, n_periods=120,
                          samples_per_period=128, n_fit=60)
    return U, (a, k1, hb.core_amplitude(U) / np.pi, A[3], A[5])


def b1_continuation(Omega=0.92):
    """Published coarse continuation and an independent fine zero location."""
    coarse_a = np.round(np.arange(1.30, 1.6001, 0.03), 8)
    coarse_rows = []
    U = None
    for a in coarse_a:
        t0 = time.perf_counter()
        U, row = continuation_point(float(a), Omega, U)
        coarse_rows.append(row)
        print(f"   b=1 continuation a={a:.3f}: core={row[2]:.3f} pi, "
              f"A3={row[3]:.5e}, A5={row[4]:.5e} ({time.perf_counter()-t0:.0f} s)", flush=True)

    # Re-enter the branch at a=1.36 and continue in smaller steps through the minimum.
    U = None
    for a in [1.30, 1.33, 1.36]:
        U, _ = continuation_point(a, Omega, U)
    fine_a = np.arange(1.365, 1.4351, 0.005)
    fine_A3 = []
    for a in fine_a:
        U, row = continuation_point(float(a), Omega, U)
        fine_A3.append(row[3])
    ac, residual = signed_zero(fine_a, fine_A3)
    return coarse_rows, ac, residual


if __name__ == "__main__":
    print(f"j01/2 = {A0:.7f}")
    print("(1) a_c(Omega) at fixed b")
    map_rows = []
    out = {}
    for b, hw in [(0.25, 0.010), (0.5, 0.030)]:
        out[b] = []
        for Omega in [0.70, 0.75, 0.80, 0.85, 0.90, 0.94]:
            print(f"   b = {b}, Omega = {Omega:.2f}", flush=True)
            ac, core, centre, residual = locate(b, Omega, hw)
            out[b].append((Omega, ac)); map_rows.append((b, Omega, ac))
            print(f"   -> a_c = {ac:.6f}; signed-fit residual={residual:.2e}; "
                  f"core={core/np.pi:.3f} pi", flush=True)

    print("\n(2) b = 1 at Omega = 0.92 by harmonic-balance continuation")
    b1_rows, ac1, res1 = b1_continuation(0.92)
    save_csv("b1_continuation_Om0.92.csv", ["a", "kappa", "core_over_pi", "A3", "A5"], b1_rows)
    map_rows.append((1.0, 0.92, ac1))
    save_csv("ac_map_b_Omega.csv", ["b", "Omega", "a_c"], map_rows)
    print(f"   -> a_c(b=1, Omega=0.92) = {ac1:.6f}; fine-fit residual={res1:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for b in out:
        O = np.array([o for o, _ in out[b]]); A = np.array([a for _, a in out[b]])
        ax[0].plot(O, A, "o-", label=f"b = {b}")
        ax[1].plot(O, (A - A0) / b ** 2, "o-", label=f"b = {b}")
    ax[1].plot([0.92], [(ac1 - A0)], "s", label="b = 1")
    OO = np.linspace(0.68, 0.99, 100)
    ax[1].plot(OO, [alpha2_model(o) for o in OO], "k--", lw=0.8, label="scan-centering model")
    ax[0].axhline(A0, color="gray", ls="--", lw=0.8)
    ax[1].axhline(0.0, color="gray", ls="--", lw=0.8)
    ax[0].set_xlabel(r"$\Omega$"); ax[0].set_ylabel("$a_c$"); ax[0].legend(fontsize=8)
    ax[0].set_title("cancellation point vs reduced frequency")
    ax[1].set_xlabel(r"$\Omega$"); ax[1].set_ylabel(r"$(a_c-j_{01}/2)/b^2$"); ax[1].legend(fontsize=8)
    ax[1].set_title(r"$O(b^2)$ displacement coefficient")
    fig.tight_layout(); fig.savefig("experiment_ac_map.png", dpi=150)
    print("\nsaved experiment_ac_map.png")
