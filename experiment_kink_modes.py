"""
experiment_kink_modes.py -- H4, part 1: the internal mode of the kink versus the HF amplitude.

If the cancellation of the breather radiation at J_0(2a) = 0 is the switching-off of the
second miniband harmonic, the same point must show in the kink.  At O(b^2) the equation is
double sine-Gordon with rho = c_2 J_0(2a)/(c_1 J_0(a)); at rho = 0 it is sine-Gordon, whose
kink has no internal mode (the linearised operator has only the translation zero mode and a
threshold state tanh x at the continuum edge).  A perturbation rho W(x) turns the threshold
state into a bound internal mode on one side only, with binding

    kappa_b = sqrt(kappa - omega_i^2) = -(rho/2) int W tanh^2 dx + O(rho^2),

so the prediction is: the internal mode of the GSL kink delocalises linearly in (a - a_k)
and disappears at a_k(b) -> j_01/2 as b -> 0, with no internal mode on the other side.

This script diagonalises L = -d^2/dx^2 + F'_{b,a}(u_k(x)) on the static kink (Fourier
collocation on a periodic box; the potential tends to kappa at both ends) and reports the
bound states below the continuum edge kappa.

Run:  python experiment_kink_modes.py    (about 4 minutes)
"""

import time
import numpy as np
from scipy import linalg, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf

L, N = 300.0, 1536


def kink_spectrum(a, b, n_states=4):
    """Lowest eigenvalues of -d^2/dx^2 + F'(u_k) on the periodic box, and kappa."""
    d = gf.DressedGSL(a, b)
    dx = L / N
    x = -L / 2 + dx * np.arange(N)
    uk = d.kink_on_grid(0.0, 2 * np.pi, x)
    V = d.dF(uk)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    col = np.real(np.fft.ifft(k * k))                 # first column of the circulant -d^2/dx^2
    C = linalg.circulant(col)
    H = C + np.diag(V)
    w = linalg.eigh(H, eigvals_only=True, subset_by_index=[0, n_states - 1])
    return w, float(d.mass2)


if __name__ == "__main__":
    j01_2 = special.jn_zeros(0, 1)[0] / 2
    print(f"box L = {L}, N = {N}; continuum edge = kappa; states are bound if below kappa - 2 (2 pi/L)^2")
    thr = 2 * (2 * np.pi / L) ** 2
    results = {}
    for b in [0.25, 0.5, 1.0]:
        a_star = gf.first_zero_mass2(b)
        a_vals = np.concatenate([np.linspace(0.0, 1.0, 11), np.linspace(1.05, 1.35, 13),
                                 np.linspace(1.4, a_star - 0.1, 8)])
        rows = []
        t0 = time.perf_counter()
        for a in a_vals:
            w, kappa = kink_spectrum(a, b)
            bound = [x for x in w if x < kappa - thr]
            zero = bound[0] if bound else np.nan
            internal = bound[1] if len(bound) > 1 else np.nan
            rows.append((a, kappa, zero, internal))
        rows = np.array(rows)
        results[b] = rows
        np.savetxt(f"results/kink_modes_b{b}.csv", rows, delimiter=",",
                   header="a,kappa,zero_mode,internal_omega2", comments="")
        print(f"\nb = {b}   (regime I ends at a_* = {a_star:.4f};  {time.perf_counter()-t0:.0f} s)")
        print("     a      kappa     zero mode   internal omega^2/kappa   kappa_b = sqrt(kappa - omega_i^2)")
        for a, kappa, z, om2 in rows:
            if np.isnan(om2):
                print(f"   {a:.3f}  {kappa:.5f}   {z:+.2e}     none")
            else:
                print(f"   {a:.3f}  {kappa:.5f}   {z:+.2e}     {om2/kappa:.6f}               {np.sqrt(max(kappa-om2,0)):.5f}")
        # where does the internal mode disappear?  linear extrapolation of kappa_b to zero
        sel = ~np.isnan(rows[:, 3]) & (rows[:, 0] > 0.6)
        if sel.sum() >= 3:
            aa = rows[sel, 0]; kb = np.sqrt(np.maximum(rows[sel, 1] - rows[sel, 3], 0))
            last = aa > aa[-1] - 0.25
            p = np.polyfit(aa[last], kb[last], 1)
            print(f"   internal mode disappears at a_k = {-p[1]/p[0]:.4f} (linear extrapolation of kappa_b; "
                  f"j_01/2 = {j01_2:.4f}; rho = c_2 J_0(2a)/(c_1 J_0(a)) vanishes at {j01_2:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for b, rows in results.items():
        ok = ~np.isnan(rows[:, 3])
        ax[0].plot(rows[ok, 0], rows[ok, 3] / rows[ok, 1], "o-", ms=3, label=f"b = {b}")
        ax[1].plot(rows[ok, 0], np.sqrt(np.maximum(rows[ok, 1] - rows[ok, 3], 0)), "o-", ms=3, label=f"b = {b}")
    for axx in ax:
        axx.axvline(j01_2, color="gray", ls="--", lw=0.8)
        axx.set_xlabel("a"); axx.legend(fontsize=8)
    ax[0].set_ylabel(r"$\omega_i^2/\kappa$"); ax[0].set_title("internal mode of the kink (1 = continuum edge)")
    ax[1].set_ylabel(r"$\kappa_b=\sqrt{\kappa-\omega_i^2}$"); ax[1].set_title("binding of the internal mode")
    fig.tight_layout(); fig.savefig("experiment_kink_modes.png", dpi=150)
    print("\nsaved experiment_kink_modes.png")
