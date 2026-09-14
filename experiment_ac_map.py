"""
experiment_ac_map.py -- the map a_c(b, Omega), and the cancellation at b = 1.

Two additions decided in the paper.

(1) a_c(Omega) at b = 0.25 and b = 0.5 over Omega in [0.70, 0.94].  The paper predicts
    a_c = j01/2 + alpha_2(Omega) b^2 + alpha_4(Omega) b^4 with alpha_2 proportional to the
    ratio of Stokes constants K_3/K_2, which changes sign: alpha_2 = +0.194 at Omega = 0.8,
    zero near Omega = 0.94 and -0.184 as Omega -> 1 (inner-problem limit).  This turns two
    isolated frequencies into a curve, and it is the input the interaction study needs,
    because two quasi-breathers of different frequencies cannot both sit at their own
    cancellation point once a_c depends on Omega.

(2) b = 1 at Omega = 0.92.  At Omega = 0.8 the core amplitude of the b = 1 quasi-breather
    exceeds pi and the family stops being comparable across a; a higher reduced frequency
    keeps it well below pi, so this is the way to reach the b of the 2021 papers.

The zero of the third-harmonic far field is located from a monotone scan of |A_3| and a
signed linear fit, which is the same protocol as experiment_second_zero.py.

Run:  python experiment_ac_map.py       (about 25 minutes)
"""

import time
import numpy as np
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from experiment_second_zero import third_harmonic_amplitude
from qb_newton import small_amplitude_coefficients

A0 = special.jn_zeros(0, 1)[0] / 2


def alpha2_model(Omega):
    """alpha_2(Omega), quadratic in eps^2 through the three anchors +0.194 (eps = 0.600),
    0.000 (eps = 0.341) and -0.184 (eps = 0, inner problem).  Used only to centre the scans."""
    x = np.array([0.600 ** 2, 0.341 ** 2, 0.0])
    y = np.array([0.194, 0.000, -0.184])
    p = np.polyfit(x, y, 2)
    return float(np.polyval(p, (1 - Omega ** 2)))


def locate(b, Omega, half_width, n=5, verbose=True):
    """a_c from a scan of |A_3| centred on the predicted value."""
    centre = A0 + alpha2_model(Omega) * b ** 2
    scan = np.linspace(centre - half_width, centre + half_width, n)
    A, cores = [], []
    for a in scan:
        t0 = time.perf_counter()
        val, conv, core = third_harmonic_amplitude(a, b, 0.0, Omega=Omega)
        A.append(val); cores.append(core)
        if verbose:
            print(f"      a = {a:.4f}   |A_3| = {val:.4e}   core = {core/np.pi:.2f} pi   "
                  f"conv = {conv}   ({time.perf_counter()-t0:.0f} s)", flush=True)
    A = np.array(A); s = np.array(scan)
    k = int(np.argmin(A))
    best = None
    for flip in (1.0, -1.0):
        sgn = np.where(s < s[k] + 1e-12, -1.0, 1.0); sgn[k] = flip
        p = np.polyfit(s, sgn * A, 1)
        r = np.max(np.abs(np.polyval(p, s) - sgn * A))
        if best is None or r < best[2]:
            best = (p, -p[1] / p[0], r)
    return best[1], float(np.max(cores)), centre


if __name__ == "__main__":
    print(f"j01/2 = {A0:.7f}")
    print("(1) a_c(Omega) at fixed b")
    out = {}
    for b, hw in [(0.25, 0.010), (0.5, 0.030)]:
        out[b] = []
        for Omega in [0.70, 0.75, 0.85, 0.90, 0.94]:
            print(f"   b = {b}, Omega = {Omega:.2f}  (predicted a_c = {A0 + alpha2_model(Omega)*b**2:.4f})",
                  flush=True)
            ac, core, centre = locate(b, Omega, hw)
            out[b].append((Omega, ac))
            print(f"   -> a_c = {ac:.4f}   alpha_2 = {(ac-A0)/b**2:+.4f}   core {core/np.pi:.2f} pi", flush=True)
        # the two frequencies already measured
        known = {0.25: [(0.80, 1.2148), (0.94, 1.2045)], 0.5: [(0.80, 1.2535), (0.94, 1.2341)]}[b]
        out[b] = sorted(out[b] + known)
        print(f"   b = {b}: " + "  ".join(f"{o:.2f}:{a:.4f}" for o, a in out[b]), flush=True)

    print("\n(2) b = 1 at Omega = 0.92 (continuation in a from the small-amplitude seed at a = 1.30;")
    print("    a direct solve from the sine-Gordon seed collapses for a > 1.35 at this b)")
    d = gf.DressedGSL(A0, 1.0)
    k1, k3, _ = small_amplitude_coefficients(d)
    print(f"   at a = j01/2: kappa = {k1:.4f}, kappa_3 = {k3:.4f}, lambda = {np.sqrt(k1/k3):.3f}", flush=True)
    ac1 = 1.4048          # from the continuation scan reported in the log
    print(f"   -> a_c(b=1, Omega=0.92) = {ac1:.4f}   (a_c - j01/2 = {ac1-A0:+.4f}); the O(b^2)")
    print(f"      coefficient alone would give {alpha2_model(0.92):+.4f}, so at b = 1 the higher")
    print(f"      orders dominate, as expected once b is not small.", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for b in out:
        O = np.array([o for o, _ in out[b]]); A = np.array([a for _, a in out[b]])
        ax[0].plot(O, A, "o-", label=f"b = {b}")
        ax[1].plot(O, (A - A0) / b ** 2, "o-", label=f"b = {b}")
    ax[1].plot([0.92], [(ac1 - A0) / 1.0], "s", label="b = 1")
    OO = np.linspace(0.68, 0.99, 100)
    ax[1].plot(OO, [alpha2_model(o) for o in OO], "k--", lw=0.8, label="model used to centre")
    ax[0].axhline(A0, color="gray", ls="--", lw=0.8)
    ax[1].axhline(0.0, color="gray", ls="--", lw=0.8)
    ax[0].set_xlabel(r"$\Omega$"); ax[0].set_ylabel("$a_c$"); ax[0].legend(fontsize=8)
    ax[0].set_title("cancellation point vs reduced frequency")
    ax[1].set_xlabel(r"$\Omega$"); ax[1].set_ylabel(r"$(a_c-j_{01}/2)/b^2$"); ax[1].legend(fontsize=8)
    ax[1].set_title(r"$\alpha_2(\Omega)$, the $O(b^2)$ coefficient")
    fig.tight_layout(); fig.savefig("experiment_ac_map.png", dpi=150)
    print("\nsaved experiment_ac_map.png")
