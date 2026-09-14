"""
experiment_collisions.py -- H4, part 2: kink-antikink collisions versus the HF amplitude.

The resonant-energy-exchange mechanism behind the capture and the fractal windows of the
2021 paper needs the internal mode of the kink.  kink_threshold.py shows that this mode
merges into the continuum at a_k(b) -> j_01/2, so the critical velocity v_cr below which a
kink-antikink pair is captured (or escapes only after several bounces) should fall towards
zero as a -> a_k, and the collisions should become elastic there.  This script measures
v_cr(a) at b = 1 (where a_k = 1.036) by bisection on the outcome of a single collision.

Classification.  u(0, t) equals 2 pi before the collision; a one-bounce escape takes it to
-2 pi once and for all; a capture leaves it oscillating; an n-bounce escape returns to
+-2 pi after n crossings.  v_cr is the largest velocity below which the outcome is not a
one-bounce escape.

Run:  python experiment_collisions.py   (about 15 minutes; results are printed as they come)
"""

import time
import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from kg_spectral import SpectralKG
from kink_threshold import a_internal_mode_vanishes

L, N, X0 = 200.0, 2048, 15.0


def collide(d, v, h=0.1):
    """Return (n_crossings of u(0,t) through 0, final |u(0)| near 2pi?, final separation growth)."""
    kappa = float(d.mass2)
    kg = SpectralKG(L, N, d.F, d.G, kappa=kappa, twist=0, G_ref=0.0)
    x = kg.x
    gam = 1.0 / np.sqrt(1.0 - v * v)
    xk, uk = d.static_kink(0.0, 2 * np.pi)
    spl = interpolate.CubicSpline(xk, uk)

    def prof(y):
        z = gam * y
        out = spl(np.clip(z, xk[0], xk[-1]))
        return np.where(z < xk[0], 0.0, np.where(z > xk[-1], 2 * np.pi, out))

    def dprof(y):
        return np.sqrt(np.maximum(2.0 * d.dG_from(prof(y), 0.0), 0.0))

    u0 = prof(x + X0) + (2 * np.pi - prof(x - X0)) - 2 * np.pi
    v0 = -gam * v * (dprof(x + X0) + dprof(x - X0))
    kg.set_initial(u0, v0)
    i0 = int(np.argmin(np.abs(x)))
    ts, uc, seps = [], [], []

    def obs(kg, u):
        ts.append(kg.t); uc.append(u[i0])
        au = np.abs(u) - np.pi
        cr = np.nonzero(au[:-1] * au[1:] < 0)[0]
        seps.append(x[cr[-1]] - x[cr[0]] if cr.size >= 2 else 0.0)

    t_end = X0 / v + min(600.0, 70.0 / v)      # after the collision the kinks travel at most 70 (no wrap-around)
    kg.run_until(h, t_end, observer=obs, every=20)
    uc = np.array(uc); seps = np.array(seps)
    n_cross = int(np.sum((uc[:-1] * uc[1:]) < 0))
    final_sep_growing = seps[-1] > seps[-10] + 1.0 and seps[-1] > 25.0
    one_bounce = (n_cross == 1) and final_sep_growing and abs(abs(uc[-1]) - 2 * np.pi) < 0.6
    return one_bounce, n_cross, seps[-1]


def critical_velocity(d, v_lo=0.02, v_hi=0.5, tol=0.004, verbose=True):
    ob_hi = collide(d, v_hi)[0]
    if not ob_hi:
        return np.nan
    ob_lo = collide(d, v_lo)[0]
    if ob_lo:
        return v_lo         # escapes even at the lowest velocity tried
    while v_hi - v_lo > tol:
        vm = 0.5 * (v_lo + v_hi)
        ob, n, sep = collide(d, vm)
        if verbose:
            print(f"      v = {vm:.4f}: {'one-bounce escape' if ob else 'capture / multi-bounce'} "
                  f"(crossings {n}, final separation {sep:.1f})", flush=True)
        if ob:
            v_hi = vm
        else:
            v_lo = vm
    return 0.5 * (v_lo + v_hi)


if __name__ == "__main__":
    b = 1.0
    a_k = a_internal_mode_vanishes(b)
    print(f"b = {b}: internal mode of the kink vanishes at a_k = {a_k:.4f}")
    a_vals = [0.0, 0.4, 0.7, 0.85, 0.95, 1.0, 1.1, 1.3, 1.6]
    vcr = []
    for a in a_vals:
        d = gf.DressedGSL(a, b)
        t0 = time.perf_counter()
        print(f"a = {a:.3f}  (kappa = {d.mass2:.4f})", flush=True)
        vc = critical_velocity(d)
        vcr.append(vc)
        print(f"   -> v_cr = {vc:.4f}   ({time.perf_counter()-t0:.0f} s)", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(a_vals, vcr, "o-")
    ax.axvline(a_k, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("a"); ax.set_ylabel("$v_{cr}$")
    ax.set_title(f"$v_{{cr}}$ for capture, b = {b}; internal mode vanishes at a = {a_k:.3f}", fontsize=10)
    fig.tight_layout(); fig.savefig("experiment_collisions.png", dpi=150)
    print("saved experiment_collisions.png")
