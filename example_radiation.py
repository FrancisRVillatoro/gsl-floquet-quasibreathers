"""
example_radiation.py -- template for the radiation measurements of hypothesis H1.

A sine-Gordon breather (in the variables rescaled by sqrt(kappa)) is used as a
seed for the dressed GSL equation in regime I, for b = 0.5 and three values of
a: a = 0 (plain GSL), a = 0.8016 (J_0(3a) = 0, third-harmonic coupling d_3 = 0)
and a = 1.5.  The energy leaving the interior is measured through the absorbing
layer.  This is a demonstration of the tooling, not a result: the seed is not a
stationary quasi-breather, so the early part of each run is a transient in which
the seed sheds radiation and relaxes.  Run:  python example_radiation.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from kg_spectral import SpectralKG, zero_crossing_frequency

b = 0.5
Omega = 0.8                 # breather frequency in units of sqrt(kappa)
periods = 300
L, N = 300.0, 2048
sponge = dict(x_s=80.0, width=50.0, sigma0=1.0)
cases = [0.0, 2.404826 / 3.0, 1.5]

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    print(f"b = {b}, Omega/sqrt(kappa) = {Omega}, {periods} periods, L = {L}, N = {N}")
    print(" a        kappa    d2/d1    d3/d1    E_int(end)/E0   P_abs/E0 per period (last 100)   omega_B start -> end")
    for a in cases:
        d = gf.DressedGSL(a, b)
        kappa = float(d.dF(0.0))
        kg = SpectralKG(L, N, d.F, d.G, kappa=kappa, sponge=sponge, G_ref=0.0)
        x = kg.x
        beta = np.sqrt(1.0 - Omega * Omega)
        u0 = np.zeros_like(x)
        v0 = np.sqrt(kappa) * 4.0 * beta / np.cosh(beta * np.sqrt(kappa) * x)   # d/dt of the rescaled SG breather at t = 0
        kg.set_initial(u0, v0)
        E0 = kg.energy()
        T = 2.0 * np.pi / (Omega * np.sqrt(kappa))
        h = min(0.1, 0.9 * kg.h_max)
        ts, Eint, Eabs, uc = [], [], [], []
        i0 = int(np.argmin(np.abs(x)))

        def obs(kg, u):
            ts.append(kg.t); uc.append(u[i0])
            if len(ts) % 20 == 1:
                Eint.append((kg.t, kg.energy(interior=(-sponge["x_s"], sponge["x_s"])), kg.absorbed))

        t0 = time.perf_counter()
        kg.run_until(h, periods * T, observer=obs, every=5)
        dt = time.perf_counter() - t0
        Eint = np.array(Eint); ts = np.array(ts); uc = np.array(uc)
        sel = Eint[:, 0] > (periods - 100) * T
        P = (Eint[sel, 2][-1] - Eint[sel, 2][0]) / ((Eint[sel, 0][-1] - Eint[sel, 0][0]) / T)
        w_start = zero_crossing_frequency(ts[ts < 20 * T], uc[ts < 20 * T])
        w_end = zero_crossing_frequency(ts[ts > (periods - 20) * T], uc[ts > (periods - 20) * T])
        print(f"{a:6.4f}  {kappa:7.4f}  {d.d[1]/d.d[0]:7.4f}  {d.d[2]/d.d[0]:7.4f}   {Eint[-1,1]/E0:12.6f}   "
              f"{P/E0:10.3e}                      {w_start/np.sqrt(kappa):.4f} -> {w_end/np.sqrt(kappa):.4f}   ({dt:.0f} s)")
        axes[0].plot(Eint[:, 0] / T, Eint[:, 1] / E0, label=f"a = {a:.4f}")
        axes[1].semilogy(Eint[1:, 0] / T, np.maximum(Eint[1:, 2] / E0, 1e-16), label=f"a = {a:.4f}")

    axes[0].set_xlabel("t / T"); axes[0].set_ylabel("interior energy / E0"); axes[0].legend()
    axes[0].set_title(f"seeded quasi-breather, b = {b}, regime I")
    axes[1].set_xlabel("t / T"); axes[1].set_ylabel("absorbed energy / E0"); axes[1].legend()
    axes[1].set_title("energy radiated into the absorbing layers")
    fig.tight_layout()
    fig.savefig("example_radiation.png", dpi=150)
    print("saved example_radiation.png")
