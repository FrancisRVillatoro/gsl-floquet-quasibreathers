"""
experiment_H1.py -- test of hypothesis H1 of the model specification.

For b fixed and the reduced frequency Omega = omega / sqrt(kappa) held fixed, the
stationary quasi-breather of the Floquet-dressed GSL equation is computed with the
Newton-Fourier method (qb_newton.HarmonicBalance) for a set of values of the HF
amplitude a in regime I.  Each converged solution is then used as initial data for a
time-domain run with absorbing layers (kg_spectral.SpectralKG), and the radiated power
is measured as the energy deposited in the layers per period after the transient.

Holding Omega fixed (rather than omega, or the amplitude) is what makes the comparison
meaningful: the small-amplitude reduction of every case is the same NLS soliton, so the
differences in radiated power are differences between the nonlinearities, not between
breather families.

Two predictors are recorded alongside the measured power:
    |d_3| = |c_3(b) J_0(3a)|      the third-harmonic coupling of the dressed nonlinearity,
    eta_5 = kappa kappa_5 / kappa_3^2 - 1,  the quintic distance to sine-Gordon.
Neither is a theory of the radiation, which is beyond all orders in the amplitude; they
are candidate predictors to be confronted with the measurement.

Run:  python experiment_H1.py     (about 6 minutes)
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from qb_newton import HarmonicBalance, sg_breather_harmonics, small_amplitude_coefficients
from kg_spectral import SpectralKG, zero_crossing_frequency

# ---------------------------------------------------------------- parameters
b = 0.5
Omega = 0.8                    # reduced breather frequency, omega / sqrt(kappa)
K = 10                         # harmonics kept in the harmonic balance
L, N = 140.0, 1024             # box for both the Newton solve and the time-domain run
sponge = dict(x_s=45.0, width=22.0, sigma0=0.8)
periods_total, periods_fit = 200, 100
a_values = [0.0, 0.3, 0.6, 0.8016, 1.0, 1.2, 1.5, 1.8]

if __name__ == "__main__":
    rows = []
    print(f"b = {b}, Omega = {Omega}, K = {K}, L = {L}, N = {N}, {periods_total} periods")
    print("   a     kappa    |d3|      eta_5     core     tail(NF)   P_rad/E0 per period   omega_B drift")
    for a in a_values:
        d = gf.DressedGSL(a, b)
        k1, k3, k5 = small_amplitude_coefficients(d)
        if k3 <= 0:
            print(f"{a:6.4f}  kappa_3 <= 0, no small-amplitude breather below the gap; skipped")
            continue
        eta5 = k1 * k5 / k3 ** 2 - 1.0
        lam = np.sqrt(k1 / k3)
        omega = Omega * np.sqrt(k1)

        # --- stationary quasi-breather -------------------------------------------------
        hb = HarmonicBalance(L, N, d.F, d.dF, K=K, kappa=k1)
        U0 = lam * sg_breather_harmonics(np.sqrt(k1) * hb.x, Omega, K)
        t0 = time.perf_counter()
        U, info = hb.solve(U0, omega, tol=1e-11, maxit=25)
        t_nf = time.perf_counter() - t0
        if not info["converged"]:
            print(f"{a:6.4f}  Newton did not converge (residual {info['residual']:.2e}); skipped")
            continue

        # --- time-domain radiation ------------------------------------------------------
        kg = SpectralKG(L, N, d.F, d.G, kappa=k1, sponge=sponge, G_ref=0.0)
        u0, v0 = hb.initial_data(U, omega, t=0.0)
        kg.set_initial(u0, v0)
        E0 = kg.energy()
        T = 2.0 * np.pi / omega
        h = min(0.1, 0.9 * kg.h_max)
        ts, uc, rec = [], [], []
        i0 = int(np.argmin(np.abs(kg.x)))

        def obs(kg, u):
            ts.append(kg.t); uc.append(u[i0])
            if len(ts) % 20 == 1:
                rec.append((kg.t, kg.energy(interior=(-sponge["x_s"], sponge["x_s"])), kg.absorbed))

        t0 = time.perf_counter()
        kg.run_until(h, periods_total * T, observer=obs, every=5)
        t_td = time.perf_counter() - t0
        rec = np.array(rec); ts = np.array(ts); uc = np.array(uc)
        sel = rec[:, 0] > (periods_total - periods_fit) * T
        p = np.polyfit(rec[sel, 0] / T, rec[sel, 2] / E0, 1)
        P = p[0]                                     # absorbed energy per period, relative to E0
        w0 = zero_crossing_frequency(ts[ts < 20 * T], uc[ts < 20 * T]) / np.sqrt(k1)
        w1 = zero_crossing_frequency(ts[ts > (periods_total - 20) * T], uc[ts > (periods_total - 20) * T]) / np.sqrt(k1)
        print(f"{a:6.4f}  {k1:7.4f}  {abs(d.d[2]):.2e}  {eta5:8.4f}  {hb.core_amplitude(U):.4f}  "
              f"{hb.tail_amplitude(U):.2e}   {P:.4e}          {w0:.4f} -> {w1:.4f}   "
              f"({t_nf:.1f}+{t_td:.0f} s)")
        rows.append(dict(a=a, kappa=k1, d3=abs(d.d[2]), eta5=eta5, core=hb.core_amplitude(U),
                         tail=hb.tail_amplitude(U), P=P, w0=w0, w1=w1, rec=rec, E0=E0, T=T))

    # ---------------------------------------------------------------- figure
    if rows:
        A = np.array([r["a"] for r in rows]); P = np.array([r["P"] for r in rows])
        D3 = np.array([r["d3"] for r in rows]); E5 = np.array([abs(r["eta5"]) for r in rows])
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for r in rows:
            ax[0].semilogy(r["rec"][1:, 0] / r["T"], np.maximum(r["rec"][1:, 2] / r["E0"], 1e-17),
                           label=f"a = {r['a']:.3f}")
        ax[0].set_xlabel("t / T"); ax[0].set_ylabel("absorbed energy / $E_0$")
        ax[0].set_title(f"radiated energy, b = {b}, $\\Omega$ = {Omega}"); ax[0].legend(fontsize=7)
        ax[1].semilogy(A, np.maximum(P, 1e-17), "o-")
        ax[1].set_xlabel("a"); ax[1].set_ylabel("$P_{rad}/E_0$ per period")
        ax[1].set_title("radiated power vs HF amplitude")
        ax[2].loglog(np.maximum(D3, 1e-8), np.maximum(P, 1e-17), "o", label="$|d_3|$")
        ax[2].loglog(np.maximum(E5, 1e-8), np.maximum(P, 1e-17), "s", label="$|\\eta_5|$")
        for r in rows:
            ax[2].annotate(f"{r['a']:.2f}", (max(r["d3"], 1e-8), max(r["P"], 1e-17)), fontsize=6)
        ax[2].set_xlabel("predictor"); ax[2].set_ylabel("$P_{rad}/E_0$ per period")
        ax[2].set_title("measurement vs candidate predictors"); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig("experiment_H1.png", dpi=150)
        print("saved experiment_H1.png")
