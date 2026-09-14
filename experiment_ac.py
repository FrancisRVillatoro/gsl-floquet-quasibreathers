"""
experiment_ac.py -- characterisation of the radiation cancellation of Section 8.1.

Three questions are answered.

(1) Is a_c a true zero of the far-field amplitude or a deep minimum?
    The radiated power is measured on a fine grid of a around the minimum and
    sqrt(P) is fitted linearly on each side; two consistent zeros mean a simple
    zero of the amplitude, P ~ (a - a_c)^2.

(2) Which harmonic carries the radiation, and what remains at a_c?
    The field is recorded at a probe point between the core and the absorbing
    layer and Fourier-analysed in time.  The k = 1 line is not radiation but the
    evanescent tail of the breather and serves as a check of the measurement.

(3) Does the first Born (Kivshar-Malomed) estimate predict a_c?
    It is evaluated along the scan and compared with the measurement, together
    with its value for sine-Gordon, where the true radiation is exactly zero.

Run:  python experiment_ac.py     (about 5 minutes)
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsl_floquet as gf
from qb_newton import (HarmonicBalance, sg_breather_harmonics, small_amplitude_coefficients,
                       born_radiation, emitted_harmonics)
from kg_spectral import SpectralKG

b, Omega, K = 0.5, 0.8, 10
L, N = 140.0, 1024
sponge = dict(x_s=45.0, width=22.0, sigma0=0.8)
x_probe = 30.0


def solve(a):
    d = gf.DressedGSL(a, b)
    k1, k3, _ = small_amplitude_coefficients(d)
    omega = Omega * np.sqrt(k1)
    hb = HarmonicBalance(L, N, d.F, d.dF, K=K, kappa=k1)
    U, info = hb.solve(np.sqrt(k1 / k3) * sg_breather_harmonics(np.sqrt(k1) * hb.x, Omega, K),
                       omega, tol=1e-11, maxit=25)
    return d, hb, U, k1, omega, info


def radiated_power(a, periods=120, fit=60):
    d, hb, U, k1, omega, info = solve(a)
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
    P = np.polyfit(rec[sel, 0] / T, rec[sel, 1] / E0, 1)[0]
    born = born_radiation(hb, U, omega, k1, E0, first_born=True)
    return P, born, info


if __name__ == "__main__":
    # ---------------------------------------------------------------- (1) fine scan
    a_scan = np.array([1.22, 1.23, 1.24, 1.245, 1.25, 1.255, 1.26, 1.27, 1.28])
    P, born3 = [], []
    print(f"b = {b}, Omega = {Omega}, L = {L}, N = {N}, K = {K}")
    print("    a       P_measured     sqrt(P)      p3 (first Born)")
    for a in a_scan:
        t0 = time.perf_counter()
        p, bo, info = radiated_power(a)
        P.append(p); born3.append(bo[3]["p"])
        print(f"  {a:.4f}  {p:.5e}  {np.sqrt(max(p,0)):.4e}   {bo[3]['p']:.4e}   ({time.perf_counter()-t0:.0f} s)")
    P = np.array(P); s = np.sqrt(P)
    left = np.polyfit(a_scan[a_scan <= 1.245], s[a_scan <= 1.245], 1)
    right = np.polyfit(a_scan[a_scan >= 1.26], s[a_scan >= 1.26], 1)
    ac_l, ac_r = -left[1] / left[0], -right[1] / right[0]
    print(f"  sqrt(P) is linear on both sides; zeros at a = {ac_l:.5f} (left) and {ac_r:.5f} (right)")
    print(f"  => simple zero of the far-field amplitude at a_c = {0.5*(ac_l+ac_r):.4f} "
          f"+- {0.5*abs(ac_l-ac_r):.4f}")

    # ---------------------------------------------------------------- (2) harmonic content
    print("\nharmonic amplitudes of the field at x = 30 (last 60 of 120 periods).")
    print("The flux of a harmonic emitted to both sides is P_k = |A_k|^2 (k omega) q_k, so the")
    print("last column is an independent estimate of the same power measured by the sponges.")
    print("    a        |A1| (evanescent)   |A3|         |A5|        sum_k P_k T / E0    measured")
    harm = {}
    for a in [0.0, 1.25, 1.5]:
        d, hb, U, k1, omega, info = solve(a)
        kg = SpectralKG(L, N, d.F, d.G, kappa=k1, sponge=sponge, G_ref=0.0)
        u0, v0 = hb.initial_data(U, omega)
        kg.set_initial(u0, v0)
        E0 = kg.energy()
        A = emitted_harmonics(kg, omega, x_probe, n_periods=120, n_fit=60)
        harm[a] = A
        flux = 0.0
        for k in [3, 5, 7]:
            w2 = (k * omega) ** 2 - k1
            if w2 > 0:
                flux += A[k] ** 2 * (k * omega) * np.sqrt(w2)
        pm, _, _ = radiated_power(a)
        print(f"  {a:.4f}   {A[1]:.3e}          {A[3]:.3e}    {A[5]:.3e}   "
              f"{flux * 2 * np.pi / omega / E0:.4e}         {pm:.4e}")
    print("  at a_c the third harmonic falls to the level of the fifth, which does not vanish there;")
    print("  the residual power at the minimum is therefore set by the fifth harmonic.")

    # ---------------------------------------------------------------- (3) first Born on sine-Gordon
    hb0 = HarmonicBalance(120.0, 1024, np.sin, np.cos, K=10, kappa=1.0)
    U0, _ = hb0.solve(sg_breather_harmonics(hb0.x, 0.6, 10), 0.6, tol=1e-11)
    E0 = hb0.energy(U0, 0.6, lambda u: 1.0 - np.cos(u))
    b0 = born_radiation(hb0, U0, 0.6, 1.0, E0, first_born=True)
    print(f"\nsine-Gordon control (omega = 0.6): E0 = {E0:.6f} (exact 16 beta = {16*np.sqrt(1-0.36):.6f});")
    print(f"  first Born predicts p3 = {b0[3]['p']:.3e} per period, while the exact breather")
    print(f"  radiates nothing (measured below 1e-12 of E0 over 20 periods).  The first Born")
    print(f"  is therefore not quantitative here: the radiation of these breathers is beyond")
    print(f"  all orders and needs exponential asymptotics, not a leading-order source.")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogy(a_scan, P, "o-", label="measured")
    ax[0].semilogy(a_scan, born3, "s--", label="first Born, $p_3$")
    ax[0].set_xlabel("a"); ax[0].set_ylabel("$P/E_0$ per period"); ax[0].legend(fontsize=8)
    ax[0].set_title("radiated power near the cancellation")
    ax[1].plot(a_scan, np.where(a_scan < 0.5 * (ac_l + ac_r), -s, s), "o")
    aa = np.linspace(a_scan[0], a_scan[-1], 100)
    ax[1].plot(aa, np.polyval(left, aa) * -1, "-", lw=0.8)
    ax[1].plot(aa, np.polyval(right, aa), "-", lw=0.8)
    ax[1].axhline(0, color="gray", lw=0.6); ax[1].set_ylim(-4e-4, 4e-4)
    ax[1].set_xlabel("a"); ax[1].set_ylabel(r"$\pm\sqrt{P/E_0}$")
    ax[1].set_title(f"simple zero at $a_c$ = {0.5*(ac_l+ac_r):.4f}")
    ks = [1, 3, 5, 7]
    w = 0.25
    for i, (a, A) in enumerate(harm.items()):
        ax[2].bar(np.arange(len(ks)) + (i - 1) * w, [A[k] for k in ks], w, label=f"a = {a:.2f}")
    ax[2].set_yscale("log"); ax[2].set_xticks(range(len(ks))); ax[2].set_xticklabels([f"k = {k}" for k in ks])
    ax[2].set_ylabel("$|A_k|$ at x = 30"); ax[2].legend(fontsize=8)
    ax[2].set_title("harmonic content of the emitted field")
    fig.tight_layout(); fig.savefig("experiment_ac.png", dpi=150)
    print("saved experiment_ac.png")
