"""
fig_gsl_floquet.py -- overview figure for the Floquet-dressed GSL nonlinearity.

Panels
  (a) m_eff^2(a, b) = F_{b,a}'(0) for several b; zeros mark the turn-over a_*(b)
      (compare Figs. 4 and 5 of Kryuchkov, Kukhar', Zav'yalov, Laser Phys. 2013,
      where delta = (b^2/2) m_eff^2).
  (b) regime map in the (a, b) plane: I (vacuum 0), II (vacua +-u*), III (vacuum pi).
  (c) dressed potential G_{b,a}(u) for b = 1 in the three regimes.
  (d) static kinks for b = 1: the 2pi-kink (regime I), the two kinks of regime II
      (through 0 and through pi) and the inverted 2pi-kink of regime III.
  (e) Fourier coefficients c_m(b) of F_b and their dressed values c_m J_0(m a).

Run:  python fig_gsl_floquet.py   -> gsl_floquet_overview.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import special

import gsl_floquet as gf

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    (axa, axb, axc), (axd, axe, axf) = axes

    # (a) effective mass
    a_grid = np.linspace(0.0, 10.0, 801)
    for b in [0.45, 1.0, 2.5, 5.0]:
        c = gf.fourier_coefficients(b)
        m = np.arange(1, c.size + 1)
        m2 = np.array([np.sum(m * c * special.j0(m * a)) for a in a_grid])
        axa.plot(a_grid, m2, label=f"b = {b}")
    axa.plot(a_grid, special.j0(a_grid), "k--", lw=1, label="b = 0: $J_0(a)$")
    axa.axhline(0, color="gray", lw=0.6)
    axa.set_xlabel("a"); axa.set_ylabel(r"$m_{\rm eff}^2(a,b)=\mathcal{F}_{b,a}'(0)$")
    axa.set_title("(a) linear mass about u = 0"); axa.legend(fontsize=8)

    # (b) regime map
    b_vals = np.linspace(0.05, 5.0, 60)
    a_vals = np.linspace(0.0, 9.0, 361)
    reg = np.zeros((b_vals.size, a_vals.size))
    code = {"I": 0, "II": 1, "III": 2, "IV": 3}
    for i, b in enumerate(b_vals):
        c = gf.fourier_coefficients(b)
        m = np.arange(1, c.size + 1)
        for j, a in enumerate(a_vals):
            dm = c * special.j0(m * a)
            k0 = np.sum(m * dm); kp = np.sum(m * dm * np.cos(m * np.pi))
            reg[i, j] = 0 if (k0 > 0 and kp < 0) else 1 if (k0 < 0 and kp < 0) else 2 if (k0 < 0 and kp > 0) else 3
    cmap = matplotlib.colors.ListedColormap(["#c7d9f2", "#f2c7c7", "#d9f2c7", "#f2e7c7"])
    axb.pcolormesh(a_vals, b_vals, reg, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest")
    for z in special.jn_zeros(0, 3):
        axb.axvline(z, color="k", ls=":", lw=0.8)
    axb.set_xlabel("a"); axb.set_ylabel("b")
    axb.set_title("(b) regimes: I blue (vac. 0), II red ($\\pm u^*$), III green (vac. $\\pi$)", fontsize=10)

    # (c) potentials for b = 1
    b = 1.0
    a1 = gf.first_zero_mass2(b, at=0.0); a2 = gf.first_zero_mass2(b, at=np.pi)
    u = np.linspace(-2 * np.pi, 2 * np.pi, 801)
    for a, lab in [(0.0, "a = 0 (GSL)"), (1.5, "a = 1.5 (I)"), (0.5 * (a1 + a2), f"a = {0.5*(a1+a2):.3f} (II)"),
                   (3.2, "a = 3.2 (III)")]:
        d = gf.DressedGSL(a, b)
        axc.plot(u, d.G(u), label=lab)
    axc.set_xlabel("u"); axc.set_ylabel(r"$G_{b,a}(u)-G_{b,a}(0)$")
    axc.set_title(f"(c) dressed potential, b = 1 ($a_*$={a1:.3f}, $a_{{**}}$={a2:.3f})", fontsize=10)
    axc.legend(fontsize=8)

    # (d) kinks for b = 1
    dI = gf.DressedGSL(1.5, b)
    x, uk = dI.static_kink(0.0, 2 * np.pi); axd.plot(x, uk, label="I: $0\\to2\\pi$, a = 1.5")
    dII = gf.DressedGSL(0.5 * (a1 + a2), b); us = dII.vacua()[0]
    x, uk = dII.static_kink(-us, us); axd.plot(x, uk, label=f"II: $-u^*\\to u^*$ (through 0), $u^*$={us:.3f}")
    x, uk = dII.static_kink(us, 2 * np.pi - us); axd.plot(x, uk, label="II: $u^*\\to2\\pi-u^*$ (through $\\pi$)")
    dIII = gf.DressedGSL(3.2, b)
    x, uk = dIII.static_kink(np.pi, 3 * np.pi); axd.plot(x, uk, label="III: $\\pi\\to3\\pi$, a = 3.2")
    axd.set_xlim(-25, 25); axd.set_xlabel("x"); axd.set_ylabel("u(x)")
    axd.set_title("(d) static kinks, b = 1 (energies: "
                  f"{dI.kink_energy(0, 2*np.pi):.3f}, {dII.kink_energy(-us, us):.3f}, "
                  f"{dII.kink_energy(us, 2*np.pi-us):.3f}, {dIII.kink_energy(np.pi, 3*np.pi):.3f})", fontsize=9)
    axd.legend(fontsize=7)

    # (e) Fourier coefficients
    for b in [0.45, 1.0, 2.5, 5.0, 20.0]:
        c = gf.fourier_coefficients(b)
        m = np.arange(1, min(c.size, 40) + 1)
        axe.semilogy(m, np.abs(c[:m.size]), "o-", ms=3, label=f"b = {b}")
    axe.set_xlabel("m"); axe.set_ylabel("$|c_m(b)|$"); axe.set_title("(e) sine coefficients of $F_b$")
    axe.legend(fontsize=8)

    # (f) dressed coefficients d_m = c_m J_0(m a) for b = 1 at several a
    c = gf.fourier_coefficients(1.0); m = np.arange(1, 13)
    for a in [0.0, 1.5, a1, 0.5 * (a1 + a2), 3.2]:
        axf.plot(m, c[:12] * special.j0(m * a), "o-", ms=3, label=f"a = {a:.3f}")
    axf.axhline(0, color="gray", lw=0.6)
    axf.set_xlabel("m"); axf.set_ylabel("$c_m(1)\\,J_0(ma)$"); axf.set_title("(f) dressed harmonic content, b = 1")
    axf.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("gsl_floquet_overview.png", dpi=150)
    print("saved gsl_floquet_overview.png")
