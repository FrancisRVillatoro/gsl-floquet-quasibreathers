"""
make_figures.py -- the six publication figures, from results/*.csv and from direct computation.

Every measured dataset lives in results/ as a CSV; the figures are built only from those
files and from the analytic parts of gsl_floquet.py, so that a reader can redraw them
without re-running the simulations.  Output goes to figures/ as PDF and PNG.

Run:  python make_figures.py     (about 30 s)
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

import gsl_floquet as gf
import paper_style as ps
from results_io import result_path

ps.use()
A0 = special.jn_zeros(0, 1)[0] / 2
A0b = special.jn_zeros(0, 2)[1] / 2
PI2 = np.pi * np.sqrt(2.0)


def load(name):
    return np.genfromtxt(result_path(f"{name}.csv"), delimiter=",", names=True)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))
    a = np.linspace(0, 10, 801)
    for b in [0.5, 1.0, 2.5]:
        c = gf.fourier_coefficients(b); m = np.arange(1, c.size + 1)
        ax[0, 0].plot(a, [np.sum(m * c * special.j0(m * x)) for x in a], label=f"$b$ = {b}")
    ax[0, 0].plot(a, special.j0(a), "k--", lw=0.7, label="$b\\to0$: $J_0(a)$")
    ax[0, 0].axhline(0, color="gray", lw=0.5)
    ax[0, 0].set_xlabel("$a$"); ax[0, 0].set_ylabel("$m_{\\rm eff}^2$"); ax[0, 0].legend()
    ps.panel(ax[0, 0], "a")

    bb = np.linspace(0.05, 3.0, 45); aa = np.linspace(0, 9, 300)
    reg = np.zeros((bb.size, aa.size))
    for i, b in enumerate(bb):
        c = gf.fourier_coefficients(b); m = np.arange(1, c.size + 1)
        for j, x in enumerate(aa):
            d = c * special.j0(m * x)
            k0 = np.sum(m * d); kp = np.sum(m * d * np.cos(m * np.pi))
            reg[i, j] = 0 if (k0 > 0 and kp < 0) else 1 if (k0 < 0 and kp < 0) else 2
    ax[0, 1].pcolormesh(aa, bb, reg, cmap=plt.matplotlib.colors.ListedColormap(
        ["#dbe6f2", "#f5dcdc", "#dcf0dc"]), vmin=-0.5, vmax=2.5, shading="nearest")
    for z in special.jn_zeros(0, 3):
        ax[0, 1].axvline(z, color="k", ls=":", lw=0.6)
    ax[0, 1].set_xlabel("$a$"); ax[0, 1].set_ylabel("$b$")
    ax[0, 1].text(1.0, 2.6, "I", fontsize=9); ax[0, 1].text(3.6, 2.6, "III", fontsize=9)
    ps.panel(ax[0, 1], "b")

    b = 1.0
    a1 = gf.first_zero_mass2(b); a2 = gf.first_zero_mass2(b, at=np.pi)
    u = np.linspace(-2 * np.pi, 2 * np.pi, 601)
    for x, lab in [(0.0, "$a$ = 0"), (1.5, "$a$ = 1.5 (I)"),
                   (0.5 * (a1 + a2), f"$a$ = {0.5*(a1+a2):.2f} (II)"), (3.2, "$a$ = 3.2 (III)")]:
        ax[1, 0].plot(u, gf.DressedGSL(x, b).G(u), label=lab)
    ax[1, 0].set_xlabel("$u$"); ax[1, 0].set_ylabel("$G_{b,a}(u)-G_{b,a}(0)$"); ax[1, 0].legend()
    ps.panel(ax[1, 0], "c")

    m = np.arange(1, 7)
    for b in [0.25, 0.5, 1.0]:
        c = gf.fourier_coefficients(b)
        ax[1, 1].semilogy(m, np.abs(c[:6]), "o-", label=f"$b$ = {b}")
    ax[1, 1].set_xlabel("$m$"); ax[1, 1].set_ylabel("$|c_m(b)|$"); ax[1, 1].legend()
    ps.panel(ax[1, 1], "d")
    fig.tight_layout(); ps.save(fig, "fig1_dressed_nonlinearity")


# ---------------------------------------------------------------- Fig. 2
def fig2():
    d = load("power_vs_a_b0.5_Om0.8"); f = load("fine_scan_b0.5_Om0.8")
    h = load("emitted_harmonics_b0.5_Om0.8")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax[0].semilogy(d["a"], d["P_per_period_over_E0"], "o-")
    ax[0].axvline(A0, color="gray", ls="--", lw=0.7)
    ax[0].set_xlabel("$a$"); ax[0].set_ylabel("$P\\,T/E_0$")
    ps.panel(ax[0], "a", x=-0.30)

    s = f["a"]; P = f["P_per_period_over_E0"]; r = np.sqrt(P)
    lo = s <= 1.245; hi = s >= 1.26
    pl = np.polyfit(s[lo], -r[lo], 1); ph = np.polyfit(s[hi], r[hi], 1)
    ac = 0.5 * (-pl[1] / pl[0] - ph[1] / ph[0])
    ax[1].plot(s, np.where(s < ac, -r, r), "o")
    xl = np.linspace(s[0], ac, 30); xh = np.linspace(ac, s[-1], 30)
    ax[1].plot(xl, np.polyval(pl, xl), "-", lw=0.7, color=ps.PALETTE[1])
    ax[1].plot(xh, np.polyval(ph, xh), "-", lw=0.7, color=ps.PALETTE[1])
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set_xlabel("$a$"); ax[1].set_ylabel("$\\pm\\sqrt{P\\,T/E_0}$")
    ax[1].set_title(f"$a_c$ = {ac:.4f}", fontsize=8)
    ax[1].ticklabel_format(axis="x", useOffset=False)
    ps.panel(ax[1], "b", x=-0.34)

    ks = [1, 3, 5, 7]; w = 0.25
    for i, a in enumerate(h["a"]):
        vals = [h["A1_evanescent"][i], h["A3"][i], h["A5"][i], h["A7"][i]]
        ax[2].bar(np.arange(4) + (i - 1) * w, vals, w, label=f"$a$ = {a:.2f}")
    ax[2].set_yscale("log"); ax[2].set_xticks(range(4))
    ax[2].set_xticklabels([f"$k$={k}" for k in ks]); ax[2].set_ylabel("$|A_k|$")
    ax[2].legend(fontsize=7)
    ps.panel(ax[2], "c", x=-0.30)
    fig.tight_layout(); ps.save(fig, "fig2_cancellation")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    o = load("power_vs_Omega_a0.6_b0.5"); bb = load("power_vs_b_a0.6_Om0.8")
    ac = load("ac_first_zero_vs_b_Om0.8")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    E = PI2 / o["epsilon"]
    ax[0].semilogy(E, o["P_per_period_over_E0"], "o")
    p = np.polyfit(E, np.log(o["P_per_period_over_E0"]), 1)
    ax[0].semilogy(E, np.exp(np.polyval(p, E)), "-", lw=0.7)
    ax[0].set_xlabel("$\\pi\\sqrt{2}/\\epsilon$"); ax[0].set_ylabel("$P\\,T/E_0$")
    ps.panel(ax[0], "a", x=-0.30)
    ax[1].loglog(bb["rho_exact"], bb["P_per_period_over_E0"], "o-")
    ax[1].loglog(bb["rho_exact"], bb["P_per_period_over_E0"][0] *
                 (bb["rho_exact"] / bb["rho_exact"][0]) ** 2, "--", lw=0.7, label="$\\rho^2$")
    ax[1].set_xlabel("$\\rho$"); ax[1].set_ylabel("$P\\,T/E_0$"); ax[1].legend()
    ps.panel(ax[1], "b", x=-0.32)
    b2 = ac["b"] ** 2
    ax[2].plot(b2, ac["a_c_measured"], "o", label="measured")
    ax[2].plot(b2, ac["a_c_predicted_V3_V4"], "s", mfc="none", label="predicted")
    pp = np.polyfit(b2[:2], ac["a_c_measured"][:2], 1)
    xx = np.linspace(0, 0.27, 20); ax[2].plot(xx, np.polyval(pp, xx), "-", lw=0.7)
    ax[2].axhline(A0, color="gray", ls="--", lw=0.7)
    ax[2].set_xlabel("$b^2$"); ax[2].set_ylabel("$a_c$"); ax[2].legend()
    ps.panel(ax[2], "c", x=-0.32)
    fig.tight_layout(); ps.save(fig, "fig3_asymptotics")


# ---------------------------------------------------------------- Fig. 4
def fig4():
    k = load("stokes_K_of_epsilon"); mp = load("ac_map_b_Omega"); sc = load("stokes_constants_inner")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    eps = k["epsilon"]; q3 = np.sqrt(8 - 9 * eps ** 2)
    y = np.log(k["K_richardson"]) + np.pi * q3 / (2 * eps)
    L0 = float(sc["Lambda0"][np.argmin(np.abs(sc["m"] - 2))])
    ax[0].plot(eps ** 2, y - np.log(L0), "o")
    c = np.linalg.lstsq(np.column_stack([eps ** 2, eps ** 4]), y - np.log(L0), rcond=None)[0]
    xx = np.linspace(0, eps.max() ** 2, 50)
    ax[0].plot(xx, c[0] * xx + c[1] * xx ** 2, "-", lw=0.7)
    ax[0].axhline(0, color="gray", ls="--", lw=0.7)
    ax[0].set_xlabel("$\\epsilon^2$")
    ax[0].set_ylabel("$\\ln\\mathcal{K}+\\pi q_3/(2\\epsilon)-\\ln\\Lambda_0$")
    ax[0].set_title(f"$\\Lambda_0$ = {L0:.4f}, $\\gamma$ = 0", fontsize=8)
    ps.panel(ax[0], "a", x=-0.26)
    for b, mk in [(0.25, "o"), (0.5, "s")]:
        sel = mp["b"] == b
        ax[1].plot(mp["Omega"][sel], (mp["a_c"][sel] - A0) / b ** 2, mk + "-", label=f"$b$ = {b}")
    ax[1].axhline(0, color="gray", ls="--", lw=0.7)
    ax[1].set_xlabel("$\\Omega$"); ax[1].set_ylabel("$(a_c-j_{0,1}/2)/b^2$"); ax[1].legend()
    ps.panel(ax[1], "b", x=-0.22)
    fig.tight_layout(); ps.save(fig, "fig4_stokes_and_map")


# ---------------------------------------------------------------- Fig. 5
def fig5():
    s = load("second_zero_scans_Om0.8"); t = load("ac_second_zero_vs_b_Om0.8")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for i, b in enumerate([0.5, 0.25, 0.125]):
        sel = s["b"] == b
        ax[i].plot(s["a"][sel], s["A3"][sel], "o-")
        ax[i].axvline(A0b, color="gray", ls="--", lw=0.7)
        j = np.argmin(np.abs(t["b"] - b))
        ax[i].axvline(t["a_c_measured"][j], color=ps.PALETTE[1], lw=0.8)
        ax[i].set_xlabel("$a$"); ax[i].set_title(f"$b$ = {b}", fontsize=8)
        ax[i].ticklabel_format(axis="x", useOffset=False)
        ax[i].tick_params(axis="x", labelrotation=30)
        ps.panel(ax[i], "abc"[i], x=-0.30)
    ax[0].set_ylabel("$|A_3|$")
    fig.tight_layout(); ps.save(fig, "fig5_second_zero")


# ---------------------------------------------------------------- Fig. 6
def fig6():
    v = load("collisions_vcr_b1_Om0.8"); ak = load("kink_a_k_vs_b")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for b in [0.25, 0.5, 1.0]:
        r = np.genfromtxt(result_path(f"kink_modes_b{b}.csv"), delimiter=",", names=True)
        ok = ~np.isnan(r["internal_omega2"])
        ax[0].plot(r["a"][ok], np.sqrt(np.maximum(r["kappa"][ok] - r["internal_omega2"][ok], 0)),
                   "o-", ms=2.5, label=f"$b$ = {b}")
        j = np.argmin(np.abs(ak["b"] - b))
        ax[0].plot([ak["a_k_measured"][j]], [0.0], "v", ms=4, color="k")
    ax[0].axvline(A0, color="gray", ls="--", lw=0.7)
    ax[0].set_xlabel("$a$"); ax[0].set_ylabel("$\\kappa_b$"); ax[0].legend()
    ps.panel(ax[0], "a", x=-0.24)
    ax[1].plot(v["a"], v["v_cr"], "o-")
    ax[1].axvline(1.03583, color="gray", ls="--", lw=0.7)
    ax[1].set_xlabel("$a$"); ax[1].set_ylabel("$v_{cr}$")
    ax[1].set_title("$b$ = 1 ($a_k$ = 1.036)", fontsize=8)
    ps.panel(ax[1], "b", x=-0.22)
    fig.tight_layout(); ps.save(fig, "fig6_kink")


if __name__ == "__main__":
    print("writing publication figures to figures/")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
