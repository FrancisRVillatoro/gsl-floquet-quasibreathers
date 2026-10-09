"""Common matplotlib style and deterministic publication-figure writer."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import matplotlib as mpl

PALETTE = ["#1f4e79", "#c0392b", "#1e8449", "#b7791f", "#6c3483", "#117a8b"]


def use():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9, "axes.linewidth": 0.7,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "text.usetex": False, "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, label, x=-0.16, y=1.02):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontweight="bold", fontsize=9)


def figures_dir() -> Path:
    p = Path(os.environ.get("GSL_FIGURES_DIR", "figures"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, name):
    """Write deterministic PDF/PNG publication figures."""
    root = figures_dir()
    # Fix PDF timestamps so reruns with identical numeric inputs can be byte-compared.
    pdf_meta = {
        "Creator": "gsl-floquet-quasibreathers make_figures.py",
        "CreationDate": datetime(2026, 9, 20, tzinfo=timezone.utc),
    }
    fig.savefig(root / f"{name}.pdf", metadata=pdf_meta)
    fig.savefig(root / f"{name}.png", metadata={"Software": "gsl-floquet-quasibreathers"})
    print(f"   {root}/{name}.pdf, .png")
