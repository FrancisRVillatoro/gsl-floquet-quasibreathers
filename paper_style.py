"""Common matplotlib style for the published figures.  Import and call `use()`."""
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


def save(fig, name):
    """Write figures/<name>.pdf and figures/<name>.png."""
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}")
    print(f"   figures/{name}.pdf, .png")
