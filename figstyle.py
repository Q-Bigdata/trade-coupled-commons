"""Publication figure mechanics: role-mapped size ladder (8/7/6 pt), open frame, outward ticks,
frameless legends, Type-42 fonts, 300 dpi."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

META_GREY = "#888888"


def apply_figure_style(sizes=(8, 7, 6)):
    base, sec, tick = sizes
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": sec,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left",
    })


def panel_letter(ax, letter, dx=-0.18, dy=1.04, size=9):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold", va="bottom", ha="left")
