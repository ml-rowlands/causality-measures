"""Shared matplotlib styling for the experiment figures.

Colors are the first three slots of a CVD-validated categorical palette
(blue, orange, aqua); the two Ay-Polani variants share aqua and differ by
dash pattern, and lagged MI is a gray reference line. Because several
measures coincide exactly on some systems, every series also carries its own
marker shape, with markers staggered along x so coincident lines stay
individually visible.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"

SERIES = {
    "mi": {"label": r"Lagged MI  $I(X_t;Y_{t+1})$", "color": GRAY, "linestyle": "-", "marker": "P", "linewidth": 1.5},
    "te": {"label": r"Transfer entropy  $I(Y_{t+1};X_t\,|\,Y_t)$", "color": BLUE, "linestyle": "-", "marker": "o"},
    "janzing": {"label": r"Janzing  $C_{X_t\to Y_{t+1}}$", "color": ORANGE, "linestyle": "-", "marker": "s"},
    "ap_imposed": {
        "label": r"Ay–Polani  $I(X_t\to Y_{t+1}\,|\,\hat{S})$",
        "color": AQUA,
        "linestyle": (0, (5, 3)),
        "marker": "D",
    },
    "ap": {"label": r"Ay–Polani  $I(X_t\to Y_{t+1})$", "color": AQUA, "linestyle": (0, (1, 2.5)), "marker": "v"},
}
DRAW_ORDER = ["mi", "te", "janzing", "ap_imposed", "ap"]


def use_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "lines.markersize": 6.5,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )


def plot_measures(ax, x, rows, keys=DRAW_ORDER, every=8):
    """Plot one line per measure; markers are staggered so overlapping lines stay visible."""
    for i, key in enumerate(keys):
        style = SERIES[key]
        ax.plot(
            x,
            [row[key] for row in rows],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style.get("linewidth", 2.0),
            marker=style["marker"],
            markevery=(i * every // len(keys), every),
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            zorder=2 + i,
        )


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")


def note(ax, x, y, text, **kwargs):
    """A sparse direct annotation in secondary text color."""
    ax.annotate(text, (x, y), fontsize=8.5, color=TEXT_SECONDARY, **kwargs)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(RESULTS.parent)}")
