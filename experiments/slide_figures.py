"""Slide-sized vector figures for ``slides/slides.tex``, drawn from the results CSVs.

Run the other experiments first, then ``uv run experiments/slide_figures.py``.
Nothing is recomputed: every number comes from ``results/``. Figures are
drawn at their final physical size so their text matches the slides.
"""

import csv
from math import log2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, ORANGE, RESULTS, SERIES, SURFACE, TEXT, TEXT_SECONDARY, use_style

OUT = Path(__file__).resolve().parent.parent / "slides" / "figures"
SWEEP_KEYS = ["te", "janzing", "ap_imposed", "ap"]


def main() -> None:
    use_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    binary_entropy()
    sweep("confounded_lags", "strength", "confounder strength $c$", (3.5, 2.35), legend_loc="upper left")
    sweep("synchronized_copy", "persistence", "persistence of $X$:  $P(X_t = X_{t-1})$", (3.5, 2.35),
          legend_loc="lower left")
    pid_by_system()
    pid_copy_sweep()
    finite_sample()


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote slides/figures/{name}")


def binary_entropy() -> None:
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    fig, ax = plt.subplots(figsize=(2.5, 1.75))
    ax.plot(p, h, color=BLUE)
    for x in (0.5, 0.9):
        y = -x * log2(x) - (1 - x) * log2(1 - x)
        ax.plot([x], [y], marker="o", color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f"{y:.2f} bits", (x, y), textcoords="offset points", xytext=(4, 4 if x == 0.5 else 6),
                    fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("$p$ = probability of heads")
    ax.set_ylabel("$h(p)$  (bits)")
    save(fig, "binary_entropy.pdf")


def sweep(name: str, parameter: str, xlabel: str, size, legend_loc: str) -> None:
    rows = read(RESULTS / "sweep_plots" / f"{name}.csv")
    x = np.array([float(r[parameter]) for r in rows])
    fig, ax = plt.subplots(figsize=size)
    for i, key in enumerate(SWEEP_KEYS):
        style = SERIES[key]
        ax.plot(x, [float(r[key]) for r in rows], label=_short_label(key), color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"], markevery=(i * 2, 8),
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3 + i)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bits")
    ax.legend(loc=legend_loc, handlelength=2.6)
    save(fig, f"sweep_{name}.pdf")


def _short_label(key: str) -> str:
    return {
        "te": "transfer entropy",
        "janzing": "Janzing strength",
        "ap_imposed": r"Ay–Polani, imposing $Y_t$",
        "ap": "Ay–Polani, nothing imposed",
    }[key]


PID_ROWS = [
    ("Confounding", "Confounding, staggered lags", "Y_t", "TE = {te}"),
    ("XOR with $W = Y_t$", "XOR, W = Y_t", "Y_t", "TE = {te}"),
    ("XOR with external $W_t$", "XOR, W = W_t: partner W_t", "W_t", "TE = {te}  (TE looks at $Y_t$)"),
    ("Synchronized copy", "Synchronized copy", "Y_t", "TE = {te}"),
]
PID_SEGMENTS = [
    ("redundancy", "redundant", BLUE, "white"),
    ("unique_source", r"unique to $X_t$", ORANGE, TEXT),
    ("synergy", "synergistic", AQUA, TEXT),
]


def pid_by_system() -> None:
    table = {(r["system"], r["partner"]): r for r in read(RESULTS / "pid" / "pid_table.csv") if r["pid"] == "WB"}
    fig, ax = plt.subplots(figsize=(5.6, 1.7))
    y = np.arange(len(PID_ROWS))[::-1]
    for yi, (label, system, partner, te_text) in zip(y, PID_ROWS):
        row = table[(system, partner)]
        left = 0.0
        for key, _, color, ink in PID_SEGMENTS:
            width = float(row[key])
            if width <= 1e-9:
                continue
            ax.barh(yi, width, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=1.5)
            ax.text(left + width / 2, yi, f"{width:.2f}", ha="center", va="center", fontsize=7, color=ink)
            left += width
        ax.text(left + 0.02, yi, te_text.format(te=f"{float(row['te']):.2f}"), va="center", fontsize=7, color=TEXT)
    ax.set_yticks(y, [row[0] for row in PID_ROWS])
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"bits of $I(Y_{t+1};\, X_t, W)$")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in PID_SEGMENTS]
    ax.legend(handles, [name for _, name, _, _ in PID_SEGMENTS], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, frameon=False, borderaxespad=0.2)
    save(fig, "pid_by_system.pdf")


def pid_copy_sweep() -> None:
    rows = read(RESULTS / "pid" / "pid_copy_sweep.csv")
    p = np.array([float(r["persistence"]) for r in rows])
    unique = np.array([float(r["unique_source"]) for r in rows])
    redundant = np.array([float(r["redundancy"]) for r in rows])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.fill_between(p, 0, unique, color=ORANGE, alpha=0.55, linewidth=0, label=r"unique to $X_t$ (= TE)")
    ax.fill_between(p, unique, unique + redundant, color=BLUE, alpha=0.55, linewidth=0, label=r"redundant with $Y_t$")
    ax.plot(p, unique, color=SURFACE, linewidth=1.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"persistence of $X$:  $P(X_t = X_{t-1})$")
    ax.set_ylabel("bits")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=0.9)
    save(fig, "pid_copy_sweep.pdf")


def finite_sample() -> None:
    rows = read(RESULTS / "finite_sample" / "finite_sample_te.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    cases = [("uncoupled", "no coupling", BLUE), ("copy_0.99", "exact copy", ORANGE)]
    for case, label, color in cases:
        sub = [r for r in rows if r["case"] == case]
        n = np.array([float(r["n"]) for r in sub])
        ax.fill_between(n, [float(r["q05"]) for r in sub], [float(r["q95"]) for r in sub], color=color, alpha=0.12,
                        linewidth=0)
        ax.plot(n, [float(r["mean"]) for r in sub], color=color, marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.0, label=f"{label}: average estimate")
        ax.axhline(float(sub[0]["exact"]), color=color, linewidth=1.0, linestyle=(0, (4, 3)),
                   label=f"{label}: exact value ({float(sub[0]['exact']):.2f})")
    ax.set_xscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_ylim(-0.01, 0.52)
    ax.set_xlabel("number of samples $n$")
    ax.set_ylabel("estimated TE (bits)")
    ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "finite_sample.pdf")


if __name__ == "__main__":
    main()
