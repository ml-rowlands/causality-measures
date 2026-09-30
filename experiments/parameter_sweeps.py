"""Sweep each system's parameter and plot how every measure responds.

Run with ``uv run experiments/parameter_sweeps.py``. For each sweep, writes
``results/sweep_plots/<name>.csv`` (the exact values) and ``<name>.png``,
plus an ``overview.png`` with all four panels.
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from plot_style import RESULTS, note, plot_measures, save, subtitle, use_style

from causality_measures import systems
from causality_measures.compare import MEASURES, evaluate

OUT = RESULTS / "sweep_plots"


@dataclass(frozen=True)
class Sweep:
    name: str
    title: str
    description: str
    parameter: str
    xlabel: str
    grid: np.ndarray
    build: Callable[[float], systems.CausalSystem]
    annotate: Callable


def _annotate_confounded(ax):
    note(ax, 0.88, 0.8, "TE = MI: spurious 'transfer'", ha="right")
    note(ax, 0.02, 0.06, "AP = AP | Ŝ = Janzing = 0\n(no arrow X_t → Y_t+1)")


def _annotate_xor_past(ax):
    note(ax, 0.02, 0.9, "TE = AP | Ŝ = Janzing = 1 − h(ε)")
    note(ax, 0.22, 0.04, "MI = naive AP = 0", ha="center")


def _annotate_xor_external(ax):
    note(ax, 0.02, 0.9, "AP | Ŝ = Janzing = 1 − h(ε)")
    note(ax, 0.22, 0.04, "TE = MI = naive AP = 0", ha="center")


def _annotate_copy(ax):
    note(ax, 0.99, 0.91, "MI = AP = AP | Ŝ = Janzing = 1 bit", ha="right")
    note(ax, 0.87, 0.4, "TE = h(persistence) → 0", ha="right")


SWEEPS = [
    Sweep(
        name="confounded_lags",
        title="Confounding with staggered lags",
        description="Hidden Z drives X one step later and Y two steps later; no X → Y arrow",
        parameter="strength",
        xlabel="confounder strength c  (correlation of Z with X and with Y)",
        grid=np.linspace(0, 1, 41),
        build=lambda c: systems.confounded_lags(strength=c),
        annotate=_annotate_confounded,
    ),
    Sweep(
        name="xor_target_past",
        title="XOR synergy with the target's past",
        description="Y_t+1 = X_t ⊕ Y_t, flipped with probability ε  (James, Barnett & Crutchfield)",
        parameter="noise",
        xlabel="XOR flip probability ε",
        grid=np.linspace(0, 0.5, 41),
        build=lambda e: systems.xor_synergy(noise=e, partner="target_past"),
        annotate=_annotate_xor_past,
    ),
    Sweep(
        name="xor_external",
        title="XOR synergy with an external partner",
        description="Y_t+1 = X_t ⊕ W_t, flipped with probability ε;  Ŝ = {Y_t, W_t}",
        parameter="noise",
        xlabel="XOR flip probability ε",
        grid=np.linspace(0, 0.5, 41),
        build=lambda e: systems.xor_synergy(noise=e, partner="external"),
        annotate=_annotate_xor_external,
    ),
    Sweep(
        name="synchronized_copy",
        title="Synchronized copy",
        description="Y_t+1 = X_t exactly, while X stays put with probability P(X_t = X_t-1)",
        parameter="persistence",
        xlabel="self-predictability of X:  persistence P(X_t = X_t−1)",
        grid=np.linspace(0.5, 1, 41),
        build=lambda p: systems.synchronized_copy(persistence=p),
        annotate=_annotate_copy,
    ),
]


def run(sweep: Sweep) -> list[dict[str, float]]:
    rows = [{sweep.parameter: float(v), **evaluate(sweep.build(float(v)))} for v in sweep.grid]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{sweep.name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[sweep.parameter, *MEASURES])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def draw(ax, sweep: Sweep, rows) -> None:
    plot_measures(ax, sweep.grid, rows)
    ax.set_xlim(sweep.grid[0], sweep.grid[-1])
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(sweep.xlabel)
    ax.set_ylabel("bits")
    ax.set_title(sweep.title, pad=22)
    subtitle(ax, sweep.description)
    sweep.annotate(ax)


def main() -> None:
    use_style()
    results = {sweep.name: run(sweep) for sweep in SWEEPS}

    for sweep in SWEEPS:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        draw(ax, sweep, results[sweep.name])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
        save(fig, OUT / f"{sweep.name}.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    for ax, sweep in zip(axes.flat, SWEEPS):
        draw(ax, sweep, results[sweep.name])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.0, w_pad=2.5)
    save(fig, OUT / "overview.png")


if __name__ == "__main__":
    main()
