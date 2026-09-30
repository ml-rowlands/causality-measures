"""Stretch goal 2: partial information decomposition of what transfer entropy counts.

For each system, decomposes I(Y_t+1 ; X_t, W) into redundant, unique and
synergistic parts with ``dit.pid`` (W = Y_t unless noted). Any PID satisfies
TE = I(Y_t+1 ; X_t | Y_t) = unique(X_t) + synergy, so the decomposition shows
which kind of information TE is reporting as "transfer".

Run with ``uv run experiments/pid_analysis.py``. Writes ``results/pid/``.
"""

import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
from plot_style import AQUA, BLUE, ORANGE, RESULTS, SURFACE, TEXT, TEXT_SECONDARY, save, subtitle, use_style

from causality_measures import systems
from causality_measures.decomposition import ATOMS, PID_MEASURES, pid_atoms
from causality_measures.measures import transfer_entropy

OUT = RESULTS / "pid"
YELLOW = "#eda100"

ROWS = [
    ("Direct chain (control)", systems.direct_chain(noise=0.1), "Y_t"),
    ("Confounding, staggered lags", systems.confounded_lags(strength=0.9), "Y_t"),
    ("XOR, W = Y_t", systems.xor_synergy(noise=0.0, partner="target_past"), "Y_t"),
    ("XOR, W = W_t: partner Y_t", systems.xor_synergy(noise=0.0, partner="external"), "Y_t"),
    ("XOR, W = W_t: partner W_t", systems.xor_synergy(noise=0.0, partner="external"), "W_t"),
    ("Synchronized copy", systems.synchronized_copy(persistence=0.95), "Y_t"),
]

# Segment order left to right; TE = the last two segments when the partner is Y_t.
SEGMENTS = [
    ("unique_partner", "unique to partner", YELLOW, TEXT),
    ("redundancy", "redundant", BLUE, "white"),
    ("unique_source", "unique to X_t", ORANGE, TEXT),
    ("synergy", "synergistic", AQUA, TEXT),
]


def main() -> None:
    warnings.filterwarnings("ignore")  # BROJA's optimizer is chatty
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    table = write_table()
    plot_by_system(table)
    plot_copy_sweep()


def write_table() -> list[dict]:
    table = []
    for label, system, partner in ROWS:
        te = transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t")
        for measure in PID_MEASURES:
            atoms = pid_atoms(system.dist, "X_t", partner, "Y_t+1", measure)
            table.append(
                {
                    "system": label,
                    "partner": partner,
                    "pid": measure,
                    **{a: f"{atoms[a]:.4f}" for a in ATOMS},
                    "total": f"{sum(atoms.values()):.4f}",
                    "unique_source+synergy": f"{atoms['unique_source'] + atoms['synergy']:.4f}",
                    "te": f"{te:.4f}",
                }
            )
    with open(OUT / "pid_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"{'system':32} {'W':4} {'pid':6} {'red':>7} {'uniqX':>7} {'uniqW':>7} {'syn':>7} {'TE':>7}")
    for row in table:
        print(
            f"{row['system']:32} {row['partner']:4} {row['pid']:6} {row['redundancy']:>7} "
            f"{row['unique_source']:>7} {row['unique_partner']:>7} {row['synergy']:>7} {row['te']:>7}"
        )
    print("wrote results/pid/pid_table.csv")
    return table


def plot_by_system(table) -> None:
    rows = [r for r in table if r["pid"] == "WB"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        left = 0.0
        for key, _, color, ink in SEGMENTS:
            width = float(row[key])
            if width <= 1e-9:
                continue
            ax.barh(yi, width, left=left, height=0.46, color=color, edgecolor=SURFACE, linewidth=2)
            if width >= 0.12:
                ax.text(left + width / 2, yi, f"{width:.3f}", ha="center", va="center", fontsize=8.5, color=ink)
            left += width
        _te_bracket(ax, yi, row)

    ax.set_yticks(y, [f"{r['system']}" for r in rows])
    ax.set_xlim(0, 1.45)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("bits of I(Y_t+1 ; X_t, W)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("What transfer entropy is made of", pad=22)
    subtitle(ax, "Williams–Beer PID of I(Y_t+1 ; X_t, W); W = Y_t unless noted.  TE = unique to X_t + synergistic")
    shown = [seg for seg in SEGMENTS if any(float(r[seg[0]]) > 1e-9 for r in rows)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in shown]
    ax.legend(handles, [name for _, name, _, _ in shown], loc="lower center", bbox_to_anchor=(0.4, -0.3), ncol=len(shown))
    ax.set_axisbelow(True)
    save(fig, OUT / "pid_by_system.png")


def _te_bracket(ax, yi, row) -> None:
    start = float(row["unique_partner"]) + float(row["redundancy"])
    end = start + float(row["unique_source+synergy"])
    y_top = yi + 0.31
    if end - start > 1e-9:
        ax.plot([start, start, end, end], [y_top - 0.05, y_top, y_top, y_top - 0.05], color=TEXT_SECONDARY, linewidth=1)
    if row["partner"] == "Y_t":
        text = f"TE = {float(row['te']):.3f}"
        if float(row["total"]) < 1e-9:
            text += "   (X_t and Y_t say nothing about Y_t+1)"
    else:
        text = f"unique + synergy = {float(row['unique_source+synergy']):.3f},  but TE = {float(row['te']):.3f}"
    ax.text(max(end, 0) + 0.02, yi, text, va="center", fontsize=8.5, color=TEXT)


def plot_copy_sweep() -> None:
    grid = np.linspace(0.5, 1.0, 41)
    rows = []
    for p in grid:
        system = systems.synchronized_copy(persistence=float(p))
        atoms = pid_atoms(system.dist, "X_t", "Y_t", "Y_t+1", "WB")
        rows.append({"persistence": float(p), **atoms, "te": transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t")})
    with open(OUT / "pid_copy_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unique = np.array([r["unique_source"] for r in rows])
    redundant = np.array([r["redundancy"] for r in rows])
    te = np.array([r["te"] for r in rows])
    assert max(r["synergy"] + r["unique_partner"] for r in rows) < 1e-9

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.fill_between(grid, 0, unique, color=ORANGE, alpha=0.55, linewidth=0, label="unique to X_t")
    ax.fill_between(grid, unique, unique + redundant, color=BLUE, alpha=0.55, linewidth=0, label="redundant with Y_t")
    ax.plot(grid, unique, color=SURFACE, linewidth=2)
    ax.plot(grid, te, color=TEXT, linewidth=1.5, linestyle=(0, (4, 3)), label="transfer entropy")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("persistence of X:  P(X_t = X_t−1)")
    ax.set_ylabel("bits")
    ax.set_title("Synchronized copy: TE keeps only the unique part", pad=22)
    subtitle(ax, "Williams–Beer PID of I(Y_t+1 ; X_t, Y_t) = 1 bit; synergy and unique-to-Y_t are 0 throughout")
    ax.text(0.985, 0.87, "what Y's own past already knows\n(X_t is predictable from X_t−1 = Y_t)", ha="right",
            fontsize=8.5, color=TEXT)
    ax.set_axisbelow(True)
    ax.text(0.53, 0.35, "new information only X_t carries", fontsize=8.5, color=TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    save(fig, OUT / "pid_copy_sweep.png")


if __name__ == "__main__":
    main()
