"""Stretch goal 1: how plug-in estimation distorts transfer entropy at small n.

Draws n i.i.d. samples of (Y_t+1, X_t, Y_t) from the exact distributions,
computes the plug-in TE with dit, and compares it with the exact value.
Run with ``uv run experiments/finite_sample_te.py``. Writes
``results/finite_sample/finite_sample_te.{csv,png}``.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from plot_style import BLUE, GRAY, ORANGE, RESULTS, SURFACE, note, save, subtitle, use_style

from causality_measures import systems
from causality_measures.estimation import null_quantile, plugin_bias, plugin_transfer_entropy
from causality_measures.measures import transfer_entropy

OUT = RESULTS / "finite_sample"
N_GRID = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000]
REPLICATES = 2000
SEED = 20260930

CASES = [
    ("uncoupled", "No coupling (X → Y channel is pure noise)", systems.direct_chain(noise=0.5), BLUE),
    ("copy_0.99", "Exact copy, persistence 0.99", systems.synchronized_copy(persistence=0.99), ORANGE),
]


def simulate(system, rng) -> list[dict]:
    exact = transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t")
    rows = []
    for n in N_GRID:
        estimates = np.array(
            [plugin_transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t", n, rng) for _ in range(REPLICATES)]
        )
        q05, q50, q95 = np.quantile(estimates, [0.05, 0.5, 0.95])
        rows.append(
            {
                "n": n,
                "exact": exact,
                "mean": estimates.mean(),
                "q05": q05,
                "median": q50,
                "q95": q95,
                "bias": estimates.mean() - exact,
            }
        )
    return rows


def main() -> None:
    use_style()
    rng = np.random.default_rng(SEED)
    results = {key: simulate(system, rng) for key, _, system, _ in CASES}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "finite_sample_te.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "n", "exact", "mean", "q05", "median", "q95", "bias"])
        writer.writeheader()
        for key, rows in results.items():
            writer.writerows({"case": key, **row} for row in rows)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    n = np.array(N_GRID)

    for key, label, _, color in CASES:
        rows = results[key]
        exact = rows[0]["exact"]
        left.fill_between(n, [r["q05"] for r in rows], [r["q95"] for r in rows], color=color, alpha=0.12, linewidth=0)
        left.plot(n, [r["mean"] for r in rows], color=color, marker="o", markeredgecolor=SURFACE,
                  label=f"{label}: mean plug-in, 5–95% band")
        left.axhline(exact, color=color, linewidth=1.2, linestyle=(0, (4, 3)), label=f"exact TE = {exact:.3f}")
    left.set_xscale("log")
    left.set_xlim(N_GRID[0], N_GRID[-1])
    left.set_ylim(-0.01, None)
    left.set_xlabel("sample size n  (i.i.d. draws of Y_t+1, X_t, Y_t)")
    left.set_ylabel("transfer entropy X → Y  (bits)")
    left.set_title("Small samples manufacture 'causality' and hide the real thing", pad=22)
    subtitle(left, f"Plug-in TE over {REPLICATES:,} replicates per n")
    left.legend(loc="upper right", fontsize=8.5)
    note(left, 250, 0.3, "For n ≤ 30 the uncoupled system's mean TE\nexceeds that of the perfectly coupled one:\nbias is upward with no coupling, downward\nfor a deterministic copy (empty cells)")

    null_rows = results["uncoupled"]
    right.plot(n, [r["mean"] for r in null_rows], color=BLUE, marker="o", markeredgecolor=SURFACE, label="mean plug-in TE")
    right.plot(n, [r["q95"] for r in null_rows], color=BLUE, marker="^", markeredgecolor=SURFACE,
               linestyle=(0, (1, 2)), label="95th percentile")
    right.plot(n, [plugin_bias(v) for v in n], color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)),
               label="theory: bias = 1 / (n ln 2)")
    right.plot(n, [null_quantile(v) for v in n], color=GRAY, linewidth=1.5, linestyle=(0, (1, 2)),
               label=r"theory: $\chi^2_2$ 95% quantile / (2n ln 2)")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(N_GRID[0], N_GRID[-1])
    right.set_xlabel("sample size n")
    right.set_ylabel("plug-in TE when exact TE = 0  (bits)")
    right.set_title("The spurious TE decays only like 1/n", pad=22)
    subtitle(right, "Uncoupled system; theory is the chi-squared limit of the G-test")
    right.legend(loc="lower left", fontsize=8.5)

    fig.tight_layout(w_pad=3)
    save(fig, OUT / "finite_sample_te.png")

    for key, label, _, _ in CASES:
        print(f"\n{label}")
        print(f"{'n':>6} {'exact':>7} {'mean':>7} {'q05':>7} {'q95':>7}")
        for r in results[key]:
            print(f"{r['n']:>6} {r['exact']:7.4f} {r['mean']:7.4f} {r['q05']:7.4f} {r['q95']:7.4f}")


if __name__ == "__main__":
    main()
