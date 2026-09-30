"""Every measure on every system at one representative setting: the disagreements at a glance.

Run with ``uv run experiments/comparison_table.py``. Writes
``results/comparison_table.csv`` and a Markdown copy for the README.
"""

import csv
from pathlib import Path

from causality_measures import systems
from causality_measures.compare import MEASURES, evaluate

RESULTS = Path(__file__).resolve().parent.parent / "results"

ROWS = [
    ("Direct chain (control)", systems.direct_chain(noise=0.1)),
    ("Confounding, staggered lags", systems.confounded_lags(strength=0.9)),
    ("  ...same data, analyst's DAG (Z unobserved)", systems.as_seen_by_analyst(systems.confounded_lags(strength=0.9))),
    ("XOR synergy, W = Y_t", systems.xor_synergy(noise=0.0, partner="target_past")),
    ("XOR synergy, W = W_t (external)", systems.xor_synergy(noise=0.0, partner="external")),
    ("Synchronized copy", systems.synchronized_copy(persistence=0.95)),
]


def main() -> None:
    header = ["system", "parameters", "arrow X_t->Y_t+1", "imposed S", *MEASURES.values()]
    table = []
    for label, system in ROWS:
        values = evaluate(system)
        params = ", ".join(f"{k}={v:g}" for k, v in system.params.items())
        table.append(
            [
                label.strip(),
                params,
                "yes" if system.has_arrow else "no",
                "{" + ", ".join(system.imposed) + "}",
                *(f"{values[key]:.4f}" for key in MEASURES),
            ]
        )

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "comparison_table.csv", "w", newline="") as f:
        csv.writer(f).writerows([header, *table])
    (RESULTS / "comparison_table.md").write_text(_markdown(table))
    print(_console(header, table))
    print("\nwrote results/comparison_table.csv and results/comparison_table.md")


MARKDOWN_HEADER = ["System", "Parameters", "Arrow X_t→Y_t+1?", "Imposed S", "MI", "TE", "AP", "AP \\| Ŝ", "Janzing"]


def _markdown(table) -> str:
    align = ["---"] * 4 + ["---:"] * len(MEASURES)
    cells = [[c.replace("|", "\\|") for c in row] for row in table]
    lines = [MARKDOWN_HEADER, align, *cells]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines) + "\n"


def _console(header, table) -> str:
    short = ["system", "params", "arrow", "S", "MI", "TE", "AP", "AP|S", "Janzing"]
    widths = [max(len(str(r[i])) for r in [short, *table]) for i in range(len(short))]
    fmt = lambda row: "  ".join(str(c).ljust(w) if i < 4 else str(c).rjust(w) for i, (c, w) in enumerate(zip(row, widths)))
    return "\n".join([fmt(short), fmt(["-" * w for w in widths]), *map(fmt, table)])


if __name__ == "__main__":
    main()
