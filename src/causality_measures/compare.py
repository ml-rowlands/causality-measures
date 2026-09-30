"""Evaluate every measure on a ``CausalSystem`` with one call."""

from causality_measures.measures import (
    ay_polani_flow,
    janzing_causal_strength,
    lagged_mutual_information,
    transfer_entropy,
)
from causality_measures.systems import CausalSystem

MEASURES = {
    "mi": "Lagged MI I(X_t;Y_t+1)",
    "te": "Transfer entropy I(Y_t+1;X_t|Y_t)",
    "ap": "Ay-Polani flow I(X_t->Y_t+1)",
    "ap_imposed": "Ay-Polani flow imposing S",
    "janzing": "Janzing strength C(X_t->Y_t+1)",
}
"""Column key -> human-readable label, in display order."""


def evaluate(system: CausalSystem) -> dict[str, float]:
    """All five quantities for the question "does X_t influence Y_t+1?", in bits."""
    d, g = system.dist, system.dag
    x, y, y_past = system.source, system.target, system.target_past
    values = {
        "mi": lagged_mutual_information(d, x, y),
        "te": transfer_entropy(d, x, y, y_past),
        "ap": ay_polani_flow(d, g, x, y),
        "ap_imposed": ay_polani_flow(d, g, x, y, system.imposed),
        "janzing": janzing_causal_strength(d, g, [(x, y)]),
    }
    return {key: _clean(value) for key, value in values.items()}


def _clean(value: float) -> float:
    """Drop float round-off so exact zeros print as 0 rather than -1e-17."""
    return round(float(value), 12) + 0.0
