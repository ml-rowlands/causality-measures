"""Finite-sample (plug-in) transfer entropy, to contrast with the exact values.

The plug-in estimator is simply the exact measure evaluated on the empirical
distribution of a sample, so it reuses ``transfer_entropy`` unchanged. Samples
are i.i.d. draws of the (Y_t+1, X_t, Y_t) triple from the exact stationary
distribution: the most favourable case, since real time series add serial
correlation on top.
"""

from collections.abc import Sequence
from math import log

import dit
import numpy as np

from causality_measures.do_operator import joint_items
from causality_measures.measures import transfer_entropy


def empirical_distribution(
    dist: dit.Distribution, names: Sequence[str], n: int, rng: np.random.Generator
) -> dit.Distribution:
    """Empirical distribution of ``n`` i.i.d. draws from the marginal of ``dist`` over ``names``."""
    items = joint_items(dist, names)
    probs = np.array([p for _, p in items])
    counts = rng.multinomial(n, probs / probs.sum())
    observed = counts > 0
    outcomes = [o for (o, _), seen in zip(items, observed) if seen]
    return dit.Distribution(outcomes, counts[observed] / n, rv_names=list(names))


def plugin_transfer_entropy(
    dist: dit.Distribution,
    source: str,
    target: str,
    target_past: str,
    n: int,
    rng: np.random.Generator,
) -> float:
    """Plug-in TE estimate from ``n`` samples drawn from ``dist``."""
    sample = empirical_distribution(dist, [target, source, target_past], n, rng)
    return transfer_entropy(sample, source, target, target_past)


def plugin_bias(n: int, source_size: int = 2, target_size: int = 2, past_size: int = 2) -> float:
    """First-order bias of the plug-in conditional mutual information, in bits.

    For large n, 2 n ln(2) * I_hat is approximately chi-squared with
    df = (|X| - 1)(|Y| - 1)|Z| degrees of freedom when the true CMI is zero,
    so the estimator overshoots by about df / (2 n ln 2) bits.
    """
    return degrees_of_freedom(source_size, target_size, past_size) / (2 * n * log(2))


def null_quantile(n: int, level: float = 0.95, source_size: int = 2, target_size: int = 2, past_size: int = 2) -> float:
    """``level`` quantile of the plug-in TE when the true TE is zero (chi-squared approximation).

    Only implemented for df = 2 (binary variables), where chi-squared is
    exponential with mean 2 and the quantile has the closed form -2 ln(1 - level).
    """
    if degrees_of_freedom(source_size, target_size, past_size) != 2:
        raise NotImplementedError("closed-form quantile only for binary variables (df = 2)")
    return -2 * log(1 - level) / (2 * n * log(2))


def degrees_of_freedom(source_size: int, target_size: int, past_size: int) -> int:
    return (source_size - 1) * (target_size - 1) * past_size
