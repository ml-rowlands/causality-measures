"""Checks of the plug-in transfer entropy estimator."""

import numpy as np
import pytest

from causality_measures import systems
from causality_measures.do_operator import joint_items
from causality_measures.estimation import (
    empirical_distribution,
    null_quantile,
    plugin_bias,
    plugin_transfer_entropy,
)
from causality_measures.measures import transfer_entropy


def test_empirical_distribution_keeps_variable_labels():
    """In the copy system Y_t+1 = X_t, so every sampled outcome must agree on those two."""
    system = systems.synchronized_copy(persistence=0.8)
    sample = empirical_distribution(system.dist, ["Y_t+1", "X_t", "Y_t"], 500, np.random.default_rng(0))
    assert sample.get_rv_names() == ("Y_t+1", "X_t", "Y_t")
    assert all(y_next == x for (y_next, x), _ in joint_items(sample, ["Y_t+1", "X_t"]))


@pytest.mark.parametrize(
    "system",
    [systems.synchronized_copy(0.9), systems.confounded_lags(0.8), systems.xor_synergy(0.1)],
    ids=lambda s: s.name,
)
def test_plugin_estimate_converges_to_exact_value(system):
    exact = transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t")
    estimate = plugin_transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t", 400_000, np.random.default_rng(1))
    assert estimate == pytest.approx(exact, abs=0.005)


def test_null_bias_matches_first_order_theory():
    """With exact TE = 0 the mean plug-in estimate is about 1 / (n ln 2) bits."""
    system = systems.direct_chain(noise=0.5)
    rng = np.random.default_rng(2)
    n = 400
    estimates = [plugin_transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t", n, rng) for _ in range(3000)]
    assert np.mean(estimates) == pytest.approx(plugin_bias(n), rel=0.1)
    assert np.quantile(estimates, 0.95) == pytest.approx(null_quantile(n), rel=0.15)
