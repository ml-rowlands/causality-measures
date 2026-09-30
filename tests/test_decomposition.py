"""Checks of the partial information decomposition used to explain transfer entropy."""

import warnings
from math import log2

import pytest

from causality_measures import systems
from causality_measures.decomposition import PID_MEASURES, pid_atoms
from causality_measures.measures import transfer_entropy

warnings.filterwarnings("ignore")  # BROJA's optimizer is chatty

TE_SYSTEMS = [
    systems.direct_chain(0.1),
    systems.confounded_lags(0.9),
    systems.xor_synergy(0.1),
    systems.synchronized_copy(0.8, copy_noise=0.05),
]


def h(p):
    return 0.0 if p in (0, 1) else -p * log2(p) - (1 - p) * log2(1 - p)


@pytest.mark.parametrize("measure", PID_MEASURES)
@pytest.mark.parametrize("system", TE_SYSTEMS, ids=lambda s: s.name)
def test_te_is_unique_plus_synergy(system, measure):
    """Consistency of any PID: I(Y_t+1 ; X_t | Y_t) = unique(X_t) + synergy."""
    atoms = pid_atoms(system.dist, "X_t", "Y_t", "Y_t+1", measure)
    te = transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t")
    tolerance = 3e-3 if measure == "BROJA" else 1e-9  # BROJA is an optimization
    assert atoms["unique_source"] + atoms["synergy"] == pytest.approx(te, abs=tolerance)


@pytest.mark.parametrize("partner", ["target_past", "external"])
def test_xor_is_pure_synergy(partner):
    system = systems.xor_synergy(0.0, partner=partner)
    w = "Y_t" if partner == "target_past" else "W_t"
    atoms = pid_atoms(system.dist, "X_t", w, "Y_t+1")
    assert atoms == pytest.approx({"redundancy": 0, "unique_source": 0, "unique_partner": 0, "synergy": 1})


@pytest.mark.parametrize("persistence", [0.6, 0.9, 0.99])
def test_copy_splits_into_redundancy_and_unique_information(persistence):
    """Y_t+1 = X_t: what Y_t = X_t-1 already predicts is redundant; TE keeps the rest."""
    system = systems.synchronized_copy(persistence)
    atoms = pid_atoms(system.dist, "X_t", "Y_t", "Y_t+1")
    assert atoms["redundancy"] == pytest.approx(1 - h(persistence))
    assert atoms["unique_source"] == pytest.approx(h(persistence))
    assert atoms["synergy"] == pytest.approx(0, abs=1e-12)
