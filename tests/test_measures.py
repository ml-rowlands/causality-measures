"""Sanity checks and closed-form values for the three causality measures."""

from math import log2

import pytest
from dit.divergences import kullback_leibler_divergence
from dit.multivariate import coinformation
from dit.shannon import conditional_entropy

from causality_measures import systems
from causality_measures.compare import evaluate
from causality_measures.do_operator import joint_from_kernels
from causality_measures.measures import (
    ay_polani_flow,
    cut_arrows,
    janzing_causal_strength,
    janzing_local_strength,
    lagged_mutual_information,
    transfer_entropy,
)

NOISES = [0.0, 0.05, 0.1, 0.25, 0.5]
STRENGTHS = [0.0, 0.3, 0.6, 0.9, 1.0]
PERSISTENCES = [0.5, 0.7, 0.9, 0.99, 1.0]


def h(p):
    """Binary entropy in bits."""
    return 0.0 if p in (0, 1) else -p * log2(p) - (1 - p) * log2(1 - p)


# --- Sanity checks requested in the brief -------------------------------------


@pytest.mark.parametrize("noise", NOISES)
def test_all_measures_agree_on_a_direct_chain(noise):
    """Unconfounded, non-synergistic X -> Y: every measure is the channel capacity."""
    values = evaluate(systems.direct_chain(noise))
    for name, value in values.items():
        assert value == pytest.approx(1 - h(noise), abs=1e-12), name


@pytest.mark.parametrize(
    "system",
    [systems.direct_chain(0.2), systems.confounded_lags(0.7)],
    ids=["chain", "confounded"],
)
def test_te_reduces_to_mutual_information_when_target_past_is_uninformative(system):
    """If Y_t is independent of (X_t, Y_t+1), conditioning on it changes nothing."""
    te = transfer_entropy(system.dist, "X_t", "Y_t+1", "Y_t")
    assert te == pytest.approx(lagged_mutual_information(system.dist, "X_t", "Y_t+1"))
    assert te > 0.1


def test_te_is_a_difference_of_conditional_entropies():
    """Cross-check dit's CMI against H(Y_t+1 | Y_t) - H(Y_t+1 | X_t, Y_t)."""
    d = systems.synchronized_copy(0.8, copy_noise=0.1).dist
    by_entropies = conditional_entropy(d, ["Y_t+1"], ["Y_t"]) - conditional_entropy(d, ["Y_t+1"], ["X_t", "Y_t"])
    assert transfer_entropy(d, "X_t", "Y_t+1", "Y_t") == pytest.approx(by_entropies)


@pytest.mark.parametrize("strength", STRENGTHS)
def test_janzing_is_zero_without_an_arrow(strength):
    system = systems.confounded_lags(strength)
    assert not system.has_arrow
    assert janzing_causal_strength(system.dist, system.dag, [("X_t", "Y_t+1")]) == pytest.approx(0, abs=1e-12)


def test_janzing_is_zero_for_an_inert_arrow():
    """The DAG has X -> Y, but Y's mechanism ignores X: cutting it changes nothing."""
    dag = {"Z": (), "X": ("Z",), "Y": ("X", "Z")}
    kernels = {
        "Z": lambda v, _: 0.5,
        "X": lambda v, pa: 0.9 if v == pa["Z"] else 0.1,
        "Y": lambda v, pa: 0.8 if v == pa["Z"] else 0.2,
    }
    d = joint_from_kernels(dag, kernels, ["Z", "X", "Y"], dict.fromkeys("ZXY", (0, 1)))
    assert lagged_mutual_information(d, "X", "Y") > 0.05
    assert janzing_causal_strength(d, dag, [("X", "Y")]) == pytest.approx(0, abs=1e-12)
    assert ay_polani_flow(d, dag, "X", "Y") == pytest.approx(0, abs=1e-12)


# --- Janzing et al. (2013) properties ------------------------------------------


@pytest.mark.parametrize(
    "system",
    [
        systems.direct_chain(0.1),
        systems.xor_synergy(0.1),
        systems.xor_synergy(0.1, partner="external"),
        systems.synchronized_copy(0.8, copy_noise=0.05),
        systems.as_seen_by_analyst(systems.confounded_lags(0.6)),
    ],
    ids=lambda s: s.name,
)
def test_janzing_local_form_equals_global_divergence(system):
    """Lemma 3: D(P || P_S) only depends on the target's family."""
    arrow = (system.source, system.target)
    assert janzing_local_strength(system.dist, system.dag, arrow) == pytest.approx(
        janzing_causal_strength(system.dist, system.dag, [arrow])
    )


def test_janzing_local_form_with_scrambled_variable_order():
    """An asymmetric family whose variables are stored in a different order than [Y, *parents]."""
    dag = {"Y": ("X", "Z"), "Z": (), "X": ("Z",)}
    kernels = {
        "Z": lambda v, _: 0.3 if v else 0.7,
        "X": lambda v, pa: (0.8 if pa["Z"] else 0.25) if v else (0.2 if pa["Z"] else 0.75),
        "Y": lambda v, pa: [[0.1, 0.7], [0.55, 0.95]][pa["X"]][pa["Z"]] if v else 1 - [[0.1, 0.7], [0.55, 0.95]][pa["X"]][pa["Z"]],
    }
    d = joint_from_kernels(dag, kernels, ["Y", "Z", "X"], dict.fromkeys("XYZ", (0, 1)))
    local = janzing_local_strength(d, dag, ("X", "Y"))
    assert local == pytest.approx(janzing_causal_strength(d, dag, [("X", "Y")]))
    cmi = coinformation(d, [["X"], ["Y"]], ["Z"])
    correction = kullback_leibler_divergence(d, cut_arrows(d, dag, [("X", "Y")]), rvs=["Y"], crvs=["Z"])
    assert local == pytest.approx(cmi + correction)
    assert correction > 1e-3


def test_janzing_two_node_strength_is_mutual_information():
    """Postulate P1: for the DAG X -> Y alone, C_{X->Y} = I(X;Y)."""
    dag = {"X": (), "Y": ("X",)}
    kernels = {"X": lambda v, _: 0.7 if v else 0.3, "Y": lambda v, pa: 0.85 if v == pa["X"] else 0.15}
    d = joint_from_kernels(dag, kernels, ["X", "Y"], dict.fromkeys("XY", (0, 1)))
    assert janzing_causal_strength(d, dag, [("X", "Y")]) == pytest.approx(lagged_mutual_information(d, "X", "Y"))


@pytest.mark.parametrize("coupling", [0.5, 0.8, 0.95])
def test_janzing_theorem_4_decomposition(coupling):
    """C_{X->Y} = I(X;Y|Z) + D[P(Y|Z) || P_S(Y|Z)]; the second term vanishes iff X is independent of Z."""
    dag = {"Z": (), "X": ("Z",), "Y": ("X", "Z")}
    kernels = {
        "Z": lambda v, _: 0.5,
        "X": lambda v, pa: coupling if v == pa["Z"] else 1 - coupling,
        "Y": lambda v, pa: 0.9 if v == (pa["X"] | pa["Z"]) else 0.1,
    }
    d = joint_from_kernels(dag, kernels, ["Z", "X", "Y"], dict.fromkeys("ZXY", (0, 1)))
    strength = janzing_causal_strength(d, dag, [("X", "Y")])
    cmi = coinformation(d, [["X"], ["Y"]], ["Z"])
    correction = kullback_leibler_divergence(d, cut_arrows(d, dag, [("X", "Y")]), rvs=["Y"], crvs=["Z"])
    assert strength == pytest.approx(cmi + correction)
    if coupling == 0.5:
        assert correction == pytest.approx(0, abs=1e-12)
    else:
        assert strength > cmi + 1e-3


def test_cutting_several_arrows_feeds_the_product_of_marginals():
    """Remark 1: cutting both arrows of X -> Y -> Z leaves P(x) P(y) P(z)."""
    dag = {"X": (), "Y": ("X",), "Z": ("Y",)}
    kernels = {
        "X": lambda v, _: 0.6 if v else 0.4,
        "Y": lambda v, pa: 0.9 if v == pa["X"] else 0.1,
        "Z": lambda v, pa: 0.8 if v == pa["Y"] else 0.2,
    }
    d = joint_from_kernels(dag, kernels, ["X", "Y", "Z"], dict.fromkeys("XYZ", (0, 1)))
    cut = cut_arrows(d, dag, [("X", "Y"), ("Y", "Z")])
    assert coinformation(cut, [["X"], ["Y"], ["Z"]]) == pytest.approx(0, abs=1e-12)
    for v in ("X", "Y", "Z"):
        assert cut.marginal([v]).pmf == pytest.approx(d.marginal([v]).pmf)


# --- Closed forms for the three canonical systems -------------------------------


@pytest.mark.parametrize("strength", STRENGTHS)
def test_confounded_lags_closed_form(strength):
    """TE = MI = 1 - h((1 - c^2) / 2) from two noisy copies of Z; no causal influence at all."""
    values = evaluate(systems.confounded_lags(strength))
    assert values["te"] == pytest.approx(1 - h((1 - strength**2) / 2), abs=1e-12)
    assert values["mi"] == pytest.approx(values["te"])
    for name in ("ap", "ap_imposed", "janzing"):
        assert values[name] == pytest.approx(0, abs=1e-12), name


@pytest.mark.parametrize("noise", NOISES)
def test_xor_with_target_past_closed_form(noise):
    """Y_t+1 = X_t XOR Y_t: invisible to MI and naive flow, 1 - h(noise) bits otherwise."""
    values = evaluate(systems.xor_synergy(noise, partner="target_past"))
    for name in ("mi", "ap"):
        assert values[name] == pytest.approx(0, abs=1e-12), name
    for name in ("te", "ap_imposed", "janzing"):
        assert values[name] == pytest.approx(1 - h(noise), abs=1e-12), name


@pytest.mark.parametrize("noise", NOISES)
def test_xor_with_external_partner_closed_form(noise):
    """Y_t+1 = X_t XOR W_t: transfer entropy misses what imposing W_t reveals."""
    values = evaluate(systems.xor_synergy(noise, partner="external"))
    for name in ("mi", "te", "ap"):
        assert values[name] == pytest.approx(0, abs=1e-12), name
    for name in ("ap_imposed", "janzing"):
        assert values[name] == pytest.approx(1 - h(noise), abs=1e-12), name


@pytest.mark.parametrize("persistence", PERSISTENCES)
def test_synchronized_copy_closed_form(persistence):
    """Exact copy: TE = h(persistence) shrinks to 0, while the causal measures stay at 1 bit."""
    values = evaluate(systems.synchronized_copy(persistence))
    assert values["te"] == pytest.approx(h(persistence), abs=1e-12)
    for name in ("mi", "ap", "ap_imposed", "janzing"):
        assert values[name] == pytest.approx(1, abs=1e-12), name


def test_perfect_synchronization_example():
    """Ay & Polani's frozen, synchronized chains (Janzing et al. Sec. 5.2): TE = 0 but C = 1."""
    values = evaluate(systems.synchronized_copy(persistence=1.0))
    assert values["te"] == pytest.approx(0, abs=1e-12)
    assert values["janzing"] == pytest.approx(1)


def test_analysts_dag_inherits_transfer_entropys_false_positive():
    """Drop the hidden driver and every measure reports the spurious TE."""
    system = systems.as_seen_by_analyst(systems.confounded_lags(0.8))
    values = evaluate(system)
    for name, value in values.items():
        assert value == pytest.approx(values["te"]), name
    assert values["te"] > 0.3


def test_ay_polani_rejects_overlapping_roles():
    system = systems.xor_synergy()
    with pytest.raises(ValueError):
        ay_polani_flow(system.dist, system.dag, "X_t", "Y_t+1", imposed=["X_t"])
