"""Checks of the do-operator against hand-derived interventional distributions."""

import dit
import pytest

from causality_measures.do_operator import (
    PositivityError,
    do,
    joint_from_kernels,
    joint_items,
    mechanisms,
    ordered_marginal,
    validate_dag,
)

BINARY = (0, 1)


def bern(p1):
    """Root kernel with P(value = 1) = p1."""
    return lambda v, _pa: p1 if v == 1 else 1 - p1


def table(parents, p1_of):
    """Kernel with P(value = 1 | parents) = p1_of[parent values]."""
    return lambda v, pa: p1_of[tuple(pa[p] for p in parents)] if v == 1 else 1 - p1_of[tuple(pa[p] for p in parents)]


# Z -> X -> Y and Z -> Y: the textbook back-door graph.
BACKDOOR_DAG = {"Z": (), "X": ("Z",), "Y": ("X", "Z")}
P_Z1 = 0.3
P_X1_GIVEN_Z = {(0,): 0.2, (1,): 0.9}
P_Y1_GIVEN_XZ = {(0, 0): 0.1, (0, 1): 0.5, (1, 0): 0.6, (1, 1): 0.95}
BACKDOOR_KERNELS = {
    "Z": bern(P_Z1),
    "X": table(["Z"], P_X1_GIVEN_Z),
    "Y": table(["X", "Z"], P_Y1_GIVEN_XZ),
}


@pytest.fixture
def backdoor():
    return joint_from_kernels(BACKDOOR_DAG, BACKDOOR_KERNELS, ["Z", "X", "Y"], dict.fromkeys("ZXY", BINARY))


def prob(dist, names, values):
    return dict(joint_items(dist, names)).get(tuple(values), 0.0)


def test_mechanisms_recover_the_generating_kernels(backdoor):
    cpts = mechanisms(backdoor, BACKDOOR_DAG)
    assert cpts["Z"][()][1] == pytest.approx(P_Z1)
    for z, p in P_X1_GIVEN_Z.items():
        assert cpts["X"][z][1] == pytest.approx(p)
    for xz, p in P_Y1_GIVEN_XZ.items():
        assert cpts["Y"][xz][1] == pytest.approx(p)


def test_joint_items_follow_the_requested_order(backdoor):
    """dit returns marginals in the distribution's own order; joint_items must not."""
    forward = dict(joint_items(backdoor, ["Z", "X"]))
    backward = dict(joint_items(backdoor, ["X", "Z"]))
    for (z, x), p in forward.items():
        assert backward[(x, z)] == pytest.approx(p)
    assert forward[(0, 1)] == pytest.approx((1 - P_Z1) * P_X1_GIVEN_Z[(0,)])
    assert ordered_marginal(backdoor, ["Y", "Z"]).get_rv_names() == ("Y", "Z")


def test_mechanisms_align_parents_listed_against_distribution_order():
    """Parents (X, Z) are listed opposite to the distribution's (Z, X), and one configuration is impossible."""
    kernels = {**BACKDOOR_KERNELS, "X": table(["Z"], {(0,): 0.0, (1,): 0.5})}
    dist = joint_from_kernels(BACKDOOR_DAG, kernels, ["Z", "X", "Y"], dict.fromkeys("ZXY", BINARY))
    cpt = mechanisms(dist, BACKDOOR_DAG)["Y"]
    assert set(cpt) == {(0, 0), (0, 1), (1, 1)}  # (X=1, Z=0) never happens
    for xz in cpt:
        assert cpt[xz][1] == pytest.approx(P_Y1_GIVEN_XZ[xz])


@pytest.mark.parametrize("x", BINARY)
def test_do_matches_backdoor_adjustment(backdoor, x):
    """P(y | do(x)) = sum_z P(y | x, z) P(z), which differs from P(y | x)."""
    expected = sum(
        P_Y1_GIVEN_XZ[(x, z)] * (P_Z1 if z else 1 - P_Z1) for z in BINARY
    )
    after = do(backdoor, BACKDOOR_DAG, {"X": x})
    assert prob(after, ["Y"], [1]) == pytest.approx(expected)
    assert prob(after, ["X"], [x]) == pytest.approx(1.0)
    observed = prob(backdoor, ["X", "Y"], [x, 1]) / prob(backdoor, ["X"], [x])
    assert observed != pytest.approx(expected)


def test_do_leaves_non_descendants_alone(backdoor):
    after = do(backdoor, BACKDOOR_DAG, {"X": 1})
    assert prob(after, ["Z"], [1]) == pytest.approx(P_Z1)


def test_do_on_a_root_equals_conditioning(backdoor):
    """With no back-door paths into Z, intervening and conditioning coincide."""
    after = do(backdoor, BACKDOOR_DAG, {"Z": 1})
    for x, y in [(0, 0), (1, 1), (1, 0)]:
        conditional = prob(backdoor, ["Z", "X", "Y"], [1, x, y]) / P_Z1
        assert prob(after, ["X", "Y"], [x, y]) == pytest.approx(conditional)


def test_do_agrees_with_mutilated_generating_model(backdoor):
    """Round trip: mechanisms read off the joint reproduce the true intervention."""
    after = do(backdoor, BACKDOOR_DAG, {"X": 0})
    mutilated = {"Z": (), "X": (), "Y": ("X", "Z")}
    kernels = {**BACKDOOR_KERNELS, "X": bern(0.0)}
    truth = joint_from_kernels(mutilated, kernels, ["Z", "X", "Y"], dict.fromkeys("ZXY", BINARY))
    assert dict(joint_items(after, ["Z", "X", "Y"])) == pytest.approx(dict(joint_items(truth, ["Z", "X", "Y"])))


def test_unidentified_intervention_raises():
    """If X always equals Z, P(Y | X=1, Z=0) is never observed, so do(X=1) is not identified."""
    kernels = {**BACKDOOR_KERNELS, "X": table(["Z"], {(0,): 0.0, (1,): 1.0})}
    dist = joint_from_kernels(BACKDOOR_DAG, kernels, ["Z", "X", "Y"], dict.fromkeys("ZXY", BINARY))
    with pytest.raises(PositivityError):
        do(dist, BACKDOOR_DAG, {"X": 1})


def test_positivity_is_only_required_where_it_matters():
    """A zero-probability parent configuration is fine if the intervention never reaches it."""
    dag = {"X": (), "Y": ("X",)}
    dist = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.5, 0.5, 0, 0], rv_names=["X", "Y"], trim=False)
    after = do(dist, dag, {"X": 0})
    assert prob(after, ["Y"], [1]) == pytest.approx(0.5)
    with pytest.raises(PositivityError):
        do(dist, dag, {"X": 1})


def test_rejects_cycles_and_mismatched_variables():
    with pytest.raises(ValueError, match="cycle"):
        validate_dag({"A": ("B",), "B": ("A",)})
    with pytest.raises(ValueError, match="do not match"):
        validate_dag({"A": ()}, ["A", "B"])
