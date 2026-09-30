"""Three information-theoretic measures of causal influence, computed exactly.

All three take a ``dit.Distribution`` with named random variables. The
causal ones also take the DAG that generated it. The causal-specific
constructions (interventions, arrow cutting) build new ``dit.Distribution``
objects; every entropy, mutual information and relative entropy is then
computed by dit itself.

* ``transfer_entropy`` (Schreiber 2000): purely observational,
  TE(X -> Y) = I(Y_{t+1} ; X_t | Y_t).
* ``ay_polani_flow`` (Ay & Polani 2008): mutual information of an
  interventional distribution, optionally imposing a set S by intervention.
* ``janzing_causal_strength`` (Janzing et al. 2013): relative entropy
  between the joint and the joint after cutting a set of arrows.
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from math import prod

import dit
from dit.divergences import kullback_leibler_divergence
from dit.multivariate import coinformation
from dit.shannon import mutual_information

from causality_measures.do_operator import (
    CPT,
    Dag,
    PositivityError,
    all_configurations,
    alphabets,
    do,
    joint_from_cpts,
    joint_items,
    mechanisms,
    ordered_marginal,
    validate_dag,
)

Arrow = tuple[str, str]


def transfer_entropy(dist: dit.Distribution, source: str, target: str, target_past: str) -> float:
    """Transfer entropy with history length 1: I(target ; source | target_past).

    ``target`` is Y_{t+1}, ``source`` is X_t and ``target_past`` is Y_t.
    Only their joint marginal is used; the causal structure is ignored.
    """
    triple = dist.marginal([target, source, target_past])
    return coinformation(triple, [[target], [source]], [target_past])


def lagged_mutual_information(dist: dit.Distribution, source: str, target: str) -> float:
    """Plain observational dependence I(source ; target), for reference."""
    return mutual_information(dist, [source], [target])


def interventional_joint(
    dist: dit.Distribution,
    dag: Dag,
    source: str,
    target: str,
    imposed: Sequence[str] = (),
) -> dit.Distribution:
    """Ay & Polani's modified distribution (their eq. 9) over (S, source, target).

        p_hat(s, x, y) = p(s) * p(x | do(s)) * p(y | do(s), do(x))

    ``p(s)`` is the observational marginal of the imposed set S. Every factor
    comes from an exact intervention on ``dist``.
    """
    imposed = tuple(imposed)
    if source == target or {source, target} & set(imposed):
        raise ValueError("source, target and the imposed set must be disjoint")
    cpts = mechanisms(dist, dag)
    outcomes, pmf = [], []
    for s, p_s_val in joint_items(dist, imposed):
        if p_s_val <= 0:
            continue
        do_s = dict(zip(imposed, s))
        after_s = do(dist, dag, do_s, cpts=cpts) if do_s else dist
        for (x,), p_x_val in joint_items(after_s, [source]):
            if p_x_val <= 0:
                continue
            after_sx = do(dist, dag, {**do_s, source: x}, cpts=cpts)
            for (y,), p_y_val in joint_items(after_sx, [target]):
                outcomes.append((*s, x, y))
                pmf.append(p_s_val * p_x_val * p_y_val)
    return dit.Distribution(outcomes, pmf, rv_names=[*imposed, source, target])


def ay_polani_flow(
    dist: dit.Distribution,
    dag: Dag,
    source: str,
    target: str,
    imposed: Sequence[str] = (),
) -> float:
    """Information flow I(source -> target | do(S)) of Ay & Polani (2008).

    Equals the conditional mutual information I(target ; source | S) under
    ``interventional_joint`` (their eq. 9). With S empty this is
    sum_x p(x) D(p(Y | do(x)) || sum_x' p(x') p(Y | do(x'))).
    """
    p_hat = interventional_joint(dist, dag, source, target, imposed)
    return coinformation(p_hat, [[target], [source]], list(imposed) or None)


def cut_arrows(dist: dit.Distribution, dag: Dag, arrows: Iterable[Arrow]) -> dit.Distribution:
    """Janzing et al.'s post-cutting distribution P_S (their Definition 1).

    Each cut arrow X_i -> X_j is removed and the open end of X_j's mechanism
    is fed an independent draw from the marginal P(X_i); several cut arrows
    into the same node are fed the *product* of their marginals:

        P_S(x_j | pa_j^keep) = sum_{pa_j^cut} P(x_j | pa_j^keep, pa_j^cut) prod_i P(x_i)

    Arrows that are not in the DAG leave P unchanged: cutting a wire that is
    not there does nothing.
    """
    cut_by_target = _group_arrows(dag, arrows)
    alphabet = alphabets(dist)
    cpts = mechanisms(dist, dag)
    reduced_dag = {n: tuple(p for p in ps if p not in cut_by_target.get(n, ())) for n, ps in dag.items()}
    reduced_cpts = dict(cpts)
    for node, cut in cut_by_target.items():
        cut_marginals = {c: _marginal_pmf(dist, c) for c in cut}
        reduced_cpts[node] = _cut_cpt(dag[node], cpts[node], cut_marginals, reduced_dag[node], alphabet)
    return joint_from_cpts(reduced_dag, reduced_cpts, dist.get_rv_names(), alphabet)


def janzing_causal_strength(dist: dit.Distribution, dag: Dag, arrows: Iterable[Arrow]) -> float:
    """Causal strength C_S = D(P || P_S) of a set of arrows (Janzing et al., Definition 2)."""
    return kullback_leibler_divergence(dist, cut_arrows(dist, dag, arrows))


def janzing_local_strength(dist: dit.Distribution, dag: Dag, arrow: Arrow) -> float:
    """Causal strength of one arrow from its target's family alone (Janzing et al., Lemma 3).

        C_{k->l} = D[ P(X_l, PA_l) || P_S(X_l | PA_l) P(PA_l) ]

    Only the marginal over X_l and its parents is touched, so this is an
    independent route to the same number as ``janzing_causal_strength``.
    """
    tail, node = arrow
    if tail not in _group_arrows(dag, [arrow]).get(node, ()):
        raise ValueError(f"{tail!r} -> {node!r} is not an arrow of the DAG")
    parents = tuple(dag[node])
    kept = tuple(p for p in parents if p != tail)
    family = ordered_marginal(dist, [node, *parents])
    cpt = mechanisms(family, {node: parents, **{p: () for p in parents}})[node]
    reduced = _cut_cpt(parents, cpt, {tail: _marginal_pmf(family, tail)}, kept, alphabets(dist))

    outcomes, pmf = [], []
    for pa, p_pa in joint_items(family, parents):
        kept_values = tuple(v for p, v in zip(parents, pa) if p in kept)
        if kept_values not in reduced:
            raise PositivityError(f"P_S({node} | {dict(zip(kept, kept_values))}) is not identified")
        for value, p_value in reduced[kept_values].items():
            outcomes.append((value, *pa))
            pmf.append(p_pa * p_value)
    cut_family = dit.Distribution(outcomes, pmf, rv_names=[node, *parents])
    return kullback_leibler_divergence(family, cut_family)


def _marginal_pmf(dist: dit.Distribution, var: str) -> dict[Hashable, float]:
    return {v: p for (v,), p in joint_items(dist, [var])}


def _group_arrows(dag: Dag, arrows: Iterable[Arrow]) -> dict[str, tuple[str, ...]]:
    """Group the arrows present in ``dag`` by their head node."""
    validate_dag(dag)
    grouped: dict[str, list[str]] = {}
    for tail, head in arrows:
        if tail not in dag or head not in dag:
            raise ValueError(f"arrow {tail!r} -> {head!r} refers to a variable not in the DAG")
        if tail in dag[head]:
            grouped.setdefault(head, []).append(tail)
    return {head: tuple(tails) for head, tails in grouped.items()}


def _cut_cpt(
    parents: Sequence[str],
    cpt: CPT,
    cut_marginals: Mapping[str, Mapping[Hashable, float]],
    kept: Sequence[str],
    alphabet: Mapping[str, Sequence[Hashable]],
) -> CPT:
    """Janzing's eq. (7): average the mechanism over independent marginals of the cut parents.

    Rows whose average would need an undefined mechanism entry are left out,
    so ``joint_from_cpts`` raises ``PositivityError`` only if they matter.
    """
    cut = tuple(cut_marginals)
    reduced: CPT = {}
    for kept_values in all_configurations(kept, alphabet):
        assignment = dict(zip(kept, kept_values))
        row: dict[Hashable, float] = {}
        for cut_values in all_configurations(cut, alphabet):
            weight = prod(cut_marginals[c].get(v, 0.0) for c, v in zip(cut, cut_values))
            if weight == 0:
                continue
            assignment.update(zip(cut, cut_values))
            full_row = cpt.get(tuple(assignment[p] for p in parents))
            if full_row is None:
                break
            for value, p in full_row.items():
                row[value] = row.get(value, 0.0) + weight * p
        else:
            reduced[tuple(kept_values)] = row
    return reduced
