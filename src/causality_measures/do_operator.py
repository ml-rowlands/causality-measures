"""Pearl's do-operator on exact, discrete joint distributions.

A causal structure is a DAG written as a mapping ``node -> parents``. Given an
observational ``dit.Distribution`` P that is Markov with respect to that DAG,
an intervention ``do(X = x)`` is computed with the truncated factorization
(Pearl 2009, eq. 3.10):

    P(v | do(X = x)) = prod_{i not in X} P(v_i | pa_i)   if v agrees with x,
                       0                                  otherwise,

where every mechanism P(v_i | pa_i) is read off P itself. The product is
assembled by ``dit.distribution_from_bayesnet``, so the result is an ordinary
``dit.Distribution`` that dit's information measures can consume.

The truncated factorization only identifies P(v | do(x)) when every mechanism
it needs is defined, i.e. when every parent configuration that has positive
probability *after* the intervention also had positive probability *before*
it. When that fails, ``PositivityError`` is raised instead of silently
guessing a mechanism the data never exhibited.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import product

import dit
import networkx as nx

Dag = Mapping[str, Sequence[str]]
"""Causal DAG as ``node -> parents``. Parent order fixes the key order of CPTs."""

CPT = dict[tuple, dict[Hashable, float]]
"""Conditional probability table: parent values (in DAG order) -> {value: prob}."""

Kernel = Callable[[Hashable, Mapping[str, Hashable]], float]
"""Mechanism as a function ``(value, {parent: parent_value}) -> probability``,
the form ``dit.distribution_from_bayesnet`` consumes."""


class PositivityError(ValueError):
    """The requested distribution is not identified from the observational one.

    Raised when a mechanism P(v_i | pa_i) is needed at a parent configuration
    that has zero observational probability, so P says nothing about it.
    """


def validate_dag(dag: Dag, variables: Iterable[str] | None = None) -> None:
    """Check that ``dag`` is acyclic, closed, and (optionally) covers ``variables``."""
    for node, parents in dag.items():
        missing = set(parents) - set(dag)
        if missing:
            raise ValueError(f"parents {sorted(missing)} of {node!r} are not nodes of the DAG")
    if not nx.is_directed_acyclic_graph(_as_digraph(dag)):
        raise ValueError("causal structure contains a directed cycle")
    if variables is not None and set(variables) != set(dag):
        raise ValueError(
            f"DAG nodes {sorted(dag)} do not match distribution variables {sorted(variables)}"
        )


def alphabets(dist: dit.Distribution) -> dict[str, tuple]:
    """Map each named random variable of ``dist`` to its alphabet."""
    return dict(zip(dist.get_rv_names(), dist.alphabet))


def joint_items(dist: dit.Distribution, names: Sequence[str]) -> list[tuple[tuple, float]]:
    """Marginal of ``dist`` over ``names`` as ``(values, probability)`` pairs.

    ``values`` is always a tuple in ``names`` order. dit itself returns a
    marginal in the distribution's own variable order, whatever order is
    requested, and uses bare scalars as outcomes of one-variable marginals.
    """
    if not names:
        return [((), 1.0)]
    marginal = dist.marginal(list(names))
    order = list(marginal.get_rv_names())
    positions = [order.index(name) for name in names]
    items = []
    for outcome, p in zip(marginal.outcomes, marginal.pmf):
        values = _as_tuple(outcome, len(names))
        items.append((tuple(values[i] for i in positions), p))
    return items


def ordered_marginal(dist: dit.Distribution, names: Sequence[str]) -> dit.Distribution:
    """Marginal over ``names`` whose variables appear in exactly that order.

    dit compares distributions positionally (e.g. in relative entropy), so any
    distribution built to be compared with another must fix its order.
    """
    items = joint_items(dist, names)
    return dit.Distribution([v for v, _ in items], [p for _, p in items], rv_names=list(names))


def mechanisms(dist: dit.Distribution, dag: Dag) -> dict[str, CPT]:
    """Read every mechanism P(v_i | pa_i) off the joint distribution.

    Only parent configurations with positive probability appear in the
    returned tables: those are the only ones the joint distribution defines.
    (dit's ``condition_on`` is not used: it lists the conditioning marginal in
    the distribution's variable order but the conditionals in the requested
    order, which misaligns them whenever the two orders differ.)
    """
    validate_dag(dag, dist.get_rv_names())
    cpts: dict[str, CPT] = {}
    for node, parents in dag.items():
        family = joint_items(dist, [*parents, node])
        p_parents: dict[tuple, float] = {}
        for values, p in family:
            p_parents[values[:-1]] = p_parents.get(values[:-1], 0.0) + p
        cpt: CPT = {}
        for values, p in family:
            pa = values[:-1]
            if p_parents[pa] > 0:
                cpt.setdefault(pa, {})[values[-1]] = p / p_parents[pa]
        cpts[node] = cpt
    return cpts


def joint_from_kernels(
    dag: Dag,
    kernels: Mapping[str, Kernel],
    rv_names: Sequence[str],
    alphabet: Mapping[str, Sequence[Hashable]],
) -> dit.Distribution:
    """Build the joint distribution prod_i P(v_i | pa_i) as a ``dit.Distribution``."""
    validate_dag(dag, rv_names)
    graph = _as_digraph(dag)
    for node in rv_names:
        graph.nodes[node]["dist"] = kernels[node]
    sample_space = dit.CartesianProduct([tuple(alphabet[v]) for v in rv_names])
    return dit.distribution_from_bayesnet(graph, nodes=list(rv_names), sample_space=sample_space)


def joint_from_cpts(
    dag: Dag,
    cpts: Mapping[str, CPT],
    rv_names: Sequence[str],
    alphabet: Mapping[str, Sequence[Hashable]],
    fixed: Mapping[str, Kernel] | None = None,
) -> dit.Distribution:
    """Build the joint from tabulated mechanisms, checking identifiability.

    ``fixed`` supplies kernels (e.g. the point masses of an intervention) that
    are used verbatim instead of a table. Table entries missing because their
    parent configuration never occurred are filled with a uniform placeholder;
    if any such configuration turns out to have positive probability in the
    result, the placeholder mattered and ``PositivityError`` is raised.
    """
    fixed = dict(fixed or {})
    kernels = {
        node: fixed.get(node) or _table_kernel(node, dag[node], cpts[node], alphabet[node])
        for node in rv_names
    }
    joint = joint_from_kernels(dag, kernels, rv_names, alphabet)
    for node in rv_names:
        if node not in fixed:
            _check_identified(joint, node, dag[node], cpts[node])
    return joint


def do(
    dist: dit.Distribution,
    dag: Dag,
    intervention: Mapping[str, Hashable],
    *,
    cpts: Mapping[str, CPT] | None = None,
) -> dit.Distribution:
    """Return the post-intervention distribution P(V | do(intervention)).

    Parameters
    ----------
    dist : the observational joint distribution, with named random variables.
    dag : the causal DAG (``node -> parents``) that ``dist`` is Markov to.
    intervention : ``{node: value}`` for every node being set.
    cpts : mechanisms already extracted with ``mechanisms(dist, dag)``; pass
        them to avoid re-extracting when intervening repeatedly.

    Raises
    ------
    PositivityError : if the intervention reaches a parent configuration the
        observational distribution never exhibits.
    """
    alphabet = alphabets(dist)
    for node, value in intervention.items():
        if node not in dag:
            raise ValueError(f"cannot intervene on unknown variable {node!r}")
        if value not in alphabet[node]:
            raise ValueError(f"{value!r} is not in the alphabet {alphabet[node]} of {node!r}")
    if cpts is None:
        cpts = mechanisms(dist, dag)
    mutilated = {node: (() if node in intervention else tuple(parents)) for node, parents in dag.items()}
    point_masses = {node: _point_mass(value) for node, value in intervention.items()}
    return joint_from_cpts(mutilated, cpts, dist.get_rv_names(), alphabet, fixed=point_masses)


def _as_digraph(dag: Dag) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(dag)
    graph.add_edges_from((parent, node) for node, parents in dag.items() for parent in parents)
    return graph


def _point_mass(value: Hashable) -> Kernel:
    return lambda v, _parents: 1.0 if v == value else 0.0


def _table_kernel(node: str, parents: Sequence[str], cpt: CPT, values: Sequence[Hashable]) -> Kernel:
    placeholder = 1.0 / len(values)

    def kernel(value: Hashable, parent_values: Mapping[str, Hashable]) -> float:
        row = cpt.get(tuple(parent_values[p] for p in parents))
        if row is None:
            return placeholder
        return row.get(value, 0.0)

    return kernel


def _check_identified(joint: dit.Distribution, node: str, parents: Sequence[str], cpt: CPT) -> None:
    if not parents:
        return
    for pa, p in joint_items(joint, parents):
        if p > 0 and pa not in cpt:
            config = dict(zip(parents, pa))
            raise PositivityError(
                f"P({node} | {config}) is needed but {config} has zero observational probability"
            )


def _as_tuple(outcome: Hashable, width: int) -> tuple:
    return tuple(outcome) if width > 1 else (outcome,)


def all_configurations(names: Sequence[str], alphabet: Mapping[str, Sequence[Hashable]]):
    """Every joint value assignment of ``names``, as tuples in ``names`` order."""
    return product(*(alphabet[n] for n in names))
