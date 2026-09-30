"""Canonical binary systems on which the causality measures disagree.

Every system is one time window of a stationary binary process, unrolled into
a small causal DAG over time-indexed variables such as ``"X_t"`` or
``"Y_t+1"``. The question is always "how much does X_t influence Y_t+1?", and
``Y_t`` is the target's own past (what transfer entropy conditions on).

Each window is chosen so that it is causally closed: any variable whose true
causes lie outside the window is a root that is independent of the other
roots, and it is given its exact stationary marginal. The joint distribution
over the window is therefore the exact stationary one, and interventions
inside the window are exact too.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from functools import reduce
from operator import xor

import dit

from causality_measures.do_operator import Dag, Kernel, joint_from_kernels, ordered_marginal

BINARY = (0, 1)


@dataclass(frozen=True)
class CausalSystem:
    """An exact joint distribution together with the causal DAG behind it."""

    name: str
    dist: dit.Distribution
    dag: Dag
    imposed: tuple[str, ...]
    """The set S imposed by intervention in the conditional Ay–Polani flow."""
    params: Mapping[str, float] = field(default_factory=dict)
    source: str = "X_t"
    target: str = "Y_t+1"
    target_past: str = "Y_t"
    hidden: tuple[str, ...] = ()

    @property
    def has_arrow(self) -> bool:
        """Whether the DAG contains a direct arrow source -> target."""
        return self.source in self.dag[self.target]


def direct_chain(noise: float = 0.1) -> CausalSystem:
    """Control case: Y_t+1 is a noisy copy of X_t, and nothing else is going on.

    X is i.i.d. uniform and Y has no memory, so Y_t is independent of both X_t
    and Y_t+1. There is no confounding and no synergy: every measure should
    report 1 - h(noise) bits.
    """
    dag = {"X_t": (), "Y_t": (), "Y_t+1": ("X_t",)}
    kernels = {"X_t": _uniform, "Y_t": _uniform, "Y_t+1": _noisy_xor(["X_t"], noise)}
    return _build("direct chain", dag, kernels, imposed=("Y_t",), params={"noise": noise})


def confounded_lags(strength: float = 0.9) -> CausalSystem:
    """A hidden driver Z reaches X after one step and Y after two, with no X -> Y arrow.

    Z is i.i.d. uniform; X_t and Y_t+1 are both noisy copies of Z_t-1, and
    Y_t is a noisy copy of Z_t-2. Every copy goes through a binary symmetric
    channel whose input/output correlation is ``strength`` (0: no coupling,
    1: exact copy). X therefore "predicts" Y one step ahead without causing it.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie in [0, 1]")
    noise = (1 - strength) / 2
    dag = {
        "Z_t-2": (),
        "Z_t-1": (),
        "X_t": ("Z_t-1",),
        "Y_t": ("Z_t-2",),
        "Y_t+1": ("Z_t-1",),
    }
    kernels = {
        "Z_t-2": _uniform,
        "Z_t-1": _uniform,
        "X_t": _noisy_xor(["Z_t-1"], noise),
        "Y_t": _noisy_xor(["Z_t-2"], noise),
        "Y_t+1": _noisy_xor(["Z_t-1"], noise),
    }
    return _build(
        "confounded lags",
        dag,
        kernels,
        imposed=("Y_t",),
        params={"strength": strength},
        hidden=("Z_t-2", "Z_t-1"),
    )


def xor_synergy(noise: float = 0.0, partner: str = "target_past") -> CausalSystem:
    """Y_t+1 = X_t XOR W XOR noise, with X_t and W independent uniform bits.

    ``partner`` selects what W is:

    * ``"target_past"``: W = Y_t, i.e. Y_t+1 = X_t XOR Y_t. This is the
      example of James, Barnett & Crutchfield (2016), where transfer entropy
      reports the purely synergistic bit as if it flowed from X alone.
    * ``"external"``: W = W_t is a separate i.i.d. driver, and Y_t is an
      irrelevant (independent, uniform) past. Transfer entropy, which only
      conditions on Y_t, then misses the influence entirely.
    """
    if partner == "target_past":
        dag = {"X_t": (), "Y_t": (), "Y_t+1": ("X_t", "Y_t")}
        kernels = {"X_t": _uniform, "Y_t": _uniform, "Y_t+1": _noisy_xor(["X_t", "Y_t"], noise)}
        return _build("XOR, partner = Y_t", dag, kernels, imposed=("Y_t",), params={"noise": noise})
    if partner == "external":
        dag = {"X_t": (), "W_t": (), "Y_t": (), "Y_t+1": ("X_t", "W_t")}
        kernels = {
            "X_t": _uniform,
            "W_t": _uniform,
            "Y_t": _uniform,
            "Y_t+1": _noisy_xor(["X_t", "W_t"], noise),
        }
        return _build("XOR, partner = W_t", dag, kernels, imposed=("Y_t", "W_t"), params={"noise": noise})
    raise ValueError(f"partner must be 'target_past' or 'external', not {partner!r}")


def synchronized_copy(persistence: float = 0.95, copy_noise: float = 0.0) -> CausalSystem:
    """Y copies X with a one-step lag while X is a slowly switching Markov chain.

    X_t = X_t-1 with probability ``persistence`` (symmetric chain, uniform
    stationary marginal); Y_t and Y_t+1 are copies of X_t-1 and X_t through a
    binary symmetric channel with flip probability ``copy_noise``. As
    persistence -> 1, Y's own past already predicts Y_t+1, so transfer entropy
    vanishes although Y_t+1 is entirely driven by X_t.
    """
    dag = {"X_t-1": (), "X_t": ("X_t-1",), "Y_t": ("X_t-1",), "Y_t+1": ("X_t",)}
    kernels = {
        "X_t-1": _uniform,
        "X_t": _noisy_xor(["X_t-1"], 1 - persistence),
        "Y_t": _noisy_xor(["X_t-1"], copy_noise),
        "Y_t+1": _noisy_xor(["X_t"], copy_noise),
    }
    return _build(
        "synchronized copy",
        dag,
        kernels,
        imposed=("Y_t",),
        params={"persistence": persistence, "copy_noise": copy_noise},
    )


def as_seen_by_analyst(system: CausalSystem) -> CausalSystem:
    """The same observations, analysed with the DAG an X/Y-only analyst would draw.

    Hidden variables are marginalized out and replaced by the complete DAG
    Y_t -> X_t -> Y_t+1 <- Y_t: "X_t and Y_t both feed Y_t+1". This is the
    causal picture transfer entropy implicitly assumes, and any interventional
    measure computed on it inherits transfer entropy's blind spots.
    """
    s, x, y = system.target_past, system.source, system.target
    dag = {s: (), x: (s,), y: (x, s)}
    return CausalSystem(
        name=f"{system.name} (analyst's DAG, {', '.join(system.hidden) or 'nothing'} unobserved)",
        dist=ordered_marginal(system.dist, [s, x, y]),
        dag=dag,
        imposed=(s,),
        params=system.params,
        source=x,
        target=y,
        target_past=s,
    )


def _build(
    name: str,
    dag: Dag,
    kernels: Mapping[str, Kernel],
    *,
    imposed: tuple[str, ...],
    params: Mapping[str, float],
    hidden: tuple[str, ...] = (),
) -> CausalSystem:
    rv_names = list(dag)
    dist = joint_from_kernels(dag, kernels, rv_names, {v: BINARY for v in rv_names})
    return CausalSystem(name=name, dist=dist, dag=dag, imposed=imposed, params=dict(params), hidden=hidden)


def _uniform(_value: Hashable, _parents: Mapping[str, Hashable]) -> float:
    return 0.5


def _noisy_xor(inputs: list[str], noise: float) -> Kernel:
    """Binary kernel: the XOR of ``inputs``, flipped with probability ``noise``.

    With a single input this is a binary symmetric channel (a noisy copy).
    """
    if not 0 <= noise <= 1:
        raise ValueError("noise must lie in [0, 1]")

    def kernel(value: Hashable, parents: Mapping[str, Hashable]) -> float:
        clean = reduce(xor, (parents[i] for i in inputs))
        return 1 - noise if value == clean else noise

    return kernel
