"""Two-source partial information decomposition (Williams & Beer 2010) via ``dit.pid``.

Splits I(target ; source, partner) into four non-overlapping atoms:

    redundancy         information both sources carry
    unique_source      information only the source carries
    unique_partner     information only the partner carries
    synergy            information only the pair carries jointly

For source = X_t, partner = Y_t and target = Y_t+1, the consistency equations
of any PID give TE(X -> Y) = I(Y_t+1 ; X_t | Y_t) = unique_source + synergy,
so the decomposition shows *which kind* of information transfer entropy is
counting.
"""

import dit
from dit.pid import PID_BROJA, PID_CCS, PID_MMI, PID_WB

PID_MEASURES = {
    "WB": PID_WB,  # Williams & Beer's I_min
    "BROJA": PID_BROJA,  # Bertschinger et al. (optimization-based)
    "CCS": PID_CCS,  # Ince's common change in surprisal
    "MMI": PID_MMI,  # minimum mutual information
}

ATOMS = ("redundancy", "unique_source", "unique_partner", "synergy")


def pid_atoms(
    dist: dit.Distribution, source: str, partner: str, target: str, measure: str = "WB"
) -> dict[str, float]:
    """The four PID atoms of I(target ; source, partner), in bits."""
    pid = PID_MEASURES[measure](dist.marginal([source, partner, target]), [[source], [partner]], [target])
    atoms = {}
    for node in pid._lattice:
        if len(node) == 2:
            atoms["redundancy"] = pid.get_pi(node)
        elif len(node[0]) == 2:
            atoms["synergy"] = pid.get_pi(node)
        elif node[0] == (source,):
            atoms["unique_source"] = pid.get_pi(node)
        else:
            atoms["unique_partner"] = pid.get_pi(node)
    return {atom: round(float(atoms[atom]), 12) + 0.0 for atom in ATOMS}
