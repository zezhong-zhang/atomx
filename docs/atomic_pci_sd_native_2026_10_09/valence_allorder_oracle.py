"""Independent finite magnetic CI+all-order equations with converged core.

All magnetic coordinates are retained. The 4-index tensor is an explicitly
bounded diagnostic oracle, never a proposed production large-space engine.
"""

import itertools

import numpy as np

from atomx.angular.sd_rho import rho_coefficient


def expand_valence(problem, plan, rows, values, column):
    labels = problem.integrals.labels
    nc = problem.integrals.noccupied
    positions = {
        key: (ip, group.index(key))
        for ip, group in enumerate(plan["groups"])
        for key in group
    }
    channels = {tuple(row): i for i, row in enumerate(rows)}
    result = np.zeros((len(labels), len(labels), nc))
    for m, n, b in itertools.product(
        range(nc, len(labels)), range(nc, len(labels)), range(nc)
    ):
        im, sm = positions[labels[m][:2]]
        jn, sn = positions[labels[n][:2]]
        for rank in range(plan["dimensions"]["KK"] + 1):
            channel = channels.get((0, im, jn, rank))
            if channel is not None:
                result[m, n, b] += (
                    rho_coefficient(
                        rank, labels[m], labels[n], labels[column], labels[b]
                    )
                    * values[sm, sn, channel]
                )
    return result


def core_numerators(problem, plan, core_physical):
    """Expand every native direct core numerator into magnetic coordinates."""
    labels = problem.integrals.labels
    nc = problem.integrals.noccupied
    if nc != 2 or len(plan["occupied_keys"]) != 1:
        raise ValueError("oracle currently requires a complete two-electron core")
    numerator = np.asarray(core_physical) * problem.denominators
    if numerator.shape != problem.source.shape or not np.all(np.isfinite(numerator)):
        raise ValueError("complete finite core vector required")
    positions = {
        key: (ip, group.index(key))
        for ip, group in enumerate(plan["groups"])
        for key in group
    }
    core_key = plan["occupied_keys"][0]
    virtual_s = [key for key in plan["virtual_keys"] if key[1] == core_key[1]]
    singles_by_key = dict(zip(virtual_s, numerator[: len(virtual_s)], strict=True))
    cursor = len(virtual_s)
    tables = {}
    for _, _, im, jn, rank in plan["channels"]:
        mkeys = [key for key in plan["groups"][im] if key in plan["virtual_keys"]]
        nkeys = [key for key in plan["groups"][jn] if key in plan["virtual_keys"]]
        size = len(mkeys) * len(nkeys)
        values = numerator[cursor : cursor + size].reshape(
            (len(mkeys), len(nkeys)), order="F"
        )
        tables[im, jn, rank] = {
            (m, n): values[i, j]
            for i, m in enumerate(mkeys)
            for j, n in enumerate(nkeys)
        }
        cursor += size
    if cursor != len(numerator):
        raise AssertionError("core coordinate coverage is incomplete")
    single = np.zeros((len(labels), nc))
    double = np.zeros((len(labels), len(labels), nc, nc))
    for m, a in itertools.product(range(nc, len(labels)), range(nc)):
        if labels[m][1:] == labels[a][1:]:
            single[m, a] = singles_by_key[labels[m][:2]]
    for m, n, a, b in itertools.product(
        range(nc, len(labels)), range(nc, len(labels)), range(nc), range(nc)
    ):
        im, _ = positions[labels[m][:2]]
        jn, _ = positions[labels[n][:2]]
        for rank in range(plan["dimensions"]["KK"] + 1):
            table = tables.get((im, jn, rank))
            if table is not None:
                double[m, n, a, b] += (
                    rho_coefficient(rank, labels[m], labels[n], labels[a], labels[b])
                    * table[labels[m][:2], labels[n][:2]]
                )
    return single, double


def action(direct, eps, core_s, core_d, column, energy, sigma, *, include_source=True):
    """Original SD gauge, fixed-energy numerator map; no valence singles term.

    This is a magnetic-state contraction of Eqs. 9/36 of Safronova et al.
    with the core/valence denominators inserted explicitly, as in valsd-ci.
    It never calls pCI diagrams or its angular recoupling services.
    """
    nc = core_s.shape[1]
    n = len(eps)
    if sigma.shape != (n, n, nc):
        raise ValueError("complete magnetic valence tensor required")
    g = direct
    gt = direct - direct.swapaxes(2, 3)
    occupied, virtual = slice(0, nc), slice(nc, n)
    es, ev = eps[occupied], eps[virtual]
    shift = energy - eps[column]
    den_s = es[None, :] - ev[:, None] + shift
    den_c = (
        es[None, None, :, None]
        + es[None, None, None, :]
        - ev[:, None, None, None]
        - ev[None, :, None, None]
        + shift
    )
    den_v = energy + es[None, None, :] - ev[:, None, None] - ev[None, :, None]
    if any(np.any(d == 0) or not np.all(np.isfinite(d)) for d in (den_s, den_c, den_v)):
        raise ValueError("singular/nonfinite fixed-energy denominator")
    cs = core_s[virtual] / den_s
    cd = core_d[virtual, virtual] / den_c
    vd = sigma[virtual, virtual] / den_v
    va = vd - vd.swapaxes(0, 1)
    ca = cd - cd.swapaxes(0, 1)
    rs = np.zeros(n, dtype=np.result_type(g, eps, core_s, core_d, sigma))
    rd = np.zeros_like(sigma)
    rs[virtual] = (
        np.einsum("mbn,nb->m", gt[virtual, occupied, column, virtual], cs)
        - np.einsum("bcn,mnbc->m", gt[occupied, occupied, column, virtual], cd)
        + np.einsum("mbnr,nrb->m", gt[virtual, occupied, virtual, virtual], vd)
    )
    rd[virtual, virtual] = (
        np.einsum("cdb,mncd->mnb", g[occupied, occupied, column, occupied], cd)
        + np.einsum("mnrs,rsb->mnb", g[virtual, virtual, virtual, virtual], vd)
        - np.einsum("cnb,mc->mnb", g[occupied, virtual, column, occupied], cs)
        + np.einsum("mnr,rb->mnb", g[virtual, virtual, column, virtual], cs)
        - np.einsum("mcb,nc->mnb", g[virtual, occupied, column, occupied], cs)
        + np.einsum("cnrb,mrc->mnb", gt[occupied, virtual, virtual, occupied], va)
        + np.einsum("mcr,rncb->mnb", gt[virtual, occupied, column, virtual], ca)
    )
    if include_source:
        rd[virtual, virtual] += g[virtual, virtual, column, occupied]
    if not all(np.all(np.isfinite(a)) for a in (rs, rd)):
        raise ValueError("nonfinite magnetic valence action")
    return rs, rd
