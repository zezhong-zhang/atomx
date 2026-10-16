"""Complete finite magnetic expansion for multiple closed-core radial shells."""

import itertools

import numpy as np

from atomx.angular.sd_rho import rho_coefficient


def expand_valence(problem, plan, rows, values, column):
    labels = problem.integrals.labels
    nc = problem.integrals.noccupied
    positions = {
        key: (i, g.index(key)) for i, g in enumerate(plan["groups"]) for key in g
    }
    core_positions = {key: i for i, key in enumerate(plan["occupied_keys"])}
    channels = {tuple(row): i for i, row in enumerate(rows)}
    result = np.zeros((len(labels), len(labels), nc))
    for m, n, b in itertools.product(
        range(nc, len(labels)), range(nc, len(labels)), range(nc)
    ):
        im, sm = positions[labels[m][:2]]
        jn, sn = positions[labels[n][:2]]
        i = core_positions[labels[b][:2]]
        for rank in range(plan["dimensions"]["KK"] + 1):
            channel = channels.get((i, im, jn, rank))
            if channel is not None:
                result[m, n, b] += (
                    rho_coefficient(
                        rank, labels[m], labels[n], labels[column], labels[b]
                    )
                    * values[sm, sn, channel]
                )
    return result


def core_numerators(problem, plan, core_physical):
    labels = problem.integrals.labels
    nc = problem.integrals.noccupied
    numerator = np.asarray(core_physical) * problem.denominators
    if numerator.shape != problem.source.shape or not np.all(np.isfinite(numerator)):
        raise ValueError("complete finite core numerator required")
    positions = {key: i for i, key in enumerate(plan["occupied_keys"])}
    radial = {key: i for i, group in enumerate(plan["groups"]) for key in group}
    singles = {}
    cursor = 0
    for a in plan["occupied_keys"]:
        for m in plan["groups"][radial[a]]:
            if m in plan["virtual_keys"]:
                singles[m, a] = numerator[cursor]
                cursor += 1
    tables = {}
    for a, b, im, jn, rank in plan["channels"]:
        mkeys = [key for key in plan["groups"][im] if key in plan["virtual_keys"]]
        nkeys = [key for key in plan["groups"][jn] if key in plan["virtual_keys"]]
        size = len(mkeys) * len(nkeys)
        block = numerator[cursor : cursor + size].reshape(
            (len(mkeys), len(nkeys)), order="F"
        )
        tables[a, b, im, jn, rank] = {
            (m, n): block[i, j]
            for i, m in enumerate(mkeys)
            for j, n in enumerate(nkeys)
        }
        cursor += size
    if cursor != len(numerator):
        raise AssertionError("complete ordered core coordinate coverage failed")
    single = np.zeros((len(labels), nc))
    double = np.zeros((len(labels), len(labels), nc, nc))
    for m, a in itertools.product(range(nc, len(labels)), range(nc)):
        if labels[m][1:] == labels[a][1:]:
            single[m, a] = singles[labels[m][:2], labels[a][:2]]
    for m, n, a, b in itertools.product(
        range(nc, len(labels)), range(nc, len(labels)), range(nc), range(nc)
    ):
        ai, bi = positions[labels[a][:2]], positions[labels[b][:2]]
        im, jn = radial[labels[m][:2]], radial[labels[n][:2]]
        for rank in range(plan["dimensions"]["KK"] + 1):
            target = (ai, bi, im, jn, rank) if ai >= bi else (bi, ai, jn, im, rank)
            table = tables.get(target)
            if table is not None:
                pair = (
                    (labels[m][:2], labels[n][:2])
                    if ai >= bi
                    else (labels[n][:2], labels[m][:2])
                )
                double[m, n, a, b] += (
                    rho_coefficient(rank, labels[m], labels[n], labels[a], labels[b])
                    * table[pair]
                )
    return single, double
