"""Finite magnetic-state CI+all-order map, independent of pCI reduced diagrams.

Implements fixed-energy, unantisymmetrized numerator contractions for a
two-electron canonical core at its first-order numerator. This diagnostic
keeps all magnetic coordinates and uses the public native Coulomb integral.
It is not a large-space engine or an experimental accuracy model.
"""

import itertools
from functools import lru_cache

import numpy as np

from atomx.angular.csf_hamiltonian import two_body_angular
from atomx.angular.sd_rho import rho_coefficient
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell


def compare_map(
    problem,
    orbitals,
    workspace,
    plan,
    rows,
    doubles,
    energy,
    column,
    response_s,
    response_d,
):
    labels = problem.integrals.labels
    nc = problem.integrals.noccupied
    if nc != 2:
        raise ValueError("finite first-order-core oracle requires two core electrons")
    eps = np.diag(problem.integrals.fock_Ha).real
    nlabels = len(labels)
    if 8 * nlabels**4 + 8 * nlabels**3 > 64 * 1024**2:
        raise MemoryError(
            "complete magnetic control tensors exceed explicit 64 MiB bound"
        )
    cache = ConvolCache(workspace)

    @lru_cache(None)
    def radial(a, b, c, d, rank):
        return compute_slater_rk_cell(
            *orbitals[a],
            *orbitals[b],
            *orbitals[c],
            *orbitals[d],
            workspace,
            rank,
            cache,
        )

    direct = np.zeros((nlabels,) * 4)
    for p, q, r, s in itertools.product(range(nlabels), repeat=4):
        a, b, c, d = (labels[i] for i in (p, q, r, s))
        direct[p, q, r, s] = sum(
            coefficient * radial(a[:2], c[:2], b[:2], d[:2], rank)
            for rank, coefficient in two_body_angular(
                a[1], a[2], b[1], b[2], c[1], c[2], d[1], d[2]
            )
        )
    antisym = direct - direct.swapaxes(2, 3)
    native_antisym_error = max(
        abs(antisym[p, q, r, s] - problem.integrals.eri(p, q, r, s))
        for p, q, r, s in itertools.product(range(nlabels), repeat=4)
    )
    if native_antisym_error > 1e-11:
        raise AssertionError(
            "independent signed direct/antisym Coulomb identity failed"
        )
    positions = {
        key: (ip, g.index(key)) for ip, g in enumerate(plan["groups"]) for key in g
    }
    channels = {tuple(row): i for i, row in enumerate(rows)}
    core_key = plan["occupied_keys"][0]

    def expand(values):
        result = np.zeros((nlabels, nlabels, nc))
        for m, n, b in itertools.product(
            range(nc, nlabels), range(nc, nlabels), range(nc)
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

    sigma = expand(doubles)
    native_response = expand(response_d)
    oracle_d = np.zeros_like(sigma)
    paper_terms = {
        name: np.zeros_like(sigma)
        for name in ("source", "core_pair", "particle_pair", "valence_cr", "core_cr")
    }
    oracle_s = np.zeros(nlabels)
    occupied, virtual = range(nc), range(nc, nlabels)
    shift = energy - eps[column]
    g, gt = direct, antisym
    for m in virtual:
        for b, c, n in itertools.product(occupied, occupied, virtual):
            oracle_s[m] -= (
                gt[b, c, column, n]
                * g[m, n, b, c]
                / (eps[b] + eps[c] - eps[m] - eps[n] + shift)
            )
        for b, n, r in itertools.product(occupied, virtual, virtual):
            oracle_s[m] += (
                gt[m, b, n, r] * sigma[n, r, b] / (energy + eps[b] - eps[n] - eps[r])
            )
    for m, n, b in itertools.product(virtual, virtual, occupied):
        paper_terms["source"][m, n, b] = g[m, n, column, b]
        for c, d in itertools.product(occupied, repeat=2):
            paper_terms["core_pair"][m, n, b] += (
                g[c, d, column, b]
                * g[m, n, c, d]
                / (eps[c] + eps[d] - eps[m] - eps[n] + shift)
            )
        for r, s in itertools.product(virtual, repeat=2):
            paper_terms["particle_pair"][m, n, b] += (
                g[m, n, r, s] * sigma[r, s, b] / (energy + eps[b] - eps[r] - eps[s])
            )
        for c, r in itertools.product(occupied, virtual):
            den_mr = energy + eps[c] - eps[m] - eps[r]
            den_nr = energy + eps[c] - eps[n] - eps[r]
            paper_terms["valence_cr"][m, n, b] += (
                gt[c, n, r, b] * sigma[m, r, c] / den_mr
            )
            paper_terms["valence_cr"][m, n, b] -= (
                g[c, n, r, b] * sigma[r, m, c] / den_mr
            )
            paper_terms["valence_cr"][m, n, b] -= (
                g[m, c, r, b] * sigma[r, n, c] / den_nr
            )
            core_nr = eps[c] + eps[b] - eps[n] - eps[r] + shift
            core_mr = eps[c] + eps[b] - eps[m] - eps[r] + shift
            paper_terms["core_cr"][m, n, b] += (
                gt[m, c, column, r] * g[r, n, c, b] / core_nr
            )
            paper_terms["core_cr"][m, n, b] -= (
                g[m, c, column, r] * g[n, r, c, b] / core_nr
            )
            paper_terms["core_cr"][m, n, b] -= (
                g[c, n, column, r] * g[m, r, c, b] / core_mr
            )
        oracle_d[m, n, b] = sum(t[m, n, b] for t in paper_terms.values())
    # Eq. 37 of the paper and the original SD (Eq. 9) store different
    # symmetric m/n gauge components of an unantisymmetrized numerator.
    # Match the original diagram gauge using the Eq. 9 tilde contractions,
    # while retaining every paper coordinate and checking its physical map.
    paper_d = oracle_d.copy()
    terms = {name: value.copy() for name, value in paper_terms.items()}
    terms["valence_cr"].fill(0)
    terms["core_cr"].fill(0)
    for m, n, b in itertools.product(virtual, virtual, occupied):
        for c, r in itertools.product(occupied, virtual):
            terms["valence_cr"][m, n, b] += (
                gt[c, n, r, b]
                * (sigma[m, r, c] - sigma[r, m, c])
                / (energy + eps[c] - eps[m] - eps[r])
            )
            terms["core_cr"][m, n, b] += (
                gt[m, c, column, r]
                * (g[r, n, c, b] - g[n, r, c, b])
                / (eps[c] + eps[b] - eps[n] - eps[r] + shift)
            )
    oracle_d = sum(terms.values())
    paper_difference = oracle_d - paper_d
    # Core singles vanish in the declared canonical first-order control.
    # No physical small-amplitude or symmetry-coordinate truncation is used.
    expected_s = np.zeros_like(oracle_s)
    for m in virtual:
        ip, slot = positions[labels[m][:2]]
        if labels[m][1:] == labels[column][1:]:
            expected_s[m] = response_s[slot]
    errors = dict(
        singles_max_Ha=float(np.max(abs(expected_s - oracle_s))),
        doubles_max_Ha=float(np.max(abs(native_response - oracle_d))),
        paper_gauge_difference_Ha=float(np.max(abs(paper_difference))),
        paper_physical_difference_Ha=float(
            np.max(abs(paper_difference - paper_difference.swapaxes(0, 1)))
        ),
        native_antisym_error_Ha=float(native_antisym_error),
        all_magnetic_double_coordinates=(nlabels - nc) ** 2 * nc,
        core_radial_key=core_key,
    )
    return errors, dict(
        **{f"oracle_{name}": value for name, value in terms.items()},
        **{f"paper_{name}": value for name, value in paper_terms.items()},
        paper_d=paper_d,
        direct=direct,
        antisym=antisym,
        input_d=sigma,
        native_d=native_response,
        oracle_d=oracle_d,
        native_s=expected_s,
        oracle_s=oracle_s,
    )
