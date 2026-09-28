"""DBSR_HF average-configuration Breit energy, ported from its source.

This follows ``hf_MOD_energy.f90:av_energy_coef`` and
``hf_energy_BR.f90:Breit_energy`` / ``SMU``. The generalized occupation
products are evaluated over the explicit jj ensemble, including
``q_i(q_i-1)`` within one subshell. The eight P/Q radial pairings and
the L-1, L, L+1 channels are retained. It is an average-configuration
energy and cannot resolve term or fine-structure Breit shifts.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache

import numpy as np

from atomx.angular.wigner import wigner_3j
from atomx.corrections.workspace import eval_orbital_pair, workspace_arrays


@lru_cache(maxsize=None)
def _cjk_squared(two_j_a: int, rank: int, two_j_b: int) -> float:
    """DBSR_HF ``Cjkj**2``; do not apply a Coulomb parity gate here."""
    three_j = wigner_3j(two_j_a, 2 * rank, two_j_b, 1, 0, -1)
    return (two_j_a + 1) * (two_j_b + 1) * three_j * three_j


def _smu(ka: int, kb: int, kc: int, kd: int,
         rank: int, radial_rank: int) -> tuple[float, ...]:
    """Literal eight-coefficient ``SMU`` formula from DBSR_HF."""
    la = ka if ka > 0 else -ka - 1
    lb = kb if kb > 0 else -kb - 1
    lc = kc if kc > 0 else -kc - 1
    ld = kd if kd > 0 else -kd - 1
    if radial_rank < 0 or (la + lc + radial_rank) % 2 == 0 or (
            lb + ld + radial_rank) % 2 == 0:
        return (0.0,) * 8
    if radial_rank == rank:
        if rank == 0:
            return (0.0,) * 8
        value = -(ka + kc) * (kb + kd) / (rank * (rank + 1))
        return (value,) * 8

    k = kc - ka
    kp = kd - kb
    if radial_rank == rank - 1:
        bp = (rank + 1) / (2 * (2 * rank - 1))
        cp = -(rank - 2) / (2 * rank * (2 * rank - 1))
        return ((rank + k) * (bp + cp * kp),
                (rank + kp) * (bp + cp * k),
                (rank - k) * (bp - cp * kp),
                (rank - kp) * (bp - cp * k),
                -(rank + k) * (bp - cp * kp),
                -(rank - kp) * (bp + cp * k),
                -(rank - k) * (bp + cp * kp),
                -(rank + kp) * (bp - cp * k))
    if radial_rank == rank + 1:
        b = rank / (2 * (2 * rank + 3))
        c = (rank + 3) / ((2 * rank + 2) * (2 * rank + 3))
        x = -rank - 1
        return ((x + kp) * (b + c * k),
                (x + k) * (b + c * kp),
                (x - kp) * (b - c * k),
                (x - k) * (b - c * kp),
                -(x - kp) * (b + c * k),
                -(x + k) * (b - c * kp),
                -(x + kp) * (b - c * k),
                -(x - k) * (b + c * kp))
    return (0.0,) * 8


def _pair_moments(orbitals, jj_configs):
    occupation = np.asarray([float(orb[2]) for orb in orbitals])
    moments = np.outer(occupation, occupation)
    np.fill_diagonal(moments, occupation * (occupation - 1))
    if jj_configs is None:
        if np.any(abs(occupation - np.rint(occupation)) > 1e-10):
            raise ValueError(
                "fractional jj occupations need an explicit configuration "
                "ensemble for DBSR_HF Breit pair coefficients")
        return moments
    configs = list(jj_configs)
    if not configs or any(len(cfg.subshells) != len(orbitals) for cfg in configs):
        raise ValueError("DBSR_HF Breit requires aligned jj configurations")
    weights = np.asarray([cfg.weight for cfg in configs], dtype=float)
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or abs(float(weights.sum()) - 1.0) > 1e-10):
        raise ValueError("DBSR_HF Breit jj weights must sum to one")
    moments.fill(0)
    for cfg in configs:
        if any(sh.label != orbital[4] for sh, orbital in
               zip(cfg.subshells, orbitals, strict=True)):
            raise ValueError("DBSR_HF Breit jj subshell labels do not match")
        q = np.asarray([sh.occupation for sh in cfg.subshells], dtype=float)
        degeneracy = np.asarray([2 * abs(int(orb[3])) for orb in orbitals])
        if (not np.all(np.isfinite(q)) or np.any(q != np.rint(q))
                or np.any(q < 0) or np.any(q > degeneracy)):
            raise ValueError("DBSR_HF Breit jj occupations must be physical integers")
        outer = np.outer(q, q)
        np.fill_diagonal(outer, q * (q - 1))
        moments += cfg.weight * outer
    return moments


def compute_dbsr_breit_energy(orbitals, workspace, *, jj_configs=None):
    """Evaluate the DBSR_HF Breit average-energy expression in Hartree.

    ``orbitals`` has ``(P,Q,occupation,kappa,label)`` tuples on a B-spline
    or radial-grid workspace. The radial integrals use atomx's common
    arbitrary-grid Y^k evaluator; angular and occupation factors match
    the DBSR_HF Fortran source.
    """
    if sum(float(orb[2]) for orb in orbitals) <= 1 + 1e-10:
        return 0.0
    moments = _pair_moments(orbitals, jj_configs)
    r, weights = workspace_arrays(workspace)
    r = np.ravel(r)
    weights = np.ravel(weights)
    waves = [tuple(np.ravel(x) for x in eval_orbital_pair(
        orb[0], orb[1], workspace)) for orb in orbitals]
    kappas = [int(orb[3]) for orb in orbitals]
    two_j = [2 * abs(kappa) - 1 for kappa in kappas]
    density_cache = OrderedDict()
    inner_cache = OrderedDict()

    def density(a, b):
        key = (a, b)
        if key not in density_cache:
            density_cache[key] = waves[a][0] * waves[b][1]
            if len(density_cache) > 256:
                density_cache.popitem(last=False)
        else:
            density_cache.move_to_end(key)
        return density_cache[key]

    def integral(a, b, c, d, rank):
        # DBSR_HF msk_ppqq stores the ordered radial triangle r1 < r2.
        # Its eight SMU terms include the reversed order explicitly; using
        # a full Y^k convolution here double counts every diagonal term.
        key = (a, b, rank)
        if key not in inner_cache:
            source = density(a, b) * r**rank
            cumulative = np.empty_like(source)
            cumulative[0] = 0.0
            cumulative[1:] = np.cumsum(
                (source[1:] + source[:-1]) * np.diff(r) / 2)
            inner_cache[key] = cumulative
            if len(inner_cache) > 256:
                inner_cache.popitem(last=False)
        else:
            inner_cache.move_to_end(key)
        return float(np.dot(
            weights, density(c, d) * inner_cache[key] / r**(rank + 1)))

    total = 0.0
    for i, ka in enumerate(kappas):
        for j, kb in enumerate(kappas):
            pair = moments[i, j]
            if pair == 0:
                continue
            if i >= j:
                indices = (i, j, i, j)
                angular_kappas = (ka, kb, ka, kb)
            else:
                indices = (i, j, j, i)
                angular_kappas = (ka, kb, kb, ka)
            i1, i2, i3, i4 = indices
            for rank in range(max(two_j) + 1):
                if i == j:
                    if rank == 0:
                        coefficient = pair / 2
                    elif rank <= two_j[i]:
                        coefficient = (-pair / 2 * _cjk_squared(
                            two_j[i], rank, two_j[i]) /
                            (two_j[i] * (two_j[i] + 1)))
                    else:
                        continue
                elif i > j:
                    if rank != 0:
                        continue
                    coefficient = pair
                else:
                    if not (abs(two_j[i] - two_j[j]) // 2 <= rank <=
                            (two_j[i] + two_j[j]) // 2):
                        continue
                    coefficient = (-pair * _cjk_squared(
                        two_j[i], rank, two_j[j]) /
                        ((two_j[i] + 1) * (two_j[j] + 1)))
                if coefficient == 0:
                    continue
                for radial_rank in (rank - 1, rank, rank + 1):
                    s = _smu(*angular_kappas, rank, radial_rank)
                    if not any(s):
                        continue
                    slots = (((i1, i2), (i3, i4)),
                             ((i2, i1), (i4, i3)),
                             ((i3, i4), (i1, i2)),
                             ((i4, i3), (i2, i1)),
                             ((i1, i4), (i3, i2)),
                             ((i4, i1), (i2, i3)),
                             ((i3, i2), (i1, i4)),
                             ((i2, i3), (i4, i1)))
                    for factor, ((a, b), (c, d)) in zip(s, slots, strict=True):
                        if factor:
                            total += coefficient * factor * integral(
                                a, c, b, d, radial_rank)
    return total
