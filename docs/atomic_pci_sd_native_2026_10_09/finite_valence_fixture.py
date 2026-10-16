"""Explicit bounded source-bound C2+ control; no production space selection."""

import itertools

import numpy as np
from build_valence_fock_control import build_fock_control
from probe_sigma2_control003 import valence_channels

from atomx.angular.sd_rho import kappa_from_partial, rho_coefficient
from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures


def prepare(
    here,
    reference,
    *,
    backend_builder=build_fock_control,
    fock_probe=True,
    virtual_rotation=0.0,
    core_virtual_rotation=0.0,
    core_options=None,
):
    if (
        not isinstance(fock_probe, bool)
        or not np.isfinite(virtual_rotation)
        or not np.isfinite(core_virtual_rotation)
    ):
        raise ValueError("explicit finite fixture operator/rotation required")
    core = DFSolver(
        Z=6, subshells=[Subshell(1, -1, 2), Subshell(2, -1, 2)], ns=40, ks=7
    ).solve()
    occupied = ((1, -1), (2, -1))
    orbitals = {key: core.orbitals_by_subshell()[key][:2] for key in occupied}
    virtual = []
    for k in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[:2]):
            key = (i + (3 if k == -1 else 2), k)
            virtual.append(key)
            orbitals[key] = state[:2]
    if virtual_rotation:
        c, s = np.cos(virtual_rotation), np.sin(virtual_rotation)
        for k in (-1, 1, -2):
            left, right = (key for key in virtual if key[1] == k)
            a, b = orbitals[left], orbitals[right]
            orbitals[left] = tuple(c * x + s * y for x, y in zip(a, b, strict=True))
            orbitals[right] = tuple(-s * x + c * y for x, y in zip(a, b, strict=True))
    if core_virtual_rotation:
        # This deliberately changes the occupied projector, not just its
        # coordinates. Construct every integral/Fock/cache identity afterwards.
        c, s = np.cos(core_virtual_rotation), np.sin(core_virtual_rotation)
        left, right = (2, -1), (3, -1)
        a, b = orbitals[left], orbitals[right]
        orbitals[left] = tuple(c * x + s * y for x, y in zip(a, b, strict=True))
        orbitals[right] = tuple(-s * x + c * y for x, y in zip(a, b, strict=True))
    core_options = {} if core_options is None else dict(core_options)
    plan = plan_core_sd(
        occupied,
        virtual,
        PolynomialCellCoulombFeatures(core._bsp_data).identity,
        **core_options,
    )
    rows = {key: valence_channels(occupied, key[1], 1, "rho") for key in virtual}
    nxv = max(
        len(valence_channels(occupied, key[1], 1, kind))
        for key in virtual
        for kind in ("x1", "x2")
    )
    nvh = max(map(len, rows.values()))
    dims = plan["dimensions"]
    extra = 8 * (
        dims["NXX"] ** 2 * (14 * nvh + 2 * nxv)
        + 10 * dims["NX"]
        + 20**4 * 3
        + 1600 * 45
    )
    if plan["complete_payload_bound_bytes"] + extra > 512 * 1024**2:
        raise MemoryError(
            "complete declared finite core/valence/oracle/Krylov storage exceeds cap"
        )
    backend_builder(
        reference / "pCI",
        here / "backend",
        dims,
        reference / "gfortran",
        nxv=nxv,
        nvh=nvh,
        runtime=reference / "toolchain/usr/lib/x86_64-linux-gnu",
        **core_options,
    )
    problem = CoreSD(
        orbitals,
        core._bsp_data,
        6,
        40,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=core._nuclear_potential,
        backend_directory=here / "backend",
        radial_representation="polynomial",
        memory_bytes=512 * 1024**2,
        **core_options,
    )
    labels = problem.integrals.labels
    fock = problem.integrals.fock_Ha.real.copy()
    for p, q in itertools.product(range(len(labels)), repeat=2):
        if fock_probe and p != q and labels[p][1:] == labels[q][1:]:
            fock[p, q] += 0.031 if (labels[p][0] + labels[q][0]) % 2 else -0.023
    alpha, beta = (
        np.zeros((dims["NK"], dims["NK"], dims["KK"] + 1), order="F") for _ in range(2)
    )
    for im, jn, rank in itertools.product(
        range(dims["NK"]), range(dims["NK"]), range(dims["KK"] + 1)
    ):
        ki, kj = kappa_from_partial(im + 1), kappa_from_partial(jn + 1)
        a = (1, ki, -2 * abs(ki) + 1)
        for mj in range(-2 * abs(kj) + 1, 2 * abs(kj), 2):
            b = (1, kj, mj)
            alpha[im, jn, rank] += rho_coefficient(rank, a, b, a, b)
            beta[im, jn, rank] += rho_coefficient(rank, b, a, a, b)
    radial = np.zeros((dims["NX"], dims["NX"], dims["KX1"]), order="F")
    for im, group in enumerate(plan["groups"]):
        for i, left in enumerate(group):
            for j, right in enumerate(group):
                p = labels.index((*left, -2 * abs(left[1]) + 1))
                q = labels.index((*right, -2 * abs(right[1]) + 1))
                radial[i, j, im] = fock[p, q]
    if fock_probe:
        problem._call("atomx_sd_set_fock", [radial, alpha, beta])
    return core, orbitals, plan, rows, nvh, extra, problem, fock
