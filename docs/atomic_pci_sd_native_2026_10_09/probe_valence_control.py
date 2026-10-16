"""Run complete small-pool pCI/native fixed-energy valence convention gates."""

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from build_valence_control import build_control

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from atomx.angular.recoupling import l_of_kappa  # noqa: E402
from atomx.angular.sd_rho import kappa_from_partial  # noqa: E402
from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)

OWNER = Path(__file__).resolve().parent
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")


def valence_channels(occupied, kappa, lmax, kind):
    rows = []
    for i, (_, ki) in enumerate(occupied):
        for im in range(1, 2 * lmax + 2):
            km = kappa_from_partial(im)
            for jn in range(1, 2 * lmax + 2):
                kn = kappa_from_partial(jn)
                a, b, c, d = (
                    (km, kappa, kn, ki) if kind != "x2" else (km, kn, kappa, ki)
                )
                low = max(abs(abs(a) - abs(b)), abs(abs(c) - abs(d)))
                high = min(abs(a) + abs(b) - 1, abs(c) + abs(d) - 1)
                for rank in range(low, high + 1):
                    parity = (
                        l_of_kappa(a) + l_of_kappa(b) + rank,
                        l_of_kappa(c) + l_of_kappa(d) + rank,
                    )
                    allowed = (
                        parity[0] % 2 == parity[1] % 2
                        if kind == "rho"
                        else not parity[0] % 2 and not parity[1] % 2
                    )
                    if allowed:
                        rows.append((i, im - 1, jn - 1, rank))
    return tuple(rows)


def main():
    here = OWNER / "valence_control001"
    here.mkdir(exist_ok=False)
    start = time.perf_counter()
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    occupied = ((1, -1),)
    od = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    virtual = []
    for k in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[:2]):
            key = (i + 2, k)
            virtual.append(key)
            od[key] = state[:2]
    feature = PolynomialCellCoulombFeatures(core._bsp_data)
    plan = plan_core_sd(occupied, virtual, feature.identity)
    rows = valence_channels(occupied, -1, 1, "rho")
    nxv = max(len(valence_channels(occupied, -1, 1, kind)) for kind in ("x1", "x2"))
    nvh = len(rows)
    if (
        plan["complete_payload_bound_bytes"]
        + 8 * plan["dimensions"]["NHF"] * plan["dimensions"]["NX"] ** 2 * 20
        > 512 * 1024**2
    ):
        raise MemoryError("complete control exceeds declared cap")
    build_control(
        REF / "pCI",
        here / "backend",
        plan["dimensions"],
        REF / "gfortran",
        nxv=nxv,
        nvh=nvh,
        runtime=REF / "toolchain/usr/lib/x86_64-linux-gnu",
    )
    problem = CoreSD(
        od,
        core._bsp_data,
        4,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=core._nuclear_potential,
        backend_directory=here / "backend",
        radial_representation="polynomial",
        memory_bytes=512 * 1024**2,
    )
    dims = plan["dimensions"]
    # Canonical validation Hamiltonian: same native two-body integrals; Fock
    # diagonals retained, all off-diagonals explicitly removed. This does not
    # change or claim results for the physical noncanonical native Hamiltonian.
    f = np.zeros((dims["NX"], dims["NX"], dims["KX1"]), order="F")
    for ip, group in enumerate(plan["groups"]):
        for i, key in enumerate(group):
            label = (*key, -2 * abs(key[1]) + 1)
            j = problem.integrals.labels.index(label)
            f[i, i, ip] = problem.integrals.fock_Ha[j, j].real
    zeros = np.zeros((dims["NK"], dims["NK"], dims["KK"] + 1), order="F")
    problem._call("atomx_sd_set_fock", [f, zeros, zeros])
    physical = problem.source / problem.denominators
    # The validation source has no occupied/virtual canonical Fock coupling.
    physical[: len(virtual) // 3] = 0
    lib = problem._backend
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    init = lib.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, pointer]
    init.restype = None
    init(2, -1, len(physical), np.asfortranarray(physical))
    channels = np.zeros((4, nvh), dtype=np.int32, order="F")
    source = np.zeros((dims["NXX"], dims["NXX"], nvh), order="F")
    energy = np.zeros(1)
    problem._call("atomx_val_info", [channels, source, energy])
    expected = np.asarray(rows, dtype=np.int32).T
    if not np.array_equal(
        channels, expected + np.array([[1], [1], [1], [0]], dtype=np.int32)
    ):
        raise AssertionError("complete valence channel identity differs")
    mapping = lib.atomx_val_map
    mapping.argtypes = [ctypes.c_double] + [pointer] * 8
    mapping.restype = None
    rng = np.random.default_rng(120092)
    singles = np.zeros(dims["NX"], order="F")
    doubles = np.zeros_like(source, order="F")
    for c, (_, im, jn, _) in enumerate(rows):
        firstm = sum(k[1] == kappa_from_partial(im + 1) for k in occupied)
        firstn = sum(k[1] == kappa_from_partial(jn + 1) for k in occupied)
        nm, nn = len(plan["groups"][im]), len(plan["groups"][jn])
        doubles[firstm:nm, firstn:nn, c] = rng.normal(
            scale=0.01, size=(nm - firstm, nn - firstn)
        )
    singles[2] = 0.02
    e = float(energy[0] + 0.075)

    def evaluate(value):
        rs, rd, drs, drd = (
            np.zeros_like(singles),
            np.zeros_like(doubles, order="F"),
            np.zeros_like(singles),
            np.zeros_like(doubles, order="F"),
        )
        mapping(
            value,
            singles,
            doubles,
            np.zeros_like(singles),
            np.zeros_like(doubles, order="F"),
            rs,
            rd,
            drs,
            drd,
        )
        if not all(np.all(np.isfinite(a)) for a in (rs, rd, drs, drd)):
            raise ValueError("nonfinite valence control response")
        return rs, rd, drs, drd

    actual = evaluate(e)
    derivative = []
    for step in (1e-3, 3e-4, 1e-4):
        plus, minus = evaluate(e + step), evaluate(e - step)
        errors = [
            float(np.max(abs((plus[i] - minus[i]) / (2 * step) - actual[i + 2])))
            for i in (0, 1)
        ]
        derivative.append(
            dict(step_Ha=step, singles_error=errors[0], doubles_error=errors[1])
        )
    np.savez_compressed(
        here / "vectors.npz",
        source=source,
        core_physical=physical,
        singles=singles,
        doubles=doubles,
        response_s=actual[0],
        response_d=actual[1],
        derivative_s=actual[2],
        derivative_d=actual[3],
        fock=f,
        channels=channels,
    )
    result = dict(
        derivative=derivative,
        nxv=nxv,
        nvh=nvh,
        energy_Ha=e,
        source_bound_core_provenance=problem.provenance,
        seconds=time.perf_counter() - start,
        load=os.getloadavg(),
        model="canonical control, not native physical Fock",
        status="prototype convention audit; no public valence precision claim",
    )
    (here / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "source_bound_core_provenance"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
