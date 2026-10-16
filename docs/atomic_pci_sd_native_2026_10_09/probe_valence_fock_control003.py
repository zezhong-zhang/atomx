"""Full reduced/magnetic and independent fermion noncanonical Fock anchors."""

import ctypes
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from build_valence_fock_control import build_fock_control  # noqa: E402
from general_rho_control import core_numerators, expand_valence  # noqa: E402
from probe_sigma2_control003 import valence_channels  # noqa: E402
from valence_fock_oracle import fermion_fock, magnetic_fock  # noqa: E402

from atomx.angular.sd_rho import kappa_from_partial, rho_coefficient  # noqa: E402
from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)

OWNER = Path(__file__).resolve().parent
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")


def main():
    here = OWNER / "valence_fock_control003"
    here.mkdir(exist_ok=False)
    archive = here / "source"
    archive.mkdir()
    names = (
        Path(__file__).name,
        "build_valence_fock_control.py",
        "valence_fock_control.f90",
        "valence_fock_oracle.py",
        "build_valence_control.py",
        "valence_control.f90",
        "valence_allorder_oracle.py",
        "general_rho_control.py",
        "probe_sigma2_control003.py",
    )
    hashes = {}
    for name in names:
        src = OWNER / name
        shutil.copy2(src, archive / name)
        hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    (here / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    core = DFSolver(
        Z=6, subshells=[Subshell(1, -1, 2), Subshell(2, -1, 2)], ns=40, ks=7
    ).solve()
    occupied = ((1, -1), (2, -1))
    od = {key: core.orbitals_by_subshell()[key][:2] for key in occupied}
    virtual = []
    for k in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[:2]):
            key = (i + (3 if k == -1 else 2), k)
            virtual.append(key)
            od[key] = state[:2]
    plan = plan_core_sd(
        occupied, virtual, PolynomialCellCoulombFeatures(core._bsp_data).identity
    )
    rows_by_key = {key: valence_channels(occupied, key[1], 1, "rho") for key in virtual}
    nxv = max(
        len(valence_channels(occupied, key[1], 1, kind))
        for key in virtual
        for kind in ("x1", "x2")
    )
    nvh = max(map(len, rows_by_key.values()))
    dims = plan["dimensions"]
    extra = 8 * (dims["NXX"] ** 2 * (14 * nvh + 2 * nxv) + 10 * dims["NX"])
    if plan["complete_payload_bound_bytes"] + extra > 512 * 1024**2:
        raise MemoryError("complete finite Fock control exceeds declared cap")
    (here / "plan.json").write_text(
        json.dumps(
            dict(core=plan, valence_nvh=nvh, nxv=nxv, extra_bytes=extra), indent=2
        )
        + "\n"
    )
    build_fock_control(
        REF / "pCI",
        here / "backend",
        dims,
        REF / "gfortran",
        nxv=nxv,
        nvh=nvh,
        runtime=REF / "toolchain/usr/lib/x86_64-linux-gnu",
    )
    problem = CoreSD(
        od,
        core._bsp_data,
        6,
        40,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=core._nuclear_potential,
        backend_directory=here / "backend",
        radial_representation="polynomial",
        memory_bytes=512 * 1024**2,
    )
    result = problem.solve(residual_tolerance_Ha=1e-12)
    (here / "core_identity.json").write_text(
        json.dumps(problem.provenance, indent=2) + "\n"
    )
    labels = problem.integrals.labels
    f_original = problem.integrals.fock_Ha.real
    eps = np.diag(f_original)
    core_s, core_d = core_numerators(problem, plan, result.amplitudes)
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
    perturb = np.zeros_like(f_original)
    for p, q in itertools.product(range(len(labels)), repeat=2):
        if p != q and labels[p][1:] == labels[q][1:]:
            perturb[p, q] = 0.031 if (labels[p][0] + labels[q][0]) % 2 else -0.023
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    init = problem._backend.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, pointer]
    init.restype = None
    function = problem._backend.atomx_val_fock_map
    function.argtypes = [ctypes.c_double] + [pointer] * 6
    function.restype = None
    rng = np.random.default_rng(912771)
    records = []
    for name, fock in (
        ("native", f_original),
        ("scalar_probe", f_original + perturb),
        ("canonical_zero", np.diag(eps)),
    ):
        radial = np.zeros((dims["NX"], dims["NX"], dims["KX1"]), order="F")
        for im, group in enumerate(plan["groups"]):
            for i, left in enumerate(group):
                for j, right in enumerate(group):
                    p = labels.index((*left, -2 * abs(left[1]) + 1))
                    q = labels.index((*right, -2 * abs(right[1]) + 1))
                    radial[i, j, im] = fock[p, q]
        problem._call("atomx_sd_set_fock", [radial, alpha, beta])
        for key in virtual:
            partial = next(i for i, g in enumerate(plan["groups"]) if key in g)
            slot = plan["groups"][partial].index(key)
            init(
                slot + 1,
                key[1],
                len(result.amplitudes),
                np.asfortranarray(result.amplitudes),
            )
            rows = rows_by_key[key]
            shape = (dims["NXX"], dims["NXX"], nvh)
            d, dd = (np.zeros(shape, order="F") for _ in range(2))
            for c, (_, im, jn, _) in enumerate(rows):
                for m, left in enumerate(plan["groups"][im]):
                    for nn, right in enumerate(plan["groups"][jn]):
                        if left in virtual and right in virtual:
                            d[m, nn, c] = rng.normal() * 0.001
                            dd[m, nn, c] = rng.normal() * 0.001
            value = float(eps[labels.index((*key, -2 * abs(key[1]) + 1))] + 0.075)
            rs, rd, ps, pd = (
                np.zeros(dims["NX"], order="F"),
                np.zeros(shape, order="F"),
                np.zeros(dims["NX"], order="F"),
                np.zeros(shape, order="F"),
            )
            function(value, d, dd, rs, rd, ps, pd)
            vectors = dict(
                fock=fock,
                eps=eps,
                core_s=core_s,
                core_d=core_d,
                radial_d=d,
                radial_prime=dd,
                radial_rs=rs,
                radial_rd=rd,
                radial_ps=ps,
                radial_pd=pd,
                labels=np.asarray(labels),
            )
            errors = []
            for mj in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                column = labels.index((*key, mj))
                sigma = expand_valence(problem, plan, rows, d, column)
                prime = expand_valence(problem, plan, rows, dd, column)
                (os_, od_, op_s, op_d), physical = magnetic_fock(
                    fock, eps, core_s, core_d, column, value, sigma, prime
                )
                native_d = expand_valence(problem, plan, rows, rd, column)
                native_pd = expand_valence(problem, plan, rows, pd, column)
                single_indices = [
                    (i, labels.index((*k, mj)))
                    for i, k in enumerate(plan["groups"][partial])
                    if k in virtual
                ]
                magnetic_error = max(
                    float(np.max(abs(native_d - od_))),
                    float(np.max(abs(native_pd - op_d))),
                    *(abs(rs[i] - os_[p]) for i, p in single_indices),
                    *(abs(ps[i] - op_s[p]) for i, p in single_indices),
                )
                fs, fd = fermion_fock(fock, *physical, column)
                fermion_error = max(
                    float(np.max(abs(fs - os_))),
                    float(np.max(abs(fd - (od_ - od_.swapaxes(0, 1))))),
                )
                errors.append(
                    dict(
                        two_m=mj,
                        full_signed_error_Ha=float(magnetic_error),
                        fermion_error_Ha=float(fermion_error),
                    )
                )
                vectors[f"magnetic_d_{mj}"] = native_d
                vectors[f"magnetic_oracle_{mj}"] = od_
                vectors[f"fermion_d_{mj}"] = fd
            np.savez_compressed(here / f"{name}_{key[0]}_{key[1]}.npz", **vectors)
            records.append(dict(operator=name, key=key, energy_Ha=value, errors=errors))
            (here / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
            if (
                max(
                    e[field]
                    for e in errors
                    for field in ("full_signed_error_Ha", "fermion_error_Ha")
                )
                > 1e-11
            ):
                raise AssertionError(
                    "complete native/magnetic/fermion Fock gate failed; full vectors retained"
                )
    print(json.dumps(dict(records=records, passed=True)), flush=True)


if __name__ == "__main__":
    main()
