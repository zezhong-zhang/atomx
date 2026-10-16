"""Run complete small-pool pCI/native fixed-energy valence convention gates."""

import ctypes
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from build_sigma2_control import build_sigma2
from scipy.sparse.linalg import LinearOperator, gmres
from sigma2_magnetic_oracle import action as sigma2_action
from sigma2_magnetic_oracle import expand_sigma2
from valence_allorder_oracle import (
    action as magnetic_action,
)
from valence_allorder_oracle import (
    core_numerators,
    expand_valence,
)
from valence_magnetic_oracle import compare_map

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
    here = OWNER / "sigma2_control002"
    here.mkdir(exist_ok=False)
    start = time.perf_counter()
    archive = here / "control_source"
    archive.mkdir()
    identity = {}
    for name in (
        Path(__file__).name,
        "build_valence_control.py",
        "build_sigma2_control.py",
        "sigma2_control.f90",
        "sigma2_magnetic_oracle.py",
        "valence_control.f90",
        "valence_magnetic_oracle.py",
        "valence_allorder_oracle.py",
    ):
        src = OWNER / name
        shutil.copy2(src, archive / name)
        identity[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    (here / "control_source_sha256.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
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
    lmax = 1
    valence_rows = {
        key: valence_channels(occupied, key[1], lmax, "rho") for key in virtual
    }
    nxv = max(
        len(valence_channels(occupied, key[1], lmax, kind))
        for key in virtual
        for kind in ("x1", "x2")
    )
    nvh = max(map(len, valence_rows.values()))
    vw_rows = []
    cn_rows = []
    for iv, v in enumerate(virtual):
        for iw, w in enumerate(virtual):
            for im in range(3):
                km = kappa_from_partial(im + 1)
                for jn in range(3):
                    kn = kappa_from_partial(jn + 1)
                    low = max(abs(abs(km) - abs(v[1])), abs(abs(kn) - abs(w[1])))
                    high = min(abs(km) + abs(v[1]) - 1, abs(kn) + abs(w[1]) - 1)
                    for rank in range(low, high + 1):
                        if (l_of_kappa(km) + l_of_kappa(v[1])) % 2 == (
                            l_of_kappa(kn) + l_of_kappa(w[1])
                        ) % 2:
                            vw_rows.append((iv, iw, im, jn, rank))
                for rank in range(
                    max(abs(1 - abs(v[1])), abs(abs(km) - abs(w[1]))),
                    min(1 + abs(v[1]) - 1, abs(km) + abs(w[1]) - 1) + 1,
                ):
                    if (
                        not (l_of_kappa(v[1]) + rank) % 2
                        and not (l_of_kappa(km) + l_of_kappa(w[1]) + rank) % 2
                    ):
                        cn_rows.append((iv, iw, 0, im, rank))
    sizes = dict(
        NZ=len(virtual),
        NXV=nxv,
        NVH=nvh,
        NVH2=sum(map(len, valence_rows.values())),
        NVWH=len(vw_rows) + 1,
        NXV2=sum(
            max(len(valence_channels(occupied, key[1], lmax, k)) for k in ("x1", "x2"))
            for key in virtual
        )
        + 1,
        NXCN=len(cn_rows) + 1,
    )
    dims = plan["dimensions"]
    # Complete core plus valence, screening and original diagram scratch bounds.
    extra = 8 * (
        2 * dims["NHF"] * dims["NX"] * sizes["NZ"]
        + 2
        * dims["NXX"] ** 2
        * (sizes["NVWH"] + sizes["NVH2"] + sizes["NXV2"] + 4 * nvh)
        + sizes["NXCN"] * dims["NXX"]
        + dims["NHF"] * dims["NX"] ** 2 * 20
    )
    if plan["complete_payload_bound_bytes"] + extra > 512 * 1024**2:
        raise MemoryError("complete declared Sigma2 control exceeds cap")
    (here / "plan.json").write_text(
        json.dumps(
            dict(
                core=plan,
                sigma2=sizes,
                payload_bound=plan["complete_payload_bound_bytes"] + extra,
            ),
            indent=2,
        )
        + "\n"
    )
    build_sigma2(
        REF / "pCI",
        here / "backend",
        dims,
        REF / "gfortran",
        sizes=sizes,
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
    labels = problem.integrals.labels
    eps = np.diag(problem.integrals.fock_Ha).real
    f = np.zeros((dims["NX"], dims["NX"], dims["KX1"]), order="F")
    for ip, group in enumerate(plan["groups"]):
        for i, key in enumerate(group):
            f[i, i, ip] = eps[labels.index((*key, -2 * abs(key[1]) + 1))]
    zeros = np.zeros((dims["NK"], dims["NK"], dims["KK"] + 1), order="F")
    problem._call("atomx_sd_set_fock", [f, zeros, zeros])
    core_source = np.zeros_like(problem.source)
    core_den = np.zeros_like(problem.denominators)
    problem._vector_call("atomx_sd_vector_info", [core_source, core_den])
    core_op = LinearOperator(
        (len(core_source),) * 2, matvec=problem.action, dtype=float
    )
    pre = LinearOperator(core_op.shape, matvec=lambda v: v / core_den, dtype=float)
    physical, info = gmres(
        core_op, core_source, M=pre, atol=1e-12, rtol=0, restart=30, maxiter=100
    )
    core_residual = problem.action(physical) - core_source
    if info != 0 or np.max(abs(core_residual)) > 1e-10:
        raise AssertionError("core canonical raw residual failed")
    core_s, core_d = core_numerators(problem, plan, physical)
    lib = problem._backend
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
    init = lib.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, pointer]
    init.restype = None
    mapping = lib.atomx_val_map
    mapping.argtypes = [ctypes.c_double] + [pointer] * 8
    mapping.restype = None
    record = []
    reduced = []
    magnetic = []
    energies = []
    direct = None
    for key in virtual:
        partial = next(i for i, g in enumerate(plan["groups"]) if key in g)
        slot = plan["groups"][partial].index(key)
        init(slot + 1, key[1], len(physical), np.asfortranarray(physical))
        count = np.zeros(1, dtype=np.int32)
        problem._call("atomx_val_count", [count])
        rows = valence_rows[key]
        if count[0] != len(rows):
            raise AssertionError("complete valence channel count differs")
        channels = np.zeros((4, nvh), dtype=np.int32, order="F")
        source = np.zeros((dims["NXX"], dims["NXX"], nvh), order="F")
        energy = np.zeros(1)
        problem._call("atomx_val_info", [channels, source, energy])
        expected = np.asarray(rows, dtype=np.int32).T + np.array(
            [[1], [1], [1], [0]], dtype=np.int32
        )
        if not np.array_equal(channels[:, : len(rows)], expected) or np.any(
            channels[:, len(rows) :]
        ):
            raise AssertionError("complete valence channel identity differs")
        coordinates = [
            (m, n, c)
            for c, (_, im, jn, _) in enumerate(rows)
            for n, nkey in enumerate(plan["groups"][jn])
            for m, mkey in enumerate(plan["groups"][im])
            if mkey in virtual and nkey in virtual
        ]
        index = np.asarray(coordinates).T
        shape = source.shape
        z = np.zeros(len(coordinates))
        zs = np.zeros(dims["NX"], order="F")
        zd = np.zeros(shape, order="F")
        value = float(energy[0] + 0.075)
        energies.append(value)

        def unpack(v, shape=shape, index=index):
            d = np.zeros(shape, order="F")
            d[tuple(index)] = v
            return d

        def evaluate(
            v,
            prime=None,
            unpack=unpack,
            shape=shape,
            index=index,
            zs=zs,
            zd=zd,
            value=value,
        ):
            d = unpack(v)
            dd = zd if prime is None else unpack(prime)
            rs, rd, drs, drd = (
                np.zeros_like(zs),
                np.zeros(shape, order="F"),
                np.zeros_like(zs),
                np.zeros(shape, order="F"),
            )
            mapping(value, zs, d, zs, dd, rs, rd, drs, drd)
            if not all(np.all(np.isfinite(a)) for a in (rs, rd, drs, drd)):
                raise ValueError("nonfinite complete valence map")
            return rs, rd[tuple(index)], drs, drd[tuple(index)]

        constant = evaluate(z)[1]
        op = LinearOperator(
            (len(z),) * 2,
            matvec=lambda v, evaluate=evaluate, constant=constant: (
                v - (evaluate(v)[1] - constant)
            ),
            dtype=float,
        )
        root, info = gmres(op, constant, atol=1e-12, rtol=0, restart=30, maxiter=100)
        residual = root - evaluate(root)[1]
        if info != 0 or np.max(abs(residual)) > 1e-10:
            raise AssertionError("complete all-column valence residual failed")
        prime, info = gmres(
            op, evaluate(root)[3], atol=1e-12, rtol=0, restart=30, maxiter=100
        )
        if info != 0 or np.max(abs(op @ prime - evaluate(root)[3])) > 1e-10:
            raise AssertionError("valence response derivative residual failed")
        d = unpack(root)
        if direct is None:
            column = labels.index((*key, -2 * abs(key[1]) + 1))
            _, vectors = compare_map(
                problem,
                od,
                core._bsp_data,
                plan,
                rows,
                d[:, :, : len(rows)],
                value,
                column,
                evaluate(root)[0],
                d[:, :, : len(rows)],
            )
            direct = vectors["direct"]
        root_errors = []
        for mval in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
            column = labels.index((*key, mval))
            native = expand_valence(problem, plan, rows, d, column)
            os_, od_ = magnetic_action(
                direct, eps, core_s, core_d, column, value, native
            )
            root_error = float(np.max(abs(native - od_)))
            single_error = max(
                abs(
                    evaluate(root)[0][plan["groups"][partial].index(k)]
                    - os_[labels.index((*k, mval))]
                )
                for k in plan["virtual_keys"]
                if k[1] == key[1]
            )
            if max(root_error, single_error) > 1e-11:
                raise AssertionError("all-multiplet signed valence equation failed")
            root_errors.append(
                dict(
                    two_m=mval,
                    doubles_error_Ha=root_error,
                    singles_error_Ha=float(single_error),
                )
            )
        reduced.append(
            (
                expected.copy(order="F"),
                d[:, :, : len(rows)].copy(order="F"),
                unpack(prime)[:, :, : len(rows)].copy(order="F"),
            )
        )
        magnetic.append(
            {
                mval: expand_valence(problem, plan, rows, d, labels.index((*key, mval)))
                for mval in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2)
            }
        )
        response = evaluate(root, prime)
        record.append(
            dict(
                key=key,
                energy_Ha=value,
                raw_residual_Ha=float(np.max(abs(residual))),
                magnetic_gates=root_errors,
                sigma1={
                    str(k): float(response[0][plan["groups"][partial].index(k)])
                    for k in virtual
                    if k[1] == key[1]
                },
            )
        )
    (here / "valence_columns.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(valence_columns=record)), flush=True)
    vwinit = lib.atomx_vw_init
    vwinit.argtypes = [ctypes.c_int, integer, pointer]
    vwinit.restype = None
    vwinit(
        len(virtual),
        np.asfortranarray(virtual, dtype=np.int32).T.copy(order="F"),
        np.asfortranarray(energies),
    )
    load = lib.atomx_vw_load
    load.argtypes = [ctypes.c_int, ctypes.c_int, integer, pointer, pointer]
    load.restype = None
    for i, (channels, d, dd) in enumerate(reduced):
        load(i + 1, channels.shape[1], channels, d, dd)
    count = np.zeros(1, dtype=np.int32)
    channels = np.zeros((5, sizes["NVWH"]), dtype=np.int32, order="F")
    response = np.zeros((dims["NXX"], dims["NXX"], sizes["NVWH"] + 1), order="F")
    derivative = np.zeros_like(response, order="F")
    problem._call("atomx_vw_evaluate", [count, channels, response, derivative])
    expected = np.asarray(vw_rows, dtype=np.int32).T + np.array(
        [[1], [1], [1], [1], [0]], dtype=np.int32
    )
    if count[0] != len(vw_rows) or not np.array_equal(
        channels[:, : count[0]], expected
    ):
        raise AssertionError("complete Sigma2 output channels differ")
    np.savez_compressed(
        here / "sigma2_raw.npz",
        channels=channels,
        result=response,
        upstream_derivative=derivative,
        core_physical=physical,
        core_residual=core_residual,
        direct=direct,
        core_s=core_s,
        core_d=core_d,
        eps=eps,
    )
    if not np.all(np.isfinite(response)):
        raise ValueError("nonfinite original Sigma2 values")
    magnetic_sigmas = np.zeros(
        (len(labels), len(labels), problem.integrals.noccupied, len(labels))
    )
    energy_by_column = eps.copy()
    for key, energy, dmap in zip(virtual, energies, magnetic, strict=True):
        for mval, sigma in dmap.items():
            column = labels.index((*key, mval))
            magnetic_sigmas[:, :, :, column] = sigma
            energy_by_column[column] = energy
    native = expand_sigma2(labels, plan, virtual, vw_rows, response)
    oracle = sigma2_action(
        direct, eps, core_s, core_d, magnetic_sigmas, energy_by_column
    )
    difference = native - oracle
    comparison = dict(
        complete_signed_error_Ha=float(np.max(abs(difference))),
        antisym_output_error_Ha=float(
            np.max(abs(difference - difference.swapaxes(0, 1)))
        ),
        antisym_both_error_Ha=float(
            np.max(
                abs(
                    difference
                    - difference.swapaxes(0, 1)
                    - difference.swapaxes(2, 3)
                    + difference.swapaxes(0, 1).swapaxes(2, 3)
                )
            )
        ),
        native_pair_exchange_error_Ha=float(
            np.max(abs(native - native.transpose(1, 0, 3, 2)))
        ),
        oracle_pair_exchange_error_Ha=float(
            np.max(abs(oracle - oracle.transpose(1, 0, 3, 2)))
        ),
        full_magnetic_coordinates=int(native.size),
        radial_channels=int(count[0]),
        seconds=time.perf_counter() - start,
        scope="complete declared finite canonical all-column control, no experimental precision claim",
    )
    np.savez_compressed(
        here / "sigma2_magnetic.npz",
        native=native,
        oracle=oracle,
        magnetic_sigmas=magnetic_sigmas,
        energy_by_column=energy_by_column,
        labels=np.asarray(labels, dtype=int),
    )
    (here / "sigma2_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    print(json.dumps(dict(sigma2_comparison=comparison)), flush=True)
    if (
        max(
            comparison["complete_signed_error_Ha"],
            comparison["native_pair_exchange_error_Ha"],
            comparison["oracle_pair_exchange_error_Ha"],
        )
        > 1e-11
    ):
        raise AssertionError("independent complete signed Sigma2 gate failed")
    print(
        json.dumps(
            dict(
                stage="original complete Sigma2 values",
                channels=int(count[0]),
                seconds=time.perf_counter() - start,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
