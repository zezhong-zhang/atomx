"""SOURCE by default; once-only full-basis native sparse/Davidson qualification."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/data/code/atomx")
HERE = Path(__file__).resolve().parent
THREADS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024**2), b""):
            h.update(part)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return plain(value.tolist()) if hasattr(value, "tolist") else value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def metadata():
    c = read(HERE / "contract.json")
    paths = {key: ROOT / value["path"] for key, value in c["inputs"].items()}
    for key, value in c["inputs"].items():
        if value["kind"] == "metadata" and sha(paths[key]) != value["sha256"]:
            raise ValueError("metadata identity changed: " + key)
    spec = importlib.util.spec_from_file_location(
        "bound_preparation", paths["preparation_driver"]
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    bc, _, source, identity = base.metadata()
    if (
        bc["source_map_sha256"] != c["source_map_sha256"]
        or bc["fixed"] != c["fixed"]
        or bc["UIDs"] != c["UIDs"]
        or bc["angular_census"] != c["angular_census"]
    ):
        raise ValueError("frozen preparation mismatch")
    for value in c["arrays"].values():
        for field in value["fields"].values():
            base.field_shapes(field)
    for arm in ("A", "B"):
        point = read(paths[arm + "_R0_record"])
        if [r["UID"] for r in point["rows"]] != [u for _, u in c["UIDs"]]:
            raise ValueError("owned five UID identity changed")
        if any(r["selected_level"][2] != 0 for r in point["rows"]):
            raise ValueError("target is not in lowest2 sector roots")
    return c, paths, source, identity, base


def execute(seal_path):
    c, paths, source, identity, base = metadata()
    cap, gates = c["resources"], c["gates"]
    seal = read(seal_path)
    expected = dict(
        execution_authorized_by_root=True,
        once_only=True,
        driver_sha256=sha(__file__),
        contract_sha256=sha(HERE / "contract.json"),
        resources=cap,
        source_map_sha256=c["source_map_sha256"],
    )
    if any(seal.get(k) != v for k, v in expected.items()):
        raise ValueError("new exact root execution seal required")
    if (
        os.path.abspath(sys.executable) != "/data/code/atomx/.venv/bin/python"
        or os.sched_getaffinity(0) != {cap["CPU"]}
        or any(os.environ.get(k) != "1" for k in THREADS)
    ):
        raise ValueError("interpreter/CPU/thread admission")
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if (
        os.getloadavg()[0] > cap["host_load1_max"]
        or available < cap["host_MemAvailable_GiB_min"] * 1024**3
    ):
        raise ValueError("host admission")
    out = HERE / "actual"
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS, (cap["AS_bytes"], cap["AS_bytes"]))
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("900s inner stop")),
    )
    signal.alarm(cap["inner_seconds"])
    started = time.monotonic()
    counts, attempts = collections.Counter(), collections.Counter()
    checks, work, points = [], {}, {}
    report = dict(
        status="RUNNING",
        checks=checks,
        work=work,
        points=points,
        source_map_sha256=c["source_map_sha256"],
        old_failures_preserved=True,
        experimental_accuracy_qualified=False,
        larger_4f12_calculated=False,
        public_iterative_owner_or_tracker_qualified=False,
        controlled_timing=False,
        host_load=os.getloadavg(),
        available_bytes=available,
    )

    def checkpoint():
        report.update(counts=dict(counts), attempted=dict(attempts))
        write(out / "partial.json", report)

    def check(name, value, limit=None):
        ok = bool(value) if limit is None else bool(value <= limit)
        checks.append(dict(name=name, observed=plain(value), limit=limit, PASS=ok))
        if not ok:
            raise RuntimeError("gate failed: " + name)

    def observe(module, name, label, after=None, before=None):
        original = getattr(module, name)

        def wrapped(*args, **kw):
            attempts[label] += 1
            if attempts[label] > cap["calls"][label]:
                raise RuntimeError("entry cap: " + label)
            if before is not None:
                before(args, kw)
            value = original(*args, **kw)
            counts[label] += 1
            if after is not None:
                after(value, args, kw)
            return value

        setattr(module, name, wrapped)

    try:
        for key, value in c["inputs"].items():
            if value["kind"] == "array" and sha(paths[key]) != value["sha256"]:
                raise ValueError("whole array-file binding changed: " + key)
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy.sparse import issparse
        from scipy.sparse.linalg import LinearOperator

        import atomx.angular.mscheme as mscheme

        observe(mscheme, "null_space", "angular_nullspace")
        highest = mscheme.highest_weight_states
        check("cold_highest_weight_cache", highest.cache_info().currsize == 0)
        allowed = {(r["twoj"], r["q"]) for r in c["angular_census"]}
        queries = set()

        def guarded_highest(tj, q):
            if (tj, q) not in allowed:
                raise ValueError("unbudgeted highest-weight query")
            queries.add((tj, q))
            return highest(tj, q)

        mscheme.highest_weight_states = guarded_highest
        import atomx.angular.csf_hamiltonian as angular
        import atomx.physics.csf.matrix as matrix
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.methods import ci as ci_module
        from atomx.numerics.bspline import orbital_optimization as opt
        from atomx.physics.csf.effective_one_body import CIReferenceBinding
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile

        def operator_only(args, kw):
            if kw.get("as_operator") is not True:
                raise RuntimeError("dense H construction forbidden")

        observe(matrix, "build_ci_matrix", "H", before=operator_only)
        observe(ci_module, "davidson_eigh", "Davidson")
        observe(angular, "block_terms", "block_terms")

        def projection_after(value, args, kw):
            check(
                "sparse_projection/" + str(counts["H_projection"]),
                issparse(value) and kw.get("sparse") is True,
            )
            work.setdefault("projections", []).append(
                dict(
                    shape=list(value.shape),
                    nnz=int(value.nnz),
                    bytes=int(
                        value.data.nbytes + value.indices.nbytes + value.indptr.nbytes
                    ),
                )
            )

        observe(angular, "csf_block_matrix", "H_projection", projection_after)

        def table_after(value, args, kw):
            keys_, table, pairs = value
            row = dict(
                raw=len(args[0]),
                keys=len(keys_),
                nnz=int(table.nnz),
                bytes=int(
                    table.data.nbytes + table.indices.nbytes + table.indptr.nbytes
                ),
                pair_bytes=pairs.nbytes,
            )
            work.setdefault("tables", []).append(row)
            check(
                "angular_table_nnz/" + str(counts["angular_table"]),
                row["nnz"],
                4_000_000,
            )
            check(
                "angular_table_total_nnz",
                sum(r["nnz"] for r in work["tables"]),
                40_000_000,
            )
            check(
                "angular_table_CSR_payload",
                sum(r["bytes"] for r in work["tables"]),
                600_000_000,
            )

        observe(angular, "_coefficient_table", "angular_table", table_after)
        raw_original = matrix.build_dirac_matrices

        def raw(*args, **kw):
            attempts["raw_hS"] += 1
            if attempts["raw_hS"] > cap["calls"]["raw_hS"]:
                raise RuntimeError("raw hS cap")
            h, ss = raw_original(*args, **kw)
            counts["raw_hS"] += 1
            check(
                "raw_original_S/" + str(counts["raw_hS"]),
                np.array_equal(ss, metric[int(args[1])]),
            )
            return h, ss

        matrix.build_dirac_matrices = raw
        for module, name, label in (
            (matrix.ConvolCache, "get_rkb_t", "factor_getter"),
            (matrix, "_bspline_pair_potential", "pair_potential"),
            (matrix, "_density_sym", "density"),
            (matrix, "_convol_case1_t", "pair_convolution"),
            (matrix, "_sum_amb_banded", "banded_contraction"),
        ):
            observe(module, name, label)
        original_radial = matrix._radial_evaluator

        def radial(*args, **kw):
            attempts["radial_evaluator"] += 1
            if attempts["radial_evaluator"] > 2:
                raise RuntimeError("radial evaluator cap")
            f = original_radial(*args, **kw)
            counts["radial_evaluator"] += 1

            def evaluate(key):
                attempts["radial_key"] += 1
                if attempts["radial_key"] > cap["calls"]["radial_key"]:
                    raise RuntimeError("radial key cap")
                value = f(key)
                counts["radial_key"] += 1
                return value

            evaluate.cache_info, evaluate.pair_cache_info = (
                f.cache_info,
                f.pair_cache_info,
            )
            return evaluate

        matrix._radial_evaluator = radial
        decoded = 0

        def archive(role):
            nonlocal decoded
            spec = c["arrays"][role]
            with zipfile.ZipFile(paths[role]) as z:
                if (
                    len(z.infolist()) > 100
                    or sum(v.file_size for v in z.infolist()) > spec["uncompressed_cap"]
                ):
                    raise ValueError("archive header/byte cap: " + role)
            result = {}
            with np.load(paths[role], allow_pickle=False) as z:
                if not set(spec["fields"]).issubset(z.files):
                    raise ValueError("archive missing fields")
                for key, field in spec["fields"].items():
                    value = z[key]
                    if (
                        list(value.shape) not in base.field_shapes(field)
                        or value.dtype.str != field["dtype"]
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(
                            "array header/finite identity: " + role + "/" + key
                        )
                    decoded += value.nbytes
                    if decoded > cap["decoded_input_bytes"]:
                        raise ValueError("decoded byte cap")
                    result[key] = value.copy()
            counts["NPZ_load"] += 1
            return result

        dense = {arm: archive(arm + "_R0") for arm in ("A", "B")}
        geometry, sm, cp = (
            archive(key) for key in ("workspace", "saved_S", "checkpoint_arrays")
        )
        data = {
            k[6:]: v.item() if v.ndim == 0 else v
            for k, v in geometry.items()
            if k.startswith("basis_")
        }
        keys = tuple(map(tuple, dense["A"]["keys"].tolist()))
        waves = {
            k: (p, q)
            for k, p, q in zip(keys, dense["A"]["P"], dense["A"]["Q"], strict=True)
        }
        check(
            "same_A_B_keys_PQ",
            all(
                np.array_equal(dense["A"][k], dense["B"][k]) for k in ("keys", "P", "Q")
            ),
        )
        metric = {k: sm["s_" + str(k)] for k in c["fixed"]["kappas"]}
        nucleus = make_nuclear_potential(70, "fermi", A=175)
        binding = asdict(
            CIReferenceBinding.capture(
                [(*waves[k], k[1]) for k in sorted(keys)], {}, data, 70, 123, nucleus
            )
        )
        original_binding = read(paths["checkpoint"])["owner"]["radial_binding"]
        check(
            "early_original_nucleus_radial_geometry", plain(binding) == original_binding
        )
        work["nucleus"] = dict(
            profile=nuclear_profile(70, "fermi", A=175), binding=binding
        )
        layout = tuple(
            (
                tuple(map(tuple, row["keys"])),
                cp["projector_" + str(i)],
                tuple(row["shape"]),
                slice(*row["slice"]),
            )
            for i, row in enumerate(read(paths["layout"]))
        )
        inverse = tuple(tuple(cp["inverse_blocks_" + str(i)]) for i in range(7))
        check(
            "original_metric",
            opt._residual_metric_sha256(layout, cp["inverse_diagonal"], inverse)
            == c["fixed"]["metric_sha256"],
        )
        for group, p, shape, sl in layout:
            cc = np.column_stack([np.concatenate(waves[k]) for k in group])
            ss, yy = metric[group[0][1]], cp["x"][sl].reshape(shape)
            check(
                "R0_P_Gram/" + str(group),
                float(np.max(abs(p.T @ ss @ p - np.eye(p.shape[1])))),
                1e-10,
            )
            check(
                "R0_radial_Gram/" + str(group),
                float(np.max(abs(cc.T @ ss @ cc - np.eye(len(group))))),
                1e-10,
            )
            check(
                "R0_coordinate_wave/" + str(group),
                float(np.max(abs(cc - p @ yy))),
                1e-12,
            )
            leak = cc - p @ (p.T @ ss @ cc)
            check(
                "R0_no_pair/" + str(group),
                float(np.max(np.einsum("ij,ij->j", leak, ss @ leak))),
                1e-12,
            )
        specs, maps = read(paths["models"]), read(paths["basis_maps"])
        sectors = set(map(tuple, specs["selected_sectors"]))
        csfs = {}
        for arm, model in (("A", "baseline"), ("B", "expanded")):
            full = [
                z
                for row in specs[model + "_configs"]
                for z in enumerate_jj_csfs(row["config"])
            ]
            selected = [z for z in full if (z.total_J2, z.parity) in sectors]
            signatures = [
                json.dumps(
                    plain((s.subshells, s.coupling, z.total_J2, z.parity)),
                    separators=(",", ":"),
                )
                for z in selected
                for s in (angular.csf_signature(z),)
            ]
            check(
                arm + "/parent_full_CSFs", len(full) == (3789 if arm == "A" else 5741)
            )
            check(
                arm + "/all_signed_CSFs",
                signatures == maps[model]["selected_signatures"],
            )
            check(arm + "/all69electrons", all(z.n_electrons == 69 for z in full))
            check(
                arm + "/all31keys",
                tuple(sorted(get_shared_orbital_basis(selected))) == keys,
            )
            csfs[arm] = selected

        for arm in ("A", "B"):
            checkpoint()
            oracle = dense[arm]
            oracle_ci = ci_module.CIResult(oracle["energies"], oracle["mixing"])
            energies, columns = ci_level_blocks(csfs[arm], oracle_ci, require_full=True)
            check(
                arm + "/dense_H_symmetric",
                float(np.max(abs(oracle["H"] - oracle["H"].T))),
                1e-12,
            )
            start_h = time.monotonic()
            operator = matrix.build_ci_matrix(
                csfs[arm],
                waves,
                keys,
                data,
                70,
                123,
                nuclear_potential=nucleus,
                include_breit=False,
                as_operator=True,
                max_determinants=2000 if arm == "A" else 2659,
            )
            check(
                arm + "/operator_shape",
                isinstance(operator, LinearOperator)
                and operator.shape == oracle["H"].shape,
            )
            check(arm + "/all5blocks", len(operator.blocks) == 5)
            point = dict(
                operator_build_seconds=time.monotonic() - start_h,
                blocks=[],
                UIDs=[],
                input_binding=binding,
                oracle_SHA256=c["inputs"][arm + "_R0"]["sha256"],
                ownership_scope="new local native LinearOperator bound here; no CIHamiltonianSnapshot fabricated",
            )
            points[arm] = point
            selected_vectors, selected_energies, selected_sectors = [], [], []
            for rows, native_block in operator.blocks:
                sector = (
                    csfs[arm][int(rows[0])].total_J2,
                    csfs[arm][int(rows[0])].parity,
                )
                check(
                    arm + "/complete_sector_rows/" + str(sector),
                    np.array_equal(
                        rows,
                        [
                            i
                            for i, z in enumerate(csfs[arm])
                            if (z.total_J2, z.parity) == sector
                        ],
                    ),
                )
                cols = columns[sector]
                oldc = oracle["mixing"][np.ix_(rows, cols)]
                oldh = oracle["H"][np.ix_(rows, rows)]
                olde = energies[cols]
                row = dict(sector=sector, CSFs=len(rows), full_columns=len(cols))
                point["blocks"].append(row)
                check(
                    arm + "/full_reference_Gram/" + str(sector),
                    float(np.max(abs(oldc.T @ oldc - np.eye(len(rows))))),
                    1e-10,
                )
                check(
                    arm + "/sparse_diagonal/" + str(sector),
                    float(np.max(abs(native_block.diagonal() - np.diag(oldh)))),
                    gates["sparse_diagonal_max_abs_Ha"],
                )
                action_max, residual_max = 0.0, 0.0

                def action(v, native_block=native_block):
                    attempts["operator_action"] += 1
                    if attempts["operator_action"] > cap["calls"]["operator_action"]:
                        raise RuntimeError("operator action cap")
                    result = native_block @ v
                    counts["operator_action"] += 1
                    return result

                for first in range(0, len(cols), gates["batch_columns"]):
                    cc = oldc[:, first : first + gates["batch_columns"]]
                    actual = action(cc)
                    action_max = max(action_max, float(np.max(abs(actual - oldh @ cc))))
                    residual_max = max(
                        residual_max,
                        float(
                            np.max(
                                abs(
                                    actual
                                    - cc * olde[first : first + gates["batch_columns"]]
                                )
                            )
                        ),
                    )
                row.update(
                    whole_reference_action_max_abs_Ha=action_max,
                    full_reference_residual_max_abs_Ha=residual_max,
                )
                check(
                    arm + "/all_column_signed_action/" + str(sector),
                    action_max,
                    gates["whole_action_max_abs_Ha"],
                )
                check(
                    arm + "/all_column_eigen_residual/" + str(sector),
                    residual_max,
                    gates["full_reference_eigenpair_max_abs_Ha"],
                )
                x = np.arange(1, len(rows) + 1, dtype=float)
                x = np.sin(x) / np.linalg.norm(np.sin(x))
                y = np.cos(np.arange(1, len(rows) + 1, dtype=float))
                y /= np.linalg.norm(y)
                check(
                    arm + "/bilinear_symmetry/" + str(sector),
                    float(abs(x @ action(y) - y @ action(x))),
                    gates["bilinear_symmetry_Ha"],
                )
                whole = np.zeros((len(csfs[arm]), 1))
                whole[rows, 0] = x
                applied = operator @ whole
                other = np.ones(len(csfs[arm]), dtype=bool)
                other[rows] = False
                check(
                    arm + "/cross_symmetry/" + str(sector),
                    float(np.max(abs(applied[other]))),
                    0.0,
                )
                bounded_block = LinearOperator(
                    native_block.shape,
                    matvec=action,
                    matmat=action,
                    rmatvec=action,
                    dtype=float,
                )
                bounded_block.diagonal = native_block.diagonal
                start_d = time.monotonic()
                ew, ev, iterations = ci_module.davidson_eigh(
                    bounded_block,
                    n_states=2,
                    tol=gates["Davidson_tol_L2_Ha"],
                    max_iter=gates["Davidson_max_iter"],
                )
                row.update(
                    Davidson_seconds=time.monotonic() - start_d,
                    iterations=iterations,
                    energies_Ha=ew.tolist(),
                    energy_difference_Ha=(ew - olde[:2]).tolist(),
                )
                direct_residual = action(ev) - ev * ew
                row["direct_residual_L2_Ha"] = np.linalg.norm(
                    direct_residual, axis=0
                ).tolist()
                row["direct_residual_max_abs_Ha"] = float(np.max(abs(direct_residual)))
                check(
                    arm + "/Davidson_residual_L2/" + str(sector),
                    float(np.max(np.linalg.norm(direct_residual, axis=0))),
                    gates["Davidson_selected_eigenpair_L2_Ha"],
                )
                check(
                    arm + "/Davidson_residual/" + str(sector),
                    float(np.max(abs(direct_residual))),
                    gates["Davidson_selected_eigenpair_max_abs_Ha"],
                )
                check(
                    arm + "/Davidson_energy/" + str(sector),
                    float(np.max(abs(ew - olde[:2]))),
                    gates["selected_energy_max_abs_Ha"],
                )
                check(
                    arm + "/Davidson_Gram/" + str(sector),
                    float(np.max(abs(ev.T @ ev - np.eye(2)))),
                    gates["selected_Gram"],
                )
                for root in range(2):
                    check(
                        arm + "/isolated_reference_root/" + str((sector, root)),
                        min(abs(olde[root] - np.delete(olde, root))) > 1e-10,
                    )
                    sign = 1.0 if oldc[:, root] @ ev[:, root] >= 0 else -1.0
                    ev[:, root] *= sign
                    check(
                        arm + "/signed_wave/" + str((sector, root)),
                        float(np.max(abs(ev[:, root] - oldc[:, root]))),
                        gates["selected_signed_wave_max_abs"],
                    )
                    check(
                        arm + "/projector/" + str((sector, root)),
                        float(
                            np.max(
                                abs(
                                    np.outer(ev[:, root], ev[:, root])
                                    - np.outer(oldc[:, root], oldc[:, root])
                                )
                            )
                        ),
                        gates["selected_projector_max_abs"],
                    )
                    whole_vector = np.zeros(len(csfs[arm]))
                    whole_vector[rows] = ev[:, root]
                    selected_vectors.append(whole_vector)
                    selected_energies.append(float(ew[root]))
                    selected_sectors.append([*sector, root])
                checkpoint()
            for oldrow in read(paths[arm + "_R0_record"])["rows"]:
                local = selected_sectors.index(oldrow["selected_level"])
                point["UIDs"].append(
                    dict(
                        UID=oldrow["UID"],
                        label=oldrow["label"],
                        selected_level=selected_sectors[local],
                        energy_Ha=selected_energies[local],
                        dense_energy_Ha=oldrow["energy_Ha"],
                        signed_overlap=float(
                            oracle["mixing"][:, oldrow["column"]]
                            @ selected_vectors[local]
                        ),
                    )
                )
            check(
                arm + "/five_owned_UIDs",
                [r["UID"] for r in point["UIDs"]] == [uid for _, uid in c["UIDs"]],
            )
            np.savez_compressed(
                out / (arm + "_selected.npz"),
                energies=np.array(selected_energies),
                mixing=np.column_stack(selected_vectors),
                levels=np.array(selected_sectors),
                keys=np.array(keys),
                P=oracle["P"],
                Q=oracle["Q"],
            )
            counts["NPZ_save"] += 1
            write(out / (arm + ".json"), point)
            del operator
        check("angular_whole_census_queries", queries == allowed)
        check("angular_whole_nullspace_count", counts["angular_nullspace"] == 43)
        for label, exact in (
            ("H", 2),
            ("Davidson", 10),
            ("H_projection", 10),
            ("angular_table", 10),
            ("NPZ_load", 5),
            ("NPZ_save", 2),
        ):
            check("complete_call_count/" + label, counts[label] == exact)
        imported = {}
        for name, module in sorted(sys.modules.items()):
            if name == "atomx" or name.startswith("atomx."):
                path = getattr(module, "__file__", None)
                if path is None:
                    continue
                relative = Path(path).resolve().relative_to(source.resolve())
                p = "src/" + str(relative)
                if (
                    p not in identity["all_atomx_python_sha256"]
                    or sha(path) != identity["all_atomx_python_sha256"][p]
                ):
                    raise ValueError("actual imported source identity: " + name)
                imported[name] = dict(path=p, sha256=sha(path))
        report["actual_imported_sources"] = imported
        metadata()
        report["status"] = "PASS_native_sparse_full_basis_Davidson_fixed_R0"
    except Exception as error:
        report.update(
            status="FAILED_preserved",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
        report.update(
            elapsed_seconds=time.monotonic() - started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            counts=dict(counts),
            attempted=dict(attempts),
        )
        write(out / "result.json", report)
        checkpoint()
    print(
        json.dumps(
            dict(
                status=report["status"],
                checks=len(checks),
                passed=sum(r["PASS"] for r in checks),
                elapsed_seconds=report["elapsed_seconds"],
                peak_RSS_MiB=report["peak_RSS_MiB"],
            )
        )
    )
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute(args.execute))
    c, _, _, identity, _ = metadata()
    print(
        json.dumps(
            dict(
                status="SOURCE_READY",
                frozen_source_files=len(identity["all_atomx_python_sha256"]),
                angular_nullspace_calls=sum(
                    r["nullspace_calls"] for r in c["angular_census"]
                ),
                science_calls=0,
                input_NPZ_loads=0,
                resources=c["resources"],
            )
        )
    )
