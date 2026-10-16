"""One complete native f043 four-point correlation-by-relaxation discriminator.

Default: SOURCE/JSON only. --execute requires the root's exact new seal.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import re
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
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if hasattr(x, "tolist"):
        return plain(x.tolist())
    return x


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def angular_census(configs):
    """Literal SOURCE loop: unique j^q and all nonnegative populated M sectors."""
    pairs = set()
    for config in configs:
        for _, letter, number in re.findall(r"(\d+)([spdfghik])(\d+)", config):
            ell, q = "spdfghik".index(letter), int(number)
            if ell == 0:
                pairs.add((1, q))
            else:
                for qm in range(max(0, q - (2 * ell + 2)), min(q, 2 * ell) + 1):
                    if qm:
                        pairs.add((2 * ell - 1, qm))
                    if q - qm:
                        pairs.add((2 * ell + 1, q - qm))
    rows = []
    for tj, q in sorted(pairs):
        counts = collections.Counter(
            map(sum, itertools.combinations(range(-tj, tj + 1, 2), q))
        )
        sectors = sorted(m for m in counts if m >= 0)
        rows.append(
            dict(
                twoj=tj,
                q=q,
                nonnegative_M2=sectors,
                nullspace_calls=len(sectors),
                maximum_M_basis=max(counts.values()),
            )
        )
    return rows


def metadata():
    c = read(HERE / "contract.json")
    for role, v in c["inputs"].items():
        if v["kind"] == "metadata" and sha(ROOT / v["path"]) != v["sha256"]:
            raise ValueError("metadata binding changed: " + role)
    source = ROOT / c["source_archive"]
    identity = read(ROOT / c["inputs"]["source_identity"]["path"])
    if identity["source_map_sha256"] != c["source_map_sha256"]:
        raise ValueError("f043 source map changed")
    for p, expected in identity["all_atomx_python_sha256"].items():
        if sha(source / p.removeprefix("src/")) != expected:
            raise ValueError("frozen source changed: " + p)
    paths = {k: ROOT / v["path"] for k, v in c["inputs"].items()}
    m, a, decision = (
        read(paths["models"]),
        read(paths["checkpoint"]),
        read(paths["decision"]),
    )
    if (
        len(m["baseline_configs"]) != 91
        or len(m["expanded_configs"]) != 100
        or m["expanded_configs"][:91] != m["baseline_configs"]
        or a["iteration"] != 13
        or a["original_stationarity_passed"] is not False
        or len(a["owner"]["CSFs"]) != 1595
        or len(a["owner"]["orbital_keys"]) != 31
        or a["objective_identity"] != c["fixed"]["objective_identity"]
        or a["metric_sha256"] != c["fixed"]["metric_sha256"]
        or a["geometry_sha256"] != c["fixed"]["geometry_sha256"]
        or decision["experiment"]["predeclared_step"]["alpha"] != c["alpha"]
        or c["alpha"] != 1 / 16
    ):
        raise ValueError("original model/point/metric/step changed")
    ref = read(paths["reference_models"])["models"]["baseline"]["configs"]
    census = angular_census([r["config"] for r in m["expanded_configs"] + ref])
    if census != c["angular_census"] or sum(r["nullspace_calls"] for r in census) != 43:
        raise ValueError("whole SOURCE angular census changed")
    if len(census) >= 512:
        raise ValueError("highest-weight LRU cannot preserve SOURCE census")
    gradient = read(paths["gradient_result"])
    if (
        gradient["status"] != c["gradient_expected_status"]
        or gradient["ordinary_FD_passed"] is not False
        or gradient["stationarity_passed"] is not False
        or gradient["comparison"]["same_metric_sha256"] != c["fixed"]["metric_sha256"]
    ):
        raise ValueError("complete conditional gradient / failed qualification changed")
    if (
        read(paths["saved_response"])["original_parent_byte_gate_remains_failed"]
        is not True
    ):
        raise ValueError("historical nested-byte failure changed")
    return c, paths, source, identity


def execute(seal_path):
    c, paths, source, identity = metadata()
    seal = read(seal_path)
    expected = dict(
        execution_authorized_by_root=True,
        once_only=True,
        driver_sha256=sha(__file__),
        contract_sha256=sha(HERE / "contract.json"),
        resources=c["resources"],
        source_map_sha256=c["source_map_sha256"],
    )
    if any(seal.get(k) != v for k, v in expected.items()):
        raise ValueError("new exact root execution seal required")
    cap = c["resources"]
    if (
        os.path.abspath(sys.executable) != "/data/code/atomx/.venv/bin/python"
        or os.sched_getaffinity(0) != {cap["CPU"]}
        or any(os.environ.get(k) != "1" for k in THREADS)
    ):
        raise ValueError("literal interpreter / CPU / thread admission")
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if os.getloadavg()[0] > 16 or available < 10 * 1024**3:
        raise ValueError("host admission")
    out = HERE / "actual"
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("900s inner stop")),
    )
    signal.alarm(900)
    started = time.monotonic()
    counts, attempts, checks, points, work = (
        collections.Counter({name: 0 for name in cap["calls"]}),
        collections.Counter({name: 0 for name in cap["calls"]}),
        [],
        {},
        {},
    )
    counts["forbidden_science"] = attempts["forbidden_science"] = 0
    report = dict(
        status="RUNNING",
        checks=checks,
        points=points,
        work=work,
        source_map_sha256=c["source_map_sha256"],
        original_failures_preserved=True,
        stationarity_or_FD_qualified=False,
        optimizer_authorized=False,
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

    def observe(module, name, label, after=None):
        original = getattr(module, name)

        def wrapped(*args, **kw):
            attempts[label] += 1
            if attempts[label] > cap["calls"][label]:
                raise RuntimeError("entry cap: " + label)
            value = original(*args, **kw)
            counts[label] += 1
            if after is not None:
                after(value, args, kw)
            return value

        setattr(module, name, wrapped)

    try:
        for role, spec in c["inputs"].items():
            if spec["kind"] == "array" and sha(paths[role]) != spec["sha256"]:
                raise ValueError("whole array-file binding changed: " + role)
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy import linalg

        import atomx.angular.mscheme as mscheme

        # The actual imported alias is observed BEFORE any enumeration/warmup.
        observe(mscheme, "null_space", "angular_nullspace")
        original_highest = mscheme.highest_weight_states
        check(
            "cold_highest_weight_cache_before_models",
            original_highest.cache_info().currsize == 0,
        )
        allowed = {(r["twoj"], r["q"]) for r in c["angular_census"]}
        queries = set()

        def highest(tj, q):
            if (tj, q) not in allowed:
                raise ValueError("unbudgeted highest-weight query: " + str((tj, q)))
            queries.add((tj, q))
            return original_highest(tj, q)

        mscheme.highest_weight_states = highest
        import atomx.angular.csf_hamiltonian as angular
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.overlap as overlap
        import atomx.physics.csf.tracking as tracking
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods import ci as ci_module
        from atomx.numerics.bspline import orbital_optimization as opt
        from atomx.physics.csf.effective_one_body import CIReferenceBinding
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile

        observe(linalg, "eigh", "CI_eigh")
        observe(matrix, "build_ci_matrix", "H")
        observe(ci_module, "ci_solve", "CI")

        def raw_after(value, args, kw):
            h, ss = value
            kappa = int(args[1])
            check(
                "raw_hS_shape/" + str(counts["raw_hS"]),
                h.shape == ss.shape == (246, 246),
            )
            check(
                "raw_S_original_identity/" + str(counts["raw_hS"]),
                np.array_equal(ss, metric[kappa]),
            )

        observe(matrix, "build_dirac_matrices", "raw_hS", raw_after)
        observe(matrix.ConvolCache, "get_rkb_t", "factor_getter")
        observe(matrix, "_bspline_pair_potential", "pair_potential")
        observe(matrix, "_density_sym", "density")
        observe(matrix, "_convol_case1_t", "pair_convolution")
        observe(matrix, "_sum_amb_banded", "banded_contraction")
        observe(overlap, "_projection", "tracking_projection")
        observe(tracking.CIBasisTracking, "resolve", "tracking_resolve")
        observe(tracking, "ci_subspace_overlaps", "tracking_overlap")
        observe(opt, "_polar_retraction", "polar_retraction")

        def forbidden(*args, **kw):
            attempts["forbidden_science"] += 1
            raise RuntimeError("new gradient/optimizer forbidden")

        opt.optimize_orbitals = forbidden
        from atomx.methods.csf import variational

        variational.CSFOrbitalFunctional.gradient_from_weights = forbidden

        def table_after(value, args, kw):
            keys, table, pairs = value
            row = dict(
                raw=len(args[0]),
                keys=len(keys),
                nnz=table.nnz,
                pairs=len(pairs),
                table_bytes=table.data.nbytes
                + table.indices.nbytes
                + table.indptr.nbytes,
                pair_bytes=pairs.nbytes,
            )
            work.setdefault("tables", []).append(row)
            check(
                "table_nnz/" + str(counts["angular_table"]), int(table.nnz), 4_000_000
            )
            check("all_table_nnz", sum(t["nnz"] for t in work["tables"]), 40_000_000)

        observe(angular, "_coefficient_table", "angular_table", table_after)
        observe(angular, "block_terms", "block_terms")
        observe(angular, "csf_block_matrix", "H_projection")
        original_radial = matrix._radial_evaluator

        def radial(*args, **kw):
            attempts["radial_evaluator"] += 1
            if attempts["radial_evaluator"] > 4:
                raise RuntimeError("radial evaluator cap")
            f = original_radial(*args, **kw)
            counts["radial_evaluator"] += 1

            def evaluate(key):
                attempts["radial_key"] += 1
                if attempts["radial_key"] > cap["calls"]["radial_key"]:
                    raise RuntimeError("complete radial-key cap")
                v = f(key)
                counts["radial_key"] += 1
                return v

            evaluate.cache_info = f.cache_info
            evaluate.pair_cache_info = f.pair_cache_info
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
                    raise ValueError("archive header/byte admission: " + role)
            result = {}
            with np.load(paths[role], allow_pickle=False) as z:
                if not set(spec["fields"]).issubset(z.files):
                    raise ValueError("missing declared fields: " + role)
                for key, h in spec["fields"].items():
                    v = z[key]
                    if (
                        list(v.shape) != h["shape"]
                        or v.dtype.str != h["dtype"]
                        or not np.isfinite(v).all()
                    ):
                        raise ValueError(
                            "full field shape/dtype/finite: " + role + "/" + key
                        )
                    decoded += v.nbytes
                    if decoded > 256 * 1024**2:
                        raise ValueError("decoded byte cap")
                    result[key] = v.copy()
            counts["NPZ_load"] += 1
            return result

        baseline, savedB, cp, geometry, sm, reference = (
            archive(k)
            for k in (
                "baseline_arrays",
                "expanded_arrays",
                "checkpoint_arrays",
                "workspace",
                "saved_S",
                "reference_arrays",
            )
        )
        data = {
            k[6:]: v.item() if v.ndim == 0 else v
            for k, v in geometry.items()
            if k.startswith("basis_")
        }
        keys = tuple(map(tuple, baseline["keys"].tolist()))
        waves0 = {
            k: (p, q)
            for k, p, q in zip(keys, baseline["P"], baseline["Q"], strict=True)
        }
        check(
            "expanded_saved_same_PQ_keys",
            all(np.array_equal(baseline[k], savedB[k]) for k in ("keys", "P", "Q")),
        )
        metric = {k: sm["s_" + str(k)] for k in c["fixed"]["kappas"]}
        nucleus = make_nuclear_potential(70, "fermi", A=175)
        binding = asdict(
            CIReferenceBinding.capture(
                [(*waves0[k], k[1]) for k in sorted(keys)], {}, data, 70, 123, nucleus
            )
        )
        original_binding = read(paths["checkpoint"])["owner"]["radial_binding"]
        check(
            "early_original_nuclear_quadrature_SHA256",
            binding["nuclear"] == original_binding["nuclear"],
        )
        check(
            "early_original_radial_geometry_nucleus_binding",
            plain(binding) == original_binding,
        )
        work["early_nuclear_identity"] = dict(
            profile=nuclear_profile(70, "fermi", A=175),
            original_quadrature_SHA256=original_binding["nuclear"],
            actual_quadrature_SHA256=binding["nuclear"],
            sampled_at_original_grid_shape=list(data["gr"].shape),
            scope="native _potential_contents sampled original gr, exact saved A owner nuclear and full radial binding before polar/enumeration/H/CI",
        )
        layout = tuple(
            (
                tuple(map(tuple, r["keys"])),
                cp["projector_" + str(i)],
                tuple(r["shape"]),
                slice(*r["slice"]),
            )
            for i, r in enumerate(read(paths["layout"]))
        )
        inverse = tuple(tuple(cp["inverse_blocks_" + str(i)]) for i in range(7))
        check(
            "original_metric_identity",
            opt._residual_metric_sha256(layout, cp["inverse_diagonal"], inverse)
            == c["fixed"]["metric_sha256"],
        )
        direction = -cp["full_normalized_pg"]
        check("all3689_direction", direction.shape == (3689,))
        work["direction_bytes_SHA256"] = hashlib.sha256(direction.tobytes()).hexdigest()
        work["alpha"] = c["alpha"]
        work["coordinate_step_norm"] = float(c["alpha"] * np.linalg.norm(direction))
        trial = cp["x"].copy()
        for _, _, shape, sl in layout:
            trial[sl] = opt._polar_retraction(
                cp["x"][sl].reshape(shape) + c["alpha"] * direction[sl].reshape(shape)
            ).ravel()

        def unpack(x):
            waves = {}
            for group, p, shape, sl in layout:
                coeff = p @ x[sl].reshape(shape)
                waves.update(
                    (k, (coeff[:123, i].copy(), coeff[123:, i].copy()))
                    for i, k in enumerate(group)
                )
            return waves

        waves1 = unpack(trial)
        for point, x, waves in (("R0", cp["x"], waves0), ("Rdelta", trial, waves1)):
            for group, p, shape, sl in layout:
                s = metric[group[0][1]]
                y = x[sl].reshape(shape)
                cc = np.column_stack([np.concatenate(waves[k]) for k in group])
                check(
                    point + "/P_Gram/" + str(group),
                    float(np.max(abs(p.T @ s @ p - np.eye(p.shape[1])))),
                    1e-10,
                )
                check(
                    point + "/coordinate_Gram/" + str(group),
                    float(np.max(abs(y.T @ y - np.eye(len(group))))),
                    1e-10,
                )
                check(
                    point + "/radial_Gram/" + str(group),
                    float(np.max(abs(cc.T @ s @ cc - np.eye(len(group))))),
                    1e-10,
                )
                check(
                    point + "/coordinate_wave/" + str(group),
                    float(np.max(abs(cc - p @ y))),
                    1e-12,
                )
                leak = cc - p @ (p.T @ s @ cc)
                check(
                    point + "/no_pair/" + str(group),
                    float(np.max(np.einsum("ij,ij->j", leak, s @ leak))),
                    1e-12,
                )
        specs = read(paths["models"])
        sectors = set(map(tuple, specs["selected_sectors"]))
        maps = read(paths["basis_maps"])
        csfs = {}

        def signatures(values):
            return [
                json.dumps(
                    plain((s.subshells, s.coupling, z.total_J2, z.parity)),
                    separators=(",", ":"),
                )
                for z in values
                for s in (angular.csf_signature(z),)
            ]

        for name, model in (("A", "baseline"), ("B", "expanded")):
            full = [
                z
                for row in specs[model + "_configs"]
                for z in enumerate_jj_csfs(row["config"])
            ]
            csfs[name] = [z for z in full if (z.total_J2, z.parity) in sectors]
            check(name + "/parent_CSFs", len(full) == (3789 if name == "A" else 5741))
            check(
                name + "/complete_CSFs",
                signatures(csfs[name]) == maps[model]["selected_signatures"],
            )
            check(name + "/all69electrons", all(z.n_electrons == 69 for z in full))
            check(
                name + "/all31keys",
                tuple(sorted(get_shared_orbital_basis(csfs[name]))) == keys,
            )
        reference_csfs = [
            z
            for r in read(paths["reference_models"])["models"]["baseline"]["configs"]
            for z in enumerate_jj_csfs(r["config"])
        ]
        check("original531_reference", len(reference_csfs) == 531)
        targets = read(paths["tracking_binding"])["targets"]
        average = CIStateAverage(
            tuple(
                (t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.0)
                for t in targets
            )
        )
        refwaves = {
            tuple(k): (p, q)
            for k, p, q in zip(
                geometry["orbital_keys"], geometry["P"], geometry["Q"], strict=True
            )
        }
        tracker = tracking.CIBasisTracking(
            average,
            reference_csfs,
            ci_module.CIResult(reference["energies_Ha"], reference["mixing"]),
            refwaves,
            metric,
            max_determinant_pairs=1_000_000,
        )

        def validate(name, model, ci, h, saved=None):
            energies, blocks = ci_level_blocks(csfs[model], ci, require_full=True)
            for sector, columns in blocks.items():
                rows = [
                    i
                    for i, z in enumerate(csfs[model])
                    if (z.total_J2, z.parity) == sector
                ]
                other = [i for i in range(len(csfs[model])) if i not in rows]
                block = h[np.ix_(rows, rows)]
                cc = ci.mixing_coefficients[np.ix_(rows, columns)]
                check(
                    name + "/residual/" + str(sector),
                    float(np.max(abs(block @ cc - cc * energies[columns]))),
                    1e-8,
                )
                check(
                    name + "/CI_Gram/" + str(sector),
                    float(np.max(abs(cc.T @ cc - np.eye(len(columns))))),
                    1e-10,
                )
                check(
                    name + "/cross_symmetry/" + str(sector),
                    float(np.max(abs(h[np.ix_(rows, other)]))),
                    0.0,
                )
                if saved is not None:
                    oldc = saved["mixing"][np.ix_(rows, columns)]
                    check(
                        name + "/saved_full_column_residual/" + str(sector),
                        float(
                            np.max(
                                abs(block @ oldc - oldc * saved["energies"][columns])
                            )
                        ),
                        1e-8,
                    )
            check(name + "/H_symmetric", float(np.max(abs(h - h.T))), 1e-12)
            return energies

        for point, waves in (("R0", waves0), ("Rdelta", waves1)):
            for arm in ("A", "B"):
                name = arm + "_" + point
                checkpoint()
                maximum = 2000 if arm == "A" else 2659
                ci = ci_module.ci_solve(
                    csfs[arm],
                    waves,
                    keys,
                    data,
                    70,
                    123,
                    solver="dense",
                    n_states=None,
                    nuclear_potential=nucleus,
                    include_breit=False,
                    max_determinants=maximum,
                    retain_hamiltonian=True,
                )
                h = ci.hamiltonian_for(
                    csfs[arm],
                    waves,
                    keys,
                    data,
                    70,
                    123,
                    nuclear_potential=nucleus,
                    max_determinants=maximum,
                )
                saved = (baseline if arm == "A" else savedB) if point == "R0" else None
                energies = validate(name, arm, ci, h, saved)
                if arm == "A" and point == "R0":
                    check(
                        "A_R0/saved_whole_H_replay",
                        float(np.max(abs(h - baseline["H"]))),
                        1e-10,
                    )
                weights, tracked = tracker.resolve(csfs[arm], ci, waves, metric)
                check(
                    name + "/five_equal_weights",
                    np.array_equal(weights[weights != 0], np.repeat(0.2, 5)),
                )
                check(
                    name + "/literal_native_UID_order",
                    plain([t["reference_levels"][0] for t in tracked])
                    == c["fixed"]["native_UID_order"],
                )
                matches = {tuple(t["reference_levels"][0]): t for t in tracked}
                check(
                    name + "/complete_UIDset",
                    set(matches) == {tuple(uid) for _, uid in c["UIDs"]},
                )
                rows = []
                for label, uid in c["UIDs"]:
                    binding = matches[tuple(uid)]
                    check(
                        name + "/rank_one_UID/" + label,
                        len(binding["reference_levels"])
                        == len(binding["selected_levels"])
                        == 1,
                    )
                    chosen = binding["selected_levels"][0]
                    cols = ci_level_blocks(csfs[arm], ci, require_full=True)[1]
                    column = int(cols[tuple(chosen[:2])][chosen[2]])
                    norm = np.longdouble(
                        ci.mixing_coefficients[:, column]
                    ) @ np.longdouble(ci.mixing_coefficients[:, column])
                    check(name + "/actual_norm/" + label, float(abs(norm - 1)), 1e-10)
                    rows.append(
                        dict(
                            label=label,
                            UID=uid,
                            selected_level=chosen,
                            column=column,
                            weight=0.2,
                            actual_norm=str(norm),
                            energy_Ha=float(energies[column]),
                            tracking=binding,
                        )
                    )
                ground = rows[0]["energy_Ha"]
                for row in rows:
                    row["excitation_cm1"] = float(
                        (np.longdouble(row["energy_Ha"]) - np.longdouble(ground))
                        * np.longdouble(HARTREE_CM1)
                    )
                multiplets = {}
                for label, lo, hi, wl, wh in (("D", 1, 2, 4, 6), ("P", 3, 4, 2, 4)):
                    multiplets[label] = dict(
                        centroid_cm1=(
                            wl * rows[lo]["excitation_cm1"]
                            + wh * rows[hi]["excitation_cm1"]
                        )
                        / (wl + wh),
                        splitting_cm1=rows[hi]["excitation_cm1"]
                        - rows[lo]["excitation_cm1"],
                    )
                points[name] = dict(
                    rows=rows,
                    multiplets=multiplets,
                    EOL_Ha=float(weights @ energies),
                    owner=ci.hamiltonian_snapshot.identity_payload(),
                    owner_input_sha256=ci.hamiltonian_snapshot.input_sha256,
                    owner_matrix_sha256=ci.hamiltonian_snapshot.matrix_sha256,
                )
                cache = matrix._SLATER_MEMO.get("_convol_cache")
                seen = set()

                def payload(value, seen=seen):
                    if isinstance(value, np.ndarray):
                        if id(value) in seen:
                            return 0
                        seen.add(id(value))
                        return value.nbytes
                    if isinstance(value, dict):
                        return sum(payload(v) for v in value.values())
                    if isinstance(value, (tuple, list)):
                        return sum(payload(v) for v in value)
                    return 0

                work.setdefault("geometry_cache_object_payload_bytes", {})[name] = (
                    payload(cache._cache)
                )
                check(
                    name + "/geometry_cache_payload",
                    work["geometry_cache_object_payload_bytes"][name],
                    1024**3,
                )
                write(out / (name + ".json"), points[name])
                np.savez_compressed(
                    out / (name + ".npz"),
                    H=h,
                    energies=energies,
                    mixing=ci.mixing_coefficients,
                    keys=np.array(keys),
                    P=np.array([waves[k][0] for k in keys]),
                    Q=np.array([waves[k][1] for k in keys]),
                )
                counts["NPZ_save"] += 1
        interaction = []
        for i in range(5):
            values = {k: p["rows"][i] for k, p in points.items()}

            def line(name, i=i):
                return np.longdouble(
                    points[name]["rows"][i]["energy_Ha"]
                ) - np.longdouble(points[name]["rows"][0]["energy_Ha"])

            interaction.append(
                dict(
                    label=values["A_R0"]["label"],
                    UID=values["A_R0"]["UID"],
                    correlation_R0_cm1=float(
                        (line("B_R0") - line("A_R0")) * HARTREE_CM1
                    ),
                    correlation_Rdelta_cm1=float(
                        (line("B_Rdelta") - line("A_Rdelta")) * HARTREE_CM1
                    ),
                    interaction_cm1=float(
                        (
                            line("B_Rdelta")
                            - line("A_Rdelta")
                            - line("B_R0")
                            + line("A_R0")
                        )
                        * HARTREE_CM1
                    ),
                    absolute_energy_changes_Ha={
                        arm: {
                            label: points[arm + "_Rdelta"]["rows"][ix]["energy_Ha"]
                            - points[arm + "_R0"]["rows"][ix]["energy_Ha"]
                            for label, ix in (("lower", 0), ("upper", i))
                        }
                        for arm in ("A", "B")
                    },
                )
            )
        report["line_interaction"] = interaction
        report["multiplet_interaction"] = {
            label: {
                observable: (
                    points["B_Rdelta"]["multiplets"][label][observable]
                    - points["A_Rdelta"]["multiplets"][label][observable]
                )
                - (
                    points["B_R0"]["multiplets"][label][observable]
                    - points["A_R0"]["multiplets"][label][observable]
                )
                for observable in ("centroid_cm1", "splitting_cm1")
            }
            for label in ("D", "P")
        }
        report["EOL_descent_diagnostic"] = {}
        for arm, action in (("A", c["EOL_actions"]["A"]), ("B", c["EOL_actions"]["B"])):
            delta = points[arm + "_Rdelta"]["EOL_Ha"] - points[arm + "_R0"]["EOL_Ha"]
            rhs = 1e-4 * c["alpha"] * action
            report["EOL_descent_diagnostic"][arm] = dict(
                delta_Ha=delta,
                directional_action_Ha=action,
                Armijo_rhs_Ha=rhs,
                Armijo_observed=delta <= rhs,
                scope="diagnostic only, no FD/optimizer/stationarity certificate",
            )
        report["whole_angular_queries"] = sorted(queries)
        report["highest_weight_cache"] = original_highest.cache_info()._asdict()
        report["decoded_input_bytes"] = decoded
        for key, expected in (
            ("H", 4),
            ("CI", 4),
            ("CI_eigh", 20),
            ("radial_evaluator", 4),
            ("tracking_resolve", 4),
            ("tracking_overlap", 5),
            ("NPZ_load", 6),
            ("NPZ_save", 4),
        ):
            check("exact_complete_calls/" + key, counts[key] == expected)
        check("whole_nullspace_budget", counts["angular_nullspace"], 43)
        report["loaded_source"] = {
            n: sha(m.__file__)
            for n, m in tuple(sys.modules.items())
            if n.startswith("atomx") and getattr(m, "__file__", "").endswith(".py")
        }
        check(
            "actual_loaded_source_origin",
            all(
                Path(sys.modules[n].__file__).resolve().is_relative_to(source)
                for n in report["loaded_source"]
            ),
        )
        metadata()
        report["status"] = (
            "PASS_complete_native_fourpoint_conditional_physical_response"
        )
    except BaseException as error:
        report.update(
            status="FAILED_preserved",
            exception=repr(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        signal.alarm(0)
        report.update(
            elapsed_uncontrolled_s=time.monotonic() - started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            driver_sha256=sha(__file__),
            contract_sha256=sha(HERE / "contract.json"),
            counts=dict(counts),
            attempted=dict(attempts),
        )
        write(out / "result.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", metavar="ROOT_SEAL")
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    else:
        c, _, _, identity = metadata()
        write(
            HERE / "preparation.json",
            dict(
                status="SOURCE_ready_no_science",
                source_files=len(identity["all_atomx_python_sha256"]),
                angular_nullspace_cap=43,
                numerical_or_NPZ_access=0,
            ),
        )
