"""Native f043 five-state, two-scale constrained directional derivative discriminator.

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
from fractions import Fraction
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


def field_shapes(header):
    """Honor exactly one predeclared shape schema; never infer from input."""
    if ("shape" in header) == ("shape_choices" in header):
        raise ValueError("field requires exactly one shape or shape_choices")
    shapes = [header["shape"]] if "shape" in header else header["shape_choices"]
    if (
        not isinstance(shapes, list)
        or not shapes
        or any(
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            for shape in shapes
        )
        or header.get("dtype") not in ("<f8", "<i8")
    ):
        raise ValueError("malformed declared shape/dtype schema")
    return shapes


def metadata():
    c = read(HERE / "contract.json")
    for spec in c["arrays"].values():
        for header in spec["fields"].values():
            field_shapes(header)
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
        or decision["experiment"]["predeclared_step"]["alpha"] != 1 / 16
        or c["steps"] != [0.002, 0.001]
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
    prior = read(paths["prior_fourpoint_result"])
    if (
        prior["status"]
        != "PASS_complete_native_fourpoint_conditional_physical_response"
    ):
        raise ValueError("prior fourpoint closure changed")
    tracking_targets = read(paths["tracking_binding"])["targets"]
    display = [
        [t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]]
        for t in tracking_targets
    ]
    native = read(paths["prior_fourpoint_contract"])["fixed"]["native_UID_order"]
    if (
        display != [uid for _, uid in c["UIDs"]]
        or native != c["fixed"]["native_UID_order"]
    ):
        raise ValueError("literal bound tracker UID display/native order changed")
    for arm in ("A", "B"):
        point = prior["points"][arm + "_R0"]
        if [r["UID"] for r in point["rows"]] != [uid for _, uid in c["UIDs"]]:
            raise ValueError("saved full UID display order changed")
        for row in point["rows"]:
            if row["weight"] != 0.2 or abs(float(row["actual_norm"]) - 1) > 1e-10:
                raise ValueError("saved UID norm/weight changed")
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

    def scientific(name, value, limit):
        checks.append(
            dict(
                name=name,
                observed=float(value),
                limit=float(limit),
                PASS=bool(value <= limit),
                scientific_FD_or_gradient_closure=True,
            )
        )

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
        from atomx.physics.csf import radial_density

        raw_hS = {}
        original_raw_after = raw_after

        def functional_raw_after(value, args, kw):
            original_raw_after(value, args, kw)
            kappa = int(args[1])
            raw_hS[kappa] = tuple(v.copy() for v in value)

        observe(variational, "build_dirac_matrices", "raw_hS", functional_raw_after)
        observe(radial_density.CIRadialDensity, "contract_bilinear", "unfiltered_RDM")
        radial_density.CIRadialDensity.contract = forbidden
        observe(variational, "_convol_case1_semiseparable_t", "gradient_components")
        observe(variational, "_banded_to_full_add", "gradient_band_add")
        observe(variational, "_density_sym", "gradient_density")
        observe(variational.ConvolCache, "get_coulomb_factors", "gradient_factors")
        observe(
            variational.CSFOrbitalFunctional,
            "_pair_potential",
            "gradient_pair_potential",
        )
        original_gradient = variational.CSFOrbitalFunctional.gradient_from_weights

        def gradient(functional, waves, weights, **kw):
            pair_targets, one_body = {}, set()
            term_refs = 0
            for term in weights:
                if term[0] == "h":
                    one_body.update(((term[1], term[2]), (term[3], term[2])))
                else:
                    _, rank, a, b, cc, d = term
                    for left, right, sa, sb in ((a, b, cc, d), (cc, d, a, b)):
                        pair_targets.setdefault(
                            (rank, *sorted((sa, sb))), set()
                        ).update((left, right))
                        term_refs += 1
            check(
                "gradient/complete_key_capacity/" + str(counts["gradient"]),
                len(weights),
                70971,
            )
            check(
                "gradient/sourcepairs/" + str(counts["gradient"]),
                len(pair_targets),
                3472,
            )
            for label, number in (
                ("one_body_MV", len(one_body)),
                ("target_MV", 2 * sum(map(len, pair_targets.values()))),
            ):
                attempts[label] += number
                if attempts[label] > cap["calls"][label]:
                    raise RuntimeError("gradient work cap: " + label)
                counts[label] += number
            work.setdefault("gradient_work", []).append(
                dict(
                    signed_keys=len(weights),
                    negative_keys=sum(v < 0 for v in weights.values()),
                    sourcepairs=len(pair_targets),
                    term_refs=term_refs,
                )
            )
            attempts["gradient"] += 1
            if attempts["gradient"] > 10:
                raise RuntimeError("gradient cap")
            value = original_gradient(functional, waves, weights, **kw)
            counts["gradient"] += 1
            return value

        variational.CSFOrbitalFunctional.gradient_from_weights = gradient

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
            if attempts["radial_evaluator"] > 10:
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
                        list(v.shape) not in field_shapes(h)
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

        baseline, savedB, cp, geometry, sm, reference, saved_gradient = (
            archive(k)
            for k in (
                "baseline_arrays",
                "expanded_arrays",
                "checkpoint_arrays",
                "workspace",
                "saved_S",
                "reference_arrays",
                "gradient_arrays",
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
        work["steps"] = c["steps"]
        work["direction_coordinate_norm"] = float(np.linalg.norm(direction))
        trials = {}
        for label, step in (
            ("plus_h", 0.002),
            ("minus_h", -0.002),
            ("plus_half", 0.001),
            ("minus_half", -0.001),
        ):
            xtrial = cp["x"].copy()
            for _, _, shape, sl in layout:
                xtrial[sl] = opt._polar_retraction(
                    cp["x"][sl].reshape(shape) + step * direction[sl].reshape(shape)
                ).ravel()
            trials[label] = xtrial

        def unpack(x):
            waves = {}
            for group, p, shape, sl in layout:
                coeff = p @ x[sl].reshape(shape)
                waves.update(
                    (k, (coeff[:123, i].copy(), coeff[123:, i].copy()))
                    for i, k in enumerate(group)
                )
            return waves

        wavebanks = {"R0": waves0, **{label: unpack(x) for label, x in trials.items()}}
        for point, x, waves in [("R0", cp["x"], waves0)] + [
            (label, trials[label], wavebanks[label]) for label in trials
        ]:
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

        functionals = {
            arm: variational.CSFOrbitalFunctional(
                csfs[arm],
                data,
                70,
                nucleus,
                state_average=average,
                level_tracking=tracker,
                max_determinants=2000 if arm == "A" else 2659,
                retain_hamiltonian=True,
            )
            for arm in ("A", "B")
        }
        check(
            "gradient_saved_keys",
            np.array_equal(saved_gradient["keys"], baseline["keys"]),
        )
        radial_direction = np.zeros((31, 246))
        key_index = {k: i for i, k in enumerate(keys)}
        for group, projector, shape, sl in layout:
            dc = projector @ direction[sl].reshape(shape)
            for j, key in enumerate(group):
                radial_direction[key_index[key]] = dc[:, j]

        def project(radial):
            raw = np.zeros(3689, dtype=radial.dtype)
            for group, projector, _shape, sl in layout:
                raw[sl] = (
                    projector.astype(radial.dtype).T
                    @ radial[[key_index[k] for k in group]].T
                ).ravel()

            def tangent(v):
                result = v.copy()
                for _, _, shape, sl in layout:
                    y = cp["x"][sl].reshape(shape).astype(v.dtype)
                    a = v[sl].reshape(shape)
                    m = y.T @ a
                    result[sl] = (a - 0.5 * y @ (m + m.T)).ravel()
                return result

            tg = tangent(raw)
            m = np.zeros_like(tg)
            for (_, _, shape, sl), blocks in zip(layout, inverse, strict=True):
                a = tg[sl].reshape(shape)
                m[sl] = np.column_stack(
                    [b.astype(tg.dtype) @ a[:, j] for j, b in enumerate(blocks)]
                ).ravel()
            pg = tangent(m)
            return raw, tg, pg

        gradients = {}

        def state_gradients(arm, ci, rows):
            functional = functionals[arm]
            full = []
            exported = {}
            for row in rows:
                vector = ci.mixing_coefficients[:, row["column"]]
                norm = sum(
                    (Fraction.from_float(float(v)) ** 2 for v in vector), Fraction()
                )
                table = functional.radial_density.contract_bilinear(vector, vector)
                weights = {}
                for key, value in table.items():
                    exact = Fraction.from_float(float(value)) / norm
                    if exact:
                        represented = float(exact)
                        if not np.isfinite(represented) or represented == 0:
                            raise ValueError(
                                "full signed coefficient overflow or underflow"
                            )
                        weights[key] = represented
                check(
                    arm + "/complete_weight_representation/" + row["label"],
                    bool(weights),
                )
                mapping = functional.gradient_from_weights(
                    waves0, weights, record_diagnostics=False
                )
                radial = np.array([mapping[k] for k in keys])
                check(
                    arm + "/full_gradient/" + row["label"],
                    radial.shape == (31, 246) and np.isfinite(radial).all(),
                )
                raw, tg, pg = project(radial)
                orbital = np.sum(
                    radial.astype(np.longdouble)
                    * radial_direction.astype(np.longdouble),
                    axis=1,
                )
                action = np.sum(orbital, dtype=np.longdouble)
                check(
                    arm + "/directional_mapping/" + row["label"],
                    float(abs(action - np.longdouble(tg) @ np.longdouble(direction))),
                    1e-10,
                )
                corrected = radial.astype(np.longdouble).copy()
                vectors = {
                    k: np.concatenate(waves0[k]).astype(np.longdouble) for k in keys
                }
                for term, coefficient in weights.items():
                    if term[0] != "h":
                        continue
                    _, na, kappa, nb = term
                    a, b = (na, kappa), (nb, kappa)
                    h, ss = (v.astype(np.longdouble) for v in raw_hS[kappa])
                    logical = functional.one_body[kappa].astype(np.longdouble)
                    va, vb = vectors[a], vectors[b]
                    sb, hb, hta = ss @ vb, h @ vb, h.T @ va
                    attempts["raw_h_correction_MV"] += 5
                    if (
                        attempts["raw_h_correction_MV"]
                        > cap["calls"]["raw_h_correction_MV"]
                    ):
                        raise RuntimeError("raw-h correction MV cap")
                    counts["raw_h_correction_MV"] += 5
                    n = vb @ sb
                    if not np.isfinite(n) or n <= 0:
                        raise ValueError("raw-h quotient radial norm")
                    numerator = va @ hb
                    corrected[key_index[a]] += coefficient * (hb / n - logical @ vb)
                    corrected[key_index[b]] += coefficient * (
                        hta / n - logical @ va - 2 * numerator * sb / n**2
                    )
                corr_action = np.sum(
                    corrected * radial_direction.astype(np.longdouble),
                    dtype=np.longdouble,
                )
                exported[row["label"]] = dict(
                    UID=row["UID"],
                    column=row["column"],
                    actualnorm_fraction=[str(norm.numerator), str(norm.denominator)],
                    actualnorm_float=float(norm),
                    signed_key_count=len(weights),
                    negative_key_count=sum(v < 0 for v in weights.values()),
                    action_Ha=float(action),
                    raw_h_quotient_action_Ha=float(corr_action),
                    raw_h_quotient_minus_native_action_Ha=float(corr_action - action),
                    reference_metric_diagnostic=dict(
                        raw_norm=float(np.linalg.norm(raw)),
                        tangent_norm=float(np.linalg.norm(tg)),
                        pg_norm=float(np.linalg.norm(pg)),
                        no_energy_error_bound=True,
                    ),
                    orbital_contributions=[
                        dict(key=k, action_Ha=float(v))
                        for k, v in zip(keys, orbital, strict=True)
                    ],
                )
                full.append(radial)
            arrays = np.array(full)
            aggregate = np.sum(arrays * 0.2, axis=0)
            mapped = project(aggregate)
            expected = (
                (
                    cp["full_normalized_radial"],
                    cp["full_normalized_raw_coordinate"],
                    cp["full_normalized_tangent"],
                    cp["full_normalized_pg"],
                )
                if arm == "A"
                else (
                    saved_gradient["full_radial"],
                    saved_gradient["full_raw"],
                    saved_gradient["full_tangent"],
                    saved_gradient["full_preconditioned"],
                )
            )
            for name, got, want, limit in zip(
                ("radial", "raw", "tangent", "pg"),
                (aggregate, *mapped),
                expected,
                (1e-9, 1e-9, 1e-10, 1e-10),
                strict=True,
            ):
                scientific(
                    arm + "/aggregate_saved_closure/" + name,
                    np.max(abs(got - want)),
                    limit,
                )
            aggregate_action = np.sum(
                aggregate.astype(np.longdouble)
                * radial_direction.astype(np.longdouble),
                dtype=np.longdouble,
            )
            scientific(
                arm + "/aggregate_saved_action",
                abs(aggregate_action - c["EOL_actions"][arm]),
                1e-10,
            )
            line_rows = []
            for index, row in enumerate(rows[1:], 1):
                difference = arrays[index] - arrays[0]
                raw, tg, pg = project(difference)
                orbital = np.sum(
                    difference.astype(np.longdouble)
                    * radial_direction.astype(np.longdouble),
                    axis=1,
                )
                line_rows.append(
                    dict(
                        label=row["label"],
                        action_Ha=float(np.sum(orbital, dtype=np.longdouble)),
                        reference_metric_diagnostic=dict(
                            tangent_norm=float(np.linalg.norm(tg)),
                            pg_norm=float(np.linalg.norm(pg)),
                            no_energy_error_bound=True,
                        ),
                        orbital_contributions=[
                            dict(key=k, action_Ha=float(v))
                            for k, v in zip(keys, orbital, strict=True)
                        ],
                    )
                )
            gradients[arm] = dict(
                states=exported,
                lines=line_rows,
                EOL_action_Ha=float(aggregate_action),
                normalization="full signed contract_bilinear; exact Fraction cTc norm/coefficient then float once; no magnitude filter",
            )
            np.savez_compressed(
                out / (arm + "_R0_state_gradients.npz"),
                keys=np.array(keys),
                state_radial=arrays,
                radial_direction=radial_direction,
                aggregate_radial=aggregate,
                aggregate_raw=mapped[0],
                aggregate_tangent=mapped[1],
                aggregate_pg=mapped[2],
            )
            counts["NPZ_save"] += 1
            write(out / (arm + "_R0_state_gradients.json"), gradients[arm])

        for point, waves in wavebanks.items():
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
                if point == "R0":
                    state_gradients(arm, ci, rows)
                    checkpoint()
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
        report["state_gradients"] = gradients
        finite = []
        wide = np.longdouble
        for arm in ("A", "B"):
            analytic = [
                gradients[arm]["states"][label]["action_Ha"] for label, _ in c["UIDs"]
            ]
            baseline_E = [wide(r["energy_Ha"]) for r in points[arm + "_R0"]["rows"]]
            derivatives = {}
            for tag, plus, minus, step in (
                ("h", "plus_h", "minus_h", 0.002),
                ("half", "plus_half", "minus_half", 0.001),
            ):
                ep = [wide(r["energy_Ha"]) for r in points[arm + "_" + plus]["rows"]]
                em = [wide(r["energy_Ha"]) for r in points[arm + "_" + minus]["rows"]]
                d = [(p - m) / (2 * wide(step)) for p, m in zip(ep, em, strict=True)]
                derivatives[tag] = d
                for i, (label, uid) in enumerate(c["UIDs"]):
                    tol = max(1e-8, 1e-5 * abs(analytic[i]))
                    error = abs(d[i] - analytic[i])
                    scientific(
                        arm + "/ordinaryFD/" + tag + "/state/" + label, error, tol
                    )
                    finite.append(
                        dict(
                            arm=arm,
                            step=step,
                            kind="state",
                            label=label,
                            UID=uid,
                            analytic_Ha=analytic[i],
                            central_Ha=float(d[i]),
                            error_Ha=float(error),
                            tolerance_Ha=tol,
                            PASS=bool(error <= tol),
                            plus_minus_R0_Ha=[
                                float(ep[i] - baseline_E[i]),
                                float(em[i] - baseline_E[i]),
                            ],
                        )
                    )
                for i, (label, uid) in enumerate(c["UIDs"]):
                    if i == 0:
                        continue
                    action = analytic[i] - analytic[0]
                    delta = d[i] - d[0]
                    tol = max(1e-8, 1e-5 * abs(action))
                    error = abs(delta - action)
                    scientific(
                        arm + "/ordinaryFD/" + tag + "/line/" + label, error, tol
                    )
                    finite.append(
                        dict(
                            arm=arm,
                            step=step,
                            kind="line",
                            label=label,
                            UID=uid,
                            analytic_Ha=action,
                            central_Ha=float(delta),
                            error_Ha=float(error),
                            tolerance_Ha=tol,
                            PASS=bool(error <= tol),
                            upper_plus_minus_R0_Ha=[
                                float(ep[i] - baseline_E[i]),
                                float(em[i] - baseline_E[i]),
                            ],
                            lower_plus_minus_R0_Ha=[
                                float(ep[0] - baseline_E[0]),
                                float(em[0] - baseline_E[0]),
                            ],
                        )
                    )
                weighted_state_fd = sum(wide(0.2) * v for v in d)
                da = (
                    wide(points[arm + "_" + plus]["EOL_Ha"])
                    - wide(points[arm + "_" + minus]["EOL_Ha"])
                ) / (2 * wide(step))
                derivatives[tag + "_EOL"] = da
                aa = gradients[arm]["EOL_action_Ha"]
                tol = max(1e-8, 1e-5 * abs(aa))
                scientific(arm + "/ordinaryFD/" + tag + "/EOL", abs(da - aa), tol)
                finite.append(
                    dict(
                        arm=arm,
                        step=step,
                        kind="EOL",
                        scope="ordinary reported binary64 weights@energies callback difference",
                        weighted_state_FD_diagnostic_Ha=float(weighted_state_fd),
                        analytic_Ha=aa,
                        central_Ha=float(da),
                        error_Ha=float(abs(da - aa)),
                        tolerance_Ha=tol,
                        PASS=bool(abs(da - aa) <= tol),
                    )
                )
            rr = [
                (4 * b - a) / 3
                for a, b in zip(derivatives["h"], derivatives["half"], strict=True)
            ]
            report.setdefault("Richardson_diagnostic", {})[arm] = dict(
                scope="predeclared separate diagnostic; cannot pass ordinary FD gate",
                state_Ha=list(map(float, rr)),
                state_minus_analytic_Ha=[
                    float(v - a) for v, a in zip(rr, analytic, strict=True)
                ],
                line_Ha=[float(v - rr[0]) for v in rr[1:]],
                EOL_Ha=float((4 * derivatives["half_EOL"] - derivatives["h_EOL"]) / 3),
                weighted_state_EOL_Ha=float(sum(wide(0.2) * v for v in rr)),
            )
        report["finite_directional_response"] = finite
        report["all_eight_displaced_points_completed"] = len(points) == 10
        report["ordinary_FD_passed"] = all(row["PASS"] for row in finite)
        report["gradient_closure_passed"] = all(
            q["PASS"]
            for q in checks
            if q.get("scientific_FD_or_gradient_closure")
            and "aggregate_saved" in q["name"]
        )
        report["whole_angular_queries"] = sorted(queries)
        report["highest_weight_cache"] = original_highest.cache_info()._asdict()
        report["decoded_input_bytes"] = decoded
        for key, expected in (
            ("H", 10),
            ("CI", 10),
            ("CI_eigh", 50),
            ("radial_evaluator", 10),
            ("tracking_resolve", 10),
            ("tracking_overlap", 11),
            ("NPZ_load", 7),
            ("NPZ_save", 12),
            ("gradient", 10),
            ("unfiltered_RDM", 10),
            ("polar_retraction", 28),
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
        passed = report["ordinary_FD_passed"] and report["gradient_closure_passed"]
        report["status"] = (
            "PASS_complete_five_state_directional_qualification"
            if passed
            else "FAILED_complete_five_state_directional_qualification_preserved"
        )
        return 0 if passed else 1
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
        raise SystemExit(execute(args.execute))
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
