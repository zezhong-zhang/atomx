"""Conditional saved2574 full-gradient discriminator; SOURCE-only by default."""

import argparse
import ast
import hashlib
import json
import math
import os
import resource
import signal
import struct
import sys
import time
import traceback
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def numpy_scalar_json(value):
    """Export scalar metadata exactly; reject arrays/unsupported scalar precision."""
    if (
        type(value).__module__.split(".")[0] == "numpy"
        and getattr(value, "ndim", None) == 0
        and callable(getattr(value, "item", None))
    ):
        scalar = value.item()
        if type(scalar) in (bool, int, float, str) or scalar is None:
            return scalar
    raise TypeError("Unsupported JSON metadata type: " + type(value).__name__)


def uid_json(value):
    """Canonical JSON container only; no permutation or component conversion."""
    return list(value)


def write(p, value):
    p.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=numpy_scalar_json) + "\n"
    )


def tuples(x):
    return tuple(tuples(v) for v in x) if isinstance(x, list) else x


def metadata():
    c = read(HERE / "contract.json")
    previous = read(ROOT / c["input_roles"]["predecessor_receipt"]["path"])
    assert previous["status"] == "foreground_closed" and previous["exit_code"] == 1
    assert previous["source_and_inputs_unchanged"] and not previous["outer_wall_stop"]
    assert previous["frozen_guard_files"] == 397 and previous["failure_sha256"] is None
    assert (
        previous["execution_log_sha256"]
        == c["input_roles"]["predecessor_log"]["sha256"]
    )
    assert c["manual_successor"][
        "physics_gradient_metric_model_weights_source_epoch_gates_caps_unchanged"
    ]
    assert not c["manual_successor"]["automatic_retry"]
    paths = {k: ROOT / v["path"] for k, v in c["input_roles"].items()}
    predecessor_log = paths["predecessor_log"].read_text()
    assert (
        "native_tracker_order" in predecessor_log
        and "int64" in predecessor_log
        and "not JSON serializable" in predecessor_log
    )
    total = 0
    for k, row in c["input_roles"].items():
        if row["kind"] == "metadata":
            total += paths[k].stat().st_size
            assert sha(paths[k]) == row["sha256"], k
    assert total <= c["resources"]["metadata_bytes_max"]
    identity = read(paths["source_identity"])
    assert identity["source_map_sha256"] == c["source_map_sha256"]
    assert len(identity["all_atomx_python_sha256"]) == 356
    source = ROOT / c["source_archive"]
    assert all(
        sha(source / p.removeprefix("src/")) == h
        for p, h in identity["all_atomx_python_sha256"].items()
    )
    readiness = read(paths["readiness"])
    assert not readiness["readiness"]["owned_snapshot_available"]
    assert len(readiness["deferred_numerical_roles"]) == 6
    for k, row in readiness["deferred_numerical_roles"].items():
        role = c["readiness_array_mapping"][k]
        assert c["input_roles"][role]["path"] == row["path"]
        assert c["input_roles"][role]["sha256"] == row["sha256_from_closed_JSON"]
    checkpoint = read(paths["checkpoint"])
    assert checkpoint["iteration"] == 13 and checkpoint["qualified"]
    assert checkpoint["source_map_sha256"] == c["source_map_sha256"]
    assert not checkpoint["original_stationarity_passed"]
    assert checkpoint["metric_sha256"] == c["fixed"]["metric_sha256"]
    assert checkpoint["geometry_sha256"] == c["fixed"]["geometry_sha256"]
    assert checkpoint["array_sha256"] == c["input_roles"]["checkpoint_arrays"]["sha256"]
    assert [r["UID"] for r in checkpoint["rows"]] == c["fixed"]["UIDs"]
    assert [r["weight"] for r in checkpoint["rows"]] == [0.2] * 5
    models, maps = read(paths["models"]), read(paths["basis_maps"])
    assert (
        len(models["baseline_configs"]) == 91 and len(models["expanded_configs"]) == 100
    )
    assert models["expanded_configs"][:91] == models["baseline_configs"]
    for name, parent, selected in [("baseline", 3789, 1595), ("expanded", 5741, 2574)]:
        assert len(maps[name]["full_parent_signatures"]) == parent
        assert len(maps[name]["selected_signatures"]) == selected
        assert len(maps[name]["selected_parent_indices"]) == selected
    tables = read(paths["angular_tables"])
    assert len(tables) == 5 and sum(r["nnz"] for r in tables) == 15298557
    assert sum(r["pairs"] for r in tables) == 2141710
    diag, receipt = read(paths["saved_failure"]), read(paths["saved_receipt"])
    assert receipt["status"] == "foreground_closed" and receipt["exit_code"] == 1
    assert receipt["source_and_inputs_unchanged"] and not receipt["outer_wall_stop"]
    assert receipt["failure_sha256"] == c["input_roles"]["saved_failure"]["sha256"]
    assert len(diag["checks"]) == 57 and sum(r["passed"] for r in diag["checks"]) == 56
    assert [r["name"] for r in diag["checks"] if not r["passed"]] == [
        "no_H_radial_or_MCDHF_producer_module_imports"
    ]
    assert diag["completed"]["complete_block_eigh"] == 5
    assert not diag["source_or_model_qualified"]
    assert (
        next(
            a["sha256"]
            for a in diag["archives"]
            if a["path"] == "saved_complete2574_spectrum.npz"
        )
        == c["input_roles"]["expanded_arrays"]["sha256"]
    )
    provenance = read(paths["saved_provenance"])
    assert provenance["n_csf"] == 2574 and provenance["hamiltonian_snapshot"] is None
    core = read(paths["core_failure"])
    assert (
        len(core["checks"]) == 158 and sum(r["passed"] for r in core["checks"]) == 157
    )
    assert [r["name"] for r in core["checks"] if not r["passed"]] == [
        "expanded_exact_nested1595_H_bytes"
    ]
    assert not core["scientific_success_claim"] and not core["expanded_MCDHF_optimized"]
    st = read(paths["saved_tracking"])
    assert (
        len(st["weights"]) == 2574 and [v for v in st["weights"] if v > 0] == [0.2] * 5
    )
    assert [r["reference_levels"][0] for r in st["tracked"]] == c["fixed"][
        "tracker_native_UID_order"
    ]
    assert set(map(tuple, c["fixed"]["tracker_native_UID_order"])) == set(
        map(tuple, c["fixed"]["UIDs"])
    )
    assert all(
        r["reference_levels"] == r["selected_levels"]
        and len(r["selected_levels"]) == 1
        and r["per_level_weight"] == 0.2
        for r in st["tracked"]
    )
    lo = read(paths["layout"])
    assert [r["keys"][0][1] for r in lo] == c["fixed"]["kappas"]
    assert [len(r["keys"]) for r in lo] == [2, 4, 6, 7, 6, 4, 2]
    assert all(r["shape"] == [119, len(r["keys"])] for r in lo)
    assert lo[0]["slice"][0] == 0 and lo[-1]["slice"][1] == 3689
    assert all(r["slice"][1] - r["slice"][0] == 119 * len(r["keys"]) for r in lo)
    assert all(lo[i]["slice"][1] == lo[i + 1]["slice"][0] for i in range(6))
    assert checkpoint["owner"]["max_determinants"] == 2000
    assert checkpoint["owner"]["hamiltonian_terms"] == ["dirac-coulomb"]
    assert len(checkpoint["owner"]["CSFs"]) == 1595
    assert checkpoint["owner"]["orbital_keys"] == sorted(
        k for r in lo for k in r["keys"]
    )
    stage = read(paths["stage_result"])
    assert (
        len(stage["checks"]) == 745 and sum(r["passed"] for r in stage["checks"]) == 743
    )
    forward, fr = read(paths["forward_result"]), read(paths["forward_receipt"])
    assert len(forward["checks"]) == 98 and all(r["passed"] for r in forward["checks"])
    assert fr["status"] == "foreground_closed" and fr["exit_code"] == 0
    assert fr["source_and_inputs_unchanged"] and not fr["outer_wall_stop"]
    assert fr["result_sha256"] == c["input_roles"]["forward_result"]["sha256"]
    assert not forward["ordinary_FD_passed"] and not forward["gradient_repaired"]
    assert not forward["optimizer_qualified"]
    assert len(c["array_access"]) == 6
    assert len(c["array_access"]["workspace"]["fields"]) == 18
    assert set(c["array_access"]) == {
        k for k, v in c["input_roles"].items() if v["kind"] == "array"
    }
    for row in c["array_access"].values():
        assert row["fields"] and row["compressed_bytes_max"] > 0
        assert set(row["fields"]).isdisjoint(row.get("extras", {}))
    return c, paths, source


def execute(binding_path):
    c, paths, source = metadata()
    binding = read(binding_path)
    assert binding["execution_authorized_by_root"] is True
    assert binding["driver_sha256"] == sha(HERE / "run.py")
    assert binding["contract_sha256"] == sha(HERE / "contract.json")
    assert binding["preparation_sha256"] == sha(HERE / "preparation.json")
    assert binding["handoff_sha256"] == sha(HERE / "handoff.json")
    assert binding["input_roles"] == c["input_roles"]
    assert binding["resources"] == c["resources"]
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    counts, attempted, checks, work = Counter(), Counter(), [], {}
    started = time.monotonic()
    caps = c["resources"]
    phase = "root binding and resource admission"
    decoded = 0
    np = None
    original_files = {}

    def progress(label):
        nonlocal phase
        phase = label
        write(
            output / "progress.json",
            {
                "phase": phase,
                "counts": dict(counts),
                "attempted": dict(attempted),
                "elapsed_uncontrolled_s": time.monotonic() - started,
            },
        )

    def check(name, condition, value=None, limit=None):
        row = {"name": name, "passed": bool(condition)}
        if value is not None:
            row["value"] = value
        if limit is not None:
            row["limit"] = limit
        checks.append(row)
        if not condition:
            raise RuntimeError("Original/fixed control failed: " + name)

    def count(name, n=1):
        attempted[name] += n
        if attempted[name] > caps["calls"][name]:
            raise RuntimeError("Whole call cap: " + name)

    def observed(name, fn, after=None):
        def call(*a, **kw):
            count(name)
            value = fn(*a, **kw)
            counts[name] += 1
            if after is not None:
                after(value, a, kw)
            return value

        return call

    def forbidden(name):
        def call(*a, **kw):
            count(name)
            raise RuntimeError("Forbidden producer entered: " + name)

        return call

    def origins():
        identity = read(paths["source_identity"])["all_atomx_python_sha256"]
        rows = {}
        for name, module in tuple(sys.modules.items()):
            if name != "atomx" and not name.startswith("atomx."):
                continue
            f = getattr(module, "__file__", None)
            if f and str(f).endswith(".py"):
                p = Path(f).resolve()
                rel = p.relative_to(source).as_posix()
                expected = identity["src/" + rel]
                check("module_origin/" + name, sha(p) == expected)
                rows[name] = {"path": rel, "sha256": expected}
        write(output / "imported_module_origins.json", rows)

    def headers(role):
        plan, p = c["array_access"][role], paths[role]
        check(
            role + "/compressed_cap", p.stat().st_size <= plan["compressed_bytes_max"]
        )
        result = {}
        allowed = dict(plan.get("extras", {}), **plan["fields"])
        with zipfile.ZipFile(p) as z:
            entries = z.infolist()
            check(
                role + "/whole_fields",
                len(entries) == len(allowed)
                and len({v.filename for v in entries}) == len(entries),
            )
            for info in entries:
                check(
                    role + "/header_name",
                    info.filename.endswith(".npy") and info.filename[:-4] in allowed,
                )
                name, spec = info.filename[:-4], allowed[info.filename[:-4]]
                with z.open(info) as f:
                    check(role + "/" + name + "/magic", f.read(6) == b"\x93NUMPY")
                    version = f.read(2)
                    check(
                        role + "/" + name + "/version",
                        version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"),
                    )
                    size = struct.unpack(
                        "<H" if version[0] == 1 else "<I",
                        f.read(2 if version[0] == 1 else 4),
                    )[0]
                    check(role + "/" + name + "/header_size", size <= 65536)
                    header = ast.literal_eval(
                        f.read(size).decode("utf8" if version[0] == 3 else "latin1")
                    )
                shape = list(header["shape"])
                dtype = header["descr"]
                check(role + "/" + name + "/dtype", dtype == spec["dtype"])
                if "shape" in spec:
                    check(role + "/" + name + "/shape", shape == spec["shape"])
                else:
                    check(
                        role + "/" + name + "/shape_choices",
                        shape in spec["shape_choices"],
                    )
                check(
                    role + "/" + name + "/layout", type(header["fortran_order"]) is bool
                )
                itemsize = int(dtype[2:])
                size_data = math.prod(shape) * itemsize
                check(
                    role + "/" + name + "/stored_size",
                    info.file_size
                    == 6 + 2 + (2 if version[0] == 1 else 4) + size + size_data,
                )
                result[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": size_data,
                    "fortran_order": header["fortran_order"],
                }
        return result

    def archive(role):
        nonlocal decoded
        count("NPZ_load")
        header = headers(role)
        fields = c["array_access"][role]["fields"]
        declared = sum(header[k]["bytes"] for k in fields)
        check(
            role + "/decoded_bound_before_load",
            decoded + declared <= caps["decoded_bytes_max"],
        )
        arrays = {}
        with np.load(paths[role], allow_pickle=False) as z:
            for name in fields:
                a = z[name]
                check(
                    role + "/" + name + "/actual_header",
                    list(a.shape) == header[name]["shape"]
                    and a.dtype.str == header[name]["dtype"]
                    and a.nbytes == header[name]["bytes"],
                )
                check(role + "/" + name + "/finite", bool(np.isfinite(a).all()))
                arrays[name] = a.copy(order="K")
                decoded += a.nbytes
        counts["NPZ_load"] += 1
        write(
            output / ("input_" + role + "_decoded.json"),
            {
                "fields": header,
                "selected_fields": list(fields),
                "cumulative_decoded_bytes": decoded,
            },
        )
        return arrays

    try:
        assert 0 <= caps["CPU"] < 32
        available = os.sched_getaffinity(0)
        check("resource/CPU_available", caps["CPU"] in available)
        available_bytes = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        load = os.getloadavg()[0]
        admission = {
            "CPU": caps["CPU"],
            "load1": load,
            "MemAvailable_bytes": available_bytes,
        }
        check("resource/host_load", load <= caps["maximum_system_load1"])
        check(
            "resource/host_available",
            available_bytes >= caps["minimum_host_available_GiB"] * 1024**3,
        )
        os.sched_setaffinity(0, {caps["CPU"]})
        resource.setrlimit(resource.RLIMIT_AS, (caps["AS_GiB"] * 1024**3,) * 2)

        def expired(*a):
            raise TimeoutError("Declared inner wall stop")

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(caps["inner_seconds"])
        for name in c["thread_environment"]:
            os.environ[name] = "1"
        write(output / "resource_admission.json", admission)
        original_files = {
            p: sha(p)
            for p in [
                HERE / "run.py",
                HERE / "contract.json",
                HERE / "preparation.json",
                HERE / "handoff.json",
                Path(binding_path),
                *paths.values(),
            ]
        }
        sys.path.insert(0, str(source))
        import numpy as numpy

        np = numpy
        scalar_probe = {
            "integer": np.int64(-7),
            "boolean": np.bool_(True),
            "false": np.bool_(False),
        }
        scalar_export = json.loads(
            json.dumps(scalar_probe, allow_nan=False, default=numpy_scalar_json)
        )
        check(
            "JSON/native_scalar_export",
            scalar_export == {"integer": -7, "boolean": True, "false": False}
            and type(scalar_export["integer"]) is int
            and type(scalar_export["boolean"]) is bool,
        )
        check(
            "UID/native_tuple_JSON_representation",
            uid_json((1, 1, 0)) == c["fixed"]["tracker_native_UID_order"][0]
            and uid_json((1, -1, 0)) != c["fixed"]["tracker_native_UID_order"][0],
        )
        import scipy.linalg as linalg

        from atomx.angular import csf_hamiltonian as angular
        from atomx.angular import mscheme
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.methods import ci as ci_module
        from atomx.methods.csf import variational
        from atomx.numerics.bspline import convol, dirac_assembly, orbital_optimization
        from atomx.physics.csf import matrix, radial_density, tracking
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        np.linalg.eigh = forbidden("eigh")
        np.linalg.eigvalsh = forbidden("eigh")
        linalg.eigh = forbidden("eigh")
        ci_module.ci_solve = forbidden("CI")
        variational.ci_solve = forbidden("CI")
        matrix.build_ci_matrix = forbidden("H")
        orbital_optimization.optimize_orbitals = forbidden("optimizer")
        orbital_optimization.OrbitalOptimizationState.from_arrays = classmethod(
            forbidden("optimizer_state")
        )
        dirac_assembly.solve_dirac = forbidden("Dirac_spectrum")
        if hasattr(orbital_optimization, "solve_dirac"):
            orbital_optimization.solve_dirac = forbidden("Dirac_spectrum")
        origins()
        progress("complete six archived inputs")
        base, expanded, checkpoint, workspace, ref, saved_s = [
            archive(k)
            for k in [
                "baseline_arrays",
                "expanded_arrays",
                "checkpoint_arrays",
                "workspace",
                "reference_arrays",
                "saved_S",
            ]
        ]
        data = {
            k[6:]: v.item() if v.ndim == 0 else v
            for k, v in workspace.items()
            if k.startswith("basis_")
        }
        check(
            "complete_basis_scalars",
            data["ns"] == 123
            and data["nv"] == 115
            and data["ks"] == 9
            and data["ksp"] in (8, 9)
            and data["ksq"] == 9,
        )
        check(
            "complete_basis_shapes",
            data["gr"].shape == data["grw"].shape
            and data["bsp_p"].shape
            == data["bsp_p_d"].shape
            == (115, data["gr"].shape[1], data["ksp"])
            and data["bsp_q"].shape
            == data["bsp_q_d"].shape
            == (115, data["gr"].shape[1], 9)
            and data["tp"].shape == (data["nsp"] + data["ksp"],)
            and data["t"].shape == (132,),
        )
        check(
            "basis_active_extent",
            data["nsp"] == 115 + data["ksp"] - 1 and data["nsq"] == 123,
        )
        keys = tuple(map(tuples, base["keys"].tolist()))
        check(
            "all31keys",
            len(keys) == len(set(keys)) == 31 and keys == tuple(sorted(keys)),
        )
        waves = {k: (p, q) for k, p, q in zip(keys, base["P"], base["Q"], strict=True)}
        for name in ("keys", "P", "Q"):
            check(
                "same_literal13_expanded_" + name,
                np.array_equal(base[name], expanded[name]),
            )
        models, maps = read(paths["models"]), read(paths["basis_maps"])
        csfs_by_model = {}
        for model in ("baseline", "expanded"):
            parent = [
                x
                for row in models[model + "_configs"]
                for x in enumerate_jj_csfs(row["config"])
            ]
            check(
                model + "/full_parent_signatures",
                [
                    json.dumps(
                        (
                            angular.csf_signature(x).subshells,
                            angular.csf_signature(x).coupling,
                            x.total_J2,
                            x.parity,
                        ),
                        separators=(",", ":"),
                    )
                    for x in parent
                ]
                == maps[model]["full_parent_signatures"],
            )
            selected = [parent[i] for i in maps[model]["selected_parent_indices"]]
            check(
                model + "/selected_signatures",
                [
                    json.dumps(
                        (
                            angular.csf_signature(x).subshells,
                            angular.csf_signature(x).coupling,
                            x.total_J2,
                            x.parity,
                        ),
                        separators=(",", ":"),
                    )
                    for x in selected
                ]
                == maps[model]["selected_signatures"],
            )
            check(
                model + "/shared_keys",
                tuple(sorted(get_shared_orbital_basis(selected))) == keys,
            )
            check(
                model + "/electron_charge", all(x.n_electrons == 69 for x in selected)
            )
            csfs_by_model[model] = selected
        csfs = csfs_by_model["expanded"]
        check("expanded2574", len(csfs) == 2574)
        check(
            "literal_old_embedding",
            expanded["old_to_expanded"].tolist()
            == read(paths["nested_embedding"])["old_to_expanded"],
        )
        check(
            "saved_legacy_owner_absent",
            read(paths["saved_provenance"])["hamiltonian_snapshot"] is None,
        )
        # Legacy CIResult is explicitly only a conditional gradient input.
        saved_ci = ci_module.CIResult(
            expanded["energies"],
            expanded["mixing"],
            provenance={
                "diagnostic": "CONDITIONAL_SAVED2574_GRADIENT",
                "owned_snapshot": False,
            },
        )
        check("no_owner_created", saved_ci.hamiltonian_snapshot is None)
        metric = {k: saved_s["s_" + str(k)] for k in c["fixed"]["kappas"]}
        reference_configs = read(paths["reference_models"])["models"]["baseline"][
            "configs"
        ]
        reference_csfs = [
            x for row in reference_configs for x in enumerate_jj_csfs(row["config"])
        ]
        check("original531_reference", len(reference_csfs) == 531)
        reference_waves = {
            tuples(k): (p, q)
            for k, p, q in zip(
                workspace["orbital_keys"].tolist(),
                workspace["P"],
                workspace["Q"],
                strict=True,
            )
        }
        average = CIStateAverage(
            tuple((uid[0], uid[1], uid[2], 1.0) for uid in c["fixed"]["UIDs"])
        )
        tracking.CIBasisTracking.resolve = observed(
            "tracker_resolve", tracking.CIBasisTracking.resolve
        )
        tracking.ci_subspace_overlaps = observed(
            "tracker_overlap", tracking.ci_subspace_overlaps
        )
        original_null = mscheme.null_space
        mscheme.null_space = observed("angular_nullspace", original_null)
        angular.block_terms = observed("angular_block_terms", angular.block_terms)
        angular.csf_block_matrix = observed(
            "angular_projection", angular.csf_block_matrix
        )
        tracker = tracking.CIBasisTracking(
            average,
            reference_csfs,
            ci_module.CIResult(ref["energies_Ha"], ref["mixing"]),
            reference_waves,
            metric,
            max_determinant_pairs=caps["max_signed_overlap_pairs"],
        )
        raw_hS = {}

        def raw_after(value, a, kw):
            kappa = int(a[1])
            h, s = value
            check("raw_hS_unique/" + str(kappa), kappa not in raw_hS)
            check(
                "S_original_bytes/" + str(kappa), s.tobytes() == metric[kappa].tobytes()
            )
            raw_hS[kappa] = (h.copy(), s.copy())

        variational.build_dirac_matrices = observed(
            "raw_hS", variational.build_dirac_matrices, raw_after
        )
        radial_density.CIRadialDensity.contract = observed(
            "retained_RDM", radial_density.CIRadialDensity.contract
        )
        radial_density.CIRadialDensity.contract_bilinear = observed(
            "unfiltered_RDM", radial_density.CIRadialDensity.contract_bilinear
        )
        variational._convol_case1_semiseparable_t = observed(
            "gradient_components", variational._convol_case1_semiseparable_t
        )
        variational._banded_to_full_add = observed(
            "gradient_band_add", variational._banded_to_full_add
        )
        variational._convol_case1_t = forbidden("full_bank_convolution")
        convol.ConvolCache.get_rkb_t = forbidden("full_bank_getter")
        original_factors = convol.ConvolCache.get_coulomb_factors

        def factors(self, rank, component="pppp"):
            missed = (rank, component, "semiseparable") not in self._cache
            if missed:
                count("factor_miss")
            result = observed("factor_getter", original_factors)(self, rank, component)
            if missed:
                counts["factor_miss"] += 1
            residency = sum(
                v.nbytes
                for kk, row in self._cache.items()
                if isinstance(row, tuple)
                for v in row
                if hasattr(v, "nbytes")
            )
            if residency > caps["semiseparable_cache_bytes_max"]:
                raise RuntimeError("Complete geometry-cache memory cap")
            work["geometry_cache_bytes_peak"] = max(
                work.get("geometry_cache_bytes_peak", 0), residency
            )
            return result

        convol.ConvolCache.get_coulomb_factors = factors
        original_moments = convol.ConvolCache._component_pair_moments

        def moments(self, rank, component):
            missed = (rank, component, "pair_moments") not in self._cache
            if missed:
                count("moment_miss")
            value = observed("moment_getter", original_moments)(self, rank, component)
            if missed:
                counts["moment_miss"] += 1
            return value

        convol.ConvolCache._component_pair_moments = moments
        variational.CSFOrbitalFunctional._pair_potential = observed(
            "pair_potential", variational.CSFOrbitalFunctional._pair_potential
        )
        original_gradient = variational.CSFOrbitalFunctional.gradient_from_weights

        def gradient(self, orbitals, weights, **kw):
            pairs, one_body = {}, set()
            nterms = 0
            for term in weights:
                if term[0] == "h":
                    one_body.update([(term[1], term[2]), (term[3], term[2])])
                else:
                    _, rank, a, b, cc, d = term
                    for left, right, sa, sb in ((a, b, cc, d), (cc, d, a, b)):
                        pairs.setdefault((rank, *sorted((sa, sb))), set()).update(
                            (left, right)
                        )
                        nterms += 1
            check("gradient/structural_key_capacity", len(weights) <= 70971)
            check("gradient/term_reference_capacity", nterms <= 141942)
            check("gradient/pair_source_capacity", len(pairs) <= 3472)
            count("one_body_MV", len(one_body))
            count("target_MV", 2 * sum(len(v) for v in pairs.values()))
            work.setdefault("gradients", []).append(
                {
                    "signed_keys": len(weights),
                    "negative_coefficients": sum(int(x < 0) for x in weights.values()),
                    "pair_sources": len(pairs),
                    "term_references": nterms,
                    "target_P_Q_MVs": 2 * sum(len(v) for v in pairs.values()),
                    "one_body_MVs": len(one_body),
                }
            )
            value = observed("gradient", original_gradient)(
                self, orbitals, weights, **kw
            )
            counts["one_body_MV"] += len(one_body)
            counts["target_MV"] += 2 * sum(len(v) for v in pairs.values())
            return value

        variational.CSFOrbitalFunctional.gradient_from_weights = gradient
        progress("construct complete native expanded functional")
        functional = variational.CSFOrbitalFunctional(
            csfs,
            data,
            70,
            make_nuclear_potential(70, "fermi", A=175),
            state_average=average,
            level_tracking=tracker,
            max_determinants=2659,
            retain_hamiltonian=False,
        )
        check("functional_full_keys", functional.keys == keys)
        physical_geometry = orbital_optimization._physical_geometry_sha256(functional)
        layout = read(paths["layout"])
        entries = tuple(
            (
                tuple(map(tuples, row["keys"])),
                checkpoint["projector_" + str(i)],
                tuple(row["shape"]),
                slice(*row["slice"]),
            )
            for i, row in enumerate(layout)
        )
        inverse = tuple(tuple(checkpoint["inverse_blocks_" + str(i)]) for i in range(7))
        check(
            "original_metric_fingerprint",
            orbital_optimization._residual_metric_sha256(
                entries, checkpoint["inverse_diagonal"], inverse
            )
            == c["fixed"]["metric_sha256"],
        )
        digest = hashlib.sha256(
            (physical_geometry + c["fixed"]["metric_sha256"]).encode()
        )
        orbital_optimization._hash_value(
            digest,
            (
                tuple(
                    (row[0][0][1], functional.one_body[row[0][0][1]]) for row in entries
                ),
                tuple(
                    (row[0][0][1], functional.metric[row[0][0][1]]) for row in entries
                ),
                "nuclear",
                0.05,
            ),
        )
        check(
            "original_complete_geometry_fingerprint",
            digest.hexdigest() == c["fixed"]["geometry_sha256"],
        )
        x = checkpoint["x"]
        for group, i_blocks in zip(entries, inverse, strict=True):
            kk, projector, shape, where = group
            Y = x[where].reshape(shape)
            C = np.column_stack([np.concatenate(waves[k]) for k in kk])
            S = metric[kk[0][1]]
            check("original_inverse_group/" + str(kk[0][1]), len(i_blocks) == len(kk))
            check(
                "original_wave_reconstruction/" + str(kk[0][1]),
                np.array_equal(projector @ Y, C),
            )
            check(
                "projector_Gram/" + str(kk[0][1]),
                float(np.max(abs(projector.T @ S @ projector - np.eye(shape[0]))))
                <= 1e-10,
            )
            check(
                "wave_Gram/" + str(kk[0][1]),
                float(np.max(abs(C.T @ S @ C - np.eye(len(kk))))) <= 1e-10,
            )
            outside = C - projector @ (projector.T @ S @ C)
            check(
                "no_pair_S_squared/" + str(kk[0][1]),
                float(np.max(abs(np.einsum("ij,ij->j", outside, S @ outside))))
                <= 1e-12,
            )
        actual_tables = []
        for structure, projection, columns in functional.radial_density.blocks:
            row = {
                "raw_determinants": len(structure.dets),
                "keys": len(structure.keys),
                "shape": list(structure.table.shape),
                "nnz": structure.table.nnz,
                "pairs": len(structure.pairs),
                "CSR_bytes": sum(
                    a.nbytes
                    for a in (
                        structure.table.data,
                        structure.table.indices,
                        structure.table.indptr,
                    )
                ),
                "pair_bytes": structure.pairs.nbytes,
            }
            actual_tables.append(row)
            check(
                "complete_angular_row/" + str(len(actual_tables)),
                row == read(paths["angular_tables"])[len(actual_tables) - 1],
            )
            check(
                "dense_projection_shape/" + str(len(actual_tables)),
                projection.shape == (len(structure.dets), len(columns)),
            )
        write(output / "complete_angular_tables.json", actual_tables)
        progress("literal native filtered full gradient")
        energy, retained, returned_ci = functional.evaluate(waves, ci_result=saved_ci)
        check(
            "saved_CI_returned_without_owner",
            returned_ci is saved_ci and saved_ci.hamiltonian_snapshot is None,
        )
        check(
            "native_tracker_order",
            [
                uid_json(r["reference_levels"][0])
                for r in functional.last_tracking_diagnostics
            ]
            == c["fixed"]["tracker_native_UID_order"],
        )
        native = dict(functional.last_weights)
        weights, diagnostics = (
            functional.last_state_weights,
            functional.last_tracking_diagnostics,
        )
        binding_by_uid = {tuples(r["reference_levels"][0]): r for r in diagnostics}
        check(
            "complete_five_UIDs",
            set(binding_by_uid) == set(map(tuples, c["fixed"]["UIDs"]))
            and np.count_nonzero(weights) == 5,
        )
        _, blocks = ci_level_blocks(csfs, saved_ci, require_full=True)
        contributions, targets = {}, []
        for uid in map(tuples, c["fixed"]["UIDs"]):
            row = binding_by_uid[uid]
            check(
                "UID_rank1/" + str(uid),
                len(row["reference_levels"]) == len(row["selected_levels"]) == 1
                and tuples(row["selected_levels"][0]) == uid
                and row["per_level_weight"] == 0.2,
            )
            column = int(blocks[uid[:2]][uid[2]])
            check("UID_weight/" + str(uid), weights[column] == 0.2)
            C = expanded["mixing"][:, column]
            outside = [
                i for i, s in enumerate(csfs) if (s.total_J2, s.parity) != uid[:2]
            ]
            check("complete_signed_sector/" + str(uid), not np.any(C[outside] != 0))
            norm = sum((Fraction.from_float(float(v)) ** 2 for v in C), Fraction())
            check("actualnorm/" + str(uid), norm > 0)
            table = functional.radial_density.contract_bilinear(C, C)
            for key, value in table.items():
                contributions.setdefault(key, []).append(
                    Fraction.from_float(0.2) * Fraction.from_float(float(value)) / norm
                )
            targets.append(
                {
                    "UID": list(uid),
                    "column": column,
                    "weight": 0.2,
                    "actualnorm": [str(norm.numerator), str(norm.denominator)],
                    "actualnorm_float": float(norm),
                    "tracking": row,
                    "unfiltered_terms": len(table),
                }
            )

        def represented(value):
            f = float(value)
            if not math.isfinite(f) or (value != 0 and f == 0):
                raise RuntimeError("Unrepresentable nonzero signed coefficient")
            return f

        full = {
            k: represented(v)
            for k, rows in contributions.items()
            if (v := sum(rows, Fraction())) != 0
        }
        ordered = list(full) + [k for k in native if k not in full]
        delta = {
            k: represented(v)
            for k in ordered
            if (
                v := Fraction.from_float(full.get(k, 0.0))
                - Fraction.from_float(native.get(k, 0.0))
            )
            != 0
        }
        write(
            output / "complete_signed_weights.json",
            {
                "targets": targets,
                "native_filtered": [[k, v] for k, v in native.items()],
                "full_actualnorm": [[k, v] for k, v in full.items()],
                "full_minus_native": [[k, v] for k, v in delta.items()],
                "normalization": "exact binary64 Fraction per UID norm and coefficient then represented once",
                "unfiltered_zero_rule": "exact zero only",
                "native_filter_preserved": 1e-15,
            },
        )
        progress("complete unfiltered and signed delta gradients")
        full_g = functional.gradient_from_weights(waves, full, record_diagnostics=False)
        delta_g = functional.gradient_from_weights(
            waves, delta, record_diagnostics=False
        )
        arrays = [
            np.array([mapping[k] for k in keys])
            for mapping in (retained, full_g, delta_g)
        ]
        check(
            "complete31_gradients",
            all(a.shape == (31, 246) and np.isfinite(a).all() for a in arrays),
        )
        check(
            "original_radial_linearity",
            float(np.max(abs(arrays[0] + arrays[2] - arrays[1]))) <= 1e-9,
        )

        def project(radial):
            lookup = dict(zip(keys, radial, strict=True))
            raw = np.zeros(3689, dtype=radial.dtype)
            for kk, P, _shape, where in entries:
                raw[where] = (
                    P.astype(radial.dtype).T @ np.column_stack([lookup[k] for k in kk])
                ).ravel()

            def tangent(v):
                result = v.copy()
                for _kk, _P, shape, where in entries:
                    Y = x[where].reshape(shape).astype(v.dtype)
                    a = v[where].reshape(shape)
                    m = Y.T @ a
                    result[where] = (a - 0.5 * Y @ (m + m.T)).ravel()
                return result

            g = tangent(raw)
            m = np.zeros_like(g)
            for (_, _, shape, where), bb in zip(entries, inverse, strict=True):
                a = g[where].reshape(shape)
                m[where] = np.column_stack(
                    [b.astype(g.dtype) @ a[:, j] for j, b in enumerate(bb)]
                ).ravel()
            pg = tangent(m)
            defect = max(
                float(
                    np.max(
                        abs(
                            x[w].reshape(s).astype(g.dtype).T @ g[w].reshape(s)
                            + g[w].reshape(s).T @ x[w].reshape(s).astype(g.dtype)
                        )
                    )
                )
                for _, _, s, w in entries
            )
            check(
                "scaled_tangent/" + str(len(checks)),
                defect / max(1, float(np.linalg.norm(raw))) <= 1e-10,
            )
            return (
                raw,
                g,
                pg,
                {
                    "raw_norm": float(np.linalg.norm(raw)),
                    "tangent_norm": float(np.linalg.norm(g)),
                    "residual": float(np.linalg.norm(pg)),
                    "indicator_Ha": 0.5 * abs(float(g @ pg)),
                    "absolute_tangent_defect": defect,
                    "no_spectrum_error_bound": True,
                },
            )

        projected = [project(a) for a in arrays]
        for index, tolerance in [(1, 1e-10), (2, 1e-10)]:
            check(
                "original_projected_linearity/" + str(index),
                float(
                    np.max(
                        abs(
                            projected[0][index]
                            + projected[2][index]
                            - projected[1][index]
                        )
                    )
                )
                <= tolerance,
            )
        baseline_native = project(checkpoint["retained_radial"])
        baseline_full = project(checkpoint["full_normalized_radial"])
        names = [
            ("raw_coordinate_gradient", "tangent_gradient", "preconditioned_gradient"),
            (
                "full_normalized_raw_coordinate",
                "full_normalized_tangent",
                "full_normalized_pg",
            ),
        ]
        for route, results in enumerate((baseline_native, baseline_full)):
            for i, name in enumerate(names[route]):
                check(
                    "baseline_complete_vector/" + name,
                    float(np.max(abs(results[i] - checkpoint[name])))
                    <= (1e-10 if i == 2 else 1e-9),
                )
            expected = read(paths["checkpoint"])["fullsigned_metrics"][
                "retained" if route == 0 else "full_normalized"
            ]
            for name in ("residual", "indicator_Ha"):
                check(
                    "baseline_metric/" + str(route) + "/" + name,
                    abs(results[3][name] - expected[name]) <= 1e-10,
                )
        wide = np.longdouble
        corrected = {k: full_g[k].astype(wide) for k in keys}
        vectors = {k: np.concatenate(waves[k]).astype(wide) for k in keys}
        for term, coefficient in full.items():
            if term[0] != "h":
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            h, S = (v.astype(wide) for v in raw_hS[kappa])
            logical = functional.one_body[kappa].astype(wide)
            va, vb = vectors[a], vectors[b]
            count("raw_h_correction_MV", 5)
            sb, hb, hta = S @ vb, h @ vb, h.T @ va
            norm = vb @ sb
            numerator = va @ hb
            check(
                "radial_actualnorm/" + str(term), bool(np.isfinite(norm) and norm > 0)
            )
            corrected[a] += coefficient * (hb / norm - logical @ vb)
            corrected[b] += coefficient * (
                hta / norm - logical @ va - 2 * numerator * sb / norm**2
            )
            counts["raw_h_correction_MV"] += 5
        corrected_array = np.array([corrected[k] for k in keys])
        corrected_projected = project(corrected_array)
        gA, pgA = baseline_full[1:3]
        gB, pgB = projected[1][1:3]

        def cosine(a, b):
            n = float(np.linalg.norm(a) * np.linalg.norm(b))
            return None if n == 0 else float(a @ b) / n

        comparison = {
            "same_metric_sha256": c["fixed"]["metric_sha256"],
            "gB_minus_gA_norm": float(np.linalg.norm(gB - gA)),
            "pgB_minus_pgA_norm": float(np.linalg.norm(pgB - pgA)),
            "g_cosine": cosine(gA, gB),
            "pg_cosine": cosine(pgA, pgB),
            "gB_dot_minus_pgA": float(gB @ (-pgA)),
            "gA_dot_minus_pgA": float(gA @ (-pgA)),
            "gB_dot_minus_pgB": float(gB @ (-pgB)),
            "baseline_full_metrics": baseline_full[3],
            "expanded_full_metrics": projected[1][3],
            "scope": "aggregate EOL only; no individual line derivatives or finite response",
        }
        count("NPZ_save")
        np.savez_compressed(
            output / "complete_gradients.npz",
            keys=np.array(keys),
            native_radial=arrays[0],
            full_radial=arrays[1],
            delta_radial=arrays[2],
            raw_h_corrected_radial=corrected_array,
            **{
                name + "_" + part: value
                for name, res in zip(
                    ("native", "full", "delta", "raw_h_corrected"),
                    (*projected, corrected_projected),
                    strict=True,
                )
                for part, value in zip(
                    ("raw", "tangent", "preconditioned"), res[:3], strict=True
                )
            },
        )
        counts["NPZ_save"] += 1
        check("exact_three_full_gradients", counts["gradient"] == 3)
        check("exact_six_archives", counts["NPZ_load"] == 6)
        check("exact_seven_raw_hS", counts["raw_hS"] == 7)
        check("exact_five_full_angular_structures", counts["angular_block_terms"] == 5)
        check(
            "exact_one_native_five_bilinear_density",
            counts["retained_RDM"] == 1 and counts["unfiltered_RDM"] == 5,
        )
        check(
            "zero_forbidden_producer_returns",
            all(counts[k] == 0 and attempted[k] == 0 for k in c["forbidden_calls"]),
        )
        origins()
        check(
            "all_bound_inputs_unchanged",
            all(sha(p) == h for p, h in original_files.items()),
        )
        progress("closed complete conditional gradients")
        write(
            output / "result.json",
            {
                "status": "closed_conditional_saved2574_full_gradient",
                "checks": checks,
                "counts": dict(counts),
                "attempted": dict(attempted),
                "work": work,
                "decoded_bytes": decoded,
                "targets": targets,
                "native_reported_EOL_Ha": energy,
                "metrics": {
                    n: r[3]
                    for n, r in zip(
                        ("native", "full", "delta", "raw_h_corrected"),
                        (*projected, corrected_projected),
                        strict=True,
                    )
                },
                "comparison": comparison,
                "gradient_archive_sha256": sha(output / "complete_gradients.npz"),
                "source_map_sha256": c["source_map_sha256"],
                "run_sha256": sha(HERE / "run.py"),
                "contract_sha256": sha(HERE / "contract.json"),
                "binding_sha256": sha(binding_path),
                "input_sha256": {k: v["sha256"] for k, v in c["input_roles"].items()},
                "original_failures_preserved": True,
                "no_owner_constructed": True,
                "source_or_optimizer_qualified": False,
                "ordinary_FD_passed": False,
                "stationarity_passed": False,
                "training_eligible": False,
                "controlled_timing": False,
                "elapsed_uncontrolled_s": time.monotonic() - started,
                "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            },
        )
        return 0
    except BaseException as error:
        write(
            output / "failure.json",
            {
                "phase": phase,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "checks": checks,
                "counts": dict(counts),
                "attempted": dict(attempted),
                "work": work,
                "decoded_bytes": decoded,
                "source_map_sha256": c["source_map_sha256"],
                "original_failures_preserved": True,
                "ordinary_FD_passed": False,
                "source_or_optimizer_qualified": False,
                "automatic_retry": False,
                "elapsed_uncontrolled_s": time.monotonic() - started,
                "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            },
        )
        return 1
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute:
        return execute(args.execute)
    c, _, _ = metadata()
    print(
        json.dumps(
            {
                "status": "SOURCE_JSON_only",
                "source_map_sha256": c["source_map_sha256"],
                "NPZ_access": 0,
                "numerical_imports": 0,
                "physical_calls": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
