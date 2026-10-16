"""Independent read-only FAC artifact and paired-statistic audit."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import resource
import statistics
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAC_OWNER = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def artifact_manifest(owner):
    manifest_path = owner / "artifact_manifest.json"
    manifest = read(manifest_path)
    failed, total = [], 0
    for row in manifest["files"]:
        path = owner / row["path"]
        if not path.is_file():
            failed.append({"path": row["path"], "reason": "missing"})
            continue
        size = path.stat().st_size
        sha = digest(path)
        total += size
        if size != row["bytes"] or sha != row["sha256"]:
            failed.append({"path": row["path"], "reason": "size or SHA mismatch"})
    assert not failed, failed
    assert len(manifest["files"]) == manifest["file_count"]
    declared_total = (
        manifest["total_bytes"] if "total_bytes" in manifest else manifest["bytes"]
    )
    assert total == declared_total
    return {
        "manifest_sha256": digest(manifest_path),
        "files_rehashed": len(manifest["files"]),
        "bytes_rehashed": total,
        "failures": failed,
        "passed": True,
    }


def gate(values, reference):
    errors = np.abs(values - reference)
    relative = errors / np.maximum(np.abs(reference), np.finfo(float).tiny)
    return {
        "max_absolute_Ha": float(errors.max(initial=0)),
        "max_relative": float(relative.max(initial=0)),
        "absolute_pass": bool(np.all(errors <= 1e-11)),
        "relative_pass": bool(np.all(relative <= 1e-10)),
        "strict_pass": bool(np.all(errors <= 1e-11) and np.all(relative <= 1e-10)),
        "count": int(values.size),
    }


def source_archive(run):
    before, after = read(run / "source_before.json"), read(run / "source_after.json")
    assert before == after, "source during-run change"
    failures, total = [], 0
    with tarfile.open(run / "actual_sources_and_build_inputs.tar.gz", "r:gz") as arc:
        for path, expected in before.items():
            stream = arc.extractfile(path.lstrip("/"))
            assert stream is not None, path
            h = hashlib.sha256()
            for block in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                total += len(block)
                h.update(block)
            if h.hexdigest() != expected:
                failures.append(path)
    first, runtime_after = (
        read(run / "runtime_first_observed.json"),
        read(run / "runtime_after.json"),
    )
    assert first == runtime_after, "runtime during-run change"
    assert not failures, failures
    return {
        "source_files": len(before),
        "source_archive_bytes_decoded": total,
        "source_archive_hash_failures": failures,
        "source_before_equals_after": True,
        "runtime_files": len(first),
        "runtime_first_equals_after": True,
        "current_runtime_all_match": all(
            Path(p).is_file() and digest(p) == sha for p, sha in first.items()
        ),
    }


def precision(run):
    result = read(run / "result.json")
    cases, array_total, coordinates, decoded_bytes = [], 0, 0, 0
    for spec in result["cases"]:
        folder = run / f"{spec['sequence']}-Z{spec['z']:03d}"
        inventory = read(folder / "array_inventory.json")
        symmetries = read(folder / "symmetry_inventory.json")
        summaries, values, reference = {}, {}, []
        with np.load(folder / "complete_signed_arrays.npz", allow_pickle=False) as data:
            assert set(data.files) == set(inventory)
            for name in data.files:
                array = data[name]
                meta = inventory[name]
                assert list(array.shape) == meta["shape"], name
                assert str(array.dtype) == meta["dtype"], name
                assert array.nbytes == meta["nbytes"], name
                assert (
                    hashlib.sha256(array.tobytes()).hexdigest()
                    == meta["logical_sha256"]
                ), name
                array_total += 1
                coordinates += array.size
                decoded_bytes += array.nbytes
            assert sum(row["dimension"] for row in symmetries) == spec["levels"]
            assert data["fac_radial_integrals_Ha"].size == spec["integrals"]
            eig_stored_difference = 0.0
            for row in symmetries:
                prefix, dim = row["matrix_prefix"], row["dimension"]
                for label, matrix_name in [
                    ("fac", prefix + "_fac_matrix_Ha"),
                    ("common_integrals", prefix + "_atomx_common_integrals_matrix_Ha"),
                    ("2000_raw", "grid_2000_" + prefix + "_raw_matrix_Ha"),
                    ("2000_aligned", "grid_2000_" + prefix + "_aligned_matrix_Ha"),
                    ("4000_raw", "grid_4000_" + prefix + "_raw_matrix_Ha"),
                    ("4000_aligned", "grid_4000_" + prefix + "_aligned_matrix_Ha"),
                ]:
                    matrix = data[matrix_name]
                    assert matrix.shape == (dim, dim)
                    assert np.array_equal(matrix, matrix.T), matrix_name
                    eig = np.linalg.eigvalsh(matrix)
                    eig_name = (
                        matrix_name.removesuffix("_matrix_Ha") + "_eigenvalues_Ha"
                    )
                    if label == "common_integrals":
                        eig_name = prefix + "_common_integrals_eigenvalues_Ha"
                    eig_stored_difference = max(
                        eig_stored_difference,
                        float(np.max(np.abs(eig - data[eig_name]))),
                    )
                    values.setdefault(label, []).append(data[eig_name])
                reference.append(data[prefix + "_fac_eigenvalues_Ha"])
            ref = np.concatenate(reference)
            summaries["common_integrals"] = gate(
                np.concatenate(values["common_integrals"]), ref
            )
            for points in ("2000", "4000"):
                summaries[points] = {
                    "raw_eigenvalues": gate(
                        np.concatenate(values[points + "_raw"]), ref
                    ),
                    "c_aligned_eigenvalues": gate(
                        np.concatenate(values[points + "_aligned"]), ref
                    ),
                    "raw_radial_integrals": gate(
                        data[f"grid_{points}_raw_radial_Ha"],
                        data["fac_radial_integrals_Ha"],
                    ),
                    "c_aligned_radial_integrals": gate(
                        data[f"grid_{points}_aligned_radial_Ha"],
                        data["fac_radial_integrals_Ha"],
                    ),
                }
        for scope, summary in summaries.items():
            recorded = spec["gates"][scope]
            targets = (
                [("all", summary, recorded)]
                if scope == "common_integrals"
                else [(name, v, recorded[name]) for name, v in summary.items()]
            )
            for name, computed, recorded_gate in targets:
                for key, value in computed.items():
                    assert value == recorded_gate[key], (
                        spec["z"],
                        scope,
                        name,
                        key,
                        value,
                        recorded_gate[key],
                    )
        process = read(folder / "process.json")
        assert process["exit_code"] == 0 and process["status"] == "exited"
        assert process["peak_threads"] == max(
            s["threads"] for s in process["thread_samples"]
        )
        assert process["thread_cap_pass"] == (process["peak_threads"] <= 8)
        assert process["load_gate_pass"] == (
            process["load_before"][0] + 1 <= process["affinity"]
            and abs(process["load_after"][0] - process["load_before"][0]) <= 4
        )
        cases.append(
            {
                "z": spec["z"],
                "sequence": spec["sequence"],
                "levels": spec["levels"],
                "integrals": spec["integrals"],
                "complete_arrays": len(inventory),
                "recomputed_gates": summaries,
                "eigenvalue_recompute_max_Ha": eig_stored_difference,
                "observed_peak_threads": process["peak_threads"],
                "load_gate_pass": process["load_gate_pass"],
                "explicit_serial_initialization": (folder / "run.sf")
                .read_text()
                .startswith("InitializeMPI(1)\n"),
            }
        )
    return {
        "run": str(run),
        "result_sha256": digest(run / "result.json"),
        "actual_source_and_runtime": source_archive(run),
        "cases": cases,
        "complete_arrays_decoded": array_total,
        "complete_coordinates_decoded": coordinates,
        "complete_array_bytes_decoded": decoded_bytes,
        "same_H_absolute_gate_Ha": 1e-11,
        "same_H_relative_gate": 1e-10,
        "failed_gates_retained": True,
        "experimental_accuracy_claim": False,
        "performance_ranking_claim": False,
    }


def science_identity(backend, science):
    if backend == "FAC":
        assert len(science["levels"]) == 17
        assert set(science["modes"]) == {"long_wavelength", "finite_wavelength"}
        assert all(len(rows) == 5 for rows in science["modes"].values())
        levels = [
            {k: level[k] for k in ("ILEV", "IBASE", "P", "VNL", "TWOJ", "config")}
            for level in science["levels"]
        ]
        lines = {
            mode: [
                {
                    k: line[k]
                    for k in (
                        "upper",
                        "lower",
                        "twoj_upper",
                        "twoj_lower",
                        "multipole",
                        "mode",
                        "gauge",
                        "Z",
                    )
                }
                for line in rows
            ]
            for mode, rows in science["modes"].items()
        }
        dimensions = sum(
            block["dim"] for rows in science["blocks"].values() for block in rows
        )
        assert dimensions == 17
        return {"levels": levels, "lines": lines, "blocks": science["blocks"]}
    assert science["converged"]
    result = {}
    for mode, configs in science["modes"].items():
        assert set(configs) == {"2p5 3s1", "2p5 3d1"}
        assert sum(len(value["upper_levels"]) for value in configs.values()) == 16
        assert sum(len(value["lines"]) for value in configs.values()) == 5
        result[mode] = {}
        for config, value in configs.items():

            def state(level):
                return {
                    key: level[key]
                    for key in ("config", "index", "two_J", "parity", "label")
                }

            result[mode][config] = {
                "upper": [state(level) for level in value["upper_levels"]],
                "lower": [state(level) for level in value["lower_levels"]],
                "lines": [
                    {
                        "upper": state(line["upper"]),
                        "lower": state(line["lower"]),
                        "multipole": line["multipole"],
                    }
                    for line in value["lines"]
                ],
            }
    return result


def performance(owner, *, metadata=None, result_path=None, archive_path=None):
    metadata = owner if metadata is None else metadata
    result_path = metadata / "result.json" if result_path is None else result_path
    archive_path = (
        owner / "actual_sources_build.tar.gz" if archive_path is None else archive_path
    )
    result = read(result_path)
    groups = read(metadata / "groups.verified.json")
    arrays_total, bytes_total, coordinates_total = 0, 0, 0
    case_checks, mapped = [], {}
    baselines = {
        (group["backend"], case["z"]): owner / case["folder"]
        for group in groups
        if group["round"] == 0 and group["workers"] == 1
        for case in group["cases"]
    }
    for group in groups:
        assert len(group["cases"]) == 4
        assert [case["z"] for case in group["cases"]] == [18, 26, 54, 82]
        assert group["seconds"] == sum(case["seconds"] for case in group["cases"])
        assert group["load_ok"] == (
            group["load_before"] + group["workers"] <= 32
            and abs(group["load_after"] - group["load_before"]) <= 4
        )
        array_comparisons = []
        identities_match = True
        for case in group["cases"]:
            folder = owner / case["folder"]
            ref_folder = baselines[group["backend"], case["z"]]
            samples = read(folder / "thread_samples.json")
            rejections = read(folder / "resource_rejections.json")
            timing = read(folder / "timing.json")
            assert case["seconds"] == timing["seconds"]
            assert case["actual_threads_peak"] == max(
                sample["threads"] for sample in samples
            )
            assert case["active_threads_peak"] == max(
                sample["running"] for sample in samples
            )
            assert case["resource_ok"] == (not rejections and case["returncode"] == 0)
            assert case["actual_threads_peak"] <= 8
            assert timing["max_rss_bytes"] <= 8 * 1024**3 and timing["seconds"] < 600
            for path, sha in read(folder / "mapped_runtime.json").items():
                assert path not in mapped or mapped[path] == sha, path
                mapped[path] = sha
            base_case = next(
                base
                for baseline in groups
                if baseline["backend"] == group["backend"]
                and baseline["round"] == 0
                and baseline["workers"] == 1
                for base in baseline["cases"]
                if base["z"] == case["z"]
            )
            if not case["science_ok"] or not base_case["science_ok"]:
                case_checks.append(
                    {
                        "folder": case["folder"],
                        "complete_arrays": 0,
                        "scientific_case_admitted": False,
                        "current_science_ok": case["science_ok"],
                        "baseline_science_ok": base_case["science_ok"],
                        "exit_code": case["exit_code"],
                        "stderr_sha256": digest(folder / "stderr.txt"),
                        "actual_process_peak_threads": case["actual_threads_peak"],
                    }
                )
                continue
            science = read(folder / "science.json")
            base_science = read(ref_folder / "science.json")
            identical = science_identity(group["backend"], science) == science_identity(
                group["backend"], base_science
            )
            identities_match &= identical
            assert identical, case["folder"]
            local_count, all_bitwise = 0, True
            with (
                np.load(folder / "complete_arrays.npz", allow_pickle=False) as current,
                np.load(
                    ref_folder / "complete_arrays.npz", allow_pickle=False
                ) as reference,
            ):
                assert current.files == reference.files
                for key in current.files:
                    a, b = current[key], reference[key]
                    assert a.shape == b.shape and a.dtype == b.dtype
                    absolute = float(np.max(np.abs(a - b))) if a.size else 0.0
                    relative = (
                        float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))
                        if a.size
                        else 0.0
                    )
                    passed = absolute <= 1e-11 and (
                        relative <= 1e-10 or absolute == 0.0
                    )
                    array_comparisons.append(
                        {
                            "z": case["z"],
                            "array": key,
                            "shape": list(a.shape),
                            "maximum_absolute": absolute,
                            "maximum_relative": relative,
                            "passed": passed,
                        }
                    )
                    all_bitwise &= a.tobytes() == b.tobytes()
                    arrays_total += 1
                    local_count += 1
                    bytes_total += a.nbytes
                    coordinates_total += a.size
            case_checks.append(
                {
                    "folder": case["folder"],
                    "complete_arrays": local_count,
                    "scientific_case_admitted": True,
                    "all_arrays_bitwise_equal_baseline": all_bitwise,
                    "complete_level_line_basis_identities_equal": identical,
                    "actual_process_peak_threads": case["actual_threads_peak"],
                    "actual_active_peak_threads": case["active_threads_peak"],
                }
            )
            if group["backend"] == "FAC":
                script = (folder / "run.sf").read_text()
                assert script.startswith(f"InitializeMPI({group['workers']})\n")
                assert "LimitArray(0,0)" in script
                assert "DumpPhoton" not in script
                assert (
                    f"0 of {group['workers']}"
                    in (folder / "stdout.txt").read_text().splitlines()[0]
                )
                assert case["actual_threads_peak"] == group["workers"]
        path = (
            owner
            / group["cases"][0]["folder"].split("/Z")[0]
            / "complete_array_comparisons.json"
        )
        assert array_comparisons == read(path)
        assert group["full_array_correctness_ok"] == (
            bool(array_comparisons) and all(row["passed"] for row in array_comparisons)
        )
        assert identities_match
    source_before, source_after = (
        read(metadata / "source_before.json"),
        read(metadata / "source_after.json"),
    )
    assert source_before == source_after
    with tarfile.open(archive_path, "r:gz") as arc:
        for path, sha in source_before.items():
            stream = arc.extractfile(path.lstrip("/"))
            assert stream is not None
            h = hashlib.sha256()
            for block in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                h.update(block)
            assert h.hexdigest() == sha, path
    runtime_match = all(
        Path(path).is_file() and digest(path) == sha for path, sha in mapped.items()
    )
    assert runtime_match
    rows = []
    for recorded in result["cpu_rows"]:
        backend = "FAC" if recorded["workload"].startswith("FAC ") else "Atomx"
        workers = recorded["workers"]
        selected = [
            group
            for group in groups
            if group["backend"] == backend and group["workers"] == workers
        ]
        ratios, times = [], []
        controlled = len(selected) == 3
        for group in selected:
            baseline = next(
                candidate
                for candidate in groups
                if candidate["backend"] == backend
                and candidate["round"] == group["round"]
                and candidate["workers"] == 1
            )
            controlled &= all(
                candidate["science_ok"]
                and candidate["resource_ok"]
                and candidate["load_ok"]
                and candidate["full_array_correctness_ok"]
                for candidate in (group, baseline)
            )
            ratios.append(baseline["seconds"] / group["seconds"])
            times.append(group["seconds"])
        assert recorded["observed_times_seconds"] == times
        assert recorded["observed_paired_speedups"] == ratios
        assert recorded["seconds_median"] == statistics.median(times)
        assert recorded["controlled"] == controlled
        assert recorded["speedup"] == (
            statistics.median(ratios) if controlled else None
        )
        assert recorded["efficiency"] == (
            statistics.median(ratios) / workers if controlled else None
        )
        rows.append(
            {
                "backend": backend,
                "workers": workers,
                "paired_speedups": ratios,
                "seconds_median": statistics.median(times),
                "speedup": recorded["speedup"],
                "efficiency": recorded["efficiency"],
                "controlled": controlled,
                "actual_process_peak_threads": recorded["actual_threads_peak"],
            }
        )
    return {
        "result_sha256": digest(result_path),
        "source_before_sha256": digest(metadata / "source_before.json"),
        "source_archive_sha256": digest(archive_path),
        "groups": len(groups),
        "complete_case_checks": case_checks,
        "complete_arrays_decoded": arrays_total,
        "complete_coordinates_decoded": coordinates_total,
        "complete_bytes_decoded": bytes_total,
        "source_files": len(source_before),
        "source_archive_all_verified": True,
        "mapped_runtime_files": len(mapped),
        "current_mapped_runtime_all_verified": runtime_match,
        "recomputed_cpu_rows": rows,
        "array_values_units": "Native array units; not all Hartree.",
        "equal_accuracy_cross_code_ranking": False,
        "default_Yk_current_epoch_measured": False,
        "fac_timed_export": "stock text tables and binary Hamiltonians; NPZ/JSON decode outside timer",
        "atomx_timed_export": "additional signed matrix/eigenvector/radial capture plus NPZ/JSON inside timer",
        "failed_children_retained": [
            case for case in case_checks if not case["scientific_case_admitted"]
        ],
        "incomplete_workflow_times_not_accepted_as_full_workflow_cost": True,
    }


def model_spread(owner):
    # Share the physical unit constant, but independently reconstruct matching
    # and every scalar instead of calling the benchmark comparison helpers.
    from atomx.constants import HARTREE_EV

    path = owner / "model_spread.json"
    recorded = read(path)
    root = FAC_OWNER.parents[2]
    total_levels, total_lines, cases = 0, 0, []
    inputs = {}
    for case in recorded["cases"]:
        fac_path = root / case["FAC_folder"] / "science.json"
        atomx_path = root / case["Atomx_folder"] / "science.json"
        inputs[str(fac_path)] = digest(fac_path)
        inputs[str(atomx_path)] = digest(atomx_path)
        fac, atomx = read(fac_path), read(atomx_path)
        assert fac["z"] == atomx["z"] == case["z"]
        science_identity("FAC", fac)
        science_identity("Atomx", atomx)
        level_rows, line_rows, ground = [], [], []
        for config in ("2p5 3s1", "2p5 3d1"):
            manifold = atomx["modes"]["long_wavelength"][config]
            energy_zero = manifold["lower_levels"][0]["energy_au"]
            ground.append(energy_zero)
            upper = manifold["upper_levels"]
            for twoj in sorted({level["two_J"] for level in upper}):
                left = sorted(
                    [row for row in fac["groups"][config] if row["TWOJ"] == twoj],
                    key=lambda row: row["ENERGY_eV"],
                )
                right = sorted(
                    [row for row in upper if row["two_J"] == twoj],
                    key=lambda row: row["energy_au"],
                )
                assert len(left) == len(right)
                for rank, (fr, ax) in enumerate(zip(left, right, strict=True)):
                    energy = (ax["energy_au"] - energy_zero) * HARTREE_EV
                    difference = energy - fr["ENERGY_eV"]
                    level_rows.append(
                        {
                            "config": config,
                            "two_J": twoj,
                            "parity": ax["parity"],
                            "energy_rank": rank,
                            "fac_level": fr["ILEV"],
                            "atomx_level": ax["index"],
                            "fac_eV": fr["ENERGY_eV"],
                            "atomx_eV": energy,
                            "difference_eV": difference,
                            "relative_difference": difference / fr["ENERGY_eV"],
                        }
                    )
            for mode in ("long_wavelength", "finite_wavelength"):
                upper_ids = {row["ILEV"] for row in fac["groups"][config]}
                left = sorted(
                    [line for line in fac["modes"][mode] if line["upper"] in upper_ids],
                    key=lambda line: line["DELTA_E_eV"],
                )
                right = sorted(
                    atomx["modes"][mode][config]["lines"],
                    key=lambda line: line["omega_au"],
                )
                assert len(left) == len(right)
                for rank, (fr, ax) in enumerate(zip(left, right, strict=True)):
                    line_rows.append(
                        {
                            "config": config,
                            "mode": mode,
                            "energy_rank": rank,
                            "two_J": ax["upper"]["two_J"],
                            "parity": ax["upper"]["parity"],
                            "atomx": ax,
                            "fac": fr,
                            "S_ratio": ax["S"]
                            / (fr["reduced_matrix_element"] ** 2 / 2),
                            "A_ratio": ax["A_per_s"] / fr["A_per_s"],
                            "energy_difference_eV": ax["omega_au"] * HARTREE_EV
                            - fr["DELTA_E_eV"],
                        }
                    )
        assert len(level_rows) == len(case["levels"]) == 16
        assert len(line_rows) == len(case["lines"]) == 10
        for computed, observed in zip(level_rows, case["levels"], strict=True):
            for key, value in computed.items():
                assert value == observed[key], (case["z"], key, value, observed[key])
        assert line_rows == case["lines"]
        assert ground == case["Atomx_ground_au_by_manifold"]
        assert case["common_zero_max_au"] == abs(ground[1] - ground[0]) == 0.0
        assert case["complete_levels_with_ground"] == 17
        assert case["complete_lines"] == 10
        maximum_energy = max(abs(row["relative_difference"]) for row in level_rows)
        maximum_a = max(abs(row["A_ratio"] - 1) for row in line_rows)
        maximum_s = max(abs(row["S_ratio"] - 1) for row in line_rows)
        assert maximum_energy == case["maximum_excitation_energy_relative_difference"]
        assert maximum_a == case["maximum_line_A_relative_difference"]
        assert maximum_s == case["maximum_line_S_relative_difference"]
        total_levels += len(level_rows)
        total_lines += len(line_rows)
        cases.append(
            {
                "z": case["z"],
                "excitation_levels_recomputed": len(level_rows),
                "E1_lines_recomputed": len(line_rows),
                "maximum_excitation_relative_spread": maximum_energy,
                "maximum_A_relative_spread": maximum_a,
                "maximum_S_relative_spread": maximum_s,
            }
        )
    assert total_levels == 64 and total_lines == 40
    return {
        "result_sha256": digest(path),
        "cases": cases,
        "excitation_levels_recomputed": total_levels,
        "E1_lines_recomputed": total_lines,
        "inputs_sha256": inputs,
        "Hartree_eV": HARTREE_EV,
        "matching": "Declared config/J/parity and energy rank only.",
        "eigenstate_overlap_identity_claim": False,
        "NIST_accuracy_claim": False,
        "equal_accuracy_software_ranking_claim": False,
        "FAC_line_strength_normalization_scope": "This pure-CI E1 fixture: S = reduced_matrix_element^2/2.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision-run", type=Path, default=FAC_OWNER / "precision/actual005"
    )
    parser.add_argument("--output", type=Path, default=HERE / "precision_audit.json")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    with (FAC_OWNER.parent / "timing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = {
            "precision": precision(args.precision_run),
            "audit_is_read_only": True,
            "new_scientific_measurements": False,
        }
        if args.final:
            owner = FAC_OWNER / "performance"
            precision_alias = read(FAC_OWNER / "precision/result.json")
            assert precision_alias["authoritative_run"] == args.precision_run.name
            assert (
                precision_alias["raw_result_sha256"]
                == output["precision"]["result_sha256"]
            )
            assert precision_alias["parent_absolute_Ha"] == 1e-11
            assert precision_alias["parent_relative"] == 1e-10
            for source_case, alias_case in zip(
                read(args.precision_run / "result.json")["cases"],
                precision_alias["cases"],
                strict=True,
            ):
                for key, value in source_case.items():
                    assert alias_case[key] == value
                assert (
                    alias_case["common_integrals_max_Ha"]
                    == source_case["gates"]["common_integrals"]["max_absolute_Ha"]
                )
                assert (
                    alias_case["common_integrals_pass"]
                    == source_case["gates"]["common_integrals"]["strict_pass"]
                )
                assert (
                    alias_case["matched_c_4000_pass"]
                    == source_case["gates"]["4000"]["c_aligned_eigenvalues"][
                        "strict_pass"
                    ]
                )
            output["precision_alias_result_sha256"] = digest(
                FAC_OWNER / "precision/result.json"
            )
            initial = performance(
                owner, result_path=owner / "initial_epoch.result.json"
            )
            fresh = performance(
                owner,
                metadata=owner / "fresh_cache002",
                archive_path=owner / "actual_sources_build.fresh_cache002.tar.gz",
            )
            output["performance_initial_epoch"] = initial
            output["performance_fresh_cache_epoch"] = fresh
            output["model_spread"] = model_spread(owner)
            source_initial = read(owner / "source_before.json")
            source_fresh = read(owner / "fresh_cache002/source_before.json")
            assert all(
                source_fresh.get(path) == sha for path, sha in source_initial.items()
            )
            output["epoch_identity_chain"] = {
                "common_files_unchanged": len(source_initial),
                "new_private_helpers": sorted(set(source_fresh) - set(source_initial)),
                "physics_source_changed": False,
            }
            official = read(owner / "result.json")
            assert official["source_guard_passed"] and official["runtime_guard_passed"]
            admitted = [
                row for row in initial["recomputed_cpu_rows"] if row["backend"] == "FAC"
            ] + fresh["recomputed_cpu_rows"]
            for row in official["cpu_rows"]:
                if row["controlled"]:
                    backend = "FAC" if row["workload"].startswith("FAC ") else "Atomx"
                    match = next(
                        value
                        for value in admitted
                        if value["backend"] == backend
                        and value["workers"] == row["workers"]
                    )
                    assert match["controlled"]
                    for key in ("seconds_median", "speedup", "efficiency"):
                        assert match[key] == row[key]
                    assert (
                        row["actual_threads_peak"]
                        == match["actual_process_peak_threads"]
                    )
                else:
                    assert row["workers"] == 8 and row["workload"].startswith("Atomx ")
                    assert (
                        row["speedup"]
                        is row["efficiency"]
                        is row["seconds_median"]
                        is None
                    )
                    warmup = next(
                        group
                        for group in read(owner / "warmup.json")
                        if group["backend"] == "Atomx" and group["workers"] == 8
                    )
                    peak = max(case["actual_threads_peak"] for case in warmup["cases"])
                    assert peak == row["actual_threads_peak"] == 15
                    assert not warmup["resource_ok"] and not warmup["science_ok"]
            output["official_result_sha256"] = digest(owner / "result.json")
            output["official_rows_independently_bound_to_admitted_epochs"] = True
            output["manifests"] = {
                "precision": artifact_manifest(FAC_OWNER / "precision"),
                "performance": artifact_manifest(owner),
            }
            assert output["precision"]["actual_source_and_runtime"][
                "current_runtime_all_match"
            ]
            output["passed"] = True
            output["passed_meaning"] = (
                "Evidence identities, completeness, arithmetic and admission checks passed; scientific target failures remain failures."
            )
            output["scientific_comparison_all_gates_passed"] = False
            output["default_Yk_current_epoch_measured"] = False
            output["scientific_failed_gates_retained"] = True
            output["highest_level_reached"] = False
            output["audit_script_sha256"] = digest(Path(__file__))
        elif args.performance:
            output["performance"] = performance(FAC_OWNER / "performance")
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arrays_decoded": output["precision"]["complete_arrays_decoded"],
                "source_files": output["precision"]["actual_source_and_runtime"][
                    "source_files"
                ],
                "cases": len(output["precision"]["cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
