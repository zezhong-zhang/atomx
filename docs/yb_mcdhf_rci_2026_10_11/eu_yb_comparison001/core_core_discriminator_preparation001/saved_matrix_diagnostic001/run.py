"""SOURCE-only by default; separately sealed finite saved-matrix diagnostic.

Unmodified saved native H, exactly five complete dense block eigensolves.
No Hamiltonian builder, retained owner, radial key, gradient or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path("/data/code/atomx")
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def digest_json(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def metadata():
    """SOURCE/JSON only; no numerical read, header access or rehash."""
    c = read(HERE / "contract.json")
    roles = {key: ROOT / row["path"] for key, row in c["input_roles"].items()}
    for key, row in c["input_roles"].items():
        if not row["deferred_numerical"] and sha(roles[key]) != row["sha256"]:
            raise ValueError("SOURCE/JSON role changed: " + key)
    identity = read(roles["source_identity"])
    source = ROOT / c["source_archive"]
    if (
        identity["source_map_sha256"] != c["source_map_sha256"]
        or len(identity["all_atomx_python_sha256"]) != 356
    ):
        raise ValueError("Frozen f043 whole-source identity changed")
    for path, expected in identity["all_atomx_python_sha256"].items():
        if sha(source / path.removeprefix("src/")) != expected:
            raise ValueError("Frozen source file changed: " + path)
    parent = ROOT / c["parent_directory"]
    manifest = read(roles["parent_manifest"])
    for path, expected in manifest["files_sha256"].items():
        if sha(parent / path) != expected:
            raise ValueError("Frozen parent preparation changed: " + path)
    receipt, failure, control = (
        read(roles["parent_receipt"]),
        read(roles["parent_failure"]),
        read(roles["parent_nested_H_control"]),
    )
    if (
        receipt["status"] != "foreground_closed"
        or receipt["exit_code"] != 1
        or receipt["outer_wall_stop"] is not False
        or receipt["source_and_inputs_unchanged"] is not True
        or receipt["failure_sha256"] != c["input_roles"]["parent_failure"]["sha256"]
        or receipt["frozen_guard_files"] != 405
    ):
        raise ValueError("Closed original failed execution lineage changed")
    failures = [v for v in failure["checks"] if not v["passed"]]
    if (
        sum(v["passed"] for v in failure["checks"]) != 157
        or len(failures) != 1
        or failures[0]["name"] != "expanded_exact_nested1595_H_bytes"
        or failure["completed"].get("H") != 1
        or failure["completed"].get("CI", 0) != 0
        or failure["completed"].get("complete_CI_eigh", 0) != 0
        or failure["original_stationarity_passed"] is not False
        or control["exact_bytes_equal"] is not False
        or control["maximum_absolute_difference_Ha"]
        != c["prerequisite"]["parent_max_nested_entry_difference_Ha"]
        or next(
            v["sha256"]
            for v in failure["archives"]
            if v["path"] == "expanded_complete_H.npz"
        )
        != c["input_roles"]["saved_expanded_H"]["sha256"]
    ):
        raise ValueError(
            "Original byte-gate failure or unmodified snapshot identity changed"
        )
    if (HERE / "preparation_manifest.json").exists():
        for path, expected in read(HERE / "preparation_manifest.json")[
            "files_sha256"
        ].items():
            if sha(HERE / path) != expected:
                raise ValueError("Frozen saved-matrix preparation changed: " + path)
    return c, roles, source


def authorize(path):
    c, roles, source = metadata()
    binding = read(path)
    expected = dict(
        execution_authorized_by_root=True,
        pending_exact_root_execution_contract=False,
        once_only=True,
        diagnostic_label="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
        driver_sha256=sha(__file__),
        contract_sha256=sha(HERE / "contract.json"),
        preparation_manifest_sha256=sha(HERE / "preparation_manifest.json"),
        source_map_sha256=c["source_map_sha256"],
        input_roles=c["input_roles"],
        resources=c["resources"],
        acceptance_sha256=digest_json(c["diagnostic_acceptance"]),
        original_byte_gate_remains_failed=True,
        new_H_authorized=False,
        retained_owner_fabrication=False,
        literal_interpreter="/data/code/atomx/.venv/bin/python",
    )
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError("Missing/mismatched saved-matrix root seal: " + key)
    if os.path.abspath(sys.executable) != expected["literal_interpreter"]:
        raise ValueError("Use the literal audited project interpreter")
    return binding, c, roles, source


def upward_fraction(value):
    """Exact nonnegative ratio and proved upward binary64 display."""
    if value < 0:
        raise ValueError("Nonnegative finite bound required")
    display = float(value)
    if Fraction.from_float(display) < value:
        display = math.nextafter(display, math.inf)
    if Fraction.from_float(display) < value:
        raise ArithmeticError("Upward display proof failed")
    return dict(
        numerator=value.numerator,
        denominator=value.denominator,
        upper_binary64=display,
        upper_binary64_hex=display.hex(),
    )


def execute(binding_path):
    binding, c, roles, source = authorize(binding_path)
    output = HERE / "actual"
    output.mkdir()
    started = time.monotonic()
    caps = c["resources"]
    gates = c["diagnostic_acceptance"]
    fixed = c["fixed"]
    attempted = Counter()
    completed = Counter()
    checks = []
    archives = []
    restored = []
    done = threading.Event()
    for key in (
        "H",
        "hS",
        "radial_key",
        "gradient",
        "Dirac_spectrum",
        "optimizer",
        "native_ci_solve",
    ):
        attempted[key] = completed[key] = 0
    work = {}
    points = {}

    def progress(stage):
        write(
            output / "progress.json",
            dict(
                stage=stage,
                attempted=attempted,
                completed=completed,
                elapsed_uncontrolled_s=time.monotonic() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
        )

    def check(name, error, limit=0.0):
        error = float(error)
        passed = math.isfinite(error) and error <= limit
        checks.append(
            dict(
                name=name,
                error=error if math.isfinite(error) else None,
                limit=limit,
                passed=passed,
            )
        )
        if not passed:
            progress("first_failed_diagnostic_gate/" + name)
            raise ValueError("Saved-matrix diagnostic gate failed: " + name)

    def save(name, **arrays):
        path = output / name
        if path.exists():
            raise FileExistsError("Preserve completed diagnostic arrays: " + name)
        temp = path.with_name(path.name + ".part")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        archives.append(dict(path=name, sha256=sha(path)))
        progress("saved/" + name)

    def observe(module, name, counter, maximum, after=None):
        original = getattr(module, name)

        def call(*args, **kwargs):
            attempted[counter] += 1
            if attempted[counter] > maximum:
                raise RuntimeError("Prospective diagnostic work stop: " + counter)
            value = original(*args, **kwargs)
            completed[counter] += 1
            if after is not None:
                after(value, args, kwargs)
            return value

        restored.append((module, name, original))
        setattr(module, name, call)

    def prohibit(module, name, counter):
        original = getattr(module, name)

        def forbidden(*args, **kwargs):
            attempted[counter] += 1
            raise RuntimeError("Forbidden saved-matrix diagnostic producer: " + counter)

        restored.append((module, name, original))
        setattr(module, name, forbidden)

    def alarm(*_):
        raise TimeoutError("180-second saved-matrix diagnostic inner stop")

    def heartbeat():
        while not done.wait(30):
            print(
                json.dumps(
                    dict(
                        heartbeat_elapsed_s=time.monotonic() - started,
                        attempted=dict(attempted),
                        completed=dict(completed),
                    )
                ),
                flush=True,
            )

    try:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[key] = "1"
        affinity = sorted(os.sched_getaffinity(0))
        mem = {
            v.split(":", 1)[0]: int(v.split()[1])
            for v in Path("/proc/meminfo").read_text().splitlines()
            if len(v.split()) >= 2 and v.split()[1].isdigit()
        }
        admission = dict(
            original_affinity=affinity,
            CPU=27,
            threads=1,
            host_available_GiB=mem["MemAvailable"] / 1024**2,
            admitted=27 in affinity
            and mem["MemAvailable"] / 1024**2 >= caps["minimum_host_available_GiB"],
            load_average=os.getloadavg(),
            timing_claim_valid=False,
        )
        write(output / "resource_admission.json", admission)
        if not admission["admitted"]:
            raise RuntimeError("Saved-matrix CPU27 / available-memory admission failed")
        os.sched_setaffinity(0, {27})
        limit = int(caps["AS_GiB"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, 180)
        threading.Thread(target=heartbeat, daemon=True).start()
        (output / "execution_source.py").write_bytes(Path(__file__).read_bytes())
        write(output / "exact_execution_binding.json", binding)
        for role, row in c["input_roles"].items():
            if row["deferred_numerical"]:
                if (
                    roles[role].stat().st_size
                    > c["array_access"][role]["compressed_bytes_max"]
                ):
                    raise RuntimeError(
                        "Before numerical hash compressed archive-byte stop: " + role
                    )
                if sha(roles[role]) != row["sha256"]:
                    raise ValueError(
                        "Unmodified saved numerical archive changed: " + role
                    )
        cache_dir = output / "numba_cache"
        cache_dir.mkdir()
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy import linalg

        import atomx.angular.csf_hamiltonian as angular
        import atomx.angular.mscheme as mscheme
        import atomx.physics.csf.overlap as overlap
        import atomx.physics.csf.tracking as tracking
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods import ci as ci_module
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage

        prohibit(angular, "exact_block_matrix", "H")
        prohibit(ci_module, "ci_solve", "native_ci_solve")
        observe(linalg, "eigh", "complete_block_eigh", 5)
        observe(
            mscheme, "null_space", "angular_null_space", caps["max_angular_null_space"]
        )
        observe(np.linalg, "svd", "tracking_SVD", 5)

        def signed_after(value, args, kwargs):
            save(
                "tracker_complete_signed_overlap_"
                + str(completed["tracker_signed_overlap"])
                + ".npz",
                overlap=value,
                bra_levels=np.asarray([(v.J2, v.parity, v.root) for v in args[3]]),
                ket_levels=np.asarray([(v.J2, v.parity, v.root) for v in args[7]]),
            )

        observe(
            tracking, "ci_subspace_overlaps", "tracker_signed_overlap", 2, signed_after
        )
        observe(tracking.CIBasisTracking, "resolve", "tracker_resolve", 1)
        observe(overlap, "_projection", "public_projection", 20)
        decoded = 0
        loads = 0

        def archive(role):
            nonlocal decoded, loads
            loads += 1
            plan = c["array_access"][role]
            if loads > caps["max_input_archive_loads"]:
                raise RuntimeError("Five numerical archive-load stop")
            names = list(plan["fields"])
            with zipfile.ZipFile(roles[role]) as z:
                infos = z.infolist()
                if len(infos) > 256 or len({v.filename for v in infos}) != len(infos):
                    raise ValueError("Duplicate/oversized numerical archive fields")
                if len(names) > plan["fields_max"]:
                    raise RuntimeError("Bound selected numerical field-count stop")
                size = sum(z.getinfo(name + ".npy").file_size for name in names)
                if size > caps["max_input_decoded_bytes"] - decoded + 16384 * len(
                    names
                ):
                    raise RuntimeError(
                        "Before selected archive decompression-byte stop"
                    )
            arrays = {}
            with np.load(roles[role], allow_pickle=False) as a:
                for name in names:
                    value = a[name]
                    spec = plan["fields"][name]
                    if (
                        value.shape != tuple(spec["shape"])
                        or value.dtype.str != spec["dtype"]
                        or value.dtype.hasobject
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(
                            "Finite complete numerical shape/type: " + role + "/" + name
                        )
                    decoded += value.nbytes
                    if decoded > caps["max_input_decoded_bytes"]:
                        raise RuntimeError("Decoded numerical-byte stop")
                    arrays[name] = value.copy()
            write(
                output / ("input_" + role + "_decoded.json"),
                dict(
                    fields=[
                        dict(name=k, shape=v.shape, dtype=v.dtype.str, bytes=v.nbytes)
                        for k, v in arrays.items()
                    ],
                    cumulative_decoded_bytes=decoded,
                ),
            )
            return arrays

        saved = archive("saved_expanded_H")
        base = archive("baseline_whole_CI")
        reference = archive("tracking_ci")
        radial = archive("geometry_workspace")
        metrics = archive("screened_geometry")
        h = saved["H"]
        old_h = base["H"]
        keys = tuple(tuple(map(int, k)) for k in saved["keys"])
        for key in ("keys", "P", "Q"):
            check(
                "same_literal31_" + key + "_bytes",
                saved[key].tobytes() != base[key].tobytes(),
            )
        check("stored_complete_H_exact_symmetric", not np.array_equal(h, h.T))
        check("baseline_H_exact_symmetric", not np.array_equal(old_h, old_h.T))
        waves = {
            k: (p.copy(), q.copy())
            for k, p, q in zip(keys, saved["P"], saved["Q"], strict=True)
        }
        reference_keys = tuple(tuple(map(int, k)) for k in radial["orbital_keys"])
        reference_waves = {
            k: (p.copy(), q.copy())
            for k, p, q in zip(reference_keys, radial["P"], radial["Q"], strict=True)
        }
        metric = {k: metrics[f"s_{k}"] for k in fixed["kappas"]}
        specs = read(roles["prepared_models"])
        maps = read(roles["parent_basis_maps"])
        selected = {}
        sectors = {tuple(v) for v in specs["selected_sectors"]}

        def signature(csf):
            s = angular.csf_signature(csf)
            return json.dumps(
                plain((s.subshells, s.coupling, csf.total_J2, csf.parity)),
                separators=(",", ":"),
            )

        for name in ("baseline", "expanded"):
            full = [
                csf
                for row in specs[name + "_configs"]
                for csf in enumerate_jj_csfs(row["config"])
            ]
            parent_signatures = [signature(csf) for csf in full]
            check(
                name + "/complete_NR_configuration_count",
                len(specs[name + "_configs"]) != (91 if name == "baseline" else 100),
            )
            check(
                name + "/complete_parent_CSF_count",
                len(full) != (3789 if name == "baseline" else 5741),
            )
            indices = [
                i for i, a in enumerate(full) if (a.total_J2, a.parity) in sectors
            ]
            selected[name] = [full[i] for i in indices]
            check(
                name + "/complete_selected_CSF_count",
                len(selected[name]) != (1595 if name == "baseline" else 2574),
            )
            check(
                name + "/all_parent_signed_signature_identity",
                parent_signatures != maps[name]["full_parent_signatures"],
            )
            check(
                name + "/all_selected_signed_basis_identity",
                [parent_signatures[i] for i in indices]
                != maps[name]["selected_signatures"],
            )
            check(
                name + "/all_parent_indices_identity",
                indices != maps[name]["selected_parent_indices"],
            )
            check(
                name + "/all_occupation_identity",
                plain([angular.csf_signature(v).pattern for v in selected[name]])
                != maps[name]["selected_occupations"],
            )
            check(
                name + "/complete_electron_and_keys",
                {v.n_electrons for v in full} != {69}
                or tuple(sorted(get_shared_orbital_basis(selected[name]))) != keys,
            )
        mapping = read(roles["parent_nested_embedding"])["old_to_expanded"]
        check(
            "complete1595_signed_embedding",
            len(mapping) != 1595
            or len(set(mapping)) != 1595
            or [maps["expanded"]["selected_signatures"][i] for i in mapping]
            != maps["baseline"]["selected_signatures"],
        )
        native_groups = ci_module._group_csfs_by_J_parity(selected["expanded"])
        check("exact5_complete_native_blocks", len(native_groups) != 5)
        for sector, indices in native_groups:
            expected = next(
                row["candidate"]["csf"]
                for row in c["upfront_census"]["complete_blocks"]
                if (row["twoJ"], row["parity"]) == sector
            )
            check(
                "native_complete_block_count/" + str(sector), len(indices) != expected
            )
            other = [
                i
                for i, a in enumerate(selected["expanded"])
                if (a.total_J2, a.parity) != sector
            ]
            check(
                "stored_exact_cross_sector_zero/" + str(sector),
                np.count_nonzero(h[np.ix_(indices, other)]),
            )
        for sector in sorted(sectors):
            indices = [
                i
                for i, value in enumerate(selected["baseline"])
                if (value.total_J2, value.parity) == sector
            ]
            other = [
                i
                for i, value in enumerate(selected["baseline"])
                if (value.total_J2, value.parity) != sector
            ]
            check(
                "baseline_exact_cross_sector_zero/" + str(sector),
                np.count_nonzero(old_h[np.ix_(indices, other)]),
            )
        embedded = h[np.ix_(mapping, mapping)]
        original_gate = dict(
            name="expanded_exact_nested1595_H_bytes",
            passed=False,
            preserved=True,
            diagnostic_acceptance_replaces_original_gate=False,
        )
        check(
            "original_parent_byte_failure_preserved",
            embedded.tobytes() == old_h.tobytes(),
        )
        # Exact row sums from exact binary64 operands, not rounded Delta values.
        rows = []
        entry_count = 0
        max_entry = Fraction(0)
        sector_bounds = {s: Fraction(0) for s in sectors}
        affected = []
        for i in range(1595):
            columns = np.flatnonzero(embedded[i] != old_h[i])
            row_sum = Fraction(0)
            for j in columns:
                delta = Fraction.from_float(
                    float(embedded[i, j])
                ) - Fraction.from_float(float(old_h[i, j]))
                value = abs(delta)
                row_sum += value
                max_entry = max(max_entry, value)
            entry_count += len(columns)
            if entry_count > caps["exact_difference_entry_capacity"]:
                raise RuntimeError("Complete exact Fraction difference work stop")
            sector = selected["baseline"][i].total_J2, selected["baseline"][i].parity
            sector_bounds[sector] = max(sector_bounds[sector], row_sum)
            if columns.size:
                affected.append(i)
            rows.append(
                dict(
                    baseline_row=i,
                    expanded_row=mapping[i],
                    sector=sector,
                    nonzero_differences=len(columns),
                    exact_absolute_row_sum=upward_fraction(row_sum),
                )
            )
        rho = max(sector_bounds.values())
        convert = Fraction.from_float(float(HARTREE_CM1))
        ground = (1, 1)
        transition_bounds = [
            dict(
                label=label,
                UID=uid,
                upper_sector=uid[:2],
                lower_sector=ground,
                ordered_excitation_bound_Ha=upward_fraction(
                    sector_bounds[tuple(uid[:2])] + sector_bounds[ground]
                ),
                ordered_excitation_bound_cm1_represented_constant=upward_fraction(
                    (sector_bounds[tuple(uid[:2])] + sector_bounds[ground]) * convert
                ),
            )
            for label, uid in fixed["original_UIDs"][1:]
        ]
        ledger = dict(
            status="EXACT_FINITE_SYMMETRIC_MATRIX_PERTURBATION_BOUND",
            original_gate=original_gate,
            all1595_squared_entries_compared=True,
            directed_nonzero_entry_count=entry_count,
            affected_rows=affected,
            maximum_entry_difference_Ha=upward_fraction(max_entry),
            rho_global_Ha=upward_fraction(rho),
            sector_bounds=[
                dict(sector=s, rho_Ha=upward_fraction(v))
                for s, v in sorted(sector_bounds.items())
            ],
            exact_rows=rows,
            ordered_excitation_bounds=transition_bounds,
            proof=c["finite_bound"],
            actual_Q_and_QQ_entries_never_modified=True,
            conceptual_comparison_matrix_never_materialized=True,
            bound_excludes_eigensolver_and_physical_errors=True,
        )
        write(output / "exact_arithmetic_perturbation_bound.json", ledger)
        progress("exact_Fraction_bound_preserved_before5_saved_matrix_eigh")
        baseline_ci = ci_module.CIResult(base["energies"], base["mixing"])
        _, base_blocks = ci_level_blocks(
            selected["baseline"], baseline_ci, require_full=True
        )
        check(
            "baseline_whole1595_residual",
            np.max(
                np.linalg.norm(
                    old_h @ base["mixing"] - base["mixing"] * base["energies"], axis=0
                )
            ),
            gates["baseline_CI_residual_Ha"],
        )
        check(
            "baseline_whole1595_Gram",
            np.max(abs(base["mixing"].T @ base["mixing"] - np.eye(1595))),
            gates["baseline_CI_Gram"],
        )
        baseline_point = read(roles["parent_baseline_point"])
        baseline_tracking = read(roles["parent_baseline_tracking"])
        check(
            "baseline_exact5_equal_weights",
            np.count_nonzero(baseline_tracking["weights"]) != 5
            or any(v != 0.2 for v in baseline_tracking["weights"] if v),
        )
        for row in baseline_point["rows"]:
            j, p, rank = row["selected_level"]
            column = int(base_blocks[j, p][rank])
            check(
                "baseline_saved_UID_energy/" + row["label"],
                column != row["column"]
                or float(base["energies"][column]).hex()
                != float(row["energy_Ha"]).hex(),
            )
        points["baseline"] = baseline_point
        # This reproduces the native f043 complete dense branch; it is a
        # diagnostic of saved H, not a public physical H builder or owner.
        values = []
        vectors = []
        for sector, indices in native_groups:
            idx = np.asarray(indices)
            block = h[np.ix_(idx, idx)]
            energies, mixing = linalg.eigh(block, subset_by_index=None, driver="evd")
            save(
                "saved_block_"
                + str(sector[0])
                + "_"
                + str(sector[1])
                + "_spectrum.npz",
                energies=energies,
                mixing=mixing,
                indices=idx,
            )
            full = np.zeros((2574, len(energies)))
            full[idx] = mixing
            values.extend(energies)
            vectors.append(full)
        order = np.argsort(values, kind="stable")
        energies = np.asarray(values)[order]
        mixing = np.column_stack(vectors)[:, order]
        del vectors
        save(
            "saved_complete2574_spectrum.npz",
            energies=energies,
            mixing=mixing,
            keys=saved["keys"],
            P=saved["P"],
            Q=saved["Q"],
            old_to_expanded=np.asarray(mapping),
        )
        provenance = dict(
            diagnostic="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
            saved_native_H_sha256=c["input_roles"]["saved_expanded_H"]["sha256"],
            solver="dense-blocked",
            n_csf=2574,
            n_blocks=5,
            block_sizes=[len(idx) for _, idx in native_groups],
            physical_ci_solve_called=False,
            hamiltonian_snapshot=None,
        )
        ci = ci_module.CIResult(energies, mixing, provenance=provenance)
        write(output / "saved_matrix_spectrum_provenance.json", provenance)
        reference_csfs = [
            v
            for row in read(roles["reference_models"])["models"]["baseline"]["configs"]
            for v in enumerate_jj_csfs(row["config"])
        ]
        check("immutable_original531_reference", len(reference_csfs) != 531)
        targets = read(roles["tracking_binding"])["targets"]
        UIDs = {tuple(uid): label for label, uid in fixed["original_UIDs"]}
        check(
            "unchanged_original5_UID_selectors",
            {(v["twoJ"], v["parity"], v["models"]["DC"]["local_root"]) for v in targets}
            != set(UIDs),
        )
        average = CIStateAverage(
            tuple(
                (v["twoJ"], v["parity"], v["models"]["DC"]["local_root"], 1.0)
                for v in targets
            )
        )
        tracker = tracking.CIBasisTracking(
            average,
            reference_csfs,
            ci_module.CIResult(reference["energies_Ha"], reference["mixing"]),
            reference_waves,
            metric,
            max_determinant_pairs=1000000,
        )
        weights, tracked = tracker.resolve(selected["expanded"], ci, waves, metric)
        write(
            output / "saved_complete2574_original531_tracking.json",
            dict(
                weights=weights,
                tracked=tracked,
                diagnostic_label="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
                original531_reference_not_rebased=True,
            ),
        )
        progress("full_spectra_and_tracking_archived_before_endpoint_gates")
        # Diagnostic endpoint gates, never substitutes for the failed parent gate.
        check(
            "saved_complete2574_residual",
            np.max(np.linalg.norm(h @ mixing - mixing * energies, axis=0)),
            gates["snapshot_CI_residual_Ha"],
        )
        check(
            "saved_complete2574_Gram",
            np.max(abs(mixing.T @ mixing - np.eye(2574))),
            gates["snapshot_CI_Gram"],
        )
        _, blocks = ci_level_blocks(selected["expanded"], ci, require_full=True)
        check(
            "saved5_equal_UID_weights",
            not (
                np.count_nonzero(weights) == 5
                and np.array_equal(weights[weights > 0], np.repeat(0.2, 5))
            ),
        )
        matches = {}
        for group in tracked:
            if (
                len(group["reference_levels"]) != 1
                or len(group["selected_levels"]) != 1
            ):
                raise ValueError(
                    "Complete native exact-cluster/rank-one UID boundary changed"
                )
            uid = tuple(group["reference_levels"][0])
            level = tuple(group["selected_levels"][0])
            if uid in matches or uid not in UIDs or uid[:2] != level[:2]:
                raise ValueError("Changed/duplicate diagnostic physical UID")
            matches[uid] = group
        check("saved_all5_original_UIDs", set(matches) != set(UIDs))
        rows = []
        oldrows = {v["label"]: v for v in baseline_point["rows"]}
        for label, uid_list in fixed["original_UIDs"]:
            uid = tuple(uid_list)
            level = matches[uid]["selected_levels"][0]
            column = int(blocks[level[0], level[1]][level[2]])
            rows.append(
                dict(
                    label=label,
                    UID=uid,
                    selected_level=level,
                    column=column,
                    weight=float(weights[column]),
                    energy_Ha=float(energies[column]),
                    tracker=matches[uid],
                )
            )
        lower = rows[0]["energy_Ha"]
        oldlower = oldrows["6s_1/2"]["energy_Ha"]
        for row in rows:
            du = row["energy_Ha"] - oldrows[row["label"]]["energy_Ha"]
            dl = lower - oldlower
            nominal = fixed["nominal_NIST_cm1"][row["label"]]
            row.update(
                delta_upper_Ha=du,
                delta_lower_Ha=dl,
                transition_delta_upper_minus_lower_cm1=(du - dl) * HARTREE_CM1,
                excitation_cm1=(row["energy_Ha"] - lower) * HARTREE_CM1,
                NIST_nominal_cm1=nominal,
                NIST_uncertainty_cm1=None,
            )
            row["nominal_gap_cm1"] = row["excitation_cm1"] - nominal
        bylabel = {v["label"]: v for v in rows}
        multiplets = {}
        for term, a, b in (("5d", "5d_3/2", "5d_5/2"), ("6p", "6p_1/2", "6p_3/2")):
            lo, hi = bylabel[a], bylabel[b]
            ga, gb = lo["UID"][0] + 1, hi["UID"][0] + 1
            multiplets[term] = dict(
                centroid_cm1=(ga * lo["excitation_cm1"] + gb * hi["excitation_cm1"])
                / (ga + gb),
                splitting_cm1=hi["excitation_cm1"] - lo["excitation_cm1"],
            )
            for key in ("centroid_cm1", "splitting_cm1"):
                multiplets[term]["delta_" + key] = (
                    multiplets[term][key] - baseline_point["multiplets"][term][key]
                )
            ra, rb = (
                sector_bounds[tuple(lo["UID"][:2])],
                sector_bounds[tuple(hi["UID"][:2])],
            )
            multiplets[term]["ordered_centroid_arithmetic_bound_cm1"] = upward_fraction(
                ((ga * ra + gb * rb) / (ga + gb) + sector_bounds[ground]) * convert
            )
            multiplets[term]["ordered_splitting_arithmetic_bound_cm1"] = (
                upward_fraction((ra + rb) * convert)
            )
        points["saved_expanded"] = dict(
            rows=rows,
            multiplets=multiplets,
            EOL_reported_Ha=float(weights @ energies),
            nominal_four_gap_RMS_cm1=math.sqrt(
                sum(v["nominal_gap_cm1"] ** 2 for v in rows[1:]) / 4
            ),
            diagnostic_label="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
        )
        write(
            output / "saved_expanded_conditional_point.json", points["saved_expanded"]
        )
        check("exactly5_complete_block_eigh", completed["complete_block_eigh"] != 5)
        check("exactly1_original_tracker_resolve", completed["tracker_resolve"] != 1)
        check(
            "zero_new_H_hS_radial_g_Dirac_optimizer_native_CI",
            any(
                attempted[v]
                for v in (
                    "H",
                    "hS",
                    "radial_key",
                    "gradient",
                    "Dirac_spectrum",
                    "optimizer",
                    "native_ci_solve",
                )
            ),
        )
        work.update(
            input_archive_loads=loads,
            input_decoded_bytes=decoded,
            complete_block_eigh=5,
            exact_difference_entries=entry_count,
            affected_rows=len(affected),
        )
        metadata()
        check(
            "no_H_radial_or_MCDHF_producer_module_imports",
            any(
                name in sys.modules
                for name in (
                    "atomx.physics.csf.matrix",
                    "atomx.physics.csf.hamiltonian_snapshot",
                    "atomx.numerics.bspline.dirac_assembly",
                    "atomx.methods.csf.variational",
                )
            ),
        )
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if (
                (name == "atomx" or name.startswith("atomx."))
                and filename
                and not Path(filename).resolve().is_relative_to(source.resolve())
            ):
                raise ValueError(
                    "Diagnostic Atomx module escaped frozen source: " + name
                )
        write(
            output / "result.json",
            dict(
                status="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC_completed",
                diagnostic_controls_passed=True,
                checks=checks,
                original_parent_byte_gate=original_gate,
                original_parent_execution_never_upgraded=True,
                source_or_model_qualified=False,
                original_stationarity_passed=False,
                expanded_MCDHF_optimized=False,
                experimental_gate_passed=False,
                full_spectrum_certified=False,
                complete_correlation_certified=False,
                attempted=attempted,
                completed=completed,
                work=work,
                archives=archives,
                points=points,
                exact_arithmetic_bound_path="exact_arithmetic_perturbation_bound.json",
                exact_arithmetic_bound_sha256=sha(
                    output / "exact_arithmetic_perturbation_bound.json"
                ),
                source_map_sha256=c["source_map_sha256"],
                saved_native_H_sha256=c["input_roles"]["saved_expanded_H"]["sha256"],
                contract_sha256=sha(HERE / "contract.json"),
                driver_sha256=sha(__file__),
                execution_binding_sha256=sha(binding_path),
                elapsed_uncontrolled_s=time.monotonic() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                timing_claim_valid=False,
            ),
        )
        progress("saved_matrix_diagnostic_complete")
        print(
            json.dumps(
                dict(
                    status="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC_completed",
                    new_H=0,
                    complete_eigh=completed["complete_block_eigh"],
                    original_byte_gate_still_failed=True,
                )
            ),
            flush=True,
        )
    except BaseException as error:
        write(
            output / "failure.json",
            dict(
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
                checks=checks,
                attempted=attempted,
                completed=completed,
                work=work,
                archives=archives,
                points=points,
                diagnostic_label="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
                original_parent_byte_gate_remains_failed=True,
                source_or_model_qualified=False,
                original_stationarity_passed=False,
                source_map_sha256=c["source_map_sha256"],
                contract_sha256=sha(HERE / "contract.json"),
                driver_sha256=sha(__file__),
                execution_binding_sha256=sha(binding_path),
                no_retry=True,
                elapsed_uncontrolled_s=time.monotonic() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
        )
        raise
    finally:
        done.set()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for module, name, original in reversed(restored):
            setattr(module, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-binding", type=Path)
    args = parser.parse_args()
    if args.execution_binding:
        execute(args.execution_binding)
    else:
        c, _, _ = metadata()
        print(
            json.dumps(
                dict(
                    status=c["status"],
                    diagnostic_label=c["diagnostic_label"],
                    SOURCE_only=True,
                    science_imports=0,
                    numerical_archive_reads=0,
                    numerical_archive_rehashes=0,
                    numerical_ZIP_header_reads=0,
                    new_H=0,
                    physical_calls=0,
                    planned_complete_eigh=5,
                    original_byte_gate_remains_failed=True,
                    driver_sha256=sha(__file__),
                    contract_sha256=sha(HERE / "contract.json"),
                )
            )
        )


if __name__ == "__main__":
    main()
