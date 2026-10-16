"""Prospective native closed68-screened f seed / fixed-literal208 RCI discriminator.

No physics has been executed by this preparation. Explicit root binding is
required before numerical imports or array reads. Uses the original native
public DC builder, full signed CSFs, and the whole qualified refinement helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import runpy
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1]
ROOT = TOP.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return plain(value.tolist()) if hasattr(value, "tolist") else value


def write(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(plain(value), indent=2) + "\n")
    temporary.replace(path)


def metadata():
    contract = load(HERE / "contract.json")
    roles = {role: ROOT / row["path"] for role, row in contract["input_roles"].items()}
    for name, expected in contract["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen scientific input changed: " + name)
    if sha(HERE / "models.json") != contract["models_sha256"]:
        raise ValueError("Predeclared complete family changed")
    for name, expected in contract["consumer_helpers_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Frozen native consumer helper changed: " + name)
    source = load(roles["candidate_source_identity"])
    if (source["source_map_sha256"] != contract["fixed"]["source_map_sha256"]
            or len(source["all_atomx_python_sha256"]) != 356):
        raise ValueError("Whole frozen f043 source identity changed")
    source_root = ROOT / contract["source"]["candidate_archive"]
    for name, expected in source["all_atomx_python_sha256"].items():
        relative = Path(name).relative_to("src")
        if sha(source_root / relative) != expected:
            raise ValueError("Frozen source changed: " + name)
    manifest, completion = load(roles["literal208_checkpoint"]), load(roles["literal208_completion"])
    fixed = contract["fixed"]
    if (manifest["global_accepted_step"] != 208 or manifest["iteration"] != 128
            or manifest["checkpoint_qualified"] is not True
            or manifest["stationarity_passed"] is not False
            or manifest["source_map_sha256"] != fixed["source_map_sha256"]
            or manifest["metric_sha256"] != fixed["checkpoint_metric_sha256"]
            or manifest["geometry_sha256"] != fixed["checkpoint_geometry_sha256"]
            or manifest["objective_sha256"] != fixed["checkpoint_objective_sha256"]
            or manifest["array_sha256"] != sha(roles["literal208_arrays"])
            or completion["all_completion_metadata_checks_passed"] is not True
            or completion["all_actual_live_controls_passed"] is not True
            or completion["total_global_accepted_steps"] != 208
            or completion["stationarity_passed"] is not False
            or completion["actual_foreground_exit_code"] != 0):
        raise ValueError("Literal208 completed point/failed-stationarity boundary changed")
    if load(roles["refinement_qualified_proof"])["all_independent_saved_array_checks_passed"] is not True:
        raise ValueError("Original complete-spectrum numerical helper lacks qualified anchor")
    old_failure = load(roles["nuclear_failed_evidence"])
    if (old_failure["scientific_success_claim"] is not False
            or old_failure["completed"]["CI_internal_hS"] != 7
            or old_failure["completed"]["H"] != 2 or old_failure["completed"]["CI"] != 2
            or old_failure["checks"][-1]["name"] != "expected_completed_CI_internal_hS"
            or old_failure["checks"][-1]["passed"] is not False
            or any(row["passed"] is not True for row in old_failure["checks"][:-1])):
        raise ValueError("Prior nuclear scientific evidence/counter failure boundary changed")
    return contract, roles, manifest, source_root


def authorize(path):
    contract, roles, manifest, source_root = metadata()
    binding = load(path)
    expected = {
        "execution_authorized_by_root": True,
        "pending_exact_root_execution_binding": False,
        "preparation_contract_sha256": sha(HERE / "contract.json"),
        "driver_sha256": sha(__file__),
        "models_sha256": sha(HERE / "models.json"),
        "scientific_inputs_sha256": contract["inputs_sha256"],
        "source_map_sha256": contract["fixed"]["source_map_sha256"],
        "checkpoint_manifest_sha256": sha(roles["literal208_checkpoint"]),
        "checkpoint_arrays_sha256": sha(roles["literal208_arrays"]),
        "signed_scalar_controls_sha256": sha(HERE / "signed_scalar_controls.py"),
        "closed68_adapter_sha256": sha(roles["closed68_adapter"]),
        "preparation_sha256": sha(HERE / "preparation.json"),
        "prospective_execution_contract_sha256": sha(HERE / "execution_contract.json"),
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Missing/mismatched exact NEW root authorization; no science")
    trigger = binding.get("actual_iteration_trigger", {})
    if (not trigger.get("message_id") or not trigger.get("delivery_proof_path")
            or sha(ROOT / trigger["delivery_proof_path"]) != trigger.get("delivery_proof_sha256")):
        raise ValueError("Actual delivery trigger not independently bound by root")
    proof = load(ROOT / trigger["delivery_proof_path"])
    if (proof.get("passed") is not True or proof.get("message_id") != trigger["message_id"]
            or proof.get("chat_id") != "oc_fb2fc627b3e98dc9555de31580878758"
            or proof.get("bot_open_id") != "ou_c697e48a528ad9aa6157f62531561b59"
            or proof.get("actual_send", {}).get("passed") is not True):
        raise ValueError("Actual saved delivery proof schema/chat/bot/message does not match")
    return binding, contract, roles, manifest, source_root


def execute(binding_path):
    binding, contract, roles, manifest, source_root = authorize(binding_path)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    (output / "execution_source.py").write_bytes(Path(__file__).read_bytes())
    caps, gates, fixed = contract["resources"], contract["acceptance"], contract["fixed"]
    started = time.monotonic()
    limits = caps["call_limits"]
    attempted, completed = dict.fromkeys(limits, 0), dict.fromkeys(limits, 0)
    checks, stages, restored = [], [], []
    hS_requests = dict(attempted=0, completed=0, cache_hits=0, cache_misses=0,
                       per_kappa={}, by_CI=[], carry_events=[])
    current_CI = [None]
    current_CI_kappas = [set()]
    scalar_summary = {}
    refinement = dict.fromkeys(("complete_spectrum_refinement_attempts", "completed_complete_spectrum_refinements",
        "near_identity_Gram_whitenings", "physical_residual_evaluations", "attempted_complete_sweeps",
        "completed_complete_sweeps", "attempted_pair_rotations", "completed_pair_rotations",
        "skipped_exact_zero_pairs"), 0)
    finished = threading.Event()

    def progress(phase):
        write(output / "progress.json", dict(phase=phase, attempted=attempted, completed=completed,
              refinement=refinement, hS_request_ledger=hS_requests, checks=checks, stages=stages,
              elapsed_uncontrolled_s=time.monotonic() - started))

    def budget(seconds):
        remaining = int(caps["wall_seconds"] - (time.monotonic() - started))
        if remaining < 1:
            raise TimeoutError("Whole prospective scientific wall budget exhausted")
        signal.alarm(min(seconds, remaining))

    def timeout(*_):
        raise TimeoutError("Prospective phase/whole wall budget exhausted")

    signal.signal(signal.SIGALRM, timeout)
    budget(caps["preflight_seconds"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    cache_dir = output / "numba_cache"
    cache_dir.mkdir()
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

    def heartbeat():
        while not finished.wait(30):
            print(json.dumps(dict(elapsed_s=time.monotonic() - started, attempted=attempted,
                  completed=completed, peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)), flush=True)

    try:
        affinity = sorted(os.sched_getaffinity(0))
        memory = {r.split(":", 1)[0]: int(r.split()[1]) for r in Path("/proc/meminfo").read_text().splitlines()
                  if len(r.split()) >= 2 and r.split()[1].isdigit()}
        admission = dict(original_affinity=affinity, selected_CPU=caps["CPU"], load_average=os.getloadavg(),
                         host_available_GiB=memory["MemAvailable"] / 1024**2,
                         timing_scope="Uncontrolled finite physical discriminator; no speedup claim")
        admission["admitted"] = (caps["CPU"] in affinity
            and admission["host_available_GiB"] >= caps["minimum_host_available_GiB"]
            and admission["load_average"][0] <= caps["maximum_system_load1"])
        write(output / "resource_admission.json", admission)
        if not admission["admitted"]:
            raise RuntimeError("Resource admission failed before physical imports/arrays")
        os.sched_setaffinity(0, {caps["CPU"]})
        address_limit = int(caps["address_space_GiB"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
        if any(name == "atomx" or name.startswith("atomx.") for name in sys.modules):
            raise ValueError("Fresh source-isolated process required")
        threading.Thread(target=heartbeat, daemon=True).start()
        sys.path.insert(0, str(source_root))
        import numpy as np
        import scipy
        import scipy.linalg as scipy_linalg

        from atomx.angular import mscheme
        from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.config.subshell import Subshell
        from atomx.constants import HARTREE_CM1
        from atomx.methods import ci as ci_module
        from atomx.methods.dirac_fock import DFSolver
        from atomx.numerics.bspline import dirac_assembly as dirac
        from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
        from atomx.physics.csf import matrix
        from atomx.physics.csf import tracking as tracking_module
        from atomx.physics.csf.frozen_core import closed_core_one_body
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(source_root.resolve())):
                raise ValueError("Atomx import escaped immutable f043: " + name)

        def check(name, error, limit=0.):
            row = dict(name=name, error=float(error), limit=limit,
                       passed=bool(np.isfinite(error) and error <= limit))
            checks.append(row)
            if not row["passed"]:
                progress("failed_" + name)
                raise ValueError("Unchanged scientific/numeric gate failed: " + name)

        def save(path, **arrays):
            if path.exists():
                raise FileExistsError("Keep every numerical archive immutable")
            temporary = path.with_name(path.name + ".part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            stages.append(dict(name=path.name, path=str(path.relative_to(HERE)), sha256=sha(path)))

        def counted(name, function, capture=False, announce=True):
            def call(*args, **kwargs):
                if attempted[name] >= limits[name]:
                    progress("resource_stop_before_" + name)
                    raise RuntimeError("Prospective call cap: " + name)
                attempted[name] += 1
                if announce:
                    progress("before_" + name)
                value = function(*args, **kwargs)
                completed[name] += 1
                if capture:
                    kappa = int(args[1])
                    raw_h, raw_s = value
                    save(output / f"raw_CI_hS_{completed[name]:02d}.npz", kappa=np.array(kappa), h=raw_h, s=raw_s)
                    logical = np.tril(raw_h) + np.tril(raw_h, -1).T
                    check("same_logical_h_" + str(completed[name]), not np.array_equal(logical, hs[f"h_{kappa}"]))
                    check("same_S_" + str(completed[name]), not np.array_equal(raw_s, hs[f"s_{kappa}"]))
                if announce:
                    progress("after_" + name)
                return value
            return call

        def patch(module, attribute, name, capture=False):
            original = getattr(module, attribute)
            restored.append((module, attribute, original))
            setattr(module, attribute, counted(name, original, capture))

        patch(mscheme, "null_space", "angular_highest_weight_nullspace")
        patch(scipy_linalg, "eigh", "full_scipy_eigh")
        patch(np.linalg, "svd", "tracking_principal_SVD")
        patch(tracking_module, "ci_subspace_overlaps", "tracking_overlap")
        patch(tracking_module.CIBasisTracking, "resolve", "tracking_resolution")
        patch(matrix, "build_ci_matrix", "H")
        patch(matrix, "build_dirac_matrices", "CI_internal_hS", capture=True)
        patch(dirac, "solve_dirac", "Dirac")
        ci_solve = counted("CI", ci_module.ci_solve)
        complement_svd = counted("complement_SVD", scipy_linalg.null_space)
        cache_factory = counted("ConvolCache", ConvolCache)
        nuclear_factory = counted("nuclear_potential_constructor", make_nuclear_potential)
        # Observational forwarding only: count every consumer request separately
        # from the native producer, and verify the original h0 cache carry rules.
        original_blocks = matrix._dirac_blocks
        original_reset = matrix._reset_memo_for_ci_build
        restored.extend(((matrix, "_dirac_blocks", original_blocks),
                         (matrix, "_reset_memo_for_ci_build", original_reset)))

        def request_blocks(ns, kappa, Z, workspace):
            if hS_requests["attempted"] >= caps["maximum_hS_consumer_requests"]:
                raise RuntimeError("Prospective h/S request budget exhausted")
            hS_requests["attempted"] += 1
            check("hS_request_exact_workspace", workspace is not data)
            check("hS_request_model", ns != 123 or Z != 70 or kappa not in metrics)
            check("hS_request_exact_nuclear_object", matrix._SLATER_MEMO.get("_nuclear_potential") is not nuclear)
            memo_key = ("h0", ns, kappa, Z)
            hit = memo_key in matrix._SLATER_MEMO and matrix._MEMO_CONTEXT is workspace
            value = original_blocks(ns, kappa, Z, workspace)
            hS_requests["completed"] += 1
            hS_requests["cache_hits" if hit else "cache_misses"] += 1
            kk = str(kappa)
            hS_requests["per_kappa"][kk] = hS_requests["per_kappa"].get(kk, 0) + 1
            current_CI_kappas[0].add(kappa)
            raw_h, overlap = value
            logical = np.tril(raw_h) + np.tril(raw_h, -1).T
            check("hS_request_same_logical_h_" + str(hS_requests["completed"]),
                  not np.array_equal(logical, hs[f"h_{kappa}"]))
            check("hS_request_same_S_" + str(hS_requests["completed"]), not np.array_equal(overlap, metrics[kappa]))
            return value

        def memo_reset(workspace, potential):
            if len(hS_requests["carry_events"]) >= caps["maximum_CI_memo_resets"]:
                raise RuntimeError("Prospective CI build memo reset budget exhausted")
            prior = {key: value for key, value in matrix._SLATER_MEMO.items()
                     if isinstance(key, tuple) and key[0] == "h0"}
            same_workspace = matrix._MEMO_CONTEXT is workspace
            same_potential = matrix._SLATER_MEMO.get("_nuclear_potential") is potential
            old_geometry = matrix._MEMO_GEOMETRY
            original_reset(workspace, potential)
            carried = {key: value for key, value in matrix._SLATER_MEMO.items()
                       if isinstance(key, tuple) and key[0] == "h0"}
            event = dict(CI=current_CI[0], same_workspace_before=same_workspace,
                same_nucleus_before=same_potential, same_geometry_before_after=old_geometry == matrix._MEMO_GEOMETRY,
                previous_h0_blocks=len(prior), carried_h0_blocks=len(carried),
                carried_tuple_identity=all(key in prior and value is prior[key] for key, value in carried.items()))
            hS_requests["carry_events"].append(event)
            expected = 0 if current_CI[0] == "baseline" else 7
            check("source_h0_carry_count_" + str(current_CI[0]), len(carried) != expected)
            if expected:
                check("source_h0_carry_identity_" + str(current_CI[0]), not (same_workspace and same_potential
                      and event["same_geometry_before_after"] and event["carried_tuple_identity"]))

        matrix._dirac_blocks = request_blocks
        matrix._reset_memo_for_ci_build = memo_reset

        write(output / "numeric_environment.json", dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
              longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant), float64_epsilon=float(np.finfo(float).eps)))
        check("wide_precision_exceeds_float64", np.finfo(np.longdouble).eps >= np.finfo(float).eps)
        specs = load(HERE / "models.json")["models"]
        sectors = {tuple(v) for v in load(HERE / "models.json")["selected_sectors"]}
        full, selected, signatures = {}, {}, {}

        def signature(csf):
            sig = csf_signature(csf)
            return json.dumps(plain((sig.subshells, sig.coupling, csf.total_J2, csf.parity)), separators=(",", ":"))

        for name, spec in specs.items():
            full[name] = [c for row in spec["configs"] for c in enumerate_jj_csfs(row["config"])]
            selected[name] = [c for c in full[name] if (c.total_J2, c.parity) in sectors]
            signatures[name] = [signature(c) for c in selected[name]]
            check(name + "/whole_parent_counts", len(full[name]) != spec["full_CSFs"])
            check(name + "/whole_parent_blocks", len({(c.total_J2, c.parity) for c in full[name]}) != spec["full_J_parity_sectors"])
            check(name + "/complete_target_count", len(selected[name]) != spec["complete_target_CSFs"])
            check(name + "/same69electrons", {c.n_electrons for c in full[name]} != {69})
            check(name + "/distinct_signed_CSFs", len(set(signatures[name])) != len(signatures[name]))
            expected = {(r["twoJ"], r["parity"]): r["CSFs_integer_multiplicity"] for r in spec["complete_target_blocks"]}
            check(name + "/five_complete_blocks", Counter((c.total_J2, c.parity) for c in selected[name]) != expected)
        check("baseline_signed1046_order", signatures["baseline"] != load(roles["screened_basis"])["balanced_virtual"]["retained_signatures"])
        old_contract = load(roles["literal208_contract"])
        original_models = load(ROOT / old_contract["input_roles"]["screened_models"]["path"])["models"]
        reference_csfs = [c for row in original_models["baseline"]["configs"] for c in enumerate_jj_csfs(row["config"])]
        check("original_reference531", len(reference_csfs) != 531)
        write(output / "basis_rows.json", {name: dict(full_signatures=[signature(c) for c in full[name]],
              retained_signatures=signatures[name], full_spectrum_computed=False) for name in specs})

        def archive(role):
            with np.load(roles[role], allow_pickle=False) as values:
                return {key: values[key].copy() for key in values.files}

        saved, radial, hs = archive("literal208_arrays"), archive("geometry_workspace"), archive("screened_geometry")
        data = {key[6:]: value.item() if value.ndim == 0 else value.copy()
                for key, value in radial.items() if key.startswith("basis_")}
        check("same_ns123", data["ns"] != 123)
        old_keys = tuple(map(tuple, fixed["original29keys"]))
        check("literal208_ordered29keys", tuple(map(tuple, saved["keys"])) != old_keys)
        waves = {key: (p.copy(), q.copy()) for key, p, q in zip(old_keys, saved["P"], saved["Q"], strict=True)}
        metrics = {k: hs[f"s_{k}"] for k in sorted({k for _, k in old_keys})}
        layout, projectors = load(roles["saved_layout"]), {}
        fingerprint = hashlib.sha256()
        for i, row in enumerate(layout):
            keys, shape = tuple(map(tuple, row["keys"])), tuple(row["shape"])
            projector = saved[f"projector_{i}"]
            fingerprint.update(repr((list(keys), shape)).encode())
            fingerprint.update(np.ascontiguousarray(projector).tobytes())
            check("originalP_shape_" + str(i), projector.shape != (246, 119))
            check("originalP_SGram_" + str(i), np.max(abs(projector.T @ metrics[keys[0][1]] @ projector
                  - np.eye(119))), gates["saved_P_metric_Gram_error"])
            coords = saved["x"][slice(*row["slice"])].reshape(shape)
            mapped = projector @ coords
            for j, key in enumerate(keys):
                check("literal208_coordinate_wave_" + str(key), not np.array_equal(mapped[:, j], np.concatenate(waves[key])))
            projectors[keys[0][1]] = projector
        fingerprint.update(np.ascontiguousarray(saved["inverse_diagonal"]).tobytes())
        for i in range(len(layout)):
            for block in saved[f"inverse_blocks_{i}"]:
                fingerprint.update(np.ascontiguousarray(block).tobytes())
        check("original3451_metric_SHA", fingerprint.hexdigest() != fixed["checkpoint_metric_sha256"])
        check("original3451_x_shape", saved["x"].shape != (3451,))
        save(output / "original_literal208_point.npz", **saved)

        def radial_check(name, current):
            for kappa, metric in metrics.items():
                vectors = np.column_stack([np.concatenate(current[key]) for key in sorted(current) if key[1] == kappa])
                check(name + "/full_samek_Gram_" + str(kappa), np.max(abs(vectors.T @ metric @ vectors
                      - np.eye(vectors.shape[1]))), gates["actual_radial_Gram_error"])
                absent = vectors - projectors[kappa] @ (projectors[kappa].T @ metric @ vectors)
                leakage = np.sum(absent * (metric @ absent), axis=0)
                check(name + "/originalP_leakage_" + str(kappa), np.max(leakage), gates["nuclear_projector_leakage_S_norm_squared"])
            for key in old_keys:
                check(name + "/literal_oldwave_bytes_" + str(key), any(a.tobytes() != b.tobytes()
                      for a, b in zip(waves[key], current[key], strict=True)))

        radial_check("original29", waves)

        def raw_groups(space, keys):
            labels = [(n, k, m) for n, k in keys for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)]
            groups, counts = {}, {}
            for sector in sorted(sectors):
                masks = set()
                for csf in space:
                    if (csf.total_J2, csf.parity) == sector:
                        masks.update(mask for mask, _ in csf_magnetic_vector(csf_signature(csf), sector[0], keys))
                groups[sector] = Counter(tuple(sorted((k, m) for i, (_, k, m) in enumerate(labels)
                                            if mask & (1 << i))) for mask in masks)
                counts[sector] = len(masks)
            return groups, counts

        reference_keys = tuple(sorted(tuple(map(int, key)) for key in radial["orbital_keys"]))
        reference_groups, reference_raw = raw_groups(reference_csfs, reference_keys)
        preflight = {"original_reference_raw": [[*sector, count] for sector, count in reference_raw.items()], "models": {}}
        for name, space in selected.items():
            keys = tuple(map(tuple, specs[name]["radial_keys"]))
            check(name + "/public_complete_radialkeys", set(get_shared_orbital_basis(space)) != set(keys))
            groups, counts = raw_groups(space, keys)
            compatible = {sector: sum(count * groups[sector].get(key, 0) for key, count in reference_groups[sector].items())
                          for sector in sorted(sectors)}
            preflight["models"][name] = dict(raw=[[*s, n] for s, n in counts.items()],
                signed_compatible_pairs=[[*s, n] for s, n in compatible.items()], total_signed_compatible_pairs=sum(compatible.values()))
            write(output / "angular_resource_preflight.json", preflight)
            check(name + "/raw_determinant_cap", max(counts.values()), caps["maximum_raw_determinants_per_block"])
            check(name + "/signed_compatible_overlap_cap", sum(compatible.values()), caps["maximum_signed_compatible_overlap_pairs_per_call"])
            for row in specs[name]["complete_target_blocks"]:
                check(name + "/raw_kinematic_upper_" + str((row["twoJ"], row["parity"])),
                      counts[row["twoJ"], row["parity"]], row["all_NR_patterns_M_equals_J_magnetic_dimension"])
        self_pairs = sum(count * count for group in reference_groups.values() for count in group.values())
        preflight["reference_self_compatible_pairs"] = self_pairs
        write(output / "angular_resource_preflight.json", preflight)
        check("reference_self_compatible_cap", self_pairs, caps["maximum_signed_compatible_overlap_pairs_per_call"])
        progress("complete_CSF_and_raw_overlap_preflight_passed")

        budget(caps["operator_scalar_seconds"])
        cache = cache_factory(data)
        field_adapter = runpy.run_path(str(roles["closed68_adapter"]), run_name="native_closed68_forwarding_adapter_only")
        fields = field_adapter["build_fields"](np, DFSolver, Subshell, workspace=data, cache=cache,
            core_rows=fixed["closed68_core_rows"], waves=waves, target_keys=fixed["new_radial_keys"],
            nuclear_h={k: hs[f"h_{k}"] for k in metrics},
            invoke=lambda name, function, *args, **kwargs: counted(name, function)(*args, **kwargs),
            check=check, archive=lambda name, **arrays: save(output / (name + ".npz"), **arrays))
        # Independent component probes use original4f and the OLD saved nuclear
        # 5f numerical seeds; qualification of this field precedes its eigensolve.
        nuclear_probe_seeds = {}
        for i, key in enumerate(map(tuple, fixed["new_radial_keys"])):
            probe = archive("nuclear_probe_seed_" + str(i))["phased_seed"]
            check("saved_nuclear_probe_shape_" + str(key), probe.shape != (246,) or not np.isfinite(probe).all())
            nuclear_probe_seeds[key] = (probe[:123].copy(), probe[123:].copy())
        radial_check("old_saved_nuclear_probe_only", waves | nuclear_probe_seeds)
        scalar_helper = runpy.run_path(str(HERE / "signed_scalar_controls.py"), run_name="independent_signed_scalar_consumer_only")
        scalar_summary = scalar_helper["validate"](np, compute_slater_rk_cell, closed_core_one_body,
            core_rows=fixed["closed68_core_rows"], waves=waves, probe_seeds=nuclear_probe_seeds,
            fields=fields, workspace=data, cache=cache, caps=caps, gates=gates,
            invoke=lambda name, function, *args, **kwargs: counted(name, function, announce=False)(*args, **kwargs),
            check=check, archive=lambda name, **arrays: save(output / (name + ".npz"), **arrays),
            write=lambda name, value: write(output / (name + ".json"), value))
        check("all32_independent_signed_scalar_controls_first", scalar_summary["ordered_bilinears"] != 32)
        progress("closed68_row_scale_and_all32_signed_scalar_controls_passed_before_spectra_CI")
        refinements = runpy.run_path(str(roles["refinement_helper"]), run_name="qualified_full_spectrum_helper_only")
        extended = {key: (p.copy(), q.copy()) for key, (p, q) in waves.items()}
        virtual_rows = []
        for index, key in enumerate(map(tuple, fixed["new_radial_keys"])):
            kappa = key[1]
            p, h, s = projectors[kappa], fields[kappa]["F"], metrics[kappa]
            h_nuclear, jk = hs[f"h_{kappa}"], fields[kappa]["JK"]
            frozen = np.column_stack([np.concatenate(waves[k]) for k in old_keys if k[1] == kappa])
            check("one_original4f_" + str(kappa), frozen.shape[1] != 1)
            complement = complement_svd(frozen.T @ s @ p)
            b = p @ complement
            check("whole118_complement_" + str(kappa), b.shape != (246, 118))
            save(output / f"virtual_{index}_original_operator.npz", originalP=p, h=h, s=s, frozen=frozen,
                 complement=complement, basis=b, logical_nuclear_h=h_nuclear, JK_Ha=jk)
            check("full_complement_SGram_" + str(kappa), np.max(abs(b.T @ s @ b - np.eye(118))), gates["saved_P_metric_Gram_error"])
            check("full_complement_frozen4f_orthogonality_" + str(kappa), np.max(abs(frozen.T @ s @ b)), gates["actual_radial_Gram_error"])
            budget(caps["numerical_method_caps"]["remaining4_refinement_wall_cap_each_s"])
            reported, initial = scipy_linalg.eigh(b.T @ h @ b, b.T @ s @ b, driver="gvd")
            raw = np.asarray(b.astype(np.longdouble) @ initial.astype(np.longdouble), dtype=float)
            wide = raw.astype(np.longdouble)
            norms = np.sum(wide * (s.astype(np.longdouble) @ wide), axis=0)
            check("raw_all118_positive_norm_" + str(kappa), np.any(norms <= 0))
            raw /= np.asarray(np.sqrt(norms), dtype=float)
            wide = raw.astype(np.longdouble)
            raw_rayleigh = np.sum(wide * (h.astype(np.longdouble) @ wide), axis=0) / np.sum(wide * (s.astype(np.longdouble) @ wide), axis=0)
            save(output / f"virtual_{index}_fullgvd_initial.npz", reported_Ha=reported, initial=initial, raw_normalized=raw, raw_Rayleigh_Ha=raw_rayleigh)
            numerical = refinements["refine"](np, h=h, s=s, b=b, initial=initial,
                caps=caps["numerical_method_caps"], gates=gates, calls=refinement, check=check,
                save=lambda name, index=index, **arrays: save(output / f"virtual_{index}_{name}.npz", **arrays))
            vectors, energies = numerical["physical"], numerical["actual_Rayleigh_Ha"]
            check("complete118_final_roots_" + str(kappa), vectors.shape != (246, 118) or len(energies) != 118)
            gap = energies[1] - energies[0]
            check("predeclared_unique_minimum_" + str(kappa), gap <= gates["virtual_minimum_separation_Ha"])
            seed = vectors[:, 0].copy()
            raw_first = raw[:, int(np.argmin(raw_rayleigh))]
            overlap_squared = float(seed.astype(np.longdouble) @ s.astype(np.longdouble) @ raw_first.astype(np.longdouble))**2
            check("same_original_minimum_span_" + str(kappa), overlap_squared <= gates["virtual_original_P_minimum_span_overlap_squared"])
            check("same_original_minimum_Rayleigh_" + str(kappa), abs(float(energies[0] - np.min(raw_rayleigh))), gates["virtual_original_P_minimum_Ha"])
            signed_coordinate = p.T @ s @ seed
            first = int(np.argmax(abs(signed_coordinate)))
            phase = 1. if signed_coordinate[first] >= 0 else -1.
            extended[key] = (phase * seed[:123], phase * seed[123:])
            save(output / f"virtual_{index}_qualified_complete.npz", **{name: value for name, value in numerical.items()
                 if hasattr(value, "dtype")}, unsigned_seed=seed, phased_seed=np.concatenate(extended[key]), phase=np.array(phase))
            virtual_rows.append(dict(key=key, actual_screened_Fock_minimum_Ha=float(energies[0]),
                 nuclear_h_expectation_Ha=float(seed.astype(np.longdouble) @ h_nuclear.astype(np.longdouble) @ seed.astype(np.longdouble)
                     / (seed.astype(np.longdouble) @ s.astype(np.longdouble) @ seed.astype(np.longdouble))),
                 JK_expectation_Ha=float(seed.astype(np.longdouble) @ jk.astype(np.longdouble) @ seed.astype(np.longdouble)
                     / (seed.astype(np.longdouble) @ s.astype(np.longdouble) @ seed.astype(np.longdouble))), actual_lowest_gap_Ha=float(gap),
                 raw_reported_minus_Rayleigh_Ha=float(reported[int(np.argmin(raw_rayleigh))] - np.min(raw_rayleigh)),
                 same_minimum_span_overlap_squared=overlap_squared, sign_coordinate_index=first, phase=phase,
                 whole_spectrum_rows=118, original4f_S_orthogonality=float(np.max(abs(frozen.T @ s @ np.concatenate(extended[key])))),
                 auxiliary_closed68_field_only=True, no_correlated_Euler_stationarity_claim=True))
            check("actual_Fock_nuclear_plus_JK_Rayleigh_" + str(kappa),
                  abs(virtual_rows[-1]["actual_screened_Fock_minimum_Ha"]
                      - virtual_rows[-1]["nuclear_h_expectation_Ha"] - virtual_rows[-1]["JK_expectation_Ha"]),
                  gates["virtual_original_P_minimum_Ha"])
            write(output / "virtual_construction_controls.json", virtual_rows)
            radial_check("after_new_seed_" + str(index), extended)
            budget(caps["preflight_seconds"])

        r, w = data["gr"].astype(np.longdouble).ravel(), data["grw"].astype(np.longdouble).ravel()
        ordering = np.argsort(r)
        r, w = r[ordering], w[ordering]

        def samples(component):
            values, count = data["bsp_" + component].astype(np.longdouble), int(data["ns" + component])
            out = np.zeros((values.shape[0], values.shape[1], 123), dtype=np.longdouble)
            for cell in range(values.shape[0]):
                width = min(values.shape[2], max(0, count - cell))
                if width:
                    out[cell, :, cell:cell + width] = values[cell, :, :width]
            return out.reshape(-1, 123)[ordering]

        bp, bq = samples("p"), samples("q")
        support_keys = ((5, 3), (5, -4), (4, 3), (4, -4), (6, -1), (5, 2), (5, -3), (6, 1), (6, -2))
        support = []
        for key in support_keys:
            pg, qg = bp @ extended[key][0].astype(np.longdouble), bq @ extended[key][1].astype(np.longdouble)
            density = pg**2 + qg**2
            norm = density @ w
            check("same_quadrature_norm_" + str(key), abs(norm - 1), gates["radial_quadrature_norm_error"])
            mass = density * w / norm
            median = r[min(int(np.searchsorted(np.cumsum(mass), .5)), len(r) - 1)]
            support.append(dict(key=key, norm=float(norm), mean_r_a0=float(mass @ r), median_quadrature_r_a0=float(median),
                rms_r_a0=float(np.sqrt(mass @ r**2)), cumulative_mass={str(radius): float(np.sum(mass[r <= radius]))
                for radius in (.25, .5, 1., 2., 5., 10.)}, adequacy_bound=False))
        old_support = {tuple(row["key"]): row for row in load(roles["nuclear_radial_support"])["rows"]}
        for row in support:
            key = tuple(row["key"])
            row["nuclear_failed_run_diagnostic_mean_r_a0"] = old_support[key]["mean_r_a0"]
            row["screened_minus_nuclear_mean_r_a0"] = row["mean_r_a0"] - old_support[key]["mean_r_a0"]
        write(output / "radial_support.json", dict(rows=support, same_saved_geometry=True,
              nuclear_comparator_overall_qualification=False,
              scope="Closed68-screened finite candidate support. Old nuclear seed support is provisional saved evidence from the preserved late counter failure; no adequacy or radial-convergence bound."))

        nuclear = nuclear_factory(70, "fermi", A=175)
        reference = archive("tracking_ci")
        reference_waves = {tuple(map(int, key)): (pp.copy(), qq.copy()) for key, pp, qq in
                           zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        targets = load(roles["tracking_binding"])["targets"]
        average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
        tracker = tracking_module.CIBasisTracking(average, reference_csfs,
            ci_module.CIResult(reference["energies_Ha"], reference["mixing"]), reference_waves, metrics,
            max_determinant_pairs=caps["maximum_signed_compatible_overlap_pairs_per_call"])
        nominal = {row["label"]: row["NIST_nominal_cm1"] for row in load(roles["nominal_NIST_rows"])["DC_rows"]}
        uid_labels = {tuple(uid): label for label, uid in fixed["target_UIDs"]}
        results = {}
        for name, current in (("baseline", waves), ("complete5f", extended)):
            budget(caps["each_CI_seconds"])
            space, keys = selected[name], tuple(sorted(current))
            current_CI[0] = name
            current_CI_kappas[0] = set()
            prior_requests = {key: hS_requests[key] for key in ("attempted", "completed", "cache_hits", "cache_misses")}
            prior_producers = completed["CI_internal_hS"]
            ci = ci_solve(space, current, keys, data, 70, 123, nuclear_potential=nuclear, convol_cache=cache,
                          solver="dense", n_states=None, include_breit=False, max_determinants=2000,
                          retain_hamiltonian=True)
            hS_requests["by_CI"].append(dict(name=name, delta={key: hS_requests[key] - value
                                                for key, value in prior_requests.items()},
                completed_native_hS_producers=completed["CI_internal_hS"]))
            hS_requests["by_CI"][-1]["native_hS_producers_this_CI"] = completed["CI_internal_hS"] - prior_producers
            hS_requests["by_CI"][-1]["covered_kappas"] = sorted(current_CI_kappas[0])
            check(name + "/hS_complete7kappa_request_coverage", current_CI_kappas[0] != set(metrics))
            write(output / "hS_requests_producers_cache_carry.json", hS_requests)
            h = ci.hamiltonian_for(space, current, keys, data, 70, 123,
                                  nuclear_potential=nuclear, max_determinants=2000)
            save(output / (name + "_whole_CI.npz"), H=h, energies=ci.state_energies, mixing=ci.mixing_coefficients,
                 keys=np.array(keys), P=np.array([current[k][0] for k in keys]), Q=np.array([current[k][1] for k in keys]))
            write(output / (name + "_real_owner.json"), plain(ci.hamiltonian_snapshot.identity_payload()))
            check(name + "/actual_owner_DC", plain(ci.hamiltonian_snapshot.identity_payload())["hamiltonian_terms"] != ["dirac-coulomb"])
            check(name + "/whole_H_Hermiticity", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
            check(name + "/all_CI_norms", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients
                  - np.eye(len(space)))), gates["CI_orthogonality"])
            check(name + "/all_CI_residual_Ha", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
            for sector in sorted(sectors):
                a = [i for i, c in enumerate(space) if (c.total_J2, c.parity) == sector]
                b = [i for i, c in enumerate(space) if (c.total_J2, c.parity) != sector]
                check(name + "/all_cross_symmetry_" + str(sector), np.max(abs(h[np.ix_(a, b)])), gates["cross_symmetry_zero_Ha"])
            weights, tracked = tracker.resolve(space, ci, current, metrics)
            write(output / (name + "_signed_tracking.json"), dict(weights=weights, tracked=tracked, original_reference_unchanged=True))
            check(name + "/five_equal_weights", not (np.count_nonzero(weights) == 5
                  and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            _, blocks = ci_level_blocks(space, ci, require_full=True)
            if name == "baseline":
                check("literal208_baseline_H_bitwise", h.tobytes() != saved["H"].tobytes())
                check("literal208_baseline_reported_roots_bitwise", ci.state_energies.tobytes() != saved["energies"].tobytes())
                check("literal208_baseline_signed_C_bitwise", ci.mixing_coefficients.tobytes() != saved["mixing"].tobytes())
                check("literal208_baseline_owner_identity", plain(ci.hamiltonian_snapshot.identity_payload()) != manifest["owner"])
                original_assignments = {tuple(row["UID"]): tuple(row["selected_level"]) for row in manifest["targets"]}
                current_assignments = {tuple(row["reference_levels"][0]): tuple(row["selected_levels"][0]) for row in tracked}
                check("literal208_baseline_tracking_UIDs", current_assignments != original_assignments)
            else:
                positions = {sig: i for i, sig in enumerate(signatures[name])}
                mapping = [positions[sig] for sig in signatures["baseline"]]
                write(output / "nested_old1046_mapping.json", dict(mapping=mapping, whole_signed_signature_match=True))
                check("nested_whole_signed1046_H", np.max(abs(h[np.ix_(mapping, mapping)] - results["baseline"]["H"])),
                      gates["nested_signed_old1046_H_error_Ha"])
                for sector in sorted(sectors):
                    old = results["baseline"]["ci"].state_energies[results["baseline"]["blocks"][sector]]
                    new = ci.state_energies[blocks[sector]][:len(old)]
                    check("ordered_whole_Ritz_" + str(sector), max(0., float(np.max(new - old))), gates["nested_ordered_Ritz_eigenvalue_increase_Ha"])
            rows = []
            for group in tracked:
                uid = tuple(group["reference_levels"][0])
                check(name + "/complete_UID_rank_" + str(uid), len(group["reference_levels"]) != 1 or len(group["selected_levels"]) != 1)
                check(name + "/UID_declared_" + str(uid), uid not in uid_labels)
                selector = tuple(group["selected_levels"][0])
                check(name + "/UID_same_sector_" + str(uid), selector[:2] != uid[:2])
                column = int(blocks[selector[:2]][selector[2]])
                rows.append(dict(label=uid_labels[uid], reference_UID=uid, selected_level=selector, column=column,
                                 energy_Ha=float(ci.state_energies[column]), principal_overlap_squared=group["principal_overlap_squared"]))
            check(name + "/five_unique_labels", {r["label"] for r in rows} != set(uid_labels.values()) or len(rows) != 5)
            by_label = {row["label"]: row for row in rows}
            lower = by_label["6s_1/2"]["energy_Ha"]
            baseline = results.get("baseline")
            for row in rows:
                excitation = (row["energy_Ha"] - lower) * HARTREE_CM1
                row.update(excitation_cm1=excitation, NIST_nominal_cm1=nominal[row["label"]], NIST_uncertainty_cm1=None,
                    nominal_gap_cm1=excitation - nominal[row["label"]], nominal_relative_gap=
                    (excitation - nominal[row["label"]]) / nominal[row["label"]] if nominal[row["label"]] else None)
                if baseline:
                    du = row["energy_Ha"] - baseline["by_label"][row["label"]]["energy_Ha"]
                    dl = lower - baseline["by_label"]["6s_1/2"]["energy_Ha"]
                    row.update(absolute_delta_upper_Ha=du, absolute_delta_lower_Ha=dl,
                               transition_delta_upper_minus_lower_cm1=(du - dl) * HARTREE_CM1)
            multiplets = []
            for label, a, b, degeneracies in (("5d", "5d_3/2", "5d_5/2", (4, 6)), ("6p", "6p_1/2", "6p_3/2", (2, 4))):
                centroid = sum(g * by_label[k]["excitation_cm1"] for g, k in zip(degeneracies, (a, b), strict=True)) / sum(degeneracies)
                splitting = by_label[b]["excitation_cm1"] - by_label[a]["excitation_cm1"]
                row = dict(label=label, degeneracy_weights=degeneracies, centroid_cm1=centroid, splitting_cm1=splitting)
                if baseline:
                    old = next(r for r in baseline["summary"]["multiplets"] if r["label"] == label)
                    row.update(delta_centroid_cm1=centroid - old["centroid_cm1"], delta_splitting_cm1=splitting - old["splitting_cm1"])
                multiplets.append(row)
            summary = dict(name=name, rows=rows, multiplets=multiplets, EOL_reported_Ha=float(weights @ ci.state_energies),
                nominal_four_gap_RMS_cm1=float(np.sqrt(np.mean([r["nominal_gap_cm1"]**2 for r in rows if r["nominal_relative_gap"] is not None]))),
                nominal_four_1percent_goal_passed=all(abs(r["nominal_relative_gap"]) <= .01 for r in rows if r["nominal_relative_gap"] is not None),
                fixed_literal208=True, full_parent_CSFs=specs[name]["full_CSFs"], complete_target_CSFs=len(space), radial_orbitals=len(current),
                original_stationarity_passed=False, expanded_stationarity_evaluated=False, full_spectrum_certified=False)
            results[name] = dict(H=h, ci=ci, blocks=blocks, by_label=by_label, summary=summary)
            write(output / (name + "_same_point_evidence.json"), summary)
            radial_check(name + "/postCI", current)
            progress(name + "_completed")
        # New prospective source contract: request != producer. Original failed
        # expected14 assertion remains unchanged in the preserved old consumer.
        for counter, expected in (("H", 2), ("CI", 2), ("CI_internal_hS", 7), ("full_scipy_eigh", 12),
                                  ("complement_SVD", 2), ("auxiliary_JK_builder", 4),
                                  ("auxiliary_angular_precompute", 2), ("closed_core_scalar", 1),
                                  ("ConvolCache", 1), ("nuclear_potential_constructor", 1)):
            check("expected_completed_" + counter, completed[counter] != expected)
        check("hS_request_completed_all", hS_requests["completed"] != hS_requests["attempted"])
        check("hS_request_all7kappa_coverage", set(hS_requests["per_kappa"]) != {str(k) for k in metrics})
        check("hS_cache_misses_equal_native_producers", hS_requests["cache_misses"] != completed["CI_internal_hS"])
        check("two_native_CI_h0_carry_events", len(hS_requests["carry_events"]) != 2)
        write(output / "hS_requests_producers_cache_carry.json", hS_requests)
        provisional = load(roles["nuclear_complete5f_diagnostic"])
        previous = {row["label"]: row for row in provisional["rows"]}
        seed_response = []
        for row in results["complete5f"]["summary"]["rows"]:
            old_row = previous[row["label"]]
            check("nuclear_saved_comparator_UID_" + row["label"], tuple(old_row["reference_UID"]) != tuple(row["reference_UID"]))
            seed_response.append(dict(label=row["label"], reference_UID=row["reference_UID"],
                screened_energy_Ha=row["energy_Ha"], provisional_nuclear_energy_Ha=old_row["energy_Ha"],
                screened_minus_nuclear_energy_Ha=row["energy_Ha"] - old_row["energy_Ha"],
                screened_minus_nuclear_excitation_cm1=row["excitation_cm1"] - old_row["excitation_cm1"],
                comparator_overall_qualified=False, additive_correction=False))
        write(output / "screened_vs_provisional_nuclear_seed_response.json", dict(rows=seed_response,
            same_original29_operator_model=True, inference="Finite seed sensitivity only; prior nuclear late counter FAIL preserved, no convergence or missing-physics bound."))
        metadata()
        write(output / "result.json", dict(status="screened_f_literal208_fixed_point_native_DC_responses_completed",
            all_original_numeric_identity_gates_passed=True, checks=checks, attempted=attempted, completed=completed,
            refinement=refinement, hS_request_ledger=hS_requests, signed_scalar_controls=scalar_summary,
            virtuals=virtual_rows, models={n: r["summary"] for n, r in results.items()},
            screened_vs_provisional_nuclear_seed_response=seed_response,
            source_map_sha256=fixed["source_map_sha256"], checkpoint_arrays_sha256=sha(roles["literal208_arrays"]),
            original_metric_sha256=fixed["checkpoint_metric_sha256"], contract_sha256=sha(HERE / "contract.json"),
            driver_sha256=sha(__file__), execution_binding_sha256=sha(binding_path), completed_archives=stages,
            original_stationarity_passed=False, expanded_MCDHF_executed=False, physical_completeness_certified=False,
            literal208_experimental_gate_passed=False, no_best_gap_selection=True,
            limits="Only this native closed68-screened radial seed and finite complete5f family at nonstationary208; auxiliary Fock is not correlated MCDHF Euler, no attribution/space convergence/global spectrum/additive cross-epoch correction",
            elapsed_uncontrolled_s=time.monotonic() - started, peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        progress("complete")
    except Exception as error:
        write(output / "failure.json", dict(status="failed_preserved_no_retry", error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), checks=checks, attempted=attempted, completed=completed, refinement=refinement,
            hS_request_ledger=hS_requests, signed_scalar_controls=scalar_summary,
            completed_archives=stages, original_stationarity_passed=False, expanded_MCDHF_executed=False,
            scientific_success_claim=False, contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
            execution_binding_sha256=sha(binding_path), elapsed_uncontrolled_s=time.monotonic() - started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        progress("failed_preserved")
        raise
    finally:
        for module, name, original in reversed(restored):
            setattr(module, name, original)
        finished.set()
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if args.execute or args.authorize:
        if args.binding is None:
            parser.error("Exact NEW root execution binding required")
        if args.execute:
            execute(args.binding)
        else:
            authorize(args.binding)
            print(json.dumps(dict(metadata_authorization_passed=True, numerical_imports=0, scientific_calls=0)))
    else:
        metadata()
        print(json.dumps(dict(prepared=True, pending_exact_root_binding=True, numerical_imports=0, array_reads=0, scientific_calls=0)))


if __name__ == "__main__":
    main()
