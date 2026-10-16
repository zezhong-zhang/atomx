"""Prepared native three-point discriminator; default is JSON/hash only.

Separate root binding permits exactly3 public CI calls,3 explicit H archives,
one native virtual extension and5 original-P gap controls; no gradient/optimizer.
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
TOP = HERE.parent
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
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(plain(value), indent=2) + "\n")
    temp.replace(path)


def save(np, path, **arrays):
    if path.exists():
        raise FileExistsError("Preserve measured arrays")
    temp = path.with_name(path.name + ".part")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def metadata():
    """Use previously reviewed metadata-only validator; never execute it."""
    c = load(HERE / "contract.json")
    for name, expected in c["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Immutable preparation input changed: " + name)
    namespace = runpy.run_path(str(TOP / "target_sector_equivalence002/run.py"),
                               run_name="metadata_validator_only")
    old, _, paths, checkpoint, proof, source_root = namespace["authorize"](
        TOP / "target_sector_equivalence002/execution_binding.json")
    fixed = c["fixed_conditions"]
    if (old["expected_source_map_sha256"] != fixed["source_map_sha256"]
            or old["expected_checkpoint_objective_sha256"] != fixed["checkpoint"]["objective_sha256"]
            or sha(paths["checkpoint_manifest"]) != fixed["checkpoint"]["manifest_sha256"]
            or sha(paths["checkpoint_arrays"]) != fixed["checkpoint"]["arrays_sha256"]
            or sha(paths["checkpoint_verification"]) != fixed["checkpoint"]["independent_verification_sha256"]
            or checkpoint["iteration"] != 160 or proof["original_stationarity_gates_passed"] is not False):
        raise ValueError("Chosen point/source/failed-stationarity boundary changed")
    source = load(paths["source_identity"])
    if len(source["all_atomx_python_sha256"]) != fixed["full_atomx_source_count"]:
        raise ValueError("Whole353 source identity changed")
    for name, expected in c["audited_critical_source_sha256"].items():
        if source["all_atomx_python_sha256"].get(name) != expected:
            raise ValueError("Audited scientific consumer changed: " + name)
    for role, row in c["input_roles"].items():
        if sha(paths[role]) != row["sha256"] or paths[role] != ROOT / row["path"]:
            raise ValueError("Original geometry/tracker role changed")
    for relative, key in (
            ("target_sector_equivalence002/saved_array_audit/verification.json", "all_independent_saved_array_checks_passed"),
            ("public531_endpoint_static_rci/saved_array_verification.json", "all_passed")):
        if load(TOP / relative).get(key) is not True:
            raise ValueError("Bound independent prior proof failed")
    return c, paths, checkpoint, proof, source_root


def authorize(path):
    """All pending/mismatched new bindings reject before heavy imports."""
    binding, c = load(path), load(HERE / "contract.json")
    fixed = c["fixed_conditions"]
    if (binding.get("execution_authorized_by_root") is not True
            or binding.get("pending_exact_root_execution_contract") is not False
            or binding.get("input_status") != "frozen_verified_public531_checkpoint"
            or binding.get("preparation_contract_sha256") != sha(HERE / "contract.json")
            or binding.get("driver_sha256") != sha(__file__)
            or binding.get("scientific_inputs_sha256") != c["inputs_sha256"]
            or binding.get("expected_checkpoint_manifest_sha256") != fixed["checkpoint"]["manifest_sha256"]
            or binding.get("expected_checkpoint_objective_sha256") != fixed["checkpoint"]["objective_sha256"]
            or binding.get("expected_source_map_sha256") != fixed["source_map_sha256"]
            or binding.get("tracker_inputs_attested_original_objective_by_root") is not True):
        raise ValueError("Missing/mismatched NEW root binding; no science")
    return binding, *metadata()


def execute(binding_path):
    binding, c, paths, checkpoint, proof, source_root = authorize(binding_path)
    if any((HERE / name).exists() for name in (
            "execution_source.py", "resource_admission.json", "result.json", "failure.json", "progress.json")):
        raise FileExistsError("Existing execution retained; require successor")
    caps, gates, fixed = c["resources"], c["acceptance"], c["fixed_conditions"]
    started = time.monotonic()
    calls = dict(attempted_complete_CI=0, completed_complete_CI=0, implicit_H_attempts=0, completed_implicit_H=0,
                 explicit_H_attempts=0, completed_explicit_H=0, virtual_generator_attempts=0,
                 completed_virtual_generator=0, native_nuclear_Dirac_attempts=0,
                 completed_native_nuclear_Dirac=0, native_complement_lowest_root_attempts=0,
                 completed_native_complement_lowest_root=0, original_P_gap_spectral_attempts=0,
                 completed_original_P_gap_spectral=0, initial_one_body_matrix_attempts=0,
                 completed_initial_one_body_matrices=0, gradients=0, optimizer_calls=0)
    checks, stages = [], []
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity, load_average = sorted(os.sched_getaffinity(0)), os.getloadavg()
    memory = {row.split(":", 1)[0]: int(row.split()[1]) for row in Path("/proc/meminfo").read_text().splitlines()
              if len(row.split()) >= 2 and row.split()[1].isdigit()}
    admission = dict(original_affinity=affinity, original_available_CPUs=len(affinity),
                     load_average=load_average, host_available_GiB=memory["MemAvailable"] / 1024**2,
                     load1_per_original_available_CPU=load_average[0] / max(len(affinity), 1),
                     admitted=len(affinity) >= caps["minimum_available_affinity_CPUs"]
                     and memory["MemAvailable"] / 1024**2 >= caps["minimum_host_available_GiB"]
                     and load_average[0] / max(len(affinity), 1) <= caps["max_load1_per_available_CPU"],
                     timing_scope="Uncontrolled diagnostic only")
    write(HERE / "resource_admission.json", admission)
    if not admission["admitted"]:
        write(HERE / "failure.json", dict(error_type="ResourceAdmissionFailure", calls=calls,
              completed_evidence=stages, checks=checks, resource_admission=admission,
              source_map_sha256=fixed["source_map_sha256"], contract_sha256=sha(HERE / "contract.json"),
              driver_sha256=sha(__file__), execution_binding_sha256=sha(binding_path),
              scientific_success_claim=False, elapsed_uncontrolled_s=time.monotonic() - started))
        raise RuntimeError("Resource admission failed before physical evaluation")
    os.sched_setaffinity(0, {affinity[-2]})
    admission.update(selected_CPU=affinity[-2], pinned_affinity=sorted(os.sched_getaffinity(0)))
    write(HERE / "resource_admission.json", admission)
    limit = int(caps["address_space_cap_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    (HERE / "execution_source.py").write_bytes(Path(__file__).read_bytes())
    done = threading.Event()

    def interrupted(*_):
        raise TimeoutError("Finite three-point wall budget exhausted")

    def budget(seconds):
        remaining = int(caps["whole_execution_wall_cap_s"] - (time.monotonic() - started))
        if remaining < 1:
            raise TimeoutError("Whole execution wall budget exhausted")
        signal.alarm(min(seconds, remaining))

    signal.signal(signal.SIGALRM, interrupted)
    budget(caps["preflight_and_virtual_wall_cap_s"])

    def progress(stage):
        write(HERE / "progress.json", dict(stage=stage, calls=calls, completed_evidence=stages,
              elapsed_uncontrolled_s=time.monotonic() - started,
              source_map_sha256=fixed["source_map_sha256"],
              checkpoint_manifest_sha256=fixed["checkpoint"]["manifest_sha256"],
              stationarity_or_experimental_claim=False))

    def heartbeat():
        while not done.wait(30):
            print(json.dumps(dict(heartbeat_elapsed_s=time.monotonic() - started, calls=calls,
                  peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)), flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        sys.path.insert(0, str(source_root))
        import numpy as np
        from scipy.linalg import eigh, null_space

        from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods.ci import CIResult, ci_solve
        from atomx.methods.csf import variational as variational_module
        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.numerics.bspline import orbital_subspace as virtual_module
        from atomx.numerics.bspline.orbital_subspace import extend_nuclear_virtuals
        from atomx.physics.csf import matrix as matrix_module
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.matrix import build_ci_matrix
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.physics.csf.tracking import CIBasisTracking
        from atomx.potentials.nuclear import make_nuclear_potential

        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(source_root.resolve())):
                raise ValueError("Atomx import escaped frozen10d481")

        def check(name, error, threshold):
            row = dict(name=name, error=float(error), limit=threshold,
                       passed=bool(np.isfinite(error) and error <= threshold))
            checks.append(row)
            if not row["passed"]:
                raise ValueError("Scientific/numeric gate failed: " + name)

        def signature(csf):
            s = csf_signature(csf)
            return json.dumps(plain((s.subshells, s.coupling, csf.total_J2, csf.parity)), separators=(",", ":"))

        def counted(module, attribute, attempted, completed, maximum):
            """Observational wrapper forwards every original arg unchanged."""
            original_call = getattr(module, attribute)

            def invoke(*args, **kwargs):
                if calls[attempted] >= maximum:
                    raise RuntimeError("Prospective internal spectral/matrix call cap: " + attempted)
                calls[attempted] += 1
                progress("before_" + attempted)
                value = original_call(*args, **kwargs)
                calls[completed] += 1
                progress("completed_" + completed)
                return value

            return original_call, invoke

        specs = load(HERE / "models.json")
        sectors = {tuple(row) for row in specs["selected_sectors"]}
        full, selected, selected_rows, signatures = {}, {}, {}, {}
        for name, spec in specs["models"].items():
            full[name] = [csf for row in spec["configs"] for csf in enumerate_jj_csfs(row["config"])]
            selected_rows[name] = [i for i, csf in enumerate(full[name]) if (csf.total_J2, csf.parity) in sectors]
            selected[name] = [full[name][i] for i in selected_rows[name]]
            signatures[name] = [signature(csf) for csf in selected[name]]
            check(name + "/full_CSF_count", abs(len(full[name]) - spec["full_CSFs"]), 0)
            check(name + "/target_CSF_count", abs(len(selected[name]) - spec["complete_target_CSFs"]), 0)
            check(name + "/signature_uniqueness", len(signatures[name]) - len(set(signatures[name])), 0)
            check(name + "/electrons69", {csf.n_electrons for csf in full[name]} != {69}, 0)
            check(name + "/full_block_count", abs(len({(csf.total_J2, csf.parity) for csf in full[name]})
                                                - spec["full_blocks"]), 0)
            counts = Counter((csf.total_J2, csf.parity) for csf in selected[name])
            expected = {(r["twoJ"], r["parity"]): r["CSFs"] for r in spec["complete_target_blocks"]}
            check(name + "/complete_target_blocks", counts != expected, 0)
        basis = load(paths["basis_identity"])
        expected = [json.dumps(plain((r["subshells"], r["coupling"], r["twoJ"], r["parity"])), separators=(",", ":"))
                    for r in basis["CSFs"]]
        check("baseline_full_signed_order", [signature(csf) for csf in full["baseline"]] != expected, 0)
        saved, original = [np.load(paths[role], allow_pickle=False)
                           for role in ("checkpoint_arrays", "original_geometry_arrays")]
        radial = np.load(paths["geometry_workspace"], allow_pickle=False)
        data = {key[6:]: radial[key].item() if radial[key].ndim == 0 else radial[key]
                for key in radial.files if key.startswith("basis_")}
        waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                 zip(saved["orbital_keys"], saved["P"], saved["Q"], strict=True)}
        old_keys, new_keys = [tuple(tuple(row) for row in rows) for rows in
                              (fixed["radial_keys"], fixed["virtual_generator"]["new_keys"])]
        check("exact_old24keys", tuple(sorted(waves)) != old_keys, 0)
        check("same_workspace_ns123", data["ns"] != fixed["ns"], 0)

        def raw_groups(csfs, keys):
            labels = [(n, k, m) for n, k in keys for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)]
            groups, raw = {}, {}
            for sector in sorted(sectors):
                masks = set()
                for csf in csfs:
                    if (csf.total_J2, csf.parity) == sector:
                        masks.update(mask for mask, _ in csf_magnetic_vector(csf_signature(csf), sector[0], keys))
                raw[sector] = len(masks)
                groups[sector] = Counter(tuple(sorted((k, m) for i, (_, k, m) in enumerate(labels)
                                                       if mask & (1 << i))) for mask in masks)
            return groups, raw

        reference_groups, _ = raw_groups(full["baseline"], old_keys)
        resource_rows = {}
        for name, space in selected.items():
            keys = tuple(sorted((*old_keys, *new_keys))) if name == "balanced_virtual" else old_keys
            check(name + "/radialkeys_complete", set(get_shared_orbital_basis(space)) != set(keys), 0)
            groups, raw = raw_groups(space, keys)
            pairs = {sector: sum(count * groups[sector][key] for key, count in reference_groups[sector].items())
                     for sector in sectors}
            check(name + "/rawcap", max(raw.values()), caps["max_raw_determinants_per_block"])
            check(name + "/paircap", sum(pairs.values()), caps["max_signed_overlap_determinant_pairs"])
            upper = {(r["twoJ"], r["parity"]): r["raw_M_equals_J_determinants"]
                     for r in specs["models"][name]["complete_target_blocks"]}
            for sector, count in raw.items():
                check(name + "/raw_kinematic_upper/" + str(sector), count, upper[sector])
            resource_rows[name] = dict(blocks=[dict(twoJ=s[0], parity=s[1], raw=count, compatible_pairs=pairs[s])
                                              for s, count in sorted(raw.items())],
                                       compatible_pairs=sum(pairs.values()), before_H_CI_virtual=True)
        write(HERE / "actual_angular_resource_preflight.json", resource_rows)
        layouts, projectors, fingerprint = load(paths["original_layout"]), {}, hashlib.sha256()
        for i, row in enumerate(layouts):
            keys, shape = tuple(tuple(key) for key in row["keys"]), tuple(row["shape"])
            projector, blocks = saved[f"projector_{i}"], saved[f"inverse_blocks_{i}"]
            check("original_P_inverse_bytes/" + str(i), not (np.array_equal(projector, original[f"projector_{i}"])
                  and np.array_equal(blocks, original[f"inverse_blocks_{i}"])), 0)
            fingerprint.update(repr((list(keys), shape)).encode())
            fingerprint.update(np.ascontiguousarray(projector).tobytes())
            coefficients = projector @ saved["x"][slice(*row["slice"])].reshape(shape)
            for j, key in enumerate(keys):
                check("coordinate_oldwave/" + str(key), not (np.array_equal(coefficients[:123, j], waves[key][0])
                      and np.array_equal(coefficients[123:, j], waves[key][1])), 0)
            projectors[keys[0][1]] = projector
        fingerprint.update(saved["inverse_diagonal"].tobytes())
        for i in range(len(layouts)):
            for block in saved[f"inverse_blocks_{i}"]:
                fingerprint.update(np.ascontiguousarray(block).tobytes())
        check("original_inverse_diagonal", not np.array_equal(saved["inverse_diagonal"], original["inverse_diagonal"]), 0)
        check("complete_original_metric_SHA", fingerprint.hexdigest() != fixed["original_metric_sha256"], 0)
        targets = load(paths["tracking_binding"])["targets"]
        selectors = tuple((r["twoJ"], r["parity"], r["models"]["DC"]["local_root"], 1.) for r in targets)
        average, nuclear = CIStateAverage(selectors), make_nuclear_potential(70, "fermi", A=175)
        original_builder, observed_builder = counted(
            variational_module, "build_dirac_matrices", "initial_one_body_matrix_attempts",
            "completed_initial_one_body_matrices", caps["initial_one_body_matrix_kappa_count"])
        variational_module.build_dirac_matrices = observed_builder
        try:
            geometry = CSFOrbitalFunctional(full["baseline"], data, 70, nuclear,
                                           state_average=average, max_determinants=2000)
        finally:
            variational_module.build_dirac_matrices = original_builder
        check("native_HS_all7kappa", len(geometry.metric) != 7, 0)
        for kappa, projector in projectors.items():
            metric = geometry.metric[kappa]
            check("original_P_SGram/" + str(kappa), np.max(abs(projector.T @ metric @ projector
                  - np.eye(projector.shape[1]))), gates["saved_P_metric_Gram_error"])
            keys = [key for key in old_keys if key[1] == kappa]
            coefficients = np.column_stack([np.concatenate(waves[key]) for key in keys])
            check("old24_SGram/" + str(kappa), np.max(abs(coefficients.T @ metric @ coefficients - np.eye(len(keys)))),
                  gates["actual_radial_Gram_error"])
            missing = coefficients - projector @ (projector.T @ metric @ coefficients)
            check("old24_P_leakage/" + str(kappa), np.max(np.sum(missing * (metric @ missing), axis=0)),
                  gates["nuclear_projector_leakage_S_norm_squared"])
        reference = np.load(paths["tracking_ci"], allow_pickle=False)
        radial_reference = np.load(paths["tracking_orbitals"], allow_pickle=False)
        reference_waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                           zip(radial_reference["orbital_keys"], radial_reference["P"], radial_reference["Q"], strict=True)}
        tracker = CIBasisTracking(average, full["baseline"], CIResult(reference["energies_Ha"], reference["mixing"]),
                                  reference_waves, geometry.metric, max_determinant_pairs=1_000_000)
        calls["virtual_generator_attempts"] += 1
        progress("before_native_virtual_extension")
        original_dirac, observed_dirac = counted(
            virtual_module, "solve_dirac", "native_nuclear_Dirac_attempts",
            "completed_native_nuclear_Dirac", caps["nuclear_Dirac_spectral_calls"])
        original_minimum, observed_minimum = counted(
            virtual_module, "eigh", "native_complement_lowest_root_attempts",
            "completed_native_complement_lowest_root", caps["nuclear_complement_lowest_root_spectral_calls"])
        virtual_module.solve_dirac, virtual_module.eigh = observed_dirac, observed_minimum
        try:
            extended, diagnostics = extend_nuclear_virtuals(waves, geometry.one_body, geometry.metric, data,
                                                            layers=1, kappas=fixed["virtual_generator"]["kappas"])
        finally:
            virtual_module.solve_dirac, virtual_module.eigh = original_dirac, original_minimum
        calls["completed_virtual_generator"] += 1
        save(np, HERE / "raw_native_virtuals.npz", radial_keys=np.array(new_keys),
             P=np.array([extended[key][0] for key in new_keys]), Q=np.array([extended[key][1] for key in new_keys]))
        check("native_virtual_complete29keys", set(extended) != set(old_keys) | set(new_keys), 0)
        virtual_controls = []
        for key in new_keys:
            kappa = key[1]
            projector, h, metric = projectors[kappa], geometry.one_body[kappa], geometry.metric[kappa]
            old = [old for old in old_keys if old[1] == kappa]
            frozen = np.column_stack([np.concatenate(waves[old]) for old in old])
            complement = null_space(frozen.T @ metric @ projector)
            check("virtual_complement_rank/" + str(key), complement.shape[1] < 2, 0)
            candidate_basis = projector @ complement
            calls["original_P_gap_spectral_attempts"] += 1
            roots = eigh(candidate_basis.T @ h @ candidate_basis, candidate_basis.T @ metric @ candidate_basis,
                         subset_by_index=(0, 1), eigvals_only=True)
            calls["completed_original_P_gap_spectral"] += 1
            gap = float(roots[1] - roots[0])
            check("virtual_unresolved_minimum/" + str(key), gap <= gates["nuclear_minimum_cluster_gap_Ha"], 0)
            vector = np.concatenate(extended[key])
            coordinate = projector.T @ metric @ vector
            anchor = int(np.argmax(abs(coordinate)))
            check("virtual_phase_anchor/" + str(key), coordinate[anchor] == 0, 0)
            phase = 1. if coordinate[anchor] > 0 else -1.
            vector *= phase
            extended[key] = vector[:123].copy(), vector[123:].copy()
            energy = float(vector @ h @ vector / (vector @ metric @ vector))
            check("virtual_original_P_minimum/" + str(key), abs(energy - roots[0]),
                  gates["nuclear_complement_minimum_energy_match_Ha"])
            missing = vector - projector @ (projector.T @ metric @ vector)
            check("virtual_original_P_leakage/" + str(key), missing @ metric @ missing,
                  gates["nuclear_projector_leakage_S_norm_squared"])
            residual = projector.T @ (h @ vector - energy * (metric @ vector))
            normal = projector.T @ metric @ frozen
            residual -= normal @ (normal.T @ residual)
            relative = np.linalg.norm(residual) / max(1., abs(energy), np.linalg.norm(projector.T @ h @ vector))
            check("virtual_projected_nuclear_residual/" + str(key), relative,
                  gates["new_spinor_projected_nuclear_residual_relative"])
            whole = np.column_stack((frozen, vector))
            check("virtual_whole_SGram/" + str(key), np.max(abs(whole.T @ metric @ whole - np.eye(whole.shape[1]))),
                  gates["actual_radial_Gram_error"])
            virtual_controls.append(dict(key=key, public_diagnostics=diagnostics[key],
                                         original_P_lowest_two_Ha=roots, minimum_gap_Ha=gap,
                                         original_P_phase_anchor=anchor, joint_phase=phase,
                                         projected_nuclear_residual_relative=relative, correlation_label_only=True))
        for key in old_keys:
            check("extended_oldwave_bitwise/" + str(key), not all(np.array_equal(a, b)
                  for a, b in zip(waves[key], extended[key], strict=True)), 0)
        keys29 = tuple(sorted(extended))
        save(np, HERE / "canonical29_orbitals.npz", radial_keys=np.array(keys29),
             P=np.array([extended[key][0] for key in keys29]), Q=np.array([extended[key][1] for key in keys29]),
             original_x=saved["x"])
        write(HERE / "virtual_construction_controls.json", virtual_controls)
        progress("native_positive_virtuals_verified")
        results, baseline_indices = {}, selected_rows["baseline"]
        nominal = {r["label"]: r["NIST_nominal_cm1"] for r in load(HERE / "same_point_evidence.json")["DC_rows"]}
        for name, space in selected.items():
            budget(caps["wall_cap_each_point_s"])
            current = extended if name == "balanced_virtual" else waves
            keys = tuple(sorted(current))
            if calls["attempted_complete_CI"] >= caps["complete_CI_calls_budget"]:
                raise RuntimeError("Three completeCI budget exhausted")
            calls["attempted_complete_CI"] += 1
            progress(name + "/before_public_CI")
            original_ci_h, observed_ci_h = counted(
                matrix_module, "build_ci_matrix", "implicit_H_attempts", "completed_implicit_H",
                caps["implicit_public_CI_H_builds_budget"])
            matrix_module.build_ci_matrix = observed_ci_h
            try:
                ci = ci_solve(space, current, keys, data, 70, 123, solver="dense", n_states=None,
                              nuclear_potential=nuclear, convol_cache=geometry.cache,
                              include_breit=False, max_determinants=2000)
            finally:
                matrix_module.build_ci_matrix = original_ci_h
            calls["completed_complete_CI"] += 1
            ci_path = HERE / (name + "_CI.npz")
            save(np, ci_path, energies_Ha=ci.state_energies, mixing=ci.mixing_coefficients,
                 radial_keys=np.array(keys), indices_in_own_full_basis=np.array(selected_rows[name]))
            stage = dict(name=name, CI_path=ci_path.name, CI_sha256=sha(ci_path), explicit_H_saved=False)
            stages.append(stage)
            progress(name + "/CI_saved_before_gates")
            calls["explicit_H_attempts"] += 1
            if calls["explicit_H_attempts"] > caps["explicit_archive_H_builds_budget"]:
                raise RuntimeError("Three archiveH budget exhausted")
            h = build_ci_matrix(space, current, keys, data, 70, 123, nuclear_potential=nuclear,
                                convol_cache=geometry.cache, include_breit=False, max_determinants=2000)
            calls["completed_explicit_H"] += 1
            h_path = HERE / (name + "_H.npz")
            save(np, h_path, matrix_Ha=h)
            stage.update(explicit_H_saved=True, H_path=h_path.name, H_sha256=sha(h_path))
            progress(name + "/H_saved_before_gates")
            check(name + "/Hermiticity", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
            check(name + "/whole_CI_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
            check(name + "/whole_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients
                  - np.eye(len(space)))), gates["CI_orthogonality"])
            for i, a in enumerate(space):
                other = [j for j, b in enumerate(space) if (a.total_J2, a.parity) != (b.total_J2, b.parity)]
                check(name + "/cross_symmetry/" + str(i), np.max(abs(h[i, other])), gates["cross_symmetry_zero_Ha"])
            weights, tracked = tracker.resolve(space, ci, current, geometry.metric)
            t_path = HERE / (name + "_tracking.json")
            write(t_path, dict(physical_tracking=tracked, state_weights=weights, original_reference_changed=False,
                              complete_CI_provenance=ci.provenance))
            stage.update(tracking_path=t_path.name, tracking_sha256=sha(t_path))
            check(name + "/equal_five_weights", not (np.count_nonzero(weights) == 5
                  and np.array_equal(weights[weights > 0], np.repeat(.2, 5))), 0)
            _, blocks = ci_level_blocks(space, ci, require_full=True)
            baseline = results.get("baseline")
            if name == "baseline":
                check("baseline_saved_principal_H", np.max(abs(h - saved["matrix_Ha"][np.ix_(baseline_indices, baseline_indices)])),
                      gates["baseline_saved_H_replay_Ha"])
                _, old_blocks = ci_level_blocks(full[name], CIResult(saved["energies_Ha"], saved["mixing"]), require_full=True)
                for sector in sorted(sectors):
                    check("baseline_saved_sector_spectrum/" + str(sector), np.max(abs(ci.state_energies[blocks[sector]]
                          - saved["energies_Ha"][old_blocks[sector]])), gates["baseline_saved_target_spectrum_replay_Ha"])
                check("baseline_same_target_assignments", plain([r["selected_levels"] for r in tracked])
                      != plain([r["selected_levels"] for r in checkpoint["physical_tracking"]]), 0)
            else:
                positions = {value: i for i, value in enumerate(signatures[name])}
                mapping = [positions[value] for value in signatures["baseline"]]
                stage["nested_old261_indices"] = mapping
                check(name + "/nested_signed_old261_H", np.max(abs(h[np.ix_(mapping, mapping)] - baseline["h"])),
                      gates["nested_signed_old261_principal_H_error_Ha"])
                for sector in sorted(sectors):
                    previous = baseline["ci"].state_energies[baseline["blocks"][sector]]
                    extended_roots = ci.state_energies[blocks[sector]][:len(previous)]
                    check(name + "/ordered_Ritz/" + str(sector), max(0., float(np.max(extended_roots - previous))),
                          gates["nested_ordered_Ritz_eigenvalue_increase_Ha"])
            rows = []
            for declaration in targets:
                sector = declaration["twoJ"], declaration["parity"]
                group = next(r for r in tracked if tuple(r["reference_levels"][0][:2]) == sector)
                check(name + "/unique_reference_rank/" + str(sector), len(group["selected_levels"]) != 1, 0)
                root = group["selected_levels"][0][2]
                column = int(blocks[sector][root])
                rows.append(dict(label=declaration["label"], twoJ=sector[0], parity=sector[1], local_root=root,
                                 column=column, energy_Ha=float(ci.state_energies[column]),
                                 original_reference_principal_squared=group["principal_overlap_squared"]))
            by_label = {row["label"]: row for row in rows}
            lower = by_label["6s_1/2"]["energy_Ha"]
            for row in rows:
                excitation = (row["energy_Ha"] - lower) * HARTREE_CM1
                row.update(excitation_cm1=excitation, NIST_nominal_cm1=nominal[row["label"]],
                           NIST_uncertainty_cm1=None, nominal_gap_cm1=excitation - nominal[row["label"]],
                           nominal_relative_gap=(excitation - nominal[row["label"]]) / nominal[row["label"]]
                           if nominal[row["label"]] else None)
                if baseline:
                    previous = {r["label"]: r for r in baseline["rows"]}
                    du = row["energy_Ha"] - previous[row["label"]]["energy_Ha"]
                    dl = lower - previous["6s_1/2"]["energy_Ha"]
                    row.update(absolute_delta_upper_Ha=du, absolute_delta_lower_Ha=dl,
                               transition_delta_upper_minus_lower_cm1=(du - dl) * HARTREE_CM1)
            multiplets = []
            for label, a, b, g in (("5d", "5d_3/2", "5d_5/2", [4, 6]), ("6p", "6p_1/2", "6p_3/2", [2, 4])):
                centroid = (g[0] * by_label[a]["excitation_cm1"] + g[1] * by_label[b]["excitation_cm1"]) / sum(g)
                splitting = by_label[b]["excitation_cm1"] - by_label[a]["excitation_cm1"]
                item = dict(label=label, degeneracy_weights=g, centroid_cm1=centroid, splitting_cm1=splitting,
                            nominal_centroid_gap_cm1=centroid - (g[0] * nominal[a] + g[1] * nominal[b]) / sum(g),
                            nominal_splitting_gap_cm1=splitting - (nominal[b] - nominal[a]))
                if baseline:
                    previous = next(r for r in baseline["multiplets"] if r["label"] == label)
                    item.update(delta_centroid_cm1=centroid - previous["centroid_cm1"],
                                delta_splitting_cm1=splitting - previous["splitting_cm1"])
                multiplets.append(item)
            stage.update(rows=rows, multiplets=multiplets, EOL_Ha=float(weights @ ci.state_energies),
                         full_model_CSFs=specs["models"][name]["full_CSFs"], retained_target_CSFs=len(space),
                         signed_basis_sha256=hashlib.sha256(json.dumps(signatures[name]).encode()).hexdigest(),
                         nominal_four_relative_gap_goal_passed=all(abs(r["nominal_relative_gap"]) <= gates["nominal_relative_NIST_goal"]
                                                                   for r in rows if r["nominal_relative_gap"] is not None),
                         nominal_four_gap_RMS_cm1=float(np.sqrt(np.mean([r["nominal_gap_cm1"]**2 for r in rows
                                                                        if r["nominal_relative_gap"] is not None]))))
            results[name] = dict(ci=ci, h=h, blocks=blocks, rows=rows, multiplets=multiplets)
            progress(name + "/complete")
        result = dict(status="three_fixed_point_native_DC_responses_completed", numeric_identity_gates_passed=True,
                      checks=checks, calls=calls, stages=stages, virtual_controls=virtual_controls,
                      source_map_sha256=fixed["source_map_sha256"], original_metric_sha256=fixed["original_metric_sha256"],
                      checkpoint_manifest_sha256=fixed["checkpoint"]["manifest_sha256"],
                      checkpoint_arrays_sha256=fixed["checkpoint"]["arrays_sha256"],
                      contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
                      execution_binding_sha256=sha(binding_path), resource_admission=admission,
                      old24_wave_bytes_preserved=True, original_tracker_rebased=False,
                      inherited_orbital_residual=proof["actual_original_metric_orbital_residual"],
                      original_stationarity_passed=False, expanded_stationarity_evaluated=False,
                      experimental_gate_passed=False, full_spectrum_or_globalground_certified=False,
                      QED_recoil_dynamic_or_static_Breit_added=False, performance_comparison_valid=False,
                      missing_physics_bound=None, independent_saved_array_verification_pending=True,
                      elapsed_uncontrolled_s=time.monotonic() - started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      scope="Separate finite-space nonstationary-point scalarDC responses only; no optimized/combined/additive bound")
        write(HERE / "result.json", result)
        print(json.dumps(dict(status=result["status"], calls=calls, experiment_or_stationarity_passed=False)))
    except Exception as error:
        write(HERE / "failure.json", dict(error_type=type(error).__name__, message=str(error),
              traceback=traceback.format_exc(), checks=checks, calls=calls, completed_evidence=stages,
              source_map_sha256=fixed["source_map_sha256"], contract_sha256=sha(HERE / "contract.json"),
              driver_sha256=sha(__file__), execution_binding_sha256=sha(binding_path),
              scientific_success_claim=False, elapsed_uncontrolled_s=time.monotonic() - started))
        raise
    finally:
        done.set()
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if args.execute or args.authorize:
        if args.binding is None:
            parser.error("NEW root execution binding required")
        if args.execute:
            execute(args.binding)
        else:
            authorize(args.binding)
            print(json.dumps(dict(metadata_authorization_passed=True, physical_imports=0, scientific_calls=0)))
    else:
        contract, _, _, _, _ = metadata()
        print(json.dumps(dict(status=contract["status"], source_and_input_hashes_verified=True,
                              pending_NEW_root_execution_binding=True, physical_imports=0, scientific_calls=0,
                              contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__))))


if __name__ == "__main__":
    main()
