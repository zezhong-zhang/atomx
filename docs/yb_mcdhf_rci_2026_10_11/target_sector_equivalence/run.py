"""Prepare, or explicitly execute, a frozen-point full531/sector261 audit.

The default command is JSON/hash only.  Missing or pending root-owned input
binding refuses execution before NumPy or Atomx imports.  This driver never
calls an optimizer, perturbs orbitals, rebases tracking, or constructs a new
physical space.  The execution budget is two full public CI/gradient calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def json_plain(value):
    if isinstance(value, dict):
        return {str(k): json_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(json_plain(value), indent=2) + "\n")
    temporary.replace(path)


def prepare():
    contract = load(HERE / "contract.json")
    hashes = {name: sha(resolve(name)) == expected
              for name, expected in contract["inputs_sha256"].items()}
    return {"status": contract["status"], "immutable_inputs_match": all(hashes.values()),
            "input_hash_checks": hashes, "pending_checkpoint": False,
            "pending_exact_root_execution_contract": True,
            "execution_started": False, "physics_imports": 0,
            "CI_calls": 0, "gradient_calls": 0, "optimizer_calls": 0,
            "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(__file__)}


def authorize(binding_path):
    """Validate exact root binding and every source/input before heavy imports."""
    binding = load(binding_path)
    contract = load(HERE / "contract.json")
    if (binding.get("execution_authorized_by_root") is not True
            or binding.get("input_status") != "frozen_verified_public531_checkpoint"
            or binding.get("preparation_contract_sha256") != sha(HERE / "contract.json")
            or binding.get("driver_sha256") != sha(__file__)):
        raise ValueError("Pending or mismatched root execution binding; no physical evaluation")
    required = {"checkpoint_manifest", "checkpoint_arrays", "checkpoint_verification",
                "source_identity", "objective_identity", "parent_execution_contract",
                "tracking_ci", "tracking_binding", "tracking_orbitals", "geometry_workspace",
                "basis_identity", "original_layout", "original_geometry_arrays"}
    roles = binding.get("files", {})
    if set(roles) != required:
        raise ValueError("Incomplete or extra scientific input roles")
    paths = {}
    for role, entry in roles.items():
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("Unfrozen input role: " + role)
        path = resolve(entry["path"])
        if sha(path) != entry["sha256"]:
            raise ValueError("Frozen input changed: " + role)
        paths[role] = path
    for name, expected in contract["inputs_sha256"].items():
        if sha(resolve(name)) != expected:
            raise ValueError("Immutable preparation input changed: " + name)
    point = contract["fixed_conditions"]["checkpoint"]
    for role, field in (("checkpoint_manifest", "manifest_sha256"),
                        ("checkpoint_arrays", "arrays_sha256"),
                        ("checkpoint_verification", "independent_verification_sha256")):
        if sha(paths[role]) != point[field]:
            raise ValueError("Prospective exact checkpoint changed: " + role)
    if (binding.get("expected_checkpoint_objective_sha256") != point["objective_sha256"]
            or binding.get("expected_source_map_sha256") != point["source_map_sha256"]):
        raise ValueError("Prospective exact objective/source epoch changed")
    checkpoint = load(paths["checkpoint_manifest"])
    proof = load(paths["checkpoint_verification"])
    if (checkpoint.get("accepted_complete_point") is not True
            or checkpoint.get("array_sha256") != sha(paths["checkpoint_arrays"])
            or checkpoint.get("original_metric_sha256") != contract["fixed_conditions"]["original_metric_sha256"]
            or checkpoint.get("source_identity_sha256") != sha(paths["source_identity"])
            or checkpoint.get("execution_contract_sha256") != sha(paths["parent_execution_contract"])
            or checkpoint.get("objective_sha256") != binding.get("expected_checkpoint_objective_sha256")):
        raise ValueError("Checkpoint does not bind the declared accepted point/source/metric/objective")
    if (proof.get("all_identity_replay_checks_passed") is not True
            or proof.get("accepted_checkpoint_sha256") != sha(paths["checkpoint_manifest"])
            or proof.get("accepted_arrays_sha256") != sha(paths["checkpoint_arrays"])
            or proof.get("new_source_identity_sha256") != sha(paths["source_identity"])
            or proof.get("execution_contract_sha256") != sha(paths["parent_execution_contract"])):
        raise ValueError("Future checkpoint independent identity replay is missing or failed")
    objective = load(paths["objective_identity"])
    if (objective.get("new_sha256") != checkpoint["objective_sha256"]
            or digest(objective["new"]) != checkpoint["objective_sha256"]
            or proof.get("objective_identity_file_sha256") != sha(paths["objective_identity"])
            or proof.get("independently_rebuilt_objective_sha256") != checkpoint["objective_sha256"]
            or binding.get("tracker_inputs_attested_original_objective_by_root") is not True):
        raise ValueError("Original tracker/target objective binding is not explicit")
    payload = objective["new"]
    if (payload.get("physical_reference_CI_sha256") != sha(paths["tracking_ci"])
            or payload.get("physical_reference_binding_sha256") != sha(paths["tracking_binding"])
            or payload.get("radial_archive_sha256") != sha(paths["tracking_orbitals"])):
        raise ValueError("Tracker inputs disagree with original full objective payload")
    basis = load(paths["basis_identity"])
    declarations = load(paths["tracking_binding"])["targets"]
    target_identity = [{"label": row["label"], "twoJ": row["twoJ"], "parity": row["parity"],
                        "root": row["models"]["DC"]["local_root"], "weight": .2}
                       for row in declarations]
    if (payload.get("CSFs") != basis["CSFs"]
            or payload.get("target_identity") != target_identity
            or payload.get("Z") != 70 or payload.get("electrons") != 69
            or payload.get("operator") != "full declared no-pair Dirac-Coulomb"
            or payload.get("nucleus") != "native005 Fermi A175"):
        raise ValueError("Complete signed CSF/target/weight/operator/nucleus objective changed")
    source = load(paths["source_identity"])
    if (source.get("source_map_sha256") != binding.get("expected_source_map_sha256")
            or digest(source["all_atomx_python_sha256"]) != source["source_map_sha256"]
            or payload.get("source_map_sha256") != source["source_map_sha256"]):
        raise ValueError("Frozen full source epoch changed")
    source_root = resolve(binding["frozen_source_root"])
    actual_source_names = {"src/" + str(p.relative_to(source_root))
                           for p in (source_root / "atomx").rglob("*.py")}
    if actual_source_names != set(source["all_atomx_python_sha256"]):
        raise ValueError("Frozen source file set changed")
    for name, expected in source["all_atomx_python_sha256"].items():
        if sha(source_root / Path(name).relative_to("src")) != expected:
            raise ValueError("Frozen source file changed: " + name)
    for name, expected in contract["audited_critical_source_sha256"].items():
        if source["all_atomx_python_sha256"].get(name) != expected:
            raise ValueError("Critical consumer changed; require a new prospective source audit: " + name)
    return binding, contract, paths, checkpoint, proof, source_root


def execute(binding_path):
    binding, contract, paths, checkpoint, proof, source_root = authorize(binding_path)
    if any((HERE / name).exists() for name in ("resource_admission.json", "progress.json", "result.json",
                                              "failure.json", "equivalence_arrays.npz")):
        raise ValueError("An execution/admission record already exists; no repeat or evidence overwrite allowed")
    caps, gates = contract["resources"], contract["acceptance"]
    started = time.monotonic()
    completed_gradients = completed_ci = attempted_gradients = attempted_ci = explicit_h_builds = 0
    checks = []
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    memory = {row.split(":", 1)[0]: int(row.split()[1])
              for row in Path("/proc/meminfo").read_text().splitlines()
              if len(row.split()) >= 2 and row.split()[1].isdigit()}
    load_average = os.getloadavg()
    admission = {"available_CPUs_before_pinning": len(affinity), "original_affinity": affinity,
                 "load_average": load_average,
                 "load1_per_original_available_CPU": load_average[0] / max(len(affinity), 1),
                 "host_available_GiB": memory["MemAvailable"] / 1024**2,
                 "affinity_policy": "original sorted affinity[-2]; fail if fewer than3cores",
                 "admitted": len(affinity) >= caps["minimum_available_affinity_CPUs"]
                 and load_average[0] / max(len(affinity), 1) <= caps["max_load1_per_available_CPU"]
                 and memory["MemAvailable"] / 1024**2 >= caps["minimum_host_available_GiB"],
                 "purpose": "Resource feasibility only; no controlled timing certificate"}
    atomic_json(HERE / "resource_admission.json", admission)
    if not admission["admitted"]:
        raise RuntimeError("Prospective CPU/load/memory admission failed; no physical evaluation")
    admission["pinned_CPU"] = affinity[-2]
    os.sched_setaffinity(0, {admission["pinned_CPU"]})
    admission["actual_pinned_affinity"] = sorted(os.sched_getaffinity(0))
    if admission["actual_pinned_affinity"] != [admission["pinned_CPU"]]:
        raise RuntimeError("Deterministic CPU1 affinity failed; no physical evaluation")
    atomic_json(HERE / "resource_admission.json", admission)
    byte_cap = int(caps["address_space_cap_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (byte_cap, byte_cap))

    def timeout(*_):
        raise TimeoutError("120s exact-sector discriminator cap; no additional solve allowed")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(caps["wall_cap_s"])
    try:
        sys.path.insert(0, str(source_root))
        import numpy as np

        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods.ci import CIResult
        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.physics.csf.levels import CILevel, ci_level_blocks
        from atomx.physics.csf.matrix import build_ci_matrix
        from atomx.physics.csf.overlap import ci_subspace_overlaps
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.physics.csf.tracking import CIBasisTracking
        from atomx.potentials.nuclear import make_nuclear_potential

        def check(name, error, limit):
            value = float(error)
            checks.append({"name": name, "error": value, "limit": limit,
                           "passed": bool(np.isfinite(value) and value <= limit)})
            if not checks[-1]["passed"]:
                raise ValueError("Scientific equivalence gate failed: " + name)

        for name, module in tuple(sys.modules.items()):
            if name == "atomx" or name.startswith("atomx."):
                location = getattr(module, "__file__", None)
                if location and not Path(location).resolve().is_relative_to(source_root.resolve()):
                    raise ValueError("Atomx import escaped frozen source epoch")
        basis = load(paths["basis_identity"])
        csfs = [c for row in basis["configs"] for c in enumerate_jj_csfs(row["config"])]
        if len(csfs) != 531 or {c.n_electrons for c in csfs} != {69}:
            raise ValueError("Complete full531 model changed")
        for csf, row in zip(csfs, basis["CSFs"], strict=True):
            sig = csf_signature(csf)
            if (sig.total_J2 != row["twoJ"] or csf.parity != row["parity"]
                    or json_plain(sig.subshells) != row["subshells"]
                    or json_plain(sig.coupling) != row["coupling"]):
                raise ValueError("Signed full CSF signature/order changed")
        kept = contract["fixed_conditions"]["sector_selection"]["indices_full531"]
        selected = [csfs[i] for i in kept]
        if len(selected) != 261:
            raise ValueError("Complete selected-sector dimension changed")
        original = np.load(paths["original_geometry_arrays"], allow_pickle=False)
        saved = np.load(paths["checkpoint_arrays"], allow_pickle=False)
        workspace = np.load(paths["geometry_workspace"], allow_pickle=False)
        data = {k[6:]: workspace[k].item() if workspace[k].ndim == 0 else workspace[k]
                for k in workspace.files if k.startswith("basis_")}
        waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                 zip(saved["orbital_keys"], saved["P"], saved["Q"], strict=True)}
        expected_keys = tuple(tuple(k) for k in contract["fixed_conditions"]["radial_keys"])
        if tuple(sorted(waves)) != expected_keys or int(data["ns"]) != 123:
            raise ValueError("Full24-key radial domain/workspace changed")
        layout_rows = load(paths["original_layout"])
        layout, inverse = [], []
        digest = hashlib.sha256()
        for i, row in enumerate(layout_rows):
            keys = tuple(tuple(k) for k in row["keys"])
            shape, where = tuple(row["shape"]), slice(*row["slice"])
            projector = saved[f"projector_{i}"]
            blocks = saved[f"inverse_blocks_{i}"]
            if (not np.array_equal(projector, original[f"projector_{i}"])
                    or not np.array_equal(blocks, original[f"inverse_blocks_{i}"])):
                raise ValueError("Original projector/inverse bytes changed")
            digest.update(repr((list(keys), shape)).encode())
            digest.update(np.ascontiguousarray(projector).tobytes())
            layout.append((keys, projector, shape, where))
            inverse.append(blocks)
        if not np.array_equal(saved["inverse_diagonal"], original["inverse_diagonal"]):
            raise ValueError("Original inverse diagonal changed")
        digest.update(saved["inverse_diagonal"].tobytes())
        for blocks in inverse:
            for block in blocks:
                digest.update(np.ascontiguousarray(block).tobytes())
        if digest.hexdigest() != contract["fixed_conditions"]["original_metric_sha256"]:
            raise ValueError("Original complete residual metric fingerprint changed")
        x = saved["x"]
        for keys, projector, shape, where in layout:
            coefficients = projector @ x[where].reshape(shape)
            for j, key in enumerate(keys):
                if not (np.array_equal(coefficients[:123, j], waves[key][0])
                        and np.array_equal(coefficients[123:, j], waves[key][1])):
                    raise ValueError("Checkpoint coordinates/physical waves are not the same exact point")
        declarations = load(paths["tracking_binding"])["targets"]
        selectors = tuple((r["twoJ"], r["parity"], r["models"]["DC"]["local_root"], 1.)
                          for r in declarations)
        expected_target = {(1, 1), (3, 1), (5, 1), (1, -1), (3, -1)}
        if len(selectors) != 5 or {(j, p) for j, p, _, _ in selectors} != expected_target:
            raise ValueError("Original five equal physical targets changed")
        average = CIStateAverage(selectors)
        nuclear = make_nuclear_potential(70, "fermi", A=175)
        geometry = CSFOrbitalFunctional(csfs, data, 70, nuclear,
                                       state_average=average, max_determinants=2000)
        reference_arrays = np.load(paths["tracking_ci"], allow_pickle=False)
        reference_radial = np.load(paths["tracking_orbitals"], allow_pickle=False)
        reference_waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                           zip(reference_radial["orbital_keys"], reference_radial["P"],
                               reference_radial["Q"], strict=True)}
        reference_ci = CIResult(reference_arrays["energies_Ha"], reference_arrays["mixing"])
        tracking = CIBasisTracking(average, csfs, reference_ci, reference_waves,
                                  geometry.metric, max_determinant_pairs=1_000_000)
        functionals = [CSFOrbitalFunctional(space, data, 70, nuclear, state_average=average,
                       level_tracking=tracking, cache=geometry.cache, max_determinants=2000)
                       for space in (csfs, selected)]
        if any(f.keys != expected_keys for f in functionals):
            raise ValueError("Sector key discovery changed24-coordinate domain; no padding/shrink allowed")
        for keys, _, _, _ in layout:
            c = np.column_stack([np.concatenate(waves[k]) for k in keys])
            check("actualSGram/" + str(keys[0][1]),
                  np.max(abs(c.T @ geometry.metric[keys[0][1]] @ c - np.eye(len(keys)))),
                  gates["actual_radial_Gram_error"])
        results = []
        for label, functional in zip(("full531", "five_sectors261"), functionals, strict=True):
            if attempted_gradients >= caps["gradient_budget"] or attempted_ci >= caps["CI_solves_budget"]:
                raise RuntimeError("Two-gradient budget exhausted")
            attempted_gradients += 1
            attempted_ci += 1
            energy, gradient, ci = functional.evaluate(waves, gradient=True)
            completed_gradients += 1
            completed_ci += 1
            h = build_ci_matrix(functional.csfs, waves, functional.keys, data, 70, 123,
                nuclear_potential=nuclear, convol_cache=geometry.cache, max_determinants=2000)
            explicit_h_builds += 1
            check(label + "/Hermiticity", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
            check(label + "/CIresidual", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
            check(label + "/CIGram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients
                  - np.eye(len(ci.state_energies)))), gates["CI_orthogonality"])
            results.append((energy, gradient, ci, h))
            atomic_json(HERE / "progress.json", {"completed_CI_calls": completed_ci,
                "completed_gradient_calls": completed_gradients, "explicit_H_builds": explicit_h_builds,
                "attempted_CI_calls": attempted_ci, "attempted_gradient_calls": attempted_gradients,
                "accepted_checkpoint_sha256": sha(paths["checkpoint_manifest"]),
                "stationarity_or_equivalence_claim": False, "completed_stage": label})
        ef, gf, cf, hf = results[0]
        es, gs, cs, hs = results[1]
        check("saved_full_H_replay", np.max(abs(hf - saved["matrix_Ha"])), gates["saved_full_matrix_replay_Ha"])
        check("saved_complete_full_spectrum_replay", np.max(abs(cf.state_energies - saved["energies_Ha"])),
              gates["saved_complete_full_spectrum_replay_Ha"])
        check("frozen_point_EOL_replay", abs(ef - checkpoint["diagnostics"]["energy_ha"]),
              gates["frozen_point_EOL_replay_Ha"])
        check("principal_whole_sector_H", np.max(abs(hs - hf[np.ix_(kept, kept)])), gates["matrix_full_subset_equivalence_Ha"])
        omitted = sorted(set(range(531)) - set(kept))
        check("retained_to_omitted_sector_H", np.max(abs(hf[np.ix_(kept, omitted)])), gates["cross_symmetry_zero_Ha"])
        check("equal_EOL_energy", abs(ef - es), gates["EOL_equivalence_Ha"])
        wf, ws = [f.last_state_weights for f in functionals]
        for label, weights in (("full", wf), ("sector", ws)):
            if np.count_nonzero(weights) != 5 or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)):
                raise ValueError(label + " original equalfive target weights changed")
        full_diag, sector_diag = [f.last_tracking_diagnostics for f in functionals]
        if ([r["selected_levels"] for r in full_diag] != [r["selected_levels"] for r in sector_diag]
                or [r["selected_levels"] for r in full_diag] != [r["selected_levels"] for r in checkpoint["physical_tracking"]]):
            raise ValueError("Frozen physical target assignment changed")
        _, fb = ci_level_blocks(csfs, cf, require_full=True)
        _, sb = ci_level_blocks(selected, cs, require_full=True)
        spectra = []
        for sector in sorted(expected_target):
            fcols, scols = fb[sector], sb[sector]
            check("whole_sector_spectrum/" + str(sector),
                  np.max(abs(cf.state_energies[fcols] - cs.state_energies[scols])),
                  gates["target_energy_equivalence_Ha"])
        for row in full_diag:
            j, p, root = row["selected_levels"][0]
            fcols, scols = fb[j, p], sb[j, p]
            roots = {root}
            changed = True
            while changed:
                before = len(roots)
                for i in tuple(roots):
                    roots.update(k for k in range(len(fcols)) if
                        min(abs(cf.state_energies[fcols[k]] - cf.state_energies[fcols[i]]),
                            abs(cs.state_energies[scols[k]] - cs.state_energies[scols[i]]))
                        <= gates["near_degenerate_cluster_gap_Ha"])
                changed = len(roots) != before
            levels = tuple(CILevel(j, p, i) for i in sorted(roots))
            overlap = ci_subspace_overlaps(csfs, cf, waves, levels, selected, cs, waves,
                                          levels, geometry.metric, max_determinant_pairs=1_000_000)
            principal = np.linalg.svd(overlap, compute_uv=False) ** 2
            check("physical_signed_span/" + str((j, p)), np.max(abs(principal - 1.)),
                  gates["squared_principal_overlap_error"])
            left = cf.mixing_coefficients[np.ix_(kept, fcols[list(sorted(roots))])]
            right = cs.mixing_coefficients[:, scols[list(sorted(roots))]]
            check("complete_span_projector/" + str((j, p)),
                  np.max(abs(left @ left.T - right @ right.T)), gates["complete_degenerate_projector_error"])
            if len(roots) == 1:
                sign = 1. if float(left[:, 0] @ right[:, 0]) >= 0 else -1.
                check("signed_target_vector/" + str((j, p)),
                      np.max(abs(left[:, 0] - sign * right[:, 0])), gates["signed_non_degenerate_vector_max"])
            spectra.append({"twoJ": j, "parity": p, "local_roots_in_complete_comparison_span": sorted(roots),
                "signed_overlap": overlap, "squared_principal_overlaps": principal,
                "target_full_energy_Ha": cf.state_energies[fcols[root]],
                "target_sector_energy_Ha": cs.state_energies[scols[root]],
                "comparison_span_does_not_change_EOL_weights": True})
        radial_full = np.array([gf[k] for k in expected_keys])
        radial_sector = np.array([gs[k] for k in expected_keys])
        check("full24_Euclidean_gradient_absolute", np.max(abs(radial_full - radial_sector)),
              gates["full_radial_gradient_max_absolute"])
        check("full24_Euclidean_gradient_relative", np.max(abs(radial_full - radial_sector))
              / max(float(np.max(abs(radial_full))), 1e-30), gates["full_radial_gradient_max_relative"])

        def original_metric_gradient(gradient):
            tangent = np.zeros_like(x)
            for keys, projector, shape, where in layout:
                y = x[where].reshape(shape)
                block = projector.T @ np.column_stack([gradient[k] for k in keys])
                multiplier = y.T @ block
                tangent[where] = (block - y @ (.5 * (multiplier + multiplier.T))).ravel()
            direction = np.zeros_like(x)
            for (_, _, shape, where), blocks in zip(layout, inverse, strict=True):
                y = x[where].reshape(shape)
                g = tangent[where].reshape(shape)
                z = np.column_stack([block @ g[:, col] for col, block in enumerate(blocks)])
                multiplier = y.T @ z
                direction[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
            return tangent, direction, float(np.linalg.norm(direction)), .5 * abs(float(tangent @ direction))

        tf, pf, rf, qf = original_metric_gradient(gf)
        ts, ps, rs, qs = original_metric_gradient(gs)
        check("original_tangent_gradient", np.max(abs(tf - ts)), gates["original_metric_tangent_gradient_max_absolute"])
        check("original_preconditioned_gradient", np.max(abs(pf - ps)), gates["original_metric_preconditioned_gradient_max_absolute"])
        check("original_residual_absolute", abs(rf - rs), gates["original_metric_residual_absolute"])
        check("original_residual_relative", abs(rf - rs) / max(rf, 1e-7), gates["original_metric_residual_relative"])
        check("checkpoint_original_residual_replay", abs(rf - checkpoint["diagnostics"]["orbital_residual"]),
              gates["original_metric_residual_absolute"])
        check("original_energy_indicator_absolute", abs(qf - qs), gates["original_metric_energy_indicator_absolute_Ha"])
        check("original_energy_indicator_relative", abs(qf - qs) / max(qf, 1e-8),
              gates["original_metric_energy_indicator_relative"])
        check("checkpoint_original_energy_indicator_replay",
              abs(qf - checkpoint["diagnostics"]["estimated_energy_error"]),
              gates["original_metric_energy_indicator_absolute_Ha"])
        union = sorted(set(functionals[0].last_weights) | set(functionals[1].last_weights), key=repr)
        rdm = [{"term": term, "full": functionals[0].last_weights.get(term, 0.),
                "sector": functionals[1].last_weights.get(term, 0.)} for term in union]
        check("coherent_RDM_union", max(abs(r["full"] - r["sector"]) for r in rdm), gates["RDM_key_union_max_absolute"])
        np.savez_compressed(HERE / "equivalence_arrays.npz", full_matrix_Ha=hf, sector_matrix_Ha=hs,
            full_energies_Ha=cf.state_energies, sector_energies_Ha=cs.state_energies,
            full_mixing=cf.mixing_coefficients, sector_mixing=cs.mixing_coefficients,
            full_radial_gradient=radial_full, sector_radial_gradient=radial_sector,
            original_full_tangent=tf, original_sector_tangent=ts,
            original_full_preconditioned=pf, original_sector_preconditioned=ps,
            original_x=x, original_inverse_diagonal=saved["inverse_diagonal"], retained_CSF_indices=kept)
        atomic_json(HERE / "coherent_RDM_comparison.json", rdm)
        result = {"equivalence_gates_passed": True, "checks": checks,
            "source_map_sha256": binding["expected_source_map_sha256"],
            "contract_sha256": sha(HERE / "contract.json"), "execution_binding_sha256": sha(binding_path),
            "accepted_checkpoint_sha256": sha(paths["checkpoint_manifest"]),
            "accepted_arrays_sha256": sha(paths["checkpoint_arrays"]), "full24_radial_keys": expected_keys,
            "original_metric_sha256": digest.hexdigest(), "target_signed_spans": spectra,
            "full_EOL_Ha": ef, "sector_EOL_Ha": es, "original_full_residual": rf,
            "original_sector_residual": rs, "original_full_energy_indicator_Ha": qf,
            "original_sector_energy_indicator_Ha": qs,
            "original_stationarity_already_passed": proof["original_stationarity_gates_passed"],
            "scope": "Exact complete scalar-H target-sector/EOL equivalence at one frozen accepted point only; no new physics, no all-spectrum/globalground or experimental certificate.",
            "completed_CI_calls": completed_ci, "completed_gradient_calls": completed_gradients,
            "attempted_CI_calls": attempted_ci, "attempted_gradient_calls": attempted_gradients,
            "explicit_H_builds": explicit_h_builds, "optimizer_calls": 0,
            "resource_admission": admission,
            "seconds_uncontrolled": time.monotonic() - started,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "parallel_or_end_to_end_speedup_claim": False, "experimental_gate_passed": False,
            "arrays_sha256": sha(HERE / "equivalence_arrays.npz")}
        atomic_json(HERE / "result.json", result)
        print(json.dumps({"equivalence_passed": True, "gradient_calls": completed_gradients,
                          "original_full_residual": rf, "original_sector_residual": rs}))
    except Exception as error:
        atomic_json(HERE / "failure.json", {"status": "failed_or_resource_stopped_unqualified",
            "error": str(error), "traceback": traceback.format_exc(), "checks": checks,
            "completed_CI_calls": completed_ci, "completed_gradient_calls": completed_gradients,
            "attempted_CI_calls": attempted_ci, "attempted_gradient_calls": attempted_gradients,
            "explicit_H_builds": explicit_h_builds, "optimizer_calls": 0,
            "equivalence_passed": False, "experimental_gate_passed": False})
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path, help="Separate exact root-owned future input/execution binding")
    args = parser.parse_args()
    if args.execute is None:
        print(json.dumps(prepare(), indent=2))
    else:
        execute(args.execute)


if __name__ == "__main__":
    main()
