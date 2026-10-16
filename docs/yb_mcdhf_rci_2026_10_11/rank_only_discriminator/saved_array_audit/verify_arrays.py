"""Independent saved complete CI/H algebra; no Atomx or new eigensolve."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import signal
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    audit = load(HERE / "contract.json")
    for name, expected in audit["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen saved rank evidence changed: " + name)
    if (HERE / "verification.json").exists():
        raise FileExistsError("Preserve completed independent rank audit")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError("Require original affinity >=3")
    os.sched_setaffinity(0, {affinity[-2]})
    cap = audit["resources"]["address_space_GiB"] * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    def timeout(*_):
        raise TimeoutError("Saved rank array audit wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(audit["resources"]["wall_s"])
    import numpy as np

    checks, outputs = [], []

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            raise ValueError("Independent rank saved-array check failed: " + name)

    contract, result = load(RUN / "contract.json"), load(RUN / "result.json")
    gates = contract["acceptance"]
    roles = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    checkpoint = np.load(roles["checkpoint_arrays"], allow_pickle=False)
    old = np.load(roles["original_geometry_arrays"], allow_pickle=False)
    layout = load(roles["original_layout"])
    point = load(roles["checkpoint_manifest"])
    basis = load(roles["basis_identity"])
    specs = load(RUN / "models.json")["models"]
    saved_rows = load(RUN / "basis_rows.json")
    sectors = {tuple(row) for row in contract["fixed_conditions"]["sectors"]}
    check("source_epoch10d481", result["source_map_sha256"] != contract["fixed_conditions"]["source_map_sha256"])
    check("original160_point", result["checkpoint_manifest_sha256"] != sha(roles["checkpoint_manifest"]))
    check("exact_contract", result["contract_sha256"] != sha(RUN / "contract.json"))
    check("exact_driver", result["driver_sha256"] != sha(RUN / "run.py"))
    check("root_execution_binding", result["execution_binding_sha256"] != sha(RUN / "execution_binding.json"))
    check("runtime_all1618_gates_passed", len(result["checks"]) != 1618 or not all(r["passed"] for r in result["checks"]))
    for name, count in result["calls"].items():
        expected = 2 if name in ("attempted_complete_CI", "completed_complete_CI", "implicit_H_attempts",
            "completed_implicit_H", "explicit_H_attempts", "completed_explicit_H") else 7 if name in (
            "initial_one_body_matrix_attempts", "completed_initial_one_body_matrices") else 0
        check("actual_count/" + name, count != expected)
    for name in ("original_stationarity_passed", "expanded_stationarity_evaluated", "experimental_gate_passed",
                 "full_spectrum_or_globalground_certified", "QED_recoil_dynamic_or_static_Breit_added",
                 "performance_comparison_valid", "original_tracker_rebased"):
        check("claim_boundary/" + name, result[name] is not False)
    check("original24_bytes_claim", result["old24_wave_bytes_preserved"] is not True)
    check("old_residual_failure", result["inherited_orbital_residual"] != point["diagnostics"]["orbital_residual"]
          or result["inherited_orbital_residual"] <= 1e-7)
    def signature(row):
        return json.dumps([row["subshells"], row["coupling"], row["twoJ"], row["parity"]], separators=(",", ":"))
    check("baseline_original_signed531", saved_rows["baseline"]["full_signatures"] != [signature(r) for r in basis["CSFs"]])
    constant_tree = ast.parse((ROOT / contract["fixed_conditions"]["frozen_source_root"] / "atomx/constants.py").read_text())
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in constant_tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("CODATA", "DEFAULT_CODATA_VERSION")}
    codata = literals["CODATA"][literals["DEFAULT_CODATA_VERSION"]]
    cm1 = codata["_Ha_J"] / (100 * codata["_hplanck"] * codata["_c"])
    data, row_by_label = {}, {}
    for stage in result["stages"]:
        name, model = stage["name"], specs[stage["name"]]
        for path, expected in ((stage["CI_path"], stage["CI_sha256"]), (stage["H_path"], stage["H_sha256"]),
                               (stage["tracking_path"], stage["tracking_sha256"])):
            check(name + "/archive_SHA/" + path, sha(RUN / path) != expected)
        ci = np.load(RUN / stage["CI_path"], allow_pickle=False)
        h = np.load(RUN / stage["H_path"], allow_pickle=False)["matrix_Ha"]
        energies, mixing = ci["energies_Ha"], ci["mixing"]
        declarations = saved_rows[name]
        full_sig, full_sector = declarations["full_signatures"], declarations["full_sectors"]
        selected = [i for i, sector in enumerate(full_sector) if tuple(sector) in sectors]
        signatures = [full_sig[i] for i in selected]
        row_sectors = [tuple(full_sector[i]) for i in selected]
        dimension = model["complete_target_CSFs"]
        check(name + "/complete_full_selected_counts", len(full_sig) != model["full_CSFs"] or len(selected) != dimension)
        check(name + "/complete_sector_subset", selected != declarations["retained_indices"]
              or signatures != declarations["retained_signatures"])
        check(name + "/selected_signature_uniqueness", len(set(signatures)) != dimension)
        check(name + "/signed_basis_SHA", hashlib.sha256(json.dumps(signatures).encode()).hexdigest() != stage["signed_basis_sha256"])
        check(name + "/selected_array_indices", not np.array_equal(ci["indices_in_own_full_basis"], selected))
        check(name + "/all24_radialkeys", not np.array_equal(ci["radial_keys"], checkpoint["orbital_keys"]))
        counts = Counter(row_sectors)
        check(name + "/all_five_complete_counts", counts != {
            (r["twoJ"], r["parity"]): r["CSFs"] for r in model["complete_target_blocks"]})
        check(name + "/matrix_spectrum_shapes", h.shape != (dimension, dimension) or mixing.shape != h.shape
              or energies.shape != (dimension,))
        check(name + "/all_finite", any(not np.isfinite(a).all() for a in (h, energies, mixing)))
        check(name + "/Hermitian", np.max(abs(h-h.T)), gates["matrix_hermiticity_Ha"])
        check(name + "/all_eigenpair_residuals", np.max(np.linalg.norm(h @ mixing-mixing*energies, axis=0)), gates["CI_residual_Ha"])
        check(name + "/whole_CI_Gram", np.max(abs(mixing.T @ mixing-np.eye(dimension))), gates["CI_orthogonality"])
        rows = {sector: [i for i, s in enumerate(row_sectors) if s == sector] for sector in sectors}
        blocks = {sector: [] for sector in sectors}
        for col in range(dimension):
            masses = {sector: float(mixing[where, col] @ mixing[where, col]) for sector, where in rows.items()}
            chosen = max(masses, key=masses.get)
            check(name + "/column_support/" + str(col), abs(masses[chosen]-1), gates["CI_orthogonality"])
            check(name + "/column_other_sectors/" + str(col), sum(v for s, v in masses.items() if s != chosen),
                  gates["CI_orthogonality"])
            blocks[chosen].append(col)
        for sector, where in rows.items():
            blocks[sector].sort(key=lambda col: (energies[col], col))
            check(name + "/whole_block_spectrum_dimension/" + str(sector), len(blocks[sector]) != len(where))
            outside = [i for i in range(dimension) if i not in where]
            check(name + "/cross_sector_exact/" + str(sector), np.max(abs(h[np.ix_(where, outside)])),
                  gates["cross_symmetry_zero_Ha"])
        tracking = load(RUN / stage["tracking_path"])
        check(name + "/immutable_reference", tracking["original_reference_changed"] is not False)
        check(name + "/original_target_reference_levels", [r["reference_levels"] for r in tracking["physical_tracking"]]
              != [r["reference_levels"] for r in point["physical_tracking"]])
        weights, selected_columns = np.zeros(dimension), []
        by_label = {row["label"]: row for row in stage["rows"]}
        lower = by_label["6s_1/2"]["energy_Ha"]
        for row in stage["rows"]:
            sector = row["twoJ"], row["parity"]
            col = blocks[sector][row["local_root"]]
            check(name + "/physical_column/" + row["label"], col != row["column"])
            check(name + "/selected_energy/" + row["label"], abs(float(energies[col])-row["energy_Ha"]))
            weights[col] = .2
            selected_columns.append(col)
            group = next(g for g in tracking["physical_tracking"] if tuple(g["reference_levels"][0][:2]) == sector)
            check(name + "/selected_level/" + row["label"], group["selected_levels"] != [[*sector, row["local_root"]]])
            principal = group["principal_overlap_squared"]
            check(name + "/physical_majority/" + row["label"], min(principal) <= .5 or max(principal) > 1+1e-8)
            check(name + "/saved_principal_replay/" + row["label"], principal != row["original_reference_principal_squared"])
            cluster = [i for i in blocks[sector] if abs(energies[i]-energies[col]) < 1e-10]
            check(name + "/complete_equal_weight_degenerate_cluster/" + row["label"], any(weights[i] != .2 for i in cluster))
            excitation = (energies[col]-lower)*cm1
            check(name + "/upper_lower_excitation/" + row["label"], abs(excitation-row["excitation_cm1"]), 1e-7)
            check(name + "/nominal_gap/" + row["label"], abs(excitation-row["NIST_nominal_cm1"]-row["nominal_gap_cm1"]), 1e-7)
        check(name + "/five_equal_weights", not np.array_equal(weights, tracking["state_weights"]))
        check(name + "/EOL_replay", abs(float(weights @ energies)-stage["EOL_Ha"]), 5e-9)
        for multiplet, a, b, g in (("5d", "5d_3/2", "5d_5/2", (4, 6)), ("6p", "6p_1/2", "6p_3/2", (2, 4))):
            report = next(row for row in stage["multiplets"] if row["label"] == multiplet)
            centroid = (g[0]*by_label[a]["excitation_cm1"]+g[1]*by_label[b]["excitation_cm1"])/sum(g)
            splitting = by_label[b]["excitation_cm1"]-by_label[a]["excitation_cm1"]
            check(name + "/centroid/" + multiplet, abs(centroid-report["centroid_cm1"]), 1e-7)
            check(name + "/splitting/" + multiplet, abs(splitting-report["splitting_cm1"]), 1e-7)
        rms = float(np.sqrt(np.mean([r["nominal_gap_cm1"]**2 for r in stage["rows"] if r["nominal_relative_gap"] is not None])))
        check(name + "/RMS_gap", abs(rms-stage["nominal_four_gap_RMS_cm1"]), 1e-7)
        data[name], row_by_label[name] = (h, energies, mixing, signatures, blocks, rows, selected_columns), by_label
    hb, eb, cb, sb, bb, rb, _ = data["baseline"]
    hr, er, cr, sr, br, rr, _ = data["rank_5p_double"]
    mapping = [sr.index(sig) for sig in sb]
    check("signed_old261_mapping", mapping != result["stages"][1]["nested_old261_indices"])
    check("whole_signed_nested261_H", np.max(abs(hr[np.ix_(mapping, mapping)]-hb)), gates["nested_signed_old261_principal_H_error_Ha"])
    baseline_indices = saved_rows["baseline"]["retained_indices"]
    check("baseline_same160_H", np.max(abs(hb-checkpoint["matrix_Ha"][np.ix_(baseline_indices, baseline_indices)])),
          gates["baseline_saved_H_replay_Ha"])
    for sector in sectors:
        check("all_ordered_nested_Ritz_roots/" + str(sector), max(0., float(np.max(
              er[br[sector]][:len(bb[sector])]-eb[bb[sector]]))), gates["nested_ordered_Ritz_eigenvalue_increase_Ha"])
    lower_delta = row_by_label["rank_5p_double"]["6s_1/2"]["energy_Ha"]-row_by_label["baseline"]["6s_1/2"]["energy_Ha"]
    for label, row in row_by_label["rank_5p_double"].items():
        oldrow = row_by_label["baseline"][label]
        upper_delta = row["energy_Ha"]-oldrow["energy_Ha"]
        response = (upper_delta-lower_delta)*cm1
        check("individual_upper_delta/" + label, abs(upper_delta-row["absolute_delta_upper_Ha"]), 1e-12)
        check("individual_lower_delta/" + label, abs(lower_delta-row["absolute_delta_lower_Ha"]), 1e-12)
        check("transition_response_upper_minus_lower/" + label,
              abs(response-row["transition_delta_upper_minus_lower_cm1"]), 1e-7)
        left, right = cb[:, oldrow["column"]], cr[mapping, row["column"]]
        signed_overlap = float(left @ right)
        outputs.append(dict(label=label, upper_delta_Ha=upper_delta, lower_delta_Ha=lower_delta,
                            transition_delta_cm1=response, same160_signed_coefficient_overlap=signed_overlap,
                            same160_embedded_baseline_principal_squared=signed_overlap**2))
    fingerprint = hashlib.sha256()
    for i, row in enumerate(layout):
        keys, shape = [tuple(k) for k in row["keys"]], tuple(row["shape"])
        p = checkpoint[f"projector_{i}"]
        check("original_P_bytes/" + str(i), not np.array_equal(p, old[f"projector_{i}"]))
        check("original_inverse_blocks/" + str(i), not np.array_equal(checkpoint[f"inverse_blocks_{i}"], old[f"inverse_blocks_{i}"]))
        fingerprint.update(repr((keys, shape)).encode())
        fingerprint.update(np.ascontiguousarray(p).tobytes())
        wave = p @ checkpoint["x"][slice(*row["slice"])].reshape(shape)
        for j, key in enumerate(keys):
            index = list(map(tuple, checkpoint["orbital_keys"])).index(key)
            check("original_point_wave/" + str(key), not np.array_equal(wave[:, j],
                  np.concatenate((checkpoint["P"][index], checkpoint["Q"][index]))))
    fingerprint.update(checkpoint["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        for block in checkpoint[f"inverse_blocks_{i}"]:
            fingerprint.update(np.ascontiguousarray(block).tobytes())
    check("original_full_metric_SHA", fingerprint.hexdigest() != result["original_metric_sha256"])
    check("no_Atomx_imported", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    signal.alarm(0)
    proof = dict(all_independent_saved_array_checks_passed=True, checks_count=len(checks), checks=checks,
                 transition_controls=outputs, bound_input_sha256=audit["inputs_sha256"],
                 audit_contract_sha256=sha(HERE / "contract.json"), audit_driver_sha256=sha(__file__),
                 new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves=0, Atomx_imports=0,
                 original_stationarity_passed=False, expanded_stationarity_evaluated=False,
                 experimental_gate_passed=False, all_spectrum_globalground_certified=False,
                 limits="Archived native complete angular signatures are hash/sector/signed-nesting audited, not independently recontracted. Original physical minor-overlap probabilities are recorded public outputs replayed, not newly recontracted. Same-point signed coefficient overlaps are independently computed. S/no-pair inherited from bound point/runtime gates; no new one-body builder.",
                 scope="Same accepted160 old24 finite DC complete5p-double response; lower/upper, centroids and splitting separately. No virtual-arm, optimized model, physics completeness, or additive correction claim.")
    (HERE / "verification.json").write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps(dict(checks=len(checks), all_passed=True, transitions=outputs)))


if __name__ == "__main__":
    main()
