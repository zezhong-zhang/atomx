"""Native 68-CSF fixed-run005-orbital core-hole RCI discriminator."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "same005_orbitals001"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), OUT / "run_source.py")
    contract = json.loads((HERE / "contract.json").read_text())
    fixed, gates = contract["fixed_conditions"], contract["acceptance"]
    inp, frozen = ROOT / fixed["input_run"], ROOT / fixed["frozen_source_root"]
    for name, expected in contract["inputs"].items():
        if sha(inp / name) != expected:
            raise ValueError(f"saved native005 input changed: {name}")
    source = json.loads((inp / "source_identity.json").read_text())
    for name, expected in source["all_atomx_python_sha256"].items():
        if sha(frozen / Path(name).relative_to("src")) != expected:
            raise ValueError(f"frozen source changed: {name}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError("prospective 900 second CV discriminator cap exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            print(json.dumps({"heartbeat_elapsed_s": time.monotonic() - started,
                              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}),
                  flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    sys.path.insert(0, str(frozen))
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
    from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import ci_solve
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile

    baseline = json.loads((inp / "result.json").read_text())
    native = np.load(inp / "ci_arrays.npz")
    radial = np.load(inp / "orbitals_workspace.npz")
    workspace = {key.removeprefix("basis_"): radial[key].item() if radial[key].ndim == 0 else radial[key]
                 for key in radial.files if key.startswith("basis_")}
    keys = tuple(map(tuple, radial["orbital_keys"].tolist()))
    orbitals = {key: (p, q) for key, p, q in zip(keys, radial["P"], radial["Q"], strict=True)}
    if len(keys) != 24 or workspace["ns"] != 123 or workspace["ks"] != 9:
        raise ValueError("native radial representation changed")
    if nuclear_profile(70, "fermi") != fixed["nucleus"]:
        raise ValueError("native A175 Fermi identity changed")
    if not baseline["stationarity_passed"]:
        raise ValueError("source005 bare MCDHF stationarity did not pass")

    csfs, family_of_csf, families = [], [], []
    for family in fixed["families"]:
        here = [csf for config in family["configs"] for csf in enumerate_jj_csfs(config)]
        if len(here) != family["expected_csfs"]:
            raise ValueError(f"complete CSF family count changed: {family['name']}")
        families.append({**family, "indices": list(range(len(csfs), len(csfs) + len(here)))})
        csfs.extend(here)
        family_of_csf.extend([family["name"]] * len(here))
    n = len(csfs)
    if n != 68 or {csf.n_electrons for csf in csfs} != {69}:
        raise ValueError("prospective full 68-CSF/69-electron space changed")
    if set(get_shared_orbital_basis(csfs)) != set(keys):
        raise ValueError("CV added or omitted native radial orbitals")
    signatures = [csf_signature(csf) for csf in csfs]
    if len(set(signatures)) != n:
        raise ValueError("duplicate signed CSF/coupling-tree identity")
    blocks = {}
    for index, csf in enumerate(csfs):
        blocks.setdefault((csf.total_J2, csf.parity), []).append(index)
    preflight_blocks = []
    for (j2, parity), indices in sorted(blocks.items()):
        patterns = {signatures[index].pattern for index in indices}
        counts = [(pattern, dict(pattern_m_counts(pattern)).get(j2, 0)) for pattern in sorted(patterns)]
        raw = sum(count for _, count in counts)
        preflight_blocks.append({"twoJ": j2, "parity": parity, "CSFs": len(indices),
                                 "raw_determinants": raw, "unique_patterns": len(patterns),
                                 "pattern_raw_counts": counts, "pair_connectivity_checks_upper_bound": raw**2})
        if raw > contract["resource"]["max_raw_determinants_per_J_parity"]:
            raise MemoryError("full model exceeds predeclared per-block determinant cap")
    metadata = [{"index": index, "family": family_of_csf[index], "label": csf.label,
                 "twoJ": csf.total_J2, "parity": csf.parity, "subshells": signatures[index].subshells,
                 "coupling": signatures[index].coupling, "canonical_pattern": signatures[index].pattern}
                for index, csf in enumerate(csfs)]
    write(OUT / "csf_basis.json", {"CSFs": metadata, "families": families,
                                   "orbital_keys": keys, "signed_basis_order": "complete explicit coupling-tree enumeration"})
    write(OUT / "preflight.json", {"contract_sha256": sha(HERE / "contract.json"),
          "source_files_verified": len(source["all_atomx_python_sha256"]),
          "source_map_sha256": source["source_map_sha256"], "inputs_verified": contract["inputs"],
          "CSFs": n, "radial_orbitals": len(keys), "electron_counts": [csf.n_electrons for csf in csfs],
          "blocks": preflight_blocks, "dense_CI_matrix_bytes": 8 * n**2,
          "radial_Dirac_matrix_dimension": 2 * workspace["ns"], "estimated_resident_GiB": 2.5,
          "load_average": os.getloadavg(), "nucleus": fixed["nucleus"],
          "cap_GiB": 4, "wall_seconds": 900, "threads": 1,
          "no_full_69_electron_unrestricted_determinant_enumeration": True})
    print(json.dumps({"preflight": "pass", "CSFs": n,
                      "blocks": [{k: row[k] for k in ("twoJ", "parity", "CSFs", "raw_determinants")}
                                 for row in preflight_blocks]}), flush=True)

    common = dict(csfs=csfs, orbitals_dict=orbitals, orbital_basis=get_shared_orbital_basis(csfs),
                  bsp_data=workspace, Z=70, ns=123, nuclear_potential=make_nuclear_potential(70, "fermi", A=175),
                  max_determinants=2000)
    stages, caught_warnings, checks = {}, {}, []

    def stage(name, call):
        before = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            answer = call()
        stages[name] = time.monotonic() - before
        caught_warnings[name] = [str(item.message) for item in caught]
        print(json.dumps({"stage": name, "seconds_uncontrolled": stages[name],
                          "elapsed_s": time.monotonic() - started}), flush=True)
        return answer

    def check(name, error, limit):
        error = float(error)
        checks.append({"name": name, "error": error, "limit": limit,
                       "passed": bool(np.isfinite(error) and error <= limit)})
        if not checks[-1]["passed"]:
            write(OUT / "failed_numeric_gate.json", checks)
            raise AssertionError(f"{name}: {error} > {limit}")

    dc = stage("public_DC_all_68_roots", lambda: ci_solve(**common, solver="dense"))
    matrix_dc = stage("public_DC_matrix_archive", lambda: build_ci_matrix(**common))
    np.savez_compressed(OUT / "dc_arrays.npz", matrix=matrix_dc, energies_Ha=dc.state_energies,
                        mixing=dc.mixing_coefficients)
    static = stage("public_static_Breit_all_68_roots", lambda: ci_solve(**common, solver="dense", include_breit=True))
    matrix_static = stage("public_static_Breit_matrix_archive", lambda: build_ci_matrix(**common, include_breit=True))
    np.savez_compressed(OUT / "static_arrays.npz", matrix=matrix_static, energies_Ha=static.state_energies,
                        mixing=static.mixing_coefficients)
    matrices = {"DC": matrix_dc, "static_Breit": matrix_static}
    solutions = {"DC": dc, "static_Breit": static}
    native_static = (native["rci_mixing"] * native["rci_energies_Ha"]) @ native["rci_mixing"].T
    check("native_DC_bare_principal_submatrix", np.max(abs(matrix_dc[:5, :5] - native["matrix_dc"])),
          gates["bare_DC_principal_submatrix_replay_Ha"])
    check("native_static_bare_principal_submatrix", np.max(abs(matrix_static[:5, :5] - native_static)),
          gates["bare_static_replay_Ha"])

    root_blocks, root_metadata = {}, {}
    for model, ci in solutions.items():
        h, e, v = matrices[model], ci.state_energies, ci.mixing_coefficients
        if e.shape != (68,) or v.shape != (68, 68) or not np.isfinite(h).all() or not np.isfinite(e).all() or not np.isfinite(v).all():
            raise ValueError("incomplete or nonfinite all-root CI solution")
        check(model + "/residual_Ha", np.max(np.linalg.norm(h @ v - v * e, axis=0)), gates["CI_residual_Ha"])
        check(model + "/orthogonality", np.max(abs(v.T @ v - np.eye(n))), gates["CI_orthogonality"])
        check(model + "/hermiticity_Ha", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
        maximum_cross = max(float(np.max(abs(h[np.ix_(indices, other)])))
                            for symmetry, indices in blocks.items()
                            for other_symmetry, other in blocks.items() if other_symmetry != symmetry)
        check(model + "/cross_symmetry_zero_Ha", maximum_cross, gates["different_symmetry_zero_Ha"])
        rb = {symmetry: [] for symmetry in blocks}
        labels = []
        for col in range(n):
            symmetry = max(blocks, key=lambda key: float(np.sum(v[blocks[key], col]**2)))
            leakage = float(1 - np.sum(v[blocks[symmetry], col]**2))
            if abs(leakage) > gates["CI_orthogonality"]:
                raise ValueError("root cannot be assigned to its complete exact symmetry block")
            local_root = len(rb[symmetry])
            rb[symmetry].append(col)
            character = {family["name"]: float(np.sum(v[family["indices"], col]**2)) for family in families}
            top = np.argsort(-abs(v[:, col]))[:6]
            labels.append({"global_column": col, "twoJ": symmetry[0], "parity": symmetry[1],
                           "local_energy_rank": local_root, "energy_Ha": float(e[col]),
                           "excitation_from_finite_model_global_ground_cm1": float((e[col] - e[0]) * HARTREE_CM1),
                           "family_weights": character, "bare_CSF_weights": v[:5, col]**2,
                           "largest_display_CSFs": [{"index": int(i), "signed_amplitude": float(v[i, col]),
                                                      "weight": float(v[i, col]**2)} for i in top],
                           "display_limit_omits_no_archived_amplitudes": True})
        for symmetry, indices in blocks.items():
            if len(rb[symmetry]) != len(indices):
                raise ValueError("missing symmetry root")
        root_blocks[model], root_metadata[model] = rb, labels
        write(OUT / (model + "_all_roots.json"), {"all_roots": labels,
                                                 "finite_model_global_ground": labels[0],
                                                 "signed_amplitudes": model + " array archive"})

    # Bind all roots one-to-one between Hamiltonians; retain assignment ambiguity.
    dc_to_static, assignment_rows = {}, []
    overlap_all = dc.mixing_coefficients.T @ static.mixing_coefficients
    for symmetry, dc_columns in root_blocks["DC"].items():
        static_columns = root_blocks["static_Breit"][symmetry]
        squared = overlap_all[np.ix_(dc_columns, static_columns)]**2
        left, right = linear_sum_assignment(-squared)
        for i, j in zip(left, right, strict=True):
            source_col, target_col = dc_columns[i], static_columns[j]
            dc_to_static[source_col] = target_col
            ordered = np.sort(squared[i])[::-1]
            weight = float(squared[i, j])
            assignment_rows.append({"twoJ": symmetry[0], "parity": symmetry[1], "DC_column": source_col,
                                    "static_column": target_col, "squared_overlap": weight,
                                    "row_best_static_column": static_columns[int(np.argmax(squared[i]))],
                                    "row_best_weight": float(ordered[0]),
                                    "top_two_weight_margin": float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.,
                                    "unambiguous_weight": weight > .5})
    bare_dc, bare_static = np.zeros((n, 5)), np.zeros((n, 5))
    bare_dc[:5], bare_static[:5] = native["replay_dc_mixing"], native["rci_mixing"]
    bare_dc_overlap = bare_dc.T @ dc.mixing_coefficients
    bare_static_overlap = bare_static.T @ static.mixing_coefficients
    np.savez_compressed(OUT / "overlap_arrays.npz", DC_to_static_signed=overlap_all,
                        bare_DC_to_full_signed=bare_dc_overlap,
                        bare_static_to_full_signed=bare_static_overlap,
                        embedded_bare_DC=bare_dc, embedded_bare_static=bare_static)
    bindings, moments = [], []
    threshold = contract["state_binding"]["unambiguous_individual_branch_minimum_squared_overlap"]
    tie = contract["state_binding"]["squared_overlap_tie_window"]
    for bare_column, row in enumerate(baseline["rows_dc"]):
        symmetry = (row["twoJ"], row["parity"])
        columns = root_blocks["DC"][symmetry]
        weights = bare_dc_overlap[bare_column, columns]**2
        selected = columns[int(np.argmax(weights))]
        assigned = dc_to_static[selected]
        static_columns = root_blocks["static_Breit"][symmetry]
        sw = bare_static_overlap[bare_column, static_columns]**2
        alternative = static_columns[int(np.argmax(sw))]
        ordered = np.sort(weights)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.
        binding = {"label": row["label"], "twoJ": symmetry[0], "parity": symmetry[1],
                   "bare_column": bare_column, "DC_column": selected,
                   "DC_local_root": root_metadata["DC"][selected]["local_energy_rank"],
                   "DC_bare_squared_overlap": float(max(weights)), "DC_top_two_weight_margin": margin,
                   "DC_unambiguous": bool(max(weights) > threshold and margin > tie),
                   "static_column": assigned,
                   "static_local_root": root_metadata["static_Breit"][assigned]["local_energy_rank"],
                   "DC_to_static_squared_overlap": float(overlap_all[selected, assigned]**2),
                   "static_bare_squared_overlap": float(bare_static_overlap[bare_column, assigned]**2),
                   "static_bare_best_column": alternative,
                   "static_bare_best_weight": float(max(sw)),
                   "static_bare_binding_disagrees": alternative != assigned,
                   "DC_complete_block_overlap_distribution": [
                       {"column": col, "local_root": root_metadata["DC"][col]["local_energy_rank"],
                        "energy_Ha": float(dc.state_energies[col]), "bare_weight": float(w)}
                       for col, w in zip(columns, weights, strict=True)],
                   "static_complete_block_bare_overlap_distribution": [
                       {"column": col, "local_root": root_metadata["static_Breit"][col]["local_energy_rank"],
                        "energy_Ha": float(static.state_energies[col]), "bare_weight": float(w)}
                       for col, w in zip(static_columns, sw, strict=True)]}
        bindings.append(binding)
        for model, bare, full, model_columns in (
            ("DC", bare_dc, dc, columns), ("static_Breit", bare_static, static, static_columns)
        ):
            v = bare[:, bare_column]
            h = matrices[model]
            center = float(v @ h @ v)
            residual = h @ v - center * v
            variance = float(residual @ residual)
            spectrum_weights = (v @ full.mixing_coefficients[:, model_columns])**2
            relative_mean = float(spectrum_weights @ (full.state_energies[model_columns] - center))
            spectral_variance = float(spectrum_weights @ (full.state_energies[model_columns] - center)**2)
            check(f"{model}/{row['label']}/spectral_weight_closure", abs(float(sum(spectrum_weights)) - 1.), 1e-8)
            check(f"{model}/{row['label']}/spectral_centroid_replay_Ha", abs(relative_mean), 1e-8)
            check(f"{model}/{row['label']}/spectral_variance_replay_Ha2", abs(variance - spectral_variance), 1e-8)
            moments.append({"model": model, "label": row["label"], "centroid_Ha": center,
                            "variance_Ha2": variance, "width_cm1": variance**.5 * HARTREE_CM1,
                            "centroid_identity": "Complete bare-reference spectral measure, not a tracked eigenenergy"})

    tracked = {model: [] for model in solutions}
    for model, ci in solutions.items():
        selected = [binding["DC_column" if model == "DC" else "static_column"] for binding in bindings]
        ground = selected[0]
        bare_rows = baseline["rows_dc" if model == "DC" else "rows_rci"]
        for i, (binding, col, old) in enumerate(zip(bindings, selected, bare_rows, strict=True)):
            energy = float(ci.state_energies[col])
            excitation = float((energy - ci.state_energies[ground]) * HARTREE_CM1)
            target_ok = bool(binding["DC_unambiguous"] and binding["DC_to_static_squared_overlap"] > threshold
                             and not binding["static_bare_binding_disagrees"])
            relative_error = (excitation - old["experimental_cm1"]) / old["experimental_cm1"] if i else None
            tracked[model].append({"label": binding["label"], "twoJ": binding["twoJ"], "parity": binding["parity"],
                                   "column": col, "local_root": root_metadata[model][col]["local_energy_rank"],
                                   "energy_Ha": energy, "delta_from_bare_same_state_Ha": energy - old["energy_Ha"],
                                   "excitation_from_tracked_6s_cm1": excitation,
                                   "excitation_from_finite_model_global_ground_cm1": float((energy - ci.state_energies[0]) * HARTREE_CM1),
                                   "delta_excitation_from_bare_cm1": excitation - old["computed_cm1"],
                                   "NIST_nominal_cm1": old["experimental_cm1"],
                                   "gap_cm1": excitation - old["experimental_cm1"], "relative_error": relative_error,
                                   "nominal_one_percent_comparison_pass": bool(abs(relative_error) < .01) if i else None,
                                   "binding_unambiguous": target_ok, "family_weights": root_metadata[model][col]["family_weights"],
                                   "full_expanded_model_MCDHF_stationarity": "not evaluated"})
        check(model + "/bare_ground_spectral_minimum_variational_bound",
              max(0., float(ci.state_energies[0] - bare_rows[0]["energy_Ha"])), 1e-8)
    multiplets = {}
    for model, rows in tracked.items():
        old = baseline["rows_dc" if model == "DC" else "rows_rci"]
        entries = {}
        for label, low, high in (("5d", 1, 2), ("6p", 3, 4)):
            degeneracy = [rows[i]["twoJ"] + 1 for i in (low, high)]
            centroid = sum(rows[i]["excitation_from_tracked_6s_cm1"] * g
                           for i, g in zip((low, high), degeneracy, strict=True)) / sum(degeneracy)
            old_centroid = sum(old[i]["computed_cm1"] * g
                               for i, g in zip((low, high), degeneracy, strict=True)) / sum(degeneracy)
            splitting = rows[high]["excitation_from_tracked_6s_cm1"] - rows[low]["excitation_from_tracked_6s_cm1"]
            old_split = old[high]["computed_cm1"] - old[low]["computed_cm1"]
            entries[label] = {"degeneracy_weights": degeneracy, "centroid_cm1": centroid,
                              "bare_centroid_cm1": old_centroid, "centroid_shift_cm1": centroid - old_centroid,
                              "splitting_cm1": splitting, "bare_splitting_cm1": old_split,
                              "splitting_shift_cm1": splitting - old_split}
        multiplets[model] = entries
    binding_data = {"prescription": contract["state_binding"], "targets": bindings,
                    "complete_DC_to_static_assignment": assignment_rows,
                    "bare_reference_spectral_moments": moments}
    write(OUT / "target_binding.json", binding_data)
    np.savez_compressed(OUT / "ci_arrays.npz", matrix_DC_Ha=matrix_dc, matrix_static_Ha=matrix_static,
                        energies_DC_Ha=dc.state_energies, energies_static_Ha=static.state_energies,
                        mixing_DC=dc.mixing_coefficients, mixing_static=static.mixing_coefficients,
                        target_columns_DC=np.array([b["DC_column"] for b in bindings]),
                        target_columns_static=np.array([b["static_column"] for b in bindings]))
    result = {"numerical_gates_passed": all(c["passed"] for c in checks),
              "diagnostic_only": True, "production_certified": False,
              "bare005_MCDHF_stationarity_passed": True, "expanded68_MCDHF_stationarity_passed": None,
              "experimental_gate_passed": False, "complete_correlation_claim": False,
              "scope": "68-CSF fixed converged five-state MCDHF orbitals RCI core-hole discriminator; no expanded-model orbital optimization",
              "input_run": fixed["input_run"], "contract_sha256": sha(HERE / "contract.json"),
              "source_map_sha256": fixed["source_map_sha256"], "inputs_sha256": contract["inputs"],
              "checks": checks, "tracked_levels": tracked, "multiplets": multiplets,
              "finite_model_global_ground": {m: rows[0] for m, rows in root_metadata.items()},
              "tracked_6s_is_finite_model_global_ground": {m: rows[0]["column"] == 0 for m, rows in tracked.items()},
              "provenance": {m: ci.provenance for m, ci in solutions.items()},
              "warnings": caught_warnings,
              "resource": {"stages_seconds_uncontrolled": stages, "total_seconds_uncontrolled": time.monotonic() - started,
                           "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                           "threads": 1, "cap_GiB": 4, "wall_seconds": 900, "controlled_performance": False},
              "artifacts_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
    write(OUT / "result.json", result)
    signal.alarm(0)
    done.set()
    print(json.dumps({"numeric_pass": result["numerical_gates_passed"],
                      "elapsed_s": result["resource"]["total_seconds_uncontrolled"],
                      "expanded_MCDHF_certified": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            write(OUT / "exception.json", {"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc(), "physical_model_shrunk": False})
        raise
