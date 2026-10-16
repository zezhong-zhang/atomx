"""Three nested balanced-family RCI diagnostics on frozen native005 orbitals."""

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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n")


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    fixed, gates = contract["fixed_conditions"], contract["acceptance"]
    if (HERE / "result.json").exists() or (HERE / "execution_source.py").exists():
        raise FileExistsError("retain completed or failed execution before a successor")
    shutil.copyfile(Path(__file__), HERE / "execution_source.py")
    for relative, expected in contract["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"source-bound native/frontier input changed: {relative}")
    inp, frozen = ROOT / fixed["native_run"], ROOT / fixed["frozen_source_root"]
    source = json.loads((inp / "source_identity.json").read_text())
    for relative, expected in source["all_atomx_python_sha256"].items():
        if sha(frozen / Path(relative).relative_to("src")) != expected:
            raise ValueError(f"frozen native source changed: {relative}")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    started, done = time.monotonic(), threading.Event()

    def timeout(*_):
        raise TimeoutError("prospective three-stage 900 second RCI cap exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)

    def heartbeat():
        while not done.wait(30):
            print(json.dumps({"heartbeat_elapsed_s": time.monotonic() - started,
                              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)

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

    frontier = json.loads((ROOT / fixed["frontier_enumeration"]).read_text())
    frontier_preflight = json.loads((ROOT / fixed["frontier_preflight"]).read_text())
    native_result = json.loads((inp / "result.json").read_text())
    native = np.load(inp / "ci_arrays.npz")
    saved = np.load(inp / "orbitals_workspace.npz")
    workspace = {key.removeprefix("basis_"): saved[key].item() if saved[key].ndim == 0 else saved[key]
                 for key in saved.files if key.startswith("basis_")}
    orbital_keys = tuple(map(tuple, saved["orbital_keys"].tolist()))
    orbitals = {key: (p, q) for key, p, q in zip(orbital_keys, saved["P"], saved["Q"], strict=True)}
    if not native_result["stationarity_passed"] or len(orbital_keys) != 24 or workspace["ns"] != 123:
        raise ValueError("fixed converged native005 orbital contract changed")
    if nuclear_profile(70, "fermi") != fixed["nucleus"]:
        raise ValueError("A175 Fermi nuclear identity changed")
    csfs, family_labels = [], []
    for config in frontier["configs"]:
        generated = enumerate_jj_csfs(config["config"])
        if len(generated) != config["CSFs"] or config["indices"] != list(range(len(csfs), len(csfs) + len(generated))):
            raise ValueError("frontier family/coupling-tree enumeration order changed")
        csfs.extend(generated)
        family_labels.extend([config["family"]] * len(generated))
    signatures = [csf_signature(csf) for csf in csfs]
    if len(csfs) != 531 or len(set(signatures)) != 531 or set(get_shared_orbital_basis(csfs)) != set(orbital_keys):
        raise ValueError("complete frontier or fixed radial pool changed")
    if {csf.n_electrons for csf in csfs} != {69}:
        raise ValueError("electron number changed")
    for index, (csf, signature, recorded) in enumerate(zip(csfs, signatures, frontier["CSFs"], strict=True)):
        actual = {"index": index, "twoJ": csf.total_J2, "parity": csf.parity,
                  "subshells": plain(signature.subshells), "coupling": plain(signature.coupling)}
        if any(recorded[key] != value for key, value in actual.items()):
            raise ValueError("signed frontier CSF identity does not match native enumeration")
    shutil.copyfile(ROOT / fixed["frontier_enumeration"], HERE / "basis_identity.json")
    stage_spaces = {}
    for plan in fixed["stages"]:
        count = plan["CSFs"]
        retained = list(range(count))
        if {family_labels[i] for i in retained} != set(plan["families"]):
            raise ValueError("nested family prefix changed")
        groups = {}
        for i in retained:
            groups.setdefault((csfs[i].total_J2, csfs[i].parity), []).append(i)
        blocks = []
        for (j2, parity), indices in sorted(groups.items()):
            patterns = {signatures[i].pattern for i in indices}
            raw = sum(dict(pattern_m_counts(pattern)).get(j2, 0) for pattern in patterns)
            if raw > 2000:
                raise MemoryError("complete model exceeds unchanged raw determinant cap")
            blocks.append({"twoJ": j2, "parity": parity, "CSFs": len(indices), "raw_M_equals_J": raw})
        expected = frontier_preflight["spaces"][plan["frontier_space"]]
        if expected["CSFs"] != count or len(expected["blocks"]) != len(blocks):
            raise ValueError("frontier nested resource model changed")
        for actual, declared in zip(blocks, expected["blocks"], strict=True):
            if (actual["twoJ"], actual["parity"], actual["CSFs"], actual["raw_M_equals_J"]) != (
                declared["twoJ"], declared["parity"], declared["CSFs"], declared["raw_M_equals_J_determinants"]
            ):
                raise ValueError("exact integer determinant count disagrees with frontier")
        stage_spaces[plan["name"]] = {"CSFs": count, "blocks": blocks,
                                      "dense_whole_CI_matrix_bytes": 8 * count**2,
                                      "families": plan["families"]}
    write(HERE / "preflight.json", {"contract_sha256": sha(HERE / "contract.json"),
          "all_input_hashes_verified": contract["inputs_sha256"], "source_files_verified": 353,
          "source_map_sha256": fixed["source_map_sha256"], "stages": stage_spaces,
          "same24_radial_orbitals": True, "electron_count": 69, "nucleus": fixed["nucleus"],
          "threads": 1, "cap_GiB": 4, "wall_seconds": 900, "resident_estimate_GiB": 2.5,
          "load_average": os.getloadavg(), "physical_model_shrunk": False})
    print(json.dumps({"preflight": "pass", "stages": {name: space["CSFs"] for name, space in stage_spaces.items()}}), flush=True)
    checks, timings, captured_warnings = [], {}, {}

    def check(name, error, tolerance):
        error = float(error)
        checks.append({"name": name, "error": error, "limit": tolerance,
                       "passed": bool(np.isfinite(error) and error <= tolerance)})
        if not checks[-1]["passed"]:
            write(HERE / "failed_numeric_gate.json", checks)
            raise AssertionError(f"{name}: {error} > {tolerance}")

    def timed(name, call):
        before = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = call()
        timings[name] = time.monotonic() - before
        captured_warnings[name] = [str(item.message) for item in caught]
        print(json.dumps({"stage": name, "seconds_uncontrolled": timings[name],
                          "elapsed_s": time.monotonic() - started}), flush=True)
        return value

    def root_information(e, v, groups, family_indices, identity):
        root_groups, metadata, near_clusters = {s: [] for s in groups}, [], []
        for col, energy in enumerate(e):
            symmetry = max(groups, key=lambda s: float(np.sum(v[groups[s], col]**2)))
            check(f"{identity}/root_symmetry_leakage/{col}", abs(1 - np.sum(v[groups[symmetry], col]**2)), 1e-8)
            rank = len(root_groups[symmetry])
            root_groups[symmetry].append(col)
            metadata.append({"column": col, "twoJ": symmetry[0], "parity": symmetry[1], "local_root": rank,
                             "energy_Ha": float(energy), "excitation_from_actual_finite_ground_cm1": float((energy - e[0]) * HARTREE_CM1),
                             "family_weights": {name: float(np.sum(v[indices, col]**2)) for name, indices in family_indices.items()},
                             "bare_reference_weights": (native["replay_dc_mixing"].T @ v[:5, col])**2})
        for symmetry, columns in root_groups.items():
            if len(columns) != len(groups[symmetry]):
                raise ValueError("incomplete symmetry spectrum")
            cluster = []
            for col in columns:
                if cluster and e[col] - e[cluster[-1]] > 1e-8:
                    if len(cluster) > 1:
                        near_clusters.append({"twoJ": symmetry[0], "parity": symmetry[1], "columns": cluster})
                    cluster = []
                cluster.append(col)
            if len(cluster) > 1:
                near_clusters.append({"twoJ": symmetry[0], "parity": symmetry[1], "columns": cluster})
        return root_groups, metadata, near_clusters

    def assignment(old, old_groups, new, new_groups):
        embedded = np.zeros((new.shape[0], old.shape[1]))
        embedded[:old.shape[0]] = old
        signed = embedded.T @ new
        mapping, ledger = {}, []
        for symmetry, source_columns in old_groups.items():
            destination_columns = new_groups[symmetry]
            weights = signed[np.ix_(source_columns, destination_columns)]**2
            left, right = linear_sum_assignment(-weights)
            for i, j in zip(left, right, strict=True):
                old_col, new_col = source_columns[i], destination_columns[j]
                mapping[old_col] = new_col
                ordered = np.sort(weights[i])[::-1]
                ledger.append({"source_column": old_col, "destination_column": new_col,
                               "twoJ": symmetry[0], "parity": symmetry[1], "squared_overlap": float(weights[i, j]),
                               "individual_best_column": destination_columns[int(np.argmax(weights[i]))],
                               "top_two_margin": float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.})
        return mapping, ledger, signed

    bare_groups = {}
    for col, row in enumerate(native_result["rows_dc"]):
        bare_groups.setdefault((row["twoJ"], row["parity"]), []).append(col)
    previous = {"name": "bare005", "n": 5, "DC": {"e": native["replay_dc_energies_Ha"], "v": native["replay_dc_mixing"],
                "h": native["matrix_dc"], "groups": bare_groups, "targets": list(range(5))},
                "static_Breit": {"e": native["rci_energies_Ha"], "v": native["rci_mixing"],
                "h": (native["rci_mixing"] * native["rci_energies_Ha"]) @ native["rci_mixing"].T,
                "groups": bare_groups, "targets": list(range(5))}}
    summaries = []
    for plan in fixed["stages"]:
        name, n = plan["name"], plan["CSFs"]
        owner = HERE / name
        owner.mkdir(exist_ok=False)
        local_csfs = csfs[:n]
        groups = {}
        for i, csf in enumerate(local_csfs):
            groups.setdefault((csf.total_J2, csf.parity), []).append(i)
        family_indices = {family: [i for i in range(n) if family_labels[i] == family]
                          for family in ("bare", "4f", "5p", "5s")}
        common = dict(csfs=local_csfs, orbitals_dict=orbitals, orbital_basis=get_shared_orbital_basis(local_csfs),
                      bsp_data=workspace, Z=70, ns=123,
                      nuclear_potential=make_nuclear_potential(70, "fermi", A=175), max_determinants=2000)
        current = {"name": name, "n": n}
        for model, breit in (("DC", False), ("static_Breit", True)):
            ci = timed(name + "/" + model + "/public_ci", lambda common=common, breit=breit: ci_solve(**common, solver="dense", include_breit=breit))
            h = timed(name + "/" + model + "/public_matrix_archive", lambda common=common, breit=breit: build_ci_matrix(**common, include_breit=breit))
            e, v = ci.state_energies, ci.mixing_coefficients
            np.savez_compressed(owner / (model + "_arrays.npz"), matrix_Ha=h, energies_Ha=e, mixing=v)
            if e.shape != (n,) or v.shape != (n, n) or not all(np.isfinite(x).all() for x in (h, e, v)):
                raise ValueError("nonfinite or incomplete all-root solution")
            check(name + "/" + model + "/residual_Ha", np.max(np.linalg.norm(h @ v - v * e, axis=0)), gates["CI_residual_Ha"])
            check(name + "/" + model + "/orthogonality", np.max(abs(v.T @ v - np.eye(n))), gates["CI_orthogonality"])
            check(name + "/" + model + "/hermiticity_Ha", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
            cross = max(float(np.max(abs(h[np.ix_(idx, other)]))) for s, idx in groups.items()
                        for t, other in groups.items() if t != s)
            check(name + "/" + model + "/cross_symmetry_Ha", cross, 1e-10)
            check(name + "/" + model + "/previous_principal_matrix_Ha",
                  np.max(abs(h[:previous["n"], :previous["n"]] - previous[model]["h"])), 1e-8)
            rg, metadata, clusters = root_information(e, v, groups, family_indices, name + "/" + model)
            current[model] = {"e": e, "v": v, "h": h, "groups": rg, "metadata": metadata,
                              "clusters": clusters, "provenance": ci.provenance}
            write(owner / (model + "_all_roots.json"), {"all_roots": metadata,
                  "same_symmetry_near_degenerate_clusters": clusters, "ground": metadata[0]})
        adjacent, adjacent_ledger, overlap_adjacent = {}, {}, {}
        for model in ("DC", "static_Breit"):
            mapping, ledger, signed = assignment(previous[model]["v"], previous[model]["groups"],
                                                  current[model]["v"], current[model]["groups"])
            adjacent[model], adjacent_ledger[model], overlap_adjacent[model] = mapping, ledger, signed
        dc_static, dc_static_ledger, overlap_operator = assignment(current["DC"]["v"], current["DC"]["groups"],
                                                                  current["static_Breit"]["v"], current["static_Breit"]["groups"])
        current["DC"]["targets"] = [adjacent["DC"][col] for col in previous["DC"]["targets"]]
        current["static_Breit"]["targets"] = [dc_static[col] for col in current["DC"]["targets"]]
        binding_rows, tracked, multiplets, moments = [], {"DC": [], "static_Breit": []}, {}, []
        for target_index, original in enumerate(native_result["rows_dc"]):
            symmetry = (original["twoJ"], original["parity"])
            info = {"label": original["label"], "twoJ": symmetry[0], "parity": symmetry[1], "models": {}}
            for model, bare_key in (("DC", "replay_dc_mixing"), ("static_Breit", "rci_mixing")):
                state, col = current[model], current[model]["targets"][target_index]
                columns = state["groups"][symmetry]
                bare = np.zeros(n)
                bare[:5] = native[bare_key][:, target_index]
                weights = (bare @ state["v"][:, columns])**2
                best = columns[int(np.argmax(weights))]
                prev_col = previous[model]["targets"][target_index]
                pw = overlap_adjacent[model][prev_col, columns]**2
                nearest = min((abs(float(state["e"][other] - state["e"][col])) for other in columns if other != col), default=None)
                adjacent_col = adjacent[model][prev_col]
                previous_overlap = float(overlap_adjacent[model][prev_col, col]**2)
                operator_overlap = float(overlap_operator[current["DC"]["targets"][target_index], current["static_Breit"]["targets"][target_index]]**2)
                same_branch = adjacent_col == col
                safe = bool(weights[columns.index(col)] > .5 and previous_overlap > .5 and operator_overlap > .5
                            and best == col and same_branch and (nearest is None or nearest > 1e-8))
                info["models"][model] = {"column": col, "local_root": state["metadata"][col]["local_root"],
                    "bare_best_column": best, "bare_weight": float(weights[columns.index(col)]),
                    "previous_same_state_column": prev_col, "previous_overlap_weight": previous_overlap,
                    "previous_static_or_DC_independent_assigned_column": adjacent_col,
                    "operator_overlap_weight": operator_overlap, "bare_binding_disagrees": best != col,
                    "adjacent_binding_disagrees": not same_branch, "minimum_same_symmetry_gap_Ha": nearest,
                    "unambiguous": safe,
                    "complete_bare_distribution": [{"column": c, "energy_Ha": float(state["e"][c]), "weight": float(w)}
                                                    for c, w in zip(columns, weights, strict=True)],
                    "complete_previous_distribution": [{"column": c, "weight": float(w)}
                                                         for c, w in zip(columns, pw, strict=True)]}
                ground = state["targets"][0]
                excitation = float((state["e"][col] - state["e"][ground]) * HARTREE_CM1)
                previous_ground = previous[model]["targets"][0]
                previous_excitation = float((previous[model]["e"][prev_col] - previous[model]["e"][previous_ground]) * HARTREE_CM1)
                bare_row = native_result["rows_dc" if model == "DC" else "rows_rci"][target_index]
                shift = float(state["e"][col] - previous[model]["e"][prev_col])
                shift_bare = float(state["e"][col] - bare_row["energy_Ha"])
                gap = excitation - bare_row["experimental_cm1"]
                relative = gap / bare_row["experimental_cm1"] if target_index else None
                tracked[model].append({"label": original["label"], "twoJ": symmetry[0], "parity": symmetry[1],
                    "column": col, "local_root": state["metadata"][col]["local_root"], "energy_Ha": float(state["e"][col]),
                    "energy_delta_from_previous_Ha": shift, "energy_delta_from_bare_Ha": shift_bare,
                    "excitation_from_tracked_6s_cm1": excitation,
                    "excitation_from_actual_finite_ground_cm1": float((state["e"][col] - state["e"][0]) * HARTREE_CM1),
                    "transition_delta_from_previous_cm1": excitation - previous_excitation,
                    "transition_delta_from_bare_cm1": excitation - bare_row["computed_cm1"],
                    "NIST_nominal_cm1": bare_row["experimental_cm1"], "gap_cm1": gap, "relative_error": relative,
                    "nominal_1percent_comparison_pass": abs(relative) < .01 if relative is not None else None,
                    "binding_unambiguous": safe, "bare_character_weight": float(weights[columns.index(col)]),
                    "family_weights": state["metadata"][col]["family_weights"]})
                center = float(bare @ state["h"] @ bare)
                residual = state["h"] @ bare - center * bare
                variance = float(residual @ residual)
                check(f"{name}/{model}/{original['label']}/spectral_closure", abs(float(sum(weights)) - 1), 1e-8)
                check(f"{name}/{model}/{original['label']}/spectral_centroid_Ha", abs(float(weights @ (state["e"][columns] - center))), 1e-8)
                check(f"{name}/{model}/{original['label']}/spectral_variance_Ha2", abs(float(weights @ (state["e"][columns] - center)**2) - variance), 1e-8)
                moments.append({"model": model, "label": original["label"], "bare_spectral_centroid_Ha": center,
                                "bare_spectral_variance_Ha2": variance, "width_cm1_not_linewidth": variance**.5 * HARTREE_CM1})
            binding_rows.append(info)
        for model, rows in tracked.items():
            old_rows = previous[model].get("tracked", native_result["rows_dc" if model == "DC" else "rows_rci"])
            entries = {}
            for label, low, high in (("5d", 1, 2), ("6p", 3, 4)):
                degeneracy = [rows[i]["twoJ"] + 1 for i in (low, high)]
                def moment(source_rows, key, low=low, high=high, degeneracy=degeneracy):
                    return sum(source_rows[i][key] * g for i, g in zip((low, high), degeneracy, strict=True)) / sum(degeneracy)
                def split(source_rows, key, low=low, high=high):
                    return source_rows[high][key] - source_rows[low][key]
                old_key = "excitation_from_tracked_6s_cm1" if "tracked" in previous[model] else "computed_cm1"
                bare_rows = native_result["rows_dc" if model == "DC" else "rows_rci"]
                centroid, splitting = moment(rows, "excitation_from_tracked_6s_cm1"), split(rows, "excitation_from_tracked_6s_cm1")
                entries[label] = {"degeneracy_weights": degeneracy, "centroid_cm1": centroid, "splitting_cm1": splitting,
                    "centroid_delta_from_previous_cm1": centroid - moment(old_rows, old_key),
                    "splitting_delta_from_previous_cm1": splitting - split(old_rows, old_key),
                    "centroid_delta_from_bare_cm1": centroid - moment(bare_rows, "computed_cm1"),
                    "splitting_delta_from_bare_cm1": splitting - split(bare_rows, "computed_cm1")}
            multiplets[model] = entries
            current[model]["tracked"] = rows
        np.savez_compressed(owner / "overlaps.npz", previous_DC_to_current_signed=overlap_adjacent["DC"],
                            previous_static_to_current_signed=overlap_adjacent["static_Breit"],
                            current_DC_to_static_signed=overlap_operator,
                            target_columns_DC=np.array(current["DC"]["targets"]),
                            target_columns_static=np.array(current["static_Breit"]["targets"]))
        write(owner / "target_binding.json", {"previous_stage": previous["name"], "targets": binding_rows,
              "complete_previous_to_current_assignment": adjacent_ledger,
              "complete_DC_to_static_assignment": dc_static_ledger, "bare_spectral_moments": moments})
        summary = {"name": name, "added_core_family": plan["added_family"], "families": plan["families"], "CSFs": n,
            "previous_stage": previous["name"], "tracked_levels": tracked, "multiplets": multiplets,
            "actual_finite_model_ground": {model: state["metadata"][0] for model, state in current.items() if model in ("DC", "static_Breit")},
            "tracked_6s_is_actual_ground": {model: current[model]["targets"][0] == 0 for model in ("DC", "static_Breit")},
            "all_target_bindings_unambiguous": all(row["binding_unambiguous"] for rows in tracked.values() for row in rows),
            "provenance": {model: current[model]["provenance"] for model in ("DC", "static_Breit")},
            "expanded_MCDHF_stationarity": "not evaluated", "diagnostic_only": True, "production_certified": False,
            "experimental_gate_passed": False, "files_sha256": {p.name: sha(p) for p in sorted(owner.iterdir()) if p.is_file()}}
        write(owner / "result.json", summary)
        summaries.append(summary)
        write(HERE / "completed_stages.json", {"stages": summaries, "scope": "Bounded partial evidence if later stage fails; not a reduced-model replacement"})
        previous = current
        print(json.dumps({"completed_stage": name, "CSFs": n, "ground_is_tracked6s": summary["tracked_6s_is_actual_ground"],
                          "elapsed_s": time.monotonic() - started}), flush=True)
    card_summary = {
        "scope": contract["scope"],
        "model": "Fixed native005 MCDHF orbitals; nested balanced-family RCI, not expanded MCDHF or complete CV",
        "stages": [{"name": stage["name"], "CSFs": stage["CSFs"],
                    "added_core_family": stage["added_core_family"],
                    "static_rows": stage["tracked_levels"]["static_Breit"],
                    "tracked_6s_is_actual_ground": stage["tracked_6s_is_actual_ground"]["static_Breit"],
                    "all_target_bindings_unambiguous": stage["all_target_bindings_unambiguous"]}
                   for stage in summaries],
        "final_static_rows": summaries[-1]["tracked_levels"]["static_Breit"],
        "expanded_MCDHF_stationarity": "not evaluated",
        "production_certified": False,
        "experimental_gate_passed": False,
        "physical_model_shrunk": False,
    }
    write(HERE / "result.json", {"scope": contract["scope"], "numerical_gates_passed": all(row["passed"] for row in checks),
          "diagnostic_only": True, "production_certified": False, "expanded_MCDHF_stationarity_passed": None,
          "bare005_MCDHF_stationarity_passed": True, "experimental_gate_passed": False, "complete_CV_claim": False,
          "contract_sha256": sha(HERE / "contract.json"), "source_map_sha256": fixed["source_map_sha256"],
          "inputs_sha256": contract["inputs_sha256"], "stages": summaries, "checks": checks,
          "card_summary": card_summary,
          "warnings": captured_warnings, "resource": {"total_seconds_uncontrolled": time.monotonic() - started,
          "stages_seconds_uncontrolled": timings, "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
          "threads": 1, "cap_GiB": 4, "wall_seconds": 900, "controlled_performance": False},
          "execution_source_sha256": sha(HERE / "execution_source.py")})
    done.set()
    signal.alarm(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(HERE / "exception.json", {"type": type(error).__name__, "message": str(error),
              "traceback": traceback.format_exc(), "physical_model_shrunk": False, "expanded_MCDHF_certified": False})
        raise
