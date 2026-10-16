"""Complete declared core-single/valence-replacement angular preflight only."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    started = time.monotonic()
    contract = json.loads((HERE / "contract.json").read_text())
    fixed = contract["fixed_conditions"]
    checks = []

    def check(name, passed, evidence):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            write("verification_failed.json", {"checks": checks})
            raise AssertionError(name)

    for path, expected in contract["inputs_sha256"].items():
        check("input/" + path, sha(ROOT / path) == expected, expected)
    native = ROOT / fixed["native_run"]
    frozen = ROOT / fixed["frozen_source_root"]
    source = json.loads((native / "source_identity.json").read_text())
    check("source_map", source["source_map_sha256"] == fixed["source_map_sha256"],
          source["source_map_sha256"])
    for name, expected in source["all_atomx_python_sha256"].items():
        if sha(frozen / Path(name).relative_to("src")) != expected:
            raise ValueError("frozen005 source changed: " + name)
    check("all_frozen_source_files", True, len(source["all_atomx_python_sha256"]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))

    def timeout(*_):
        raise TimeoutError("prospective 360 second angular-preflight cap exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["wall_cap_s"])
    sys.path.insert(0, str(frozen))
    import numpy as np

    from atomx.angular.csf_hamiltonian import (
        csf_signature,
        pattern_m_counts,
        verify_csf_span,
    )
    from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
    from atomx.config.csf import (
        count_j_resolved_csfs,
        enumerate_jj_csfs,
        get_shared_orbital_basis,
    )
    from atomx.config.parsing import parse_nr_shells
    from atomx.config.subshell import L_LETTERS
    from atomx.physics.csf.radial_density import CIRadialDensity
    from atomx.physics.csf.radial_symmetry import canonical_static_radial_key

    radial = np.load(native / "orbitals_workspace.npz")
    orbital_keys = tuple(map(tuple, radial["orbital_keys"].tolist()))
    core = parse_nr_shells(fixed["core"])
    pool = [parse_nr_shells(shell + "1")[0][:2] for shell in fixed["particle_shell_pool"]]
    refs = [fixed["core"] + " " + valence for valence in fixed["bare_valence_refs"]]
    hole_shells = [parse_nr_shells(shell + "1")[0][:2] for shell in fixed["core_hole_shells"]]
    pair_pool = list(itertools.combinations_with_replacement(pool, 2))
    check("complete_pair_pool", len(pair_pool) == 6, pair_pool)
    configs = []
    for ref in refs:
        configs.append({"family": "bare", "config": ref, "parents": [ref], "pair": None})
    for hole, hole_label in zip(hole_shells, fixed["core_hole_shells"], strict=True):
        for pair in pair_pool:
            final_configs = []
            for ref in refs:
                # Configuration substitution only. Fermion/J algebra is owned by public generators.
                occupations = {(n, ell): q for n, ell, q in parse_nr_shells(ref)}
                valence = parse_nr_shells(ref)[-1][:2]
                occupations[valence] -= 1
                occupations[hole] -= 1
                for shell in pair:
                    occupations[shell] = occupations.get(shell, 0) + 1
                ordering = [(n, ell) for n, ell, _ in core] + pool
                config = " ".join(f"{n}{L_LETTERS[ell]}{occupations[(n, ell)]}"
                                  for n, ell in ordering if occupations.get((n, ell), 0))
                final_configs.append(config)
            check(f"all_ref_substitutions_equal/{hole_label}/{pair}", len(set(final_configs)) == 1,
                  final_configs)
            configs.append({"family": hole_label, "config": final_configs[0], "parents": refs,
                            "pair": [f"{n}{L_LETTERS[ell]}" for n, ell in pair]})
    check("NR_configuration_identity", len(configs) == 21 and len({c["config"] for c in configs}) == 21,
          {"bare": 3, "candidate": 18, "total": len(configs)})
    csfs, metadata, config_rows = [], [], []
    for config_index, row in enumerate(configs):
        local = enumerate_jj_csfs(row["config"])
        local_counts = Counter(c.total_J2 for c in local)
        check(f"public_count_generator/{config_index}", dict(local_counts) == count_j_resolved_csfs(row["config"]),
              {str(key): count for key, count in sorted(local_counts.items())})
        indices = list(range(len(csfs), len(csfs) + len(local)))
        for index, csf in zip(indices, local, strict=True):
            sig = csf_signature(csf)
            metadata.append({"index": index, "config_index": config_index, "family": row["family"],
                             "label": csf.label, "twoJ": csf.total_J2, "parity": csf.parity,
                             "subshells": sig.subshells, "coupling": sig.coupling,
                             "canonical_pattern": sig.pattern})
        patterns = {csf_signature(c).pattern for c in local}
        m_counts = Counter()
        for pattern in patterns:
            m_counts.update(dict(pattern_m_counts(pattern)))
        inferred = {j2: m_counts[j2] - m_counts[j2 + 2] for j2 in sorted(local_counts)}
        check(f"independent_integer_MJ_difference/{config_index}", inferred == dict(local_counts), inferred)
        complete_dim = sum((j2 + 1) * count for j2, count in local_counts.items())
        check(f"complete_magnetic_dimension/{config_index}", complete_dim == sum(m_counts.values()),
              {"CSF_degeneracies": complete_dim, "integer_pattern_M_counts": sum(m_counts.values())})
        config_rows.append({**row, "index": config_index, "indices": indices, "CSFs": len(local),
                            "parity": local[0].parity, "twoJ_counts": dict(sorted(local_counts.items())),
                            "distinct_jj_patterns": len(patterns), "magnetic_dimension": complete_dim})
        csfs.extend(local)
    signatures = [csf_signature(c) for c in csfs]
    check("signed_CSF_signature_uniqueness", len(set(signatures)) == len(csfs), len(csfs))
    check("all_69_electrons", {c.n_electrons for c in csfs} == {69}, sorted({c.n_electrons for c in csfs}))
    check("same24_radial_keys", set(get_shared_orbital_basis(csfs)) == set(orbital_keys) and len(orbital_keys) == 24,
          orbital_keys)
    bare_indices = [m["index"] for m in metadata if m["family"] == "bare"]
    check("five_bare_refs", len(bare_indices) == 5, bare_indices)

    def space(indices):
        blocks = {}
        for index in indices:
            c = csfs[index]
            blocks.setdefault((c.total_J2, c.parity), []).append(index)
        block_rows = []
        for (j2, parity), block in sorted(blocks.items()):
            patterns = sorted({signatures[index].pattern for index in block})
            counts = [(pattern, dict(pattern_m_counts(pattern)).get(j2, 0)) for pattern in patterns]
            raw = sum(count for _, count in counts)
            block_rows.append({"twoJ": j2, "parity": parity, "CSFs": len(block),
                               "unique_patterns": len(patterns), "raw_M_equals_J_determinants": raw,
                               "pattern_raw_counts": counts, "raw_pair_work_upper": raw**2,
                               "dense_CSF_matrix_bytes": 8 * len(block)**2,
                               "dense_raw_determinant_matrix_bytes": 8 * raw**2})
        return {"CSFs": len(indices), "configuration_indices": sorted({metadata[i]["config_index"] for i in indices}),
                "blocks": block_rows, "symmetry_blocks": len(block_rows),
                "max_block_CSFs": max((r["CSFs"] for r in block_rows), default=0),
                "max_block_raw_M_equals_J": max((r["raw_M_equals_J_determinants"] for r in block_rows), default=0),
                "raw_pair_work_upper": sum(r["raw_pair_work_upper"] for r in block_rows),
                "dense_CSF_matrix_bytes_all_blocks": sum(r["dense_CSF_matrix_bytes"] for r in block_rows),
                "dense_raw_matrix_bytes_all_blocks": sum(r["dense_raw_determinant_matrix_bytes"] for r in block_rows),
                "full_magnetic_dimension": sum(csfs[i].total_J2 + 1 for i in indices)}

    families = {family: [m["index"] for m in metadata if m["family"] == family]
                for family in ("bare", "4f", "5p", "5s")}
    spaces = {family: space(indices) for family, indices in families.items()}
    for family in ("4f", "5p", "5s"):
        spaces["bare+" + family] = space(bare_indices + families[family])
    nested = list(bare_indices)
    for family in ("4f", "5p", "5s"):
        nested += families[family]
        spaces["nested_" + family] = space(nested)
    spaces["complete_union"] = space(list(range(len(csfs))))
    pool_capacity = sum(twoj_of_kappa(kappa) + 1 for n, kappa in orbital_keys
                        if (n, l_of_kappa(kappa)) in pool)
    hole_capacities = {label: sum(twoj_of_kappa(kappa) + 1 for n, kappa in orbital_keys
                                 if (n, l_of_kappa(kappa)) == shell)
                       for label, shell in zip(fixed["core_hole_shells"], hole_shells, strict=True)}
    # Independent occupation-only dimension; no coupling-tree/J recurrence is used.
    for family in ("4f", "5p", "5s"):
        expected = hole_capacities[family] * math.comb(pool_capacity, 2)
        check("independent_binomial_magnetic_dimension/" + family,
              spaces[family]["full_magnetic_dimension"] == expected,
              {"particle_magnetic_states": pool_capacity, "hole_capacity": hole_capacities[family],
               "binomial_dimension": expected})
    check("preflight_raw_cap", spaces["complete_union"]["max_block_raw_M_equals_J"] <= 2000,
          spaces["complete_union"]["max_block_raw_M_equals_J"])
    basis_record = {"configs": config_rows, "CSFs": metadata, "orbital_keys": orbital_keys,
                    "complete_J_parity_enumeration": True, "original_reference_provenance_retained": True}
    write("enumeration.json", basis_record)
    write("preflight.json", {"spaces": spaces, "contract_sha256": sha(HERE / "contract.json"),
                             "enumeration_sha256": sha(HERE / "enumeration.json"),
                             "resources": contract["resources"], "load_average": os.getloadavg(),
                             "warning": "Storage arithmetic and raw-pair work counts do not certify resident memory or timing."})
    print(json.dumps({"enumeration_complete": True, "families": {k: len(v) for k, v in families.items()},
                      "total": len(csfs), "max_raw_M_equals_J": spaces["complete_union"]["max_block_raw_M_equals_J"]}), flush=True)
    if "--counts-only" in sys.argv:
        return

    coupling_rows = []
    threshold = contract["acceptance"]["angular_nonzero_robust_threshold"]
    for family in ("4f", "5p", "5s"):
        for bare in bare_indices:
            target = csfs[bare]
            block_indices = [bare] + [i for i in families[family]
                                     if (csfs[i].total_J2, csfs[i].parity) == (target.total_J2, target.parity)]
            local_csfs = [csfs[i] for i in block_indices]
            density = CIRadialDensity(local_csfs, max_determinants=2000)
            check(f"one_complete_target_block/{family}/{bare}", len(density.blocks) == 1, len(block_indices))
            structure, projection, _ = density.blocks[0]
            verify_csf_span(structure.patterns, structure.j2, projection)
            gram_error = float(np.max(abs(projection.T @ projection - np.eye(len(block_indices)))))
            check(f"public_signed_J_span_and_Gram/{family}/{bare}", gram_error <= 1e-12,
                  {"Gram_error": gram_error, "raw_dimension": len(structure.dets), "CSFs": len(block_indices)})
            left = np.zeros(len(block_indices))
            left[0] = 1
            for local_index, index in enumerate(block_indices[1:], start=1):
                right = np.zeros(len(block_indices))
                right[local_index] = 1
                weights = density.contract_bilinear(left, right)
                grouped = {}
                for key, value in weights.items():
                    canonical, sign = canonical_static_radial_key(key)
                    grouped.setdefault(canonical, []).append(sign * value)
                coefficients = {key: math.fsum(values) for key, values in grouped.items()}
                largest = max((abs(value) for value in coefficients.values()), default=0.0)
                classification = ("robust_nonzero_angular_support" if largest > threshold else
                                  "cancellation_sized_ambiguous" if largest != 0 else "angular_zero")
                coupling_rows.append({"bare_index": bare, "target_label": target.label,
                                      "twoJ": target.total_J2, "parity": target.parity,
                                      "candidate_index": index, "config_index": metadata[index]["config_index"],
                                      "family": family, "classification": classification,
                                      "max_absolute_angular_coefficient": largest,
                                      "all_raw_weights": [[key, value] for key, value in weights.items()],
                                      "all_canonical_weights": [[key, value] for key, value in coefficients.items()]})
            print(json.dumps({"angular_complete_family": family, "bare_index": bare,
                              "target_CSF_block": len(block_indices), "elapsed_s": time.monotonic() - started}), flush=True)

    coupling_summary = []
    for bare in bare_indices:
        for config in config_rows[3:]:
            subset = [r for r in coupling_rows if r["bare_index"] == bare and r["config_index"] == config["index"]]
            counts = Counter(r["classification"] for r in subset)
            coupling_summary.append({"bare_index": bare, "bare_label": csfs[bare].label,
                                     "config_index": config["index"], "family": config["family"], "pair": config["pair"],
                                     "same_J_parity_CSFs": len(subset),
                                     "robust_nonzero_angular_support_CSFs": counts["robust_nonzero_angular_support"],
                                     "cancellation_sized_ambiguous_CSFs": counts["cancellation_sized_ambiguous"],
                                     "angular_zero_CSFs": counts["angular_zero"],
                                     "max_absolute_angular_coefficient": max((r["max_absolute_angular_coefficient"] for r in subset), default=0.0)})
    write("angular_support.json", {"scope": contract["coupling_scope"], "threshold_for_classification_only": threshold,
                                   "no_small_weight_removed": True, "rows": coupling_rows, "summary": coupling_summary})
    family_target_coupling = []
    names = {(1, 1): "6s1/2", (3, 1): "5d3/2", (5, 1): "5d5/2", (1, -1): "6p1/2", (3, -1): "6p3/2"}
    for bare in sorted(bare_indices, key=lambda index: fixed["target_symmetries"].index(
            [csfs[index].total_J2, csfs[index].parity])):
        for family in ("4f", "5p", "5s"):
            subset = [r for r in coupling_summary if r["bare_index"] == bare and r["family"] == family]
            family_target_coupling.append({"target": names[(csfs[bare].total_J2, csfs[bare].parity)],
                                          "twoJ": csfs[bare].total_J2, "parity": csfs[bare].parity,
                                          "family": family,
                                          "robust_angular_coupled_CSFs": sum(r["robust_nonzero_angular_support_CSFs"] for r in subset),
                                          "ambiguous_CSFs": sum(r["cancellation_sized_ambiguous_CSFs"] for r in subset),
                                          "robust_pair_families": [r["pair"] for r in subset if r["robust_nonzero_angular_support_CSFs"]]})
    staged_plan = [{"stage": 1, "space": "nested_4f", "CSFs": 294,
                   "purpose": "Complete same-pool 4f family against prior unbalanced 68-CSF result; test newly present 6s coupling, including 4f13 5d6p, with matched lower/upper states."},
                  {"stage": 2, "space": "nested_5p", "CSFs": 468,
                   "purpose": "Add full 5p-hole pair family to complete 4f space and measure differential 6s/D/P changes, retaining inter-family interference."},
                  {"stage": 3, "space": "nested_5s", "CSFs": 531,
                   "purpose": "Add full 5s-hole pair family to nested common space, including direct lower-state ss/dd/pp coupling; measure same-pool finite-model differential trend."}]
    source_used = {str(Path(mod.__file__).resolve()): sha(mod.__file__) for name, mod in sorted(sys.modules.items())
                   if name.startswith("atomx") and getattr(mod, "__file__", None) and mod.__file__.endswith(".py")}
    check("imports_from_frozen005", all(str(frozen) in path for path in source_used), len(source_used))
    write("source_identity.json", {"source_map_sha256": source["source_map_sha256"], "all_frozen_files_verified": 353,
                                   "imported_atomx_files_sha256": source_used, "runner_sha256": sha(__file__)})
    result = {"status": "verified_preflight_only", "computed_spectrum": False,
              "expanded_MCDHF_run": False, "experimental_gate_passed": False, "highest_level_reached": False,
              "complete_correlation_claim": False, "native_orbital_baseline": fixed["native_run"],
              "source_map_sha256": source["source_map_sha256"], "contract_sha256": sha(HERE / "contract.json"),
              "NR_configurations": 21, "candidate_NR_configurations": 18, "electrons": 69, "radial_orbitals": 24,
              "family_counts": {key: len(indices) for key, indices in families.items()}, "spaces": spaces,
              "coupling_summary": coupling_summary, "family_target_coupling": family_target_coupling,
              "next_staged_plan": staged_plan,
              "next_stage_common_protocol": {
                  "orbitals": "same stationary native005 orbitals, fixed; no new MCDHF certification",
                  "Hamiltonians": "public Dirac-Coulomb and complete static Breit; unchanged nucleus/metric; no QED/recoil/dynamic Breit/fits",
                  "spectral_closure": "all roots of every complete J/parity block, not five truncated target eigenpairs",
                  "state_tracking": "Embed prior signed CSF vectors into nested space; track exact J/parity and full overlap/character; report ground identity and intruders independently",
                  "observables": "Separate absolute lower/upper energy changes, same-model excitation differences, D/P centroids and splittings, bare-character weights and state-overlap ambiguity",
                  "controls": "Replay native005 bare principal matrix; verify CI residuals, Hermiticity, orthogonality and cross-symmetry zero before comparison",
                  "stopping_rule": "Failed numerical/state-identity gates stop qualification. No gap fitting, spliced states, summed isolated family increments, complete-correlation or expanded-MCDHF-stationarity claims.",
                  "conditional_followup": "Balanced differential RCI evidence first; then select bounded native multi-reference EOL MCDHF objective and estimate response/gradient workload."},
              "card_summary": {"label": "Balanced CV frontier: enumeration/angular preflight only; no spectra computed",
                               "electrons": 69, "orbital_pool": 24, "configuration_count": 21,
                               "candidate_configuration_count": 18, "CSFs": len(csfs),
                               "family_CSFs": {key: len(indices) for key, indices in families.items()},
                               "max_block_CSFs": spaces["complete_union"]["max_block_CSFs"],
                               "max_raw_M_equals_J": spaces["complete_union"]["max_block_raw_M_equals_J"],
                               "symmetry_blocks": spaces["complete_union"]["symmetry_blocks"],
                               "ground_6s_robust_DC_angular_couplings": {row["family"]: row["robust_angular_coupled_CSFs"]
                                                                        for row in family_target_coupling if row["target"] == "6s1/2"},
                               "nested_CSF_counts": [294, 468, 531],
                               "next_step": "Fixed005-orbital nested DC/static RCI with matched lower/upper differences; no new MCDHF run here",
                               "computed_spectrum": False, "complete_correlation": False},
              "resource_observation": {"seconds_uncontrolled": time.monotonic() - started,
                                       "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                       "threads": 1, "cap_GiB": 4, "wall_cap_s": 360},
              "inputs_sha256": contract["inputs_sha256"]}
    result["artifacts_sha256"] = {name: sha(HERE / name) for name in
                                 ("enumeration.json", "preflight.json", "angular_support.json", "source_identity.json", "run.py")}
    write("result.json", result)
    write("verification.json", {"all_passed": all(c["passed"] for c in checks), "checks": checks,
                                "result_sha256": sha(HERE / "result.json"),
                                "artifacts_sha256": result["artifacts_sha256"],
                                "scope": "integer space identity and public signed angular span/support only"})
    signal.alarm(0)
    print(json.dumps({"preflight_verified": True, "CSFs": len(csfs), "result_sha256": sha(HERE / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
