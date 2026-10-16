"""Independent metadata/AST/refusal controls only; no H, CI, gradient or optimizer."""
import ast
import copy
import hashlib
import json
import runpy
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    rows = []

    def check(name, passed):
        rows.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError("Independent preparation control: " + name)

    contract, models = load(HERE / "contract.json"), load(HERE / "models.json")
    source = load(ROOT / contract["input_roles"]["source_identity"]["path"])
    frozen = ROOT / contract["fixed_conditions"]["frozen_source_root"]
    check("full_source_map", hashlib.sha256(json.dumps(source["all_atomx_python_sha256"], sort_keys=True,
          separators=(",", ":")).encode()).hexdigest() == contract["fixed_conditions"]["source_map_sha256"])
    check("whole353_file_set", {"src/" + str(p.relative_to(frozen)) for p in (frozen / "atomx").rglob("*.py")}
          == set(source["all_atomx_python_sha256"]) and len(source["all_atomx_python_sha256"]) == 353)
    for name, expected in source["all_atomx_python_sha256"].items():
        check("frozen_source/" + name, sha(frozen / Path(name).relative_to("src")) == expected)
    for name, expected in contract["inputs_sha256"].items():
        check("input/" + name, sha(ROOT / name) == expected)
    fixed, caps, gates = contract["fixed_conditions"], contract["resources"], contract["acceptance"]
    check("prospective_160_failed_original_stationarity", fixed["checkpoint"]["stationarity_gates_passed"] is False
          and fixed["checkpoint"]["original_residual"] > gates["original_stationarity_residual"]
          and fixed["checkpoint"]["manifest_sha256"] == "1ce49cd2bcc49bb22ed7bf21ec8f1e21fe3f37f945a2ef5ca3a3669e1a15c93c")
    check("complete_three_models", list(models["models"]) == ["baseline", "rank_5p_double", "balanced_virtual"])
    for name, nr, full, selected, radial in (
            ("baseline", 21, 531, 261, 24), ("rank_5p_double", 30, 2483, 1240, 24),
            ("balanced_virtual", 69, 2163, 1046, 29)):
        model = models["models"][name]
        check(name + "/declared_complete_counts", len(model["configs"]) == nr
              and len({r["config"] for r in model["configs"]}) == nr
              and model["full_CSFs"] == full and model["complete_target_CSFs"] == selected
              and model["radial_spinors"] == radial)
        for i, row in enumerate(model["configs"]):
            parsed = []
            for token in row["config"].split():
                position = next(j for j, char in enumerate(token) if char.isalpha())
                ell = "spdfghiklmnoq".index(token[position])
                parsed.append((token[:position+1], int(token[position+1:]), ell))
            check(name + f"/config{i}/69e", sum(q for _, q, _ in parsed) == 69)
            check(name + f"/config{i}/Pauli_NR", all(0 < q <= 2 * (2 * ell + 1) for _, q, ell in parsed)
                  and len({shell for shell, _, _ in parsed}) == len(parsed))
        check(name + "/all_old_NR_retained", model["configs"][:21] == models["models"]["baseline"]["configs"])
        check(name + "/five_complete_blocks", sum(r["CSFs"] for r in model["complete_target_blocks"]) == selected
              and {(r["twoJ"], r["parity"]) for r in model["complete_target_blocks"]}
              == {tuple(row) for row in models["selected_sectors"]})
    overlap = load(HERE / "overlap_budget_preflight.json")
    check("independent_raw15_controls", overlap["all_controls_passed"] is True and len(overlap["controls"]) == 15)
    for name, upper in (("baseline", 1536), ("rank_5p_double", 3712), ("balanced_virtual", 6091)):
        check(name + "/compatible_pair_upper_cap", overlap["models"][name]["total_compatible_pairs_upper"] == upper
              and upper <= caps["max_signed_overlap_determinant_pairs"])
    check("unchanged_raw_pair_caps", caps["max_raw_determinants_per_block"] == 2000
          and caps["max_signed_overlap_determinant_pairs"] == 1_000_000)
    check("explicit_complete_call_budgets", caps["complete_CI_calls_budget"] == 3
          and caps["implicit_public_CI_H_builds_budget"] == 3 and caps["explicit_archive_H_builds_budget"] == 3
          and caps["total_many_electron_H_builds_budget"] == 6
          and caps["orbital_gradients_budget"] == caps["optimizer_calls_budget"] == 0)
    check("virtual_spectral_budgets", caps["nuclear_Dirac_spectral_calls"] == 5
          and caps["nuclear_complement_lowest_root_spectral_calls"] == 5
          and caps["original_P_complement_lowest_two_spectral_controls"] == 5
          and caps["initial_one_body_matrix_kappa_count"] == 7)
    tree = ast.parse((HERE / "run.py").read_text())
    execution = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    check("no_top_level_heavy_imports", all(not (isinstance(node, ast.ImportFrom) and node.module
          and node.module.startswith(("atomx", "numpy", "scipy"))) for node in tree.body))
    check("all_physical_imports_inside_execute", all(node in list(ast.walk(execution)) for node in ast.walk(tree)
          if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("atomx", "scipy"))))
    names = [node.func.id for node in ast.walk(execution) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    check("public_CI_H_virtual_only_lexical_calls", names.count("ci_solve") == names.count("build_ci_matrix")
          == names.count("extend_nuclear_virtuals") == 1)
    check("no_gradient_or_optimizer_calls", not any(isinstance(node, ast.Call) and
          ((isinstance(node.func, ast.Attribute) and node.func.attr in ("evaluate", "gradient_from_weights", "optimize"))
           or (isinstance(node.func, ast.Name) and node.func.id.startswith(("optimize", "gradient"))))
          for node in ast.walk(tree)))
    namespace = runpy.run_path(str(HERE / "run.py"), run_name="metadata_and_refusal_controls_only")
    namespace["metadata"]()
    check("default_metadata_all_inputs_replay", True)
    proposal = load(HERE / "input_binding_proposal.json")
    check("proposal_remains_pending", proposal["execution_authorized_by_root"] is False
          and proposal["pending_exact_root_execution_contract"] is True)
    check("proposal_binds_final_driver", proposal["driver_sha256"] == sha(HERE / "run.py"))
    mutations = {
        "pending": {},
        "wrong_driver": dict(execution_authorized_by_root=True, pending_exact_root_execution_contract=False,
                             driver_sha256="deliberately_wrong"),
        "wrong_point": dict(execution_authorized_by_root=True, pending_exact_root_execution_contract=False,
                            driver_sha256=sha(HERE / "run.py"), expected_checkpoint_manifest_sha256="wrong_point"),
        "wrong_source": dict(execution_authorized_by_root=True, pending_exact_root_execution_contract=False,
                             driver_sha256=sha(HERE / "run.py"), expected_source_map_sha256="wrong_epoch"),
        "missing_original_tracker_attestation": dict(execution_authorized_by_root=True,
                             pending_exact_root_execution_contract=False, driver_sha256=sha(HERE / "run.py"))}
    with tempfile.TemporaryDirectory(prefix="readonly_controls_", dir=HERE) as temporary:
        for name, change in mutations.items():
            candidate = copy.deepcopy(proposal)
            candidate.update(change)
            path = Path(temporary) / (name + ".json")
            path.write_text(json.dumps(candidate))
            rejected = False
            try:
                namespace["authorize"](path)
            except ValueError:
                rejected = True
            check("reject/" + name, rejected)
    check("no_actual_execution_evidence", not any((HERE / name).exists() for name in
          ("resource_admission.json", "execution_source.py", "progress.json", "result.json", "failure.json")))
    check("zero_physical_imports_in_verifier", not any(name == "numpy" or name == "scipy"
          or name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    result = dict(all_preparation_controls_passed=True, checks_count=len(rows), checks=rows,
                  contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(HERE / "run.py"),
                  verifier_sha256=sha(__file__), source_map_sha256=fixed["source_map_sha256"],
                  point_manifest_sha256=fixed["checkpoint"]["manifest_sha256"],
                  H_builds=0, CI_solves=0, virtual_constructions=0, gradients=0, optimizer_calls=0,
                  heavy_imports=0, execution_authorized=False, scope="Independent preparation/hash/AST/refusal only",
                  extra_owned_artifact_sha256={name: sha(HERE / name) for name in
                      ("models.json", "same_point_evidence.json", "overlap_budget_preflight.json",
                       "count_overlap_budget.py", "prepare_contract.py", "MATHEMATICAL_SOURCE_AUDIT.md", "README.md")})
    path = HERE / "preparation_verification.json"
    if path.exists():
        raise FileExistsError("Preserve frozen independent preparation proof")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(all_preparation_controls_passed=True, checks=len(rows), new_physical_calls=0,
                         proof_sha256=sha(path))))


if __name__ == "__main__":
    main()
