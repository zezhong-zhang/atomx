"""SOURCE/JSON/AST and descriptor admission only; no numerical imports or NPZ bytes."""
import ast
import hashlib
import importlib.util
import json
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 2**20, 1024 * 2**20))
    signal.alarm(30)
    contract = load(HERE / "contract.json")
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise AssertionError(name)

    forbidden = {"numpy", "numba", "atomx", "scipy", "cupy"}
    check("entry_has_no_numerical_modules", not forbidden.intersection(sys.modules))
    trees = {}
    for name in ("run.py", "launch.py", "prepare.py"):
        text = (HERE / name).read_text()
        trees[name] = ast.parse(text)
        compile(text, str(HERE / name), "exec")
        imports = [n for n in trees[name].body if isinstance(n, (ast.Import, ast.ImportFrom))]
        check("top_level_stdlib_only_" + name, all(
            not forbidden.intersection([a.name.split(".")[0] for a in n.names]) if isinstance(n, ast.Import)
            else n.module.split(".")[0] not in forbidden for n in imports))
    check("no_exec_AST_or_dynamic_text", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
          and n.func.id in ("exec", "eval") for t in trees.values() for n in ast.walk(t)))
    spec = importlib.util.spec_from_file_location("private31_metadata_preparation", HERE / "run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    actual_contract, roles, source, deferred = driver.metadata(arrays=False)
    check("default_metadata_contract", actual_contract == contract)
    check("exact_six_NPZ_roles_deferred_unread", set(deferred) == {
        "workspace", "original_hS", "tracking_CI", "recovered_point", "recovered_CI", "saved_gradient_metric"})
    for name, row in contract["input_roles"].items():
        if roles[name].suffix != ".npz":
            check("input_source_JSON_" + name, sha(roles[name]) == row["sha256"])
    actual_directional = driver.admit_directional(contract["actual_native31_directional_prerequisite"], contract['fixed'])
    check("actual31_FD730_complete_controls", len(actual_directional['result']['checks']) == 730
          and all(r['passed'] for r in actual_directional['result']['checks']))
    check("not_scientifically_authorized", contract["execution_authorized"] is False)
    old = load(roles["old_FD_failure"])
    check("old_representation_failure_not_FD_anchor", not old.get("derivative_domains") and len(old["points"]) == 1)
    public_identity = load(roles["shared_source_identity"])
    public_map = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/atomx").rglob("*.py"))}
    actual_public_identity = load(roles["current_shared_source_identity"])
    bound_map = actual_public_identity["all_atomx_python_sha256"]
    check("shared_source_full356_unchanged", public_map == bound_map and len(public_map) == 356)
    public_map_sha256 = hashlib.sha256(json.dumps(public_map, sort_keys=True).encode()).hexdigest()
    # Public source identity may use compact separators; preserve its declared map domain.
    check("shared_source_identity_map", public_identity["source_map_sha256"] == "bef2f5d6b66d81efb0ce54ec8b2bd37c8f40a1855b07ed6ed3057912b79a50d3")
    check("current_shared_epoch_is_not_f043_replay", actual_public_identity["f043_actual_qualification_source_unchanged"]
          and contract["shared_source_disposition"]["not_source_replay_or_adoption"])
    check("private_source356_map", len(load(roles["f043_identity"])["all_atomx_python_sha256"]) == 356)

    def named(tree, name):
        rows = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        if len(rows) != 1:
            raise ValueError("ambiguous named source function " + name)
        return rows[0]

    def ast_body(node, aliases=None):
        node = ast.parse(ast.unparse(node)).body[0]
        if aliases:
            class Rename(ast.NodeTransformer):
                def visit_Name(self, item):
                    if item.id in aliases:
                        item.id = aliases[item.id]
                    return item
            node = Rename().visit(node)
        return ast.dump(node, include_attributes=False)

    reference_tree = ast.parse(roles["native31_driver"].read_text())
    for name in ("coordinate", "tangent", "metric_action", "represent"):
        check("native438_literal_" + name, ast_body(named(trees["run.py"], name), {"group": "keys"})
              == ast_body(named(reference_tree, name)))
    gradient_source = (source / "atomx/methods/csf/variational.py").read_text()
    producer_source = roles["native31_driver"].read_text()
    check("public_g_246_contiguous_source", "np.concatenate(orbitals[key])" in gradient_source)
    check("native_saved31_246_producer", 'full_array = np.array([full[k] for k in functional.keys])' in producer_source
          and 'full_normalized_radial=full_array' in producer_source)
    driver_source = (HERE / "run.py").read_text()
    check("saved_and_new_g_shape_admitted_no_2x123", 'shape != (31, 246)' in driver_source and '(31, 2, 123)' not in driver_source)
    check("original_metric_and_derivative_domains_kept", contract["gates"]["g"] == 1e-9 and contract["gates"]["pg"] == 1e-10
          and contract["gates"]["CI_residual_Ha"] == 1e-8 and contract["gates"]["CI_Gram"] == 1e-10
          and contract["gates"]["signed_span"] == 1e-8 and contract["gates"]["cluster_gap_Ha"] == 1e-7)
    check("captured_mask_set1_before_public_setup", driver_source.index('previous_NumBa_mask = nb.get_num_threads()')
          < driver_source.index('nb.set_num_threads(1)') < driver_source.index('phase = "complete_public_setup"'))
    check("private_CPU4_finally_restores_mask", 'finally:\n                nb.set_num_threads(mask)' in driver_source)
    check("outer_mask_and_affinity_restore", 'nb.set_num_threads(previous_NumBa_mask)' in driver_source
          and 'os.sched_setaffinity(0, previous_affinity)' in driver_source)
    check("CPU4_same_factory_after_identity_guard", driver_source.index('CPU4_same_source_factory_plan')
          < driver_source.index('candidate.executor.kernels = CPU1_engine.kernels'))
    check("all_four_bands_original_order_observed", 'ordinary_PP_PQ_QP_QQ_order_' in driver_source
          and 'checked_components[0] != 2824' in driver_source)
    check("hS_factors_banks_moment_requests_not_producer_inference", all(n in driver_source for n in
          ('"hS", "factors", "dense_banks", "component_moments"', 'ledger["hits" if hit else "misses"]')))
    check("all_phase_kernel_producers_observed", all(n in driver_source for n in
          ('"_build_rkb_sym"', '"_build_rkb_pqpq"', '"_build_rkb_qpqp"', '"_assemble_near_rkb"', '"_triang_sym"', '"_triang_pqpq"')))
    check("no_shared_default_install_or_fallback", not any(x in driver_source for x in ('pip install', 'except ImportError', 'setattr(functional,', 'np.linalg.eigh(')))
    check("hard_caps_fresh_single_point", contract["resources"]["calls"]["H"] == 1
          and contract["resources"]["calls"]["CI"] == 1 and contract["resources"]["calls"]["public_full_gradient"] == 1
          and contract["resources"]["calls"]["private_full_gradient"] == 3 and contract["resources"]["calls"]["optimizer"] == 0)
    check("CUDA_RSS_virtual_device_domains_distinct", contract["resources"]["virtual_AS_bytes"] == 64*2**30
          and contract["resources"]["maximum_RSS_bytes"] == int(3.95*2**30)
          and contract["resources"]["device_payload_bytes"] == 512*2**20)
    check("foreground_guard_samples_process_group", 'process_group_RSS(child.pid)' in (HERE / "launch.py").read_text()
          and 'os.killpg' in (HERE / "launch.py").read_text())
    check("CPU4_actual_affinity_admission", set(contract["resources"]["CPU4"]) == {28, 29, 30, 31}
          and contract["resources"]["CPU1"] == 30)
    check("literal_interpreter_exists", Path(contract["resources"]["interpreter_literal"]).is_file())
    for name in ("NumPy_literal_file", "Numba_literal_file"):
        check("runtime_literal_path_" + name, Path(contract["runtime"][name]).is_file())
    runtime = ROOT / contract["runtime"]["isolated_runtime"]
    check("isolated_existing_CUDA_runtime_paths", (runtime / 'numba_cuda/numba/cuda/__init__.py').is_file()
          and (runtime / 'nvidia/cu13/nvvm/libdevice/libdevice.10.bc').is_file())

    # Descriptor-only mocked documents, not numerical values or a fabricated real proof.
    expected = {"source_sha256": "source_map_sha256", "metric_sha256": "new_metric_sha256",
                "geometry_sha256": "new_geometry_sha256", "state_sha256": "new_state_sha256", "objective_sha256": "objective_identity"}
    domains = {h: dict(ordinary_reported=dict(h=float(h), passed=False),
              independent_centered=[dict(reference="fixedLambda_ideal", passed=True), dict(reference="public_retained", passed=True)],
              production_callback=False, no_uniform_error_bound=True) for h in ("1e-05", "5e-06")}
    result = dict(status="native31_single_combined_direction_two_h_FD_completed", all_declared_controls_passed=False,
                  points={k: {} for k in ("baseline", "h0_minus", "h0_plus", "h1_minus", "h1_plus")}, completed=dict(H=5, CI=5),
                  derivative_domains=domains, single_direction_finite_measurement_only=True, full31_space_certificate=False,
                  no_energy_callback=True, ordinary_reported_objective_unchanged=True,
                  **{value: contract["fixed"][value] for value in expected.values()})
    receipt = dict(status="foreground_closed", exit_code=1)
    prefix = "docs/yb_mcdhf_rci_2026_10_11/screened_f_next_step208/descriptor_mock_ONLY/"
    bound = dict(result=dict(path=prefix+'result.json', sha256='r'), domain_ledger=dict(path=prefix+'derivative_domains.json',sha256='d'),
                 root_receipt=dict(path=prefix+'root_execution.json',sha256='e'), recorded_domains=domains,
                 recorded_controls_passed=False, raw_kernel_diagnostic_only=True,
                 assertions={k:dict(document='result',pointer=[v]) for k,v in expected.items()})
    documents = {'result.json':result,'derivative_domains.json':domains,'root_execution.json':receipt}
    old_sha, old_load = driver.sha, driver.load
    driver.sha = lambda p: {'result.json':'r','derivative_domains.json':'d','root_execution.json':'e'}[p.name]
    driver.load = lambda p: documents[p.name]
    try:
        check("descriptor_closed_failed_ordinary_retained_raw_diagnostic_only", driver.admit_directional(bound, contract['fixed'])['result'] is result)
        for name, altered in (("deferred_role", None), ("old29_path", dict(bound, result=dict(path='docs/old29/result.json',sha256='r'))),
                              ("production_admission",dict(bound,raw_kernel_diagnostic_only=False))):
            try:
                driver.admit_directional(altered, contract['fixed'])
            except (ValueError, KeyError, TypeError):
                check("descriptor_reject_"+name, True)
            else:
                check("descriptor_reject_"+name, False)
        for name, field, value in (("baseline_only",'points',{'baseline':{}}), ("empty_domains",'derivative_domains',{}),
                                   ("fullspace_claim",'full31_space_certificate',True), ("wrong_metric",'new_metric_sha256','wrong')):
            original = result[field]
            result[field] = value
            try:
                driver.admit_directional(bound, contract['fixed'])
            except (ValueError, KeyError, TypeError):
                check("descriptor_reject_"+name, True)
            else:
                check("descriptor_reject_"+name, False)
            finally:
                result[field] = original
    finally:
        driver.sha, driver.load = old_sha, old_load
    check("exit_still_zero_numerical_imports", not forbidden.intersection(sys.modules))
    signal.alarm(0)
    output = dict(status="SOURCE_JSON_AST_DESCRIPTOR_ONLY_NO_SCIENTIFIC_QUALIFICATION", all_passed=True,
                  checks_count=len(checks), checks=checks, driver_sha256=sha(HERE/'run.py'), launcher_sha256=sha(HERE/'launch.py'),
                  contract_sha256=sha(HERE/'contract.json'), prepare_sha256=sha(__file__),
                  actual_FD_prerequisite=contract["actual_native31_directional_prerequisite"], six_NPZ_roles_deferred_unread=deferred,
                  historical_public_source_identity_sha256=public_identity['source_map_sha256'],
                  current_public_source_identity_sha256=actual_public_identity['source_map_sha256'],
                  public_map_defaultJSON_sha256=public_map_sha256,
                  arrays_or_NPZ_bytes_reads=0, numerical_imports_or_calls=0, GPU_queries_JIT_calls=0)
    (HERE/'preparation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,checks_count=len(checks),source_only=True)))


if __name__ == '__main__':
    main()
