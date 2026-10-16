"""Read immutable source/JSON/AST only; no Atomx import or numerical execution."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def classes(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def named_calls(tree, name):
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


def method_calls(tree, name):
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == name]


def main():
    contract, identity = load(HERE / 'contract.json'), load(HERE / 'source_identity.json')
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    current = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/atomx').rglob('*.py'))}
    check('complete_current353_source_map_preserved', current == identity['all_atomx_python_sha256']
          == contract['full_source_identity'] and len(current) == 353)
    for name, expected in contract['full_source_identity'].items():
        check('actual_immutable_source:' + name, sha(ROOT / name) == expected)
    for name, expected in contract['inputs_sha256'].items():
        check('actual_immutable_input:' + name, sha(ROOT / name) == expected)
    trees = {}
    for name, expected in contract['source_identity_sha256'].items():
        archived = HERE / 'reference_source' / Path(name).relative_to('src')
        check('archived_relevant_source:' + name, sha(archived) == expected)
        trees[name] = ast.parse(archived.read_text())
    ci = trees['src/atomx/methods/ci.py']
    result = classes(ci)['CIResult']
    fields = {node.target.id for node in result.body if isinstance(node, ast.AnnAssign)}
    check('CIResult_current_contains_eigenpairs_notes_not_retained_H', fields == {
        'state_energies', 'mixing_coefficients', 'ground_energy', 'provenance'})
    solve = functions(ci)['ci_solve']
    check('ci_solve_current_single_public_H_producer_site', len(named_calls(solve, 'build_ci_matrix')) == 1)
    kwargs = {arg.arg for arg in solve.args.kwonlyargs}
    check('current_optional_radial_snapshot_supported', 'radial_snapshot' in kwargs)
    check('retention_API_is_not_yet_implemented', 'retain_hamiltonian' not in kwargs)
    functional = classes(trees['src/atomx/methods/csf/variational.py'])['CSFOrbitalFunctional']
    evaluate = functions(functional)['evaluate']
    check('functional_current_one_CI_producer_site', len(named_calls(evaluate, 'ci_solve')) == 1)
    check('functional_does_not_directly_rebuild_H', not named_calls(evaluate, 'build_ci_matrix'))
    optimizer = functions(trees['src/atomx/numerics/bspline/orbital_optimization.py'])['optimize_orbitals']
    accepted_reuse = [node for node in ast.walk(optimizer) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'evaluate'
                      and any(key.arg == 'ci_result' and isinstance(key.value, ast.Name)
                              and key.value.id == 'trial_ci' for key in node.keywords)]
    check('optimizer_already_passes_accepted_trial_CI_to_gradient', len(accepted_reuse) == 1)
    check('optimizer_has_no_public_H_build', not named_calls(optimizer, 'build_ci_matrix'))
    live_driver = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/run.py'
    driver = ast.parse(live_driver.read_text())
    checkpoints = [node for node in ast.walk(driver) if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint']
    check('current_checkpoint_one_full_H_rebuild_site', len(checkpoints) == 1
          and len(named_calls(checkpoints[0], 'build_ci_matrix')) == 1)
    radial = classes(trees['src/atomx/physics/csf/radial_snapshot.py'])['CIRadialSnapshot']
    radial_solve = functions(radial)['solve']
    check('radial_snapshot_solve_uses_public_CI_and_forwards_options',
          len(named_calls(radial_solve, 'ci_solve')) == 1
          and any(key.arg is None and isinstance(key.value, ast.Name) and key.value.id == 'options'
                  for call in ast.walk(radial_solve) if isinstance(call, ast.Call)
                  for key in call.keywords))
    check('radial_snapshot_has_no_H_matrix_owner_field', not any(isinstance(node, ast.Attribute)
          and node.attr in ('matrix_Ha', 'hamiltonian_snapshot') for node in ast.walk(radial)))
    binding = classes(trees['src/atomx/physics/csf/effective_one_body.py'])['CIReferenceBinding']
    check('public_CIReferenceBinding_capture_and_validate_exist',
          {'capture', 'validate'} <= set(functions(binding)))
    finite = functions(trees['src/atomx/methods/ci_effective.py'])['build_finite_ci_effective_hamiltonian']
    check('full_finiteCI_producer_reuses_public_reference_binding',
          len(method_calls(finite, 'capture')) == 1 and len(named_calls(finite, 'build_ci_matrix')) == 1)
    signature = classes(trees['src/atomx/angular/csf_hamiltonian.py'])['CSFSignature']
    check('CSF_signature_key_does_not_imply_parity', not any(isinstance(node, ast.Attribute)
          and node.attr == 'parity' for node in ast.walk(signature)))
    previous = load(ROOT / 'docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run002/result_unverified.json')
    check('completed_run002_recorded_call_envelope', len(previous['checkpoints']) == 161
          and previous['energy_trial_calls'] == 202 and previous['optimizer_full_gradient_calls'] == 161
          and previous['baseline_gradient_calls'] == 2 and previous['total_run_full_gradient_calls'] == 163)
    check('full531_dense_memory_exact', contract['fixed_conditions']['dense_matrix_bytes_float64'] == 531*531*8)
    gates = contract['original_gates']
    check('original_scientific_gates_preserved', gates['orbital_residual'] == 1e-7
          and gates['energy_indicator_Ha'] == 1e-8 and gates['CI_residual_Ha'] == 1e-8
          and gates['CI_Gram'] == gates['radial_Gram'] == gates['no_pair_leakage'] == 1e-10
          and gates['samepoint_projected_g_max'] == 1e-9 and gates['samepoint_preconditioned_g_max'] == 1e-10)
    check('no_source_implementation_or_runtime_artifacts', not (ROOT / 'src/atomx/physics/csf/hamiltonian_snapshot.py').exists()
          and not any((HERE / name).exists() for name in ('result.json', 'checkpoints', 'benchmark.json')))
    readme = (HERE / 'README.md').read_text()
    proposal = (HERE / 'proposal.diff').read_text()
    check('proposal_is_explicit_nonexecuted_sketch', proposal.startswith('NONEXECUTABLE IMPLEMENTATION SKETCH'))
    check('README_design_not_measured_speedup', 'not an implemented optimization or a measured' in readme)
    check('complete_point_checked_getter_and_no_override', 'matrix_for' in proposal
          and 'No public constructor from arbitrary arrays/matrix_override' in proposal
          and 'do not silently rebuild' in proposal)
    check('independent_fresh_replay_preserved', 'independent fresh endpoint replay continues' in proposal)
    for link in re.findall(r'\]\(([^)]+)\)', readme):
        check('README_link:' + link, (HERE / link).exists() or link in ('source_audit.json', 'verification.json'))
    audit = {'reference_source_map_sha256': identity['reference_source_map_sha256'],
             'read_only_source_files_archived': len(trees),
             'public_CI_builder_sites': named_calls(solve, 'build_ci_matrix'),
             'functional_CI_builder_sites': named_calls(evaluate, 'ci_solve'),
             'accepted_trial_CI_forwarding_sites': [node.lineno for node in accepted_reuse],
             'checkpoint_redundant_H_builder_sites': named_calls(checkpoints[0], 'build_ci_matrix'),
             'existing_public_binding_used_by_full_finiteCI': True,
             'complete_CIResult_current_fields': sorted(fields),
             'historical_run002_observed_counters': {key: previous[key] for key in
                 ('energy_trial_calls', 'optimizer_full_gradient_calls', 'baseline_gradient_calls', 'total_run_full_gradient_calls')},
             'historical_run002_complete_checkpoints': len(previous['checkpoints']),
             'hypothetical_total_normal_path_builds': 2+1+202+161,
             'hypothetical_count_is_not_instrumented': True,
             'call_count_goal': 'One producer build per successful CI solve, zero extra accepted gradient/callback builds; independent fresh controls retained.',
             'matrix_storage_float64_bytes': 531*531*8, 'one_owner_matrix_MiB': 531*531*8/1024**2,
             'Atomx_imports': 0, 'physical_H_builds': 0, 'CI_solves': 0, 'gradients': 0,
             'optimizer_calls': 0, 'physics_tests_run': 0,
             'limitations': 'AST call sites and archived JSON only. No executed owner/getter, no new physical matching or measured costs.'}
    (HERE / 'source_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    result = {'all_passed': all(item['passed'] for item in checks), 'checks_count': len(checks), 'checks': checks,
              'reference_source_map_sha256': identity['reference_source_map_sha256'],
              'Atomx_imports': 0, 'physical_H_builds': 0, 'CI_solves': 0, 'gradients': 0,
              'optimizer_calls': 0, 'physics_tests_run': 0,
              'files_sha256': {name: sha(HERE / name) for name in
                  ('contract.json', 'source_identity.json', 'source_audit.json', 'proposal.diff', 'README.md', 'verify_design.py')},
              'scope': 'Read-only design/source/JSON/AST/link audit only; physical reuse and performance remain unimplemented/unmeasured.'}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'all_passed': result['all_passed'], 'checks_count': len(checks),
                      'failed': [item for item in checks if not item['passed']], 'verification_sha256': sha(HERE / 'verification.json')}))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
