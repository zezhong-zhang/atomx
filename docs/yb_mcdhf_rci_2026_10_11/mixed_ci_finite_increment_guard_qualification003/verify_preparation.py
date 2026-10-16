"""Metadata/AST preparation only: no NumPy, Atomx, pytest or physics imports."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
PRIOR = BASE.parent / 'mixed_ci_finite_increment_implementation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((BASE / 'contract.json').read_text())
    checks = []
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
    for name, expected in contract['inputs_sha256'].items():
        check('bound_' + name, sha(REPO / name) == expected)
    identity = json.loads((PRIOR / 'candidate_source_identity.json').read_text())
    check('complete_candidate_map', identity['source_map_sha256'] == contract['source_map_sha256'])
    for name, expected in identity['all_atomx_python_sha256'].items():
        check('current_' + name, sha(REPO / name) == expected)
        check('archive_' + name, sha(PRIOR / 'source' / name.removeprefix('src/')) == expected)
    test = ast.parse((BASE / 'test_saved_guards.py').read_text())
    runner = ast.parse((BASE / 'run.py').read_text())
    check('runner_has_explicit_main_guard', isinstance(runner.body[-1], ast.If))
    forbidden = {'ci_solve', 'solve_dirac', 'optimize_orbitals', 'bspline_dc_radial_increment',
                 '_polar_retraction', 'gradient_from_weights', 'evaluate'}
    actual_calls = [node for node in ast.walk(test) if isinstance(node, ast.Call)]
    names = [node.func.id if isinstance(node.func, ast.Name) else
             node.func.attr if isinstance(node.func, ast.Attribute) else '' for node in actual_calls]
    check('test_has_no_forbidden_physical_call', not forbidden.intersection(names))
    check('one_genuine_public_snapshot_call', names.count('build_ci_hamiltonian_snapshot') == 1)
    check('zero_arbitrary_matrix_restore', not any('restore' in name for name in names))
    check('negative_shell_only_one', names.count('__new__') == 1)
    functions = {node.name: node for node in test.body if isinstance(node, ast.FunctionDef)}
    check('exact_three_test_functions', sorted(name for name in functions if name.startswith('test_')) == [
        'test_legacy_fixed_density_and_constrained_mixed_guards',
        'test_saved_medium_step_original_derivative_gate',
        'test_unsupported_scope_rejects_without_physical_calls'])
    check('predeclared19cases', contract['pytest_cases'] == 19)
    check('original_derivative_gate', contract['derivative_gate'] ==
          {'selected_h': 1e-6, 'relative': 1e-5, 'absolute': 1e-8})
    check('original_full_eigenpair_gate', contract['eigenpair_gate'] ==
          {'max_column_Euclidean_residual': 1e-8, 'Gram': 1e-10})
    check('failed_predecessor_preserved', json.loads((PRIOR / 'actual/result.json').read_text())['status'] == 'failed')
    check('accepted_causal_proof_passed', json.loads((BASE.parent /
          'mixed_ci_finite_increment_saved_scale_audit/verification.json').read_text())['all_passed'])
    mapping = json.loads((BASE.parent / 'mixed_ci_one_body_mapping_audit/actual/result.json').read_text())
    check('zero_builder_mapping_admission_passed', mapping['all_passed'] is True)
    check('zero_builder_mapping_scope', mapping['H_CI_gradient_radial_builders'] == 0
          and mapping['Atomx_imports'] == 0 and mapping['eigensolves'] == 0 and mapping['SVD'] == 0)
    failure = json.loads((BASE.parent / 'mixed_ci_finite_increment_guard_qualification002/actual/result.json').read_text())
    check('guard002_failure_preserved', failure['status'] == 'failed')
    check('guard002_stopped_before_H', failure['actual_calls']['completed']['H_build'] == 0
          and failure['actual_calls']['completed']['one_body_assembly'] == 3)
    proof = dict(all_passed=all(row['passed'] for row in checks), checks_count=len(checks), checks=checks,
                 contract_sha256=sha(BASE / 'contract.json'), runner_sha256=sha(BASE / 'run.py'),
                 test_sha256=sha(BASE / 'test_saved_guards.py'), verifier_sha256=sha(Path(__file__)),
                 source_map_sha256=identity['source_map_sha256'],
                 Atomx_imports=0, NumPy_imports=0, pytest_calls=0, H_CI_gradient_radial_calls=0)
    (BASE / 'preparation_verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'checks'}))
    if not proof['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
