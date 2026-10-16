"""Replay structural/read-only preparation gates, never evaluate physics."""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    preflight = json.loads((HERE / 'preflight.json').read_text())
    contract = json.loads((HERE / 'contract.json').read_text())
    check('preflight_contract_bound', preflight['contract_sha256'] == sha(HERE / 'contract.json'))
    check('complete_count_caps', preflight['CSFs'] == 531 and preflight['max_raw_M_equals_J'] == 232
          and preflight['selected_overlap_pair_work_upper'] == 227582)
    check('input_source_identity_checked', preflight['input_hashes_verified'] == 18
          and preflight['old353_reference_source_files_verified'] == 353)
    check('old_metric_exact', preflight['original_metric_sha256']
          == contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256'])
    check('no_compute_in_preflight', preflight['functional_evaluations']
          == preflight['CI_solves'] == preflight['optimizer_calls'] == 0)
    check('no_execution_epoch', not (HERE / 'source').exists() and not (HERE / 'source_identity.json').exists())
    check('no_endpoint_claim', not (HERE / 'result_unverified.json').exists() and not (HERE / 'checkpoints').exists())
    caps = contract['resources']
    check('baseline_optimizer_total_gradient_budget_scope', caps['baseline_full_gradient_calls'] == 2
          and caps['max_optimizer_full_gradient_calls'] == caps['max_gradient_evaluations'] == 162
          and caps['max_total_run_full_gradient_calls'] == 164)
    for name in ('run.py', 'verify_endpoint.py'):
        tree = ast.parse((HERE / name).read_text())
        check(name + '/syntax', tree is not None)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'optimize_orbitals']
        if name == 'run.py':
            check('one_actual_public_optimizer_call', len(calls) == 1 and len(calls[0].args) == 2)
            keywords = {key.arg: key.value for key in calls[0].keywords}
            check('original_strict_gates', keywords['tolerance'].value == 1e-7
                  and keywords['energy_tolerance'].value == 1e-8)
            check('typed_resume_explicit_target_checkpoint', {'resume_state', 'objective_identity',
                  'allow_objective_change', 'checkpoint_callback', 'max_wall_time', 'return_state'} <= set(keywords))
        else:
            check('verifier_no_optimizer_call', not calls)
            named_calls = {name: [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name) and node.func.id == name]
                           for name in ('checkpoint_bindings', 'rebuilt_objective_payload',
                                        'saved_eigenpair_checks', 'CSFOrbitalFunctional')}
            check('verifier_checkpoint_objective_saved_eigenpair_checks_called',
                  all(named_calls[name] for name in named_calls))
            check('verifier_identity_and_saved_eigenpairs_before_functional',
                  max(min(named_calls[name]) for name in ('checkpoint_bindings', 'rebuilt_objective_payload',
                                                         'saved_eigenpair_checks'))
                  < min(named_calls['CSFOrbitalFunctional']))
    for args in ((str(HERE / 'run.py'), 'solve'), (str(HERE / 'verify_endpoint.py'),)):
        response = subprocess.run([str(ROOT / '.venv/bin/python'), *args], capture_output=True, text=True)
        check('fail_closed_entrance:' + args[-1], response.returncode == 2)
    response = subprocess.run([str(ROOT / '.venv/bin/ruff'), 'check', str(HERE / 'run.py'),
                               str(HERE / 'verify_endpoint.py'), str(HERE / 'verify_endpoint_guards.py'),
                               str(Path(__file__))], capture_output=True, text=True)
    check('Ruff_prepared_scripts', response.returncode == 0)
    response = subprocess.run([str(ROOT / '.venv/bin/python'), str(HERE / 'verify_endpoint_guards.py')],
                              capture_output=True, text=True)
    check('synthetic_endpoint_guard_runner_passed', response.returncode == 0)
    guards = json.loads((HERE / 'endpoint_guard_verification.json').read_text())
    check('synthetic_endpoint_guards_bound', guards['all_passed'] and guards['checks_count'] == 49
          and guards['verifier_sha256'] == sha(HERE / 'verify_endpoint.py')
          and guards['guard_runner_sha256'] == sha(HERE / 'verify_endpoint_guards.py')
          and guards['contract_sha256'] == sha(HERE / 'contract.json'))
    check('synthetic_guards_no_physics_or_endpoint_claim', guards['Atomx_imports'] == guards['physical_H_builds']
          == guards['CI_solves'] == guards['full_gradients'] == guards['optimizer_calls'] == 0
          and guards['actual_endpoint_verified'] is False)
    previous = json.loads((HERE / 'preparation_verification_revision001.json').read_text())
    for name, expected in previous['frozen_owner_files_sha256'].items():
        original = Path(name)
        archived = HERE / (original.stem + '_revision001' + original.suffix)
        check('unchanged_revision001_archive:' + name, sha(archived) == expected)
    readme = (HERE / 'README.md').read_text()
    for link in re.findall(r'\]\(([^)]+)\)', readme):
        check('README_link:' + link, (HERE / link).exists() or link == 'preparation_verification.json')
    check('README_no_execution_claim', 'No new optimizer call, CI solve, gradient' in readme)
    result = {'all_passed': all(c['passed'] for c in checks), 'checks_count': len(checks), 'checks': checks,
              'preparation_revision': 2,
              'previous_preparation_proof_sha256': sha(HERE / 'preparation_verification_revision001.json'),
              'no_new_scientific_solves': True, 'source_epoch_sealed': False,
              'frozen_owner_files_sha256': {name: sha(HERE / name) for name in
                  ('contract.json', 'planning_contract001.json', 'run.py', 'verify_endpoint.py', 'README.md',
                   'preflight.json', 'preflight.log', 'verify_preparation.py', 'verify_endpoint_guards.py',
                   'endpoint_guard_verification.json', 'preparation_verification_revision001.json')},
              'synthetic_endpoint_guard_checks_count': guards['checks_count'],
              'public_API_declaration_observed': preflight['public_interface'],
              'resource_read_only_admission': preflight['resources'],
              'scope': 'Structural/read-only preparation only; no resume invariant/physics/endpoint gate exercised here.'}
    (HERE / 'preparation_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks),
                      'failed': [c for c in checks if not c['passed']],
                      'verification_sha256': sha(HERE / 'preparation_verification.json')}))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
