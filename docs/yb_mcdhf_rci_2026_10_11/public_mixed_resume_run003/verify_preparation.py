"""Read-only continuation preparation controls; no physical solves permitted."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import run as driver

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = 'docs/yb_mcdhf_rci_2026_10_11/'


def main():
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    contract = driver.load(HERE / 'contract.json')
    preflight = driver.load(HERE / 'preflight.json')
    source = driver.load(ROOT / contract['fixed_conditions']['frozen_source_identity_file'])
    point = driver.start_checkpoint_status(contract)
    check('complete160_start_provenance_and_independent_replay', point['ready'])
    check('known_nonstationary160_start_is_retained', point['original_stationarity_passed'] is False
          and point['starting_original_residual'] == 4.717047646600387e-5
          and point['starting_original_indicator_Ha'] == 6.8897239137767995e-9)
    check('preflight_contract_bound', preflight['contract_sha256'] == driver.sha(HERE / 'contract.json'))
    check('preflight_start_binding', preflight['starting_full531_checkpoint'] == point)
    check('complete_count_caps', preflight['CSFs'] == 531 and preflight['radial_orbitals'] == 24
          and preflight['max_raw_M_equals_J'] == 232
          and preflight['selected_overlap_pair_work_upper'] == 227582)
    check('sixteen_full_symmetry_blocks', len(preflight['blocks']) == 16
          and sum(b['CSFs'] for b in preflight['blocks']) == 531)
    check('input_source_identity_checked', preflight['input_hashes_verified'] == len(contract['inputs_sha256'])
          and preflight['starting353_source_files_verified'] == 353)
    for path, expected in contract['inputs_sha256'].items():
        check('actual_immutable_input:' + path, driver.sha(ROOT / path) == expected)
    runtime_map = {}
    for name, expected in source['all_atomx_python_sha256'].items():
        current = driver.sha(ROOT / name)
        archived = driver.sha(ROOT / contract['fixed_conditions']['frozen_source_root'] / Path(name).relative_to('src'))
        runtime_map[name] = current
        check('unchanged_current_and_parent353_source:' + name, current == expected == archived)
    check('complete_scientific_source_epoch_unchanged', driver.digest(runtime_map)
          == source['source_map_sha256'] == contract['scientific_source_map_required_sha256']
          == preflight['planning_runtime_source_map_sha256'])
    check('original_metric_exact', preflight['original_metric_sha256']
          == contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256'])
    check('authoritative_coordinate_shape', preflight['coordinates_shape'] == [2856])
    check('no_compute_in_preflight', preflight['functional_evaluations']
          == preflight['CI_solves'] == preflight['optimizer_calls'] == 0)
    check('no_execution_epoch', not (HERE / 'source').exists() and not (HERE / 'source_identity.json').exists()
          and not (HERE / 'execution_contract.json').exists())
    check('no_endpoint_or_trial_claim', not any((HERE / name).exists() for name in
          ('result_unverified.json', 'checkpoints', 'result_partial.json', 'latest_checkpoint.json',
           'new_source_baseline_controls.json', 'independent_endpoint_verification.json')))
    caps, gates = contract['resources'], contract['acceptance']
    check('baseline_optimizer_total_gradient_budget_scope', caps['baseline_full_gradient_calls'] == 2
          and caps['max_optimizer_full_gradient_calls'] == caps['max_gradient_evaluations'] == 162
          and caps['max_total_run_full_gradient_calls'] == 164)
    check('same_point_gradient_equality_gates', gates['full531_same_start_projected_gradient_replay_max'] == 1e-9
          and gates['full531_same_start_preconditioned_gradient_replay_max'] == 1e-10)
    check('unchanged_strict_stationarity_gates', gates['original_orbital_tolerance'] == 1e-7
          and gates['original_estimated_energy_tolerance_Ha'] == 1e-8)
    check('no_bare005_at_changed_start_gate', not any(name.startswith('bare_') for name in gates))
    check('single_CPU_bounded_resources', caps['threads'] == 1 and caps['cpu_affinity_single_core'] is True
          and caps['address_space_cap_GiB'] == 3.95 and caps['max_iterations'] == 160
          and caps['accepted_boundary_soft_stop_s'] == 830 and caps['wall_cap_s'] == 890)
    check('separate_endpoint_one_gradient_only', contract['verification_resources']['max_new_full_gradient_evaluations'] == 1
          and contract['verification_resources']['optimizer_calls'] == 0)
    check('current_public_resume_API_available', preflight['public_interface']['available'] is True
          and not preflight['public_interface']['missing_keywords'])
    basis = driver.load(ROOT / (PREFIX + 'balanced_cv_replay/basis_identity.json'))
    targets = driver.load(ROOT / (PREFIX + 'balanced_cv_replay/stage_4f5p5s/target_binding.json'))['targets']
    parent_objective = driver.load(ROOT / contract['start_checkpoint']['parent_objective_identity'])
    payload = {'Z': 70, 'electrons': 69, 'operator': 'full declared no-pair Dirac-Coulomb',
               'nucleus': 'native005 Fermi A175',
               'radial_archive_sha256': driver.sha(ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment/orbitals_workspace.npz')),
               'target_identity': [{'label': t['label'], 'twoJ': t['twoJ'], 'parity': t['parity'],
                                    'root': t['models']['DC']['local_root'], 'weight': .2} for t in targets],
               'CSFs': basis['CSFs'], 'source_map_sha256': driver.digest(runtime_map),
               'physical_reference_CI_sha256': driver.sha(ROOT / (PREFIX + 'balanced_cv_replay/stage_4f5p5s/DC_arrays.npz')),
               'physical_reference_binding_sha256': driver.sha(ROOT / (PREFIX + 'balanced_cv_replay/stage_4f5p5s/target_binding.json'))}
    check('independent_plain_manifest_objective_payload', payload == parent_objective['new'])
    check('same_full531_objective_SHA', driver.digest(payload) == parent_objective['new_sha256']
          == contract['fixed_conditions']['same_objective_sha256'])
    manifest = driver.load(ROOT / contract['start_checkpoint']['manifest'])
    proof = driver.load(ROOT / contract['start_checkpoint']['independent_verification'])
    check('samepoint_parent_saved_and_independent_target_UIDs',
          driver.canonical_physical_target_groups(manifest['physical_tracking'])
          == driver.canonical_physical_target_groups(proof['physical_tracking']))

    for name in ('run.py', 'verify_endpoint.py', 'verify_endpoint_guards.py', 'verify_start_checkpoint_guards.py'):
        tree = ast.parse((HERE / name).read_text())
        check(name + '/valid_AST', tree is not None)
        optimizer_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == 'optimize_orbitals']
        if name == 'run.py':
            check('one_actual_public_optimizer_call', len(optimizer_calls) == 1 and len(optimizer_calls[0].args) == 2)
            keywords = {key.arg: key.value for key in optimizer_calls[0].keywords}
            check('explicit_same_objective_and_original_strict_gates', keywords['tolerance'].value == 1e-7
                  and keywords['energy_tolerance'].value == 1e-8
                  and keywords['allow_objective_change'].value is False
                  and keywords['energy_difference'].value is None
                  and keywords['linear_energy_increments'].value is False)
            check('typed_resume_checkpoint_and_budget_explicit', {'resume_state', 'objective_identity',
                  'checkpoint_callback', 'max_wall_time', 'return_state'} <= set(keywords)
                  and keywords['return_state'].value is True and keywords['history_size'].value == 10)
            functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            preflight_calls = {node.func.id for node in ast.walk(functions['prospective_preflight'])
                               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            check('readonly_preflight_has_no_science_or_seal_call', not preflight_calls & {
                'CSFOrbitalFunctional', 'build_ci_matrix', 'ci_solve', 'optimize_orbitals', 'seal_source', 'execute'})
        else:
            check(name + '/no_optimizer_call', not optimizer_calls)
            if name == 'verify_endpoint.py':
                named_calls = {key: [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                                     and isinstance(node.func, ast.Name) and node.func.id == key]
                               for key in ('checkpoint_bindings', 'rebuilt_objective_payload',
                                           'saved_eigenpair_checks', 'CSFOrbitalFunctional')}
                check('endpoint_identity_objective_saved_eigenpair_guards_present',
                      all(named_calls[key] for key in named_calls))
                check('endpoint_identity_guards_precede_functional',
                      max(min(named_calls[key]) for key in ('checkpoint_bindings', 'rebuilt_objective_payload',
                                                           'saved_eigenpair_checks'))
                      < min(named_calls['CSFOrbitalFunctional']))
    for args in ((str(HERE / 'run.py'), 'solve'), (str(HERE / 'verify_endpoint.py'),)):
        response = subprocess.run([str(ROOT / '.venv/bin/python'), *args], capture_output=True, text=True)
        check('fail_closed_missing_root_execution_contract:' + args[-1], response.returncode == 2)
    for name, output in (('verify_endpoint_guards.py', 'endpoint_guard_verification.json'),
                         ('verify_start_checkpoint_guards.py', 'starting_checkpoint_guard_verification.json')):
        response = subprocess.run([str(ROOT / '.venv/bin/python'), str(HERE / name)], capture_output=True, text=True)
        check('constructed_guard_runner_success:' + name, response.returncode == 0)
        guards = driver.load(HERE / output)
        check('constructed_guard_proof_bound:' + name, guards['all_passed']
              and guards['contract_sha256'] == driver.sha(HERE / 'contract.json')
              and guards.get('guard_runner_sha256', guards.get('script_sha256')) == driver.sha(HERE / name))
        for item in guards['checks']:
            check('constructed_guard:' + item['name'], item['passed'])
        check('constructed_guards_no_Atomx_or_science:' + name, guards['Atomx_imports'] == guards['physical_H_builds']
              == guards['CI_solves'] == guards.get('full_gradients', guards.get('gradients')) == guards['optimizer_calls'] == 0)
        if name == 'verify_endpoint_guards.py':
            check('constructed_endpoint_guard_verifier_bound', guards['verifier_sha256'] == driver.sha(HERE / 'verify_endpoint.py')
                  and guards['actual_endpoint_verified'] is False and guards['checks_count'] == 49)
        else:
            check('constructed_start_guard_driver_bound', guards['driver_sha256'] == driver.sha(HERE / 'run.py')
                  and guards['numerical_arrays_loaded'] == 0)
    response = subprocess.run([str(ROOT / '.venv/bin/ruff'), 'check', *(str(p) for p in sorted(HERE.glob('*.py')))],
                              capture_output=True, text=True)
    check('Ruff_all_prepared_scripts', response.returncode == 0)
    readme = (HERE / 'README.md').read_text()
    for link in re.findall(r'\]\(([^)]+)\)', readme):
        check('README_link:' + link, (HERE / link).exists() or link == 'preparation_verification.json')
    check('README_explicit_no_scientific_execution', 'No new optimizer call, CI solve, gradient' in readme)
    check('README_fresh_secant_and_same_point_scope', 'fresh history10' in readme
          and 'There is no bare005 functional' in readme and 'maximum total\nrun full gradients is164' in readme)
    files = ['contract.json', 'run.py', 'verify_endpoint.py', 'verify_endpoint_guards.py',
             'verify_start_checkpoint_guards.py', 'verify_preparation.py', 'README.md', 'preflight.json',
             'preflight.log', 'endpoint_guard_verification.json', 'starting_checkpoint_guard_verification.json']
    result = {'all_passed': all(c['passed'] for c in checks), 'checks_count': len(checks), 'checks': checks,
              'successor_run_number': 3, 'preparation_revision': 1,
              'no_new_scientific_solves': True, 'source_epoch_sealed': False,
              'frozen_owner_files_sha256': {name: driver.sha(HERE / name) for name in files},
              'starting_complete_checkpoint': point,
              'public_API_declaration_observed': preflight['public_interface'],
              'resource_read_only_admission': preflight['resources'],
              'Atomx_functional_evaluations': 0, 'physical_H_builds': 0, 'CI_solves': 0,
              'gradients': 0, 'optimizer_calls': 0,
              'scope': 'Source/hash/JSON/AST/link and constructed array guards only; original160 unchanged. '
                       'Preflight enumeration and exact archive P*x checks performed, no new physical endpoint certified.'}
    (HERE / 'preparation_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks),
                      'failed': [c for c in checks if not c['passed']],
                      'verification_sha256': driver.sha(HERE / 'preparation_verification.json')}))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
