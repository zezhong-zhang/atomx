#!/usr/bin/env python3
"""Read-only hashes/AST/document links; never execute scientific source."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import resource
import signal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    contract = read_json(HERE / 'contract.json')
    caps = contract['resource_policy']
    cap = int(caps['design_verifier_address_space_GiB'] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.alarm(caps['design_verifier_wall_timeout_s'])
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    identity = read_json(HERE / 'source_identity.json')
    check('source_identity_SHA', sha(HERE / 'source_identity.json') == contract['source_identity_sha256'])
    sources = identity['all_atomx_python_sha256']
    expected_map = contract['fixed_model']['source_map_sha256']
    check('complete_source_map_SHA', digest(sources) == expected_map == identity['source_map_sha256'])
    check('complete_python_source_count', len(sources) == 353 == identity['source_files'])
    actual = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/atomx').rglob('*.py'))}
    check('current_source_map_exactly_unchanged', actual == sources)
    archived = ROOT / identity['archived_root']
    archive_map = {'src/' + str(p.relative_to(archived)): sha(p)
                   for p in sorted((archived / 'atomx').rglob('*.py'))}
    check('runtime_archive_map_exactly_unchanged', archive_map == sources)
    for path, expected in sources.items():
        check('current_source_SHA/' + path, sha(ROOT / path) == expected)
        check('archived_source_SHA/' + path, sha(archived / Path(path).relative_to('src')) == expected)
    for path, expected in contract['inputs_sha256'].items():
        check('input_SHA/' + path, sha(ROOT / path) == expected)

    authorization = contract['authorization']
    check('design_only_no_execution_authorization', not any(authorization.values()))
    fixed = contract['fixed_model']
    check('fixed_whole531_model', (fixed['CSFs'], fixed['electrons'], fixed['radial_spinors'], fixed['ns'])
          == (531, 69, 24, 123))
    check('unchanged_original_stationarity_gates', fixed['gates']['original_orbital_tolerance'] == 1e-7
          and fixed['gates']['original_estimated_energy_tolerance_Ha'] == 1e-8)
    check('unchanged_original_CI_and_Armijo_gates', fixed['gates']['CI_residual_Ha'] == 1e-8
          and fixed['gates']['CI_orthogonality'] == 1e-10 and fixed['gates']['Armijo'] == 1e-4)
    check('exact_five_equal_weights', fixed['equal_target_weights'] == [.2] * 5)

    def method_source(path, class_name, method_name):
        text = (ROOT / path).read_text()
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
        return ast.get_source_segment(text, fn), fn

    functional = 'src/atomx/methods/csf/variational.py'
    fixed_text, _ = method_source(functional, 'CSFOrbitalFunctional', '_fixed_density_energy_difference')
    centered_text, _ = method_source(functional, 'CSFOrbitalFunctional', 'constrained_energy_difference')
    for name, body in [('fixed_density', fixed_text), ('constrained', centered_text)]:
        check(name + '_guard_retains_mixed_tracked_rejection', 'self.level_tracking is not None' in body
              and 'len(columns) != 1' in body and 'return None' in body)
    density = 'src/atomx/physics/csf/radial_density.py'
    ordinary, _ = method_source(density, 'CIRadialDensity', 'contract')
    bilinear, _ = method_source(density, 'CIRadialDensity', 'contract_bilinear')
    check('ordinary_density_filter_reported', 'abs(coefficient) > 1e-15' in ordinary)
    check('bilinear_exact_zero_only_and_real_guard', 'if value != 0' in bilinear
          and 'np.iscomplexobj(left)' in bilinear and 'np.iscomplexobj(right)' in bilinear)
    check('bilinear_preserves_signed_symmetry', '.5 * (a[i] * b[j] + b[i] * a[j])' in bilinear)
    energy_for_weights, _ = method_source(functional, 'CSFOrbitalFunctional', 'energy_for_weights')
    check('raw_onebody_polynomial_caveat_source_matched', 'value = a @ self.one_body[kappa] @ b'
          in energy_for_weights)
    tree = ast.parse((ROOT / functional).read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    check('proposed_mixed_increment_not_implemented', 'mixed_ci_energy_increment' not in names
          and '_signed_weights_radial_increment' not in names)
    ci_tree = ast.parse((ROOT / 'src/atomx/methods/ci.py').read_text())
    ci_result = next(n for n in ci_tree.body if isinstance(n, ast.ClassDef) and n.name == 'CIResult')
    fields = {n.target.id for n in ci_result.body if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name)}
    check('CIResult_has_no_assumed_matrix_attribute', 'matrix' not in fields and 'H' not in fields)
    driver = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run002/run.py'
    calls = [n for n in ast.walk(ast.parse(driver.read_text())) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'optimize_orbitals']
    check('active_run_one_public_optimizer_call', len(calls) == 1)
    kwargs = {k.arg: k.value for k in calls[0].keywords}
    check('active_run_stable_callback_explicitly_off', isinstance(kwargs['energy_difference'], ast.Constant)
          and kwargs['energy_difference'].value is None)
    check('active_run_explicit_increment_mode_off', isinstance(kwargs['linear_energy_increments'], ast.Constant)
          and kwargs['linear_energy_increments'].value is False)

    algebra = read_json(ROOT / 'docs/yb_mcdhf_rci_2026_10_11/mixed_ci_stationarity_probe/mixed_difference_algebra.json')
    check('historical239_not_new_physics_claim', algebra['all_passed'] and algebra['checks_count'] == 239
          and algebra['matched_target_pairs'] == 45 and not algebra['arithmetic']['radial_polynomial_implemented'])
    check('historical_precision_not_assumed_IEEE_quad', algebra['arithmetic']['mantissa_bits'] == 63)

    annex_contract = read_json(HERE / 'endpoint_annex_contract.json')
    check('separate_endpoint_annex_original_contract_preserved', annex_contract['parent_contract_sha256']
          == sha(HERE / 'contract.json'))
    for path, expected in annex_contract['inputs_sha256'].items():
        check('endpoint_annex_SHA/' + path, sha(ROOT / path) == expected)
    annex = read_json(HERE / 'endpoint_annex.json')
    check('endpoint_annex_contract_SHA', annex['endpoint_annex_contract_sha256']
          == sha(HERE / 'endpoint_annex_contract.json'))
    run_dir = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run002'
    result = read_json(run_dir / 'result_unverified.json')
    endpoint = read_json(run_dir / 'independent_endpoint_verification.json')
    history = result['diagnostics']['energy_history']
    check('all161_actual_saved_EOL_values_retained', len(history) == 161
          and history == annex['all161_stored_EOL_values_retained'])
    ratios = []
    for i, row in enumerate(annex['accepted_delta_rows']):
        a, b = history[i:i+2]
        quantum = max(math.ulp(a), math.ulp(b))
        ratio = (a-b) / quantum
        ratios.append(ratio)
        check(f'accepted_step{i+1}_actual_delta_and_ulps', b-a == row['reported_delta_Ha']
              and quantum == row['max_endpoint_ulp_Ha'] and ratio == row['decrease_endpoint_ulps'])
        check(f'accepted_step{i+1}_strictly_decreasing', b < a)
    check('actual_minimum81_and_final1282_ulps', min(ratios) == 81.0 and ratios[-1] == 1282.0)
    check('iteration_cap_without_line_search_fallback', result['diagnostics']['termination_reason']
          == 'maximum_iterations' and result['diagnostics']['line_search_fallbacks'] == 0)
    check('prior988_identity_does_not_certify_stationarity', endpoint['all_identity_replay_checks_passed']
          and endpoint['checks_count'] == 988 and not endpoint['original_stationarity_gates_passed'])
    check('endpoint_actual_original_residual_fails_indicator_passes',
          endpoint['actual_original_metric_orbital_residual'] > 1e-7
          and endpoint['actual_original_metric_energy_indicator_Ha'] < 1e-8)
    check('no_design_physics_calls_claimed', all(annex[k] == 0 for k in
          ['new_H_builds', 'new_CI_solves', 'new_gradients', 'new_optimizer_calls']))
    check('actual_gradient_accounting', result['baseline_gradient_calls'] == 2
          and result['optimizer_full_gradient_calls'] == 161 and result['total_run_full_gradient_calls'] == 163)

    markdown = (HERE / 'README.md').read_text()
    for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', markdown):
        path = (HERE / link.split('#', 1)[0]).resolve()
        check('README_local_link/' + link, path.is_file() or path == HERE / 'verification.json')
    check('README_explicit_conditional_scope', 'not an implementation or a new calculation' in markdown
          and 'no demonstrated' in annex['conclusion'] and '81' in markdown and '1,282' in markdown)
    self_tree = ast.parse(Path(__file__).read_text())
    imports = {n.module.split('.')[0] for n in ast.walk(self_tree) if isinstance(n, ast.ImportFrom)
               and n.module} | {a.name.split('.')[0] for n in ast.walk(self_tree)
                               if isinstance(n, ast.Import) for a in n.names}
    check('verifier_stdlib_only_no_scientific_imports', not imports & {'atomx', 'numpy', 'scipy', 'mpmath'})
    check('verifier_syntax_valid', isinstance(self_tree, ast.Module))
    signal.alarm(0)
    output = {'status': 'read_only_source_document_and_saved_JSON_verification_only',
              'all_passed': all(c['passed'] for c in checks), 'checks_count': len(checks), 'checks': checks,
              'owner_files_sha256': {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file()
                                     and p.name != 'verification.json'},
              'new_H_builds': 0, 'new_CI_solves': 0, 'new_gradients': 0, 'new_optimizer_calls': 0,
              'scientific_suite_newly_executed': False, 'implementation_qualified': False,
              'stationarity_certified': False, 'roundoff_obstruction_demonstrated': False}
    (HERE / 'verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['all_passed', 'checks_count', 'implementation_qualified',
                                           'roundoff_obstruction_demonstrated']}))
    if not output['all_passed']:
        raise RuntimeError('Preserved failed read-only design verification')


if __name__ == '__main__':
    main()
