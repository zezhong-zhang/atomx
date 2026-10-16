"""Bounded source/JSON review; never call the website builder or physical code."""
import ast
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    spec = importlib.util.spec_from_file_location('atlas009_source_review', HERE / 'prepare.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    c = builder.load(HERE / 'contract.json')
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        assert passed, name

    allowed = {'argparse', 'copy', 'hashlib', 'json', 'math', 'os', 'resource', 'shutil', 'pathlib'}
    for name in ('prepare.py', 'verify_source.py'):
        source = (HERE / name).read_text()
        tree = ast.parse(source)
        compile(tree, str(HERE / name), 'exec')
        imports = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imports.update(a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
        check(name + '_AST_and_stdlib_imports', imports <= allowed | {'ast', 'importlib', 're'})
    for name, expected in c['authoring_sources_sha256'].items():
        check(name + '_source_bytes', builder.sha(HERE / name) == expected)
    actual, closed, receipt, points = builder.validate_inputs(c)
    check('closed_native31_source_JSON_scalar_chain', len(points) == 17)
    check('all1582_runtime_all77_scalar_controls', len(actual['checks']) == 1582 and len(closed['checks']) == 77)
    check('all17_original_stationarity_FAIL_preserved', all(not p['original_stationarity_passed'] for p in points))
    before = builder.load(HERE / 'original_overlay.json')
    check('baseline_v9_overlay_total23rows92', before['total_models'] == 23 and before['total_rows'] == 92)
    check('old277records1385states', sum(len(r['trajectory']) for r in before['runs']) == 277 and
          sum(len(p['targets']) for r in before['runs'] for p in r['trajectory']) == 1385)
    old_ids = {tuple(p['saved_point_identity']) for r in before['runs'] for p in r['trajectory']}
    check('old262_saved_identities', len(old_ids) == 262)
    model, run = builder.native31(before, c, actual, closed, receipt, points)
    check('literal_last_qualified16', model['native31_terminal_local_step'] == 16 and run['terminal_iteration'] == 16)
    check('chronological17_points_and85states', [p['iteration'] for p in run['trajectory']] == list(range(17)) and
          sum(len(p['targets']) for p in run['trajectory']) == 85)
    check('new17_manifest_domain_preserves_old_domain', len({tuple(p['saved_point_identity']) for p in run['trajectory']}) == 17 and
          all(p['saved_point_identity'][0] == 'native31_saved_manifest_v1' for p in run['trajectory']))
    check('no_global208_to224_index_claim', all(p['global_accepted_step'] is None for p in run['trajectory']))
    check('four_original_unique_NIST_branches', len(model['rows']) == 4 and
          all(r['matching'] == 'unique_nominal_branch' and not r['goal_pass'] and r['uncertainty_cm1'] is None for r in model['rows']))
    check('no_inherited_overlap_values', all(t['physical_reference_overlap_squared'] is None for p in run['trajectory'] for t in p['targets']) and
          all(r['physical_reference_overlap_squared'] is None for r in model['rows']))
    check('separate3689_metric_identity', model['metric_sha256'] != before['models'][0]['metric_sha256'] and
          model['active_coordinate_count'] == 3689 and model['radial_orbital_count'] == 31)
    check('full1595_target_scope_and2194_remaining', model['target_sector_csf_count'] == 1595 and
          model['declared_full_model_csf_count'] == 3789 and model['other_parent_CSFs_not_computed'] == 2194)
    check('no_certification_current_source_GPU_or_default_upgrade', not any(model[k] for k in
          ('stationarity_passed', 'experimental_gate_passed', 'physical_completeness_certified', 'production_certified',
           'highest_level_reached', 'current_source_adopted', 'GPU_adopted', 'is_current')))
    check('terminal1283_769912_RMS_two_stationarity_gatesFAIL', model['nominal_gap_RMS_cm1'] == 1283.7699121117464 and
          model['orbital_residual'] == .68254770188387 and model['energy_indicator_Ha'] == .6018594158445788)
    check('terminal_shifts_bound_to_scalar_closure', all(
          abs(t['delta_upper_from_initial_Ha'] - change['delta_energy_Ha']) <= 1e-14 and
          abs(t['transition_shift_from_initial_cm1'] - change['transition_shift_cm1']) <= 1e-9
          for t, change in zip(model['targets'], closed['terminal_target_changes'], strict=True)))
    page = (HERE / 'original_page.html').read_text()
    block = (HERE / 'native31_summary.js').read_text()
    updated = builder.page_delta(page, block)
    check('all_six_renderer_seams_unique', len(updated) > len(page) and updated.count('function native31Summary(update){') == 1)
    check('all_old29_fixed208_function_bodies_retained', all(
          updated[updated.index('function ' + name + '('):updated.index('\nfunction ', updated.index('function ' + name + '(') + 1)] ==
          page[page.index('function ' + name + '('):page.index('\nfunction ', page.index('function ' + name + '(') + 1)]
          for name in ('fixed208Summary', 'correlatedSummary', 'correlatedShifts')))
    check('native31_render_all17_85_failed_gates_and_shifts', all(s in block for s in
          ('native31-trajectory-points', 'native31-trajectory-states', 'native31Shifts', 'maximum_iterations',
           'data-stationarity="failed"', 'data-experiment="failed"', '3689', '2194')))
    manifest = builder.load(ROOT / c['published_manifest'])
    check('saved_v9_6019gzip_6128file_manifest', len(manifest['files']) == 6128 and
          sum(p.endswith('.json.gz') for p in manifest['files']) == 6019)
    check('baseline_page_overlay_bound_to_saved_v9_hashes',
          builder.sha(HERE / 'original_page.html') == manifest['files'][c['page_path']]['sha256'] and
          builder.sha(HERE / 'original_overlay.json') == manifest['files'][c['overlay_path']]['sha256'])
    check('baseline_ownerprivate_hosting_hash_retained', builder.sha(Path(c['site']) / '.openai/hosting.json') == c['hosting_sha256'])
    check('prospective_delta_not_executed', not (HERE / 'generated_delta001').exists())
    link_count = 0
    for name in ('README.md', 'EARS.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (HERE / name).read_text()):
            if target.startswith('http'):
                continue
            check(name + '_link_' + target, (HERE / target.split('#')[0]).exists())
            link_count += 1
    result = dict(status='source_and_JSON_preparation_verified_no_build', all_checks_passed=True,
        checks=checks, checks_count=len(checks), local_links_checked=link_count,
        contract_sha256=builder.sha(HERE / 'contract.json'), inputs_sha256=c['inputs_sha256'],
        authored_files_sha256=c['authoring_sources_sha256'],
        source_review_prospective_models=24, prospective_rows=96, old_runs=5, old_records=277,
        new_run_records=17, total_records=294, total_five_state_records=1470,
        old_saved_point_identity_domain=262, new_manifest_identity_domain=17,
        verification_calls=dict(Atomx=0, NumPy=0, NPZ_read_or_rehash=0, science=0, browser=0, API=0),
        old6019gzip_rehashed=False, compiled_Python_AST=True, JavaScript_syntax_or_browser_executed=False,
        isolated_site_delta_built=False, published_site_mutated=False)
    builder.write(HERE / 'source_verification.json', result)
    print(json.dumps(dict(status=result['status'], checks_PASS=len(checks), links=link_count,
                          verification_sha256=builder.sha(HERE / 'source_verification.json'))))


if __name__ == '__main__':
    main()
