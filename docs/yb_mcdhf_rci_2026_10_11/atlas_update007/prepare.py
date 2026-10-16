"""Append two closed fork endpoints and saved JSON trajectories; no physical import."""
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LABELS = ('6s_1/2', '5d_3/2', '5d_5/2', '6p_1/2', '6p_3/2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def point_identity(manifest):
    return tuple(manifest[k] for k in ('source_map_sha256', 'metric_sha256', 'geometry_sha256',
                  'objective_sha256', 'state_sha256', 'owner_input_sha256', 'owner_H_sha256'))


def append_branch(branch, original, contract, site):
    root = REPO / contract['run_root']
    run = root / branch
    completion, actual, receipt, scientific = [load(run / p) for p in
        ('completion.json', 'actual/result.json', 'root_execution.json', 'contract.json')]
    assert receipt['status'] == 'foreground_closed' and receipt['exit_code'] == 0
    assert completion['all_completion_metadata_checks_passed'] and completion['checks'] == 15
    assert completion['actual_live_controls'] == len(actual['controls']) == 2219
    assert not completion['failed_live_controls'] and not completion['forensic_unqualified_candidates']
    assert actual['comparison_branch'] == branch and actual['search_memory_policy'] == scientific['search_memory']
    assert scientific['search_memory']['initial_history_size'] == 0
    assert actual['source_map_sha256'] == scientific['source']['candidate356_map_sha256'] == contract['source_map_sha256']
    assert scientific['fixed']['target_CSFs'] == 1046 and len(scientific['fixed']['keys29']) == 29
    assert scientific['fixed']['five_weights'] == [.2] * 5 and scientific['fixed']['all29_allowed_to_relax']
    assert not actual['energy_callback_used'] and not actual['ordinary_reported_objective_changed']
    assert not actual['stationarity_passed'] and not actual['experimental_certification']
    assert completion['actual_new_accepted_steps'] == 32 and completion['total_global_accepted_steps'] == 240
    assert completion['all_qualified_finite_stored_H_acceptances_qualified']
    assert [r['iteration'] for r in actual['trajectory']] == list(range(33))
    model = copy.deepcopy(original['models'][0])
    assert model['id'] == 'correlated29_run005_endpoint208_DC'
    references = {r['label']: r for r in model['rows']}
    uids = {t['label']: t['UID'] for t in model['targets']}
    trajectory, identities = [], []
    for record, closed in zip(actual['trajectory'], completion['trajectory'], strict=True):
        path = REPO / record['manifest_path']
        manifest = load(path)
        assert sha(path) == record['manifest_sha256'] and manifest['array_sha256'] == record['array_sha256']
        assert manifest['checkpoint_qualified'] and manifest['iteration'] == closed['iteration'] == record['iteration']
        assert manifest['global_accepted_step'] == 208 + record['iteration']
        for key in ('source_map', 'metric', 'geometry', 'objective'):
            assert manifest[key + '_sha256'] == contract[key + '_sha256']
        identities.append(point_identity(manifest))
        targets = {t['label']: copy.deepcopy(t) for t in manifest['targets']}
        assert set(targets) == set(LABELS) and all(targets[label]['UID'] == uids[label] for label in LABELS)
        lower = targets['6s_1/2']['energy_Ha']
        first = {t['label']: t for t in trajectory[0]['targets']} if trajectory else copy.deepcopy(targets)
        for label in LABELS:
            t = targets[label]
            assert t['tracking']['reference_levels'] == [t['UID']]
            assert t['tracking']['selected_levels'] == [t['selected_level']]
            assert t['selected_level'][:2] == t['UID'][:2] and t['tracking']['per_level_weight'] == .2
            assert abs((t['energy_Ha'] - lower) * original['Hartree_cm1'] - t['excitation_cm1']) <= 1e-8
            assert t['NIST_uncertainty_cm1'] is None
            assert t['nominal_gap_cm1'] == t['excitation_cm1'] - t['NIST_nominal_cm1']
            if label in references:
                assert t['NIST_nominal_cm1'] == references[label]['reference_cm1']
            t.update(delta_upper_from_initial_Ha=t['energy_Ha'] - first[label]['energy_Ha'],
                     delta_lower_from_initial_Ha=lower - first['6s_1/2']['energy_Ha'],
                     transition_shift_from_initial_cm1=t['excitation_cm1'] - first[label]['excitation_cm1'],
                     state_average_weight=.2)
        moments = copy.deepcopy(model['multiplets'])
        for moment in moments:
            js = (3, 5) if moment['label'] == '5d' else (1, 3)
            lo, hi = [targets[f"{moment['label']}_{j}/2"] for j in js]
            center = ((js[0] + 1) * lo['excitation_cm1'] + (js[1] + 1) * hi['excitation_cm1']) / sum(j + 1 for j in js)
            split = hi['excitation_cm1'] - lo['excitation_cm1']
            saved = next(m for m in manifest['multiplets'] if m['label'] == moment['label'])
            assert abs(center - saved['centroid_cm1']) <= 1e-9 and abs(split - saved['splitting_cm1']) <= 1e-9
            moment.update(centroid_cm1=center, splitting_cm1=split,
                          centroid_gap_cm1=center - moment['reference_centroid_cm1'],
                          splitting_gap_cm1=split - moment['reference_splitting_cm1'])
        rms = math.sqrt(sum(targets[label]['nominal_gap_cm1']**2 for label in LABELS[1:]) / 4)
        d = manifest['diagnostics']
        assert abs(rms - closed['nominal_four_gap_RMS_cm1']) <= 1e-9 and d['energy_ha'] == closed['reported_EO_Ha']
        assert not manifest['stationarity_passed'] and not manifest['experimental_certification']
        trajectory.append({'iteration': record['iteration'], 'global_accepted_step': manifest['global_accepted_step'],
                           'fork_branch': branch, 'reported_EOL_Ha': d['energy_ha'],
                           'orbital_residual': d['orbital_residual'], 'energy_indicator_Ha': d['estimated_energy_error'],
                           'nominal_four_gap_RMS_cm1': rms, 'targets': [targets[label] for label in LABELS],
                           'multiplets': moments, 'source_manifest': record['manifest_path'],
                           'source_manifest_sha256': record['manifest_sha256'], 'source_arrays_sha256': record['array_sha256'],
                           'saved_state_sha256': manifest['state_sha256'], 'saved_point_identity': list(identities[-1]),
                           'stationarity_passed': False, 'experimental_certification': False,
                           'exact_stored_H_acceptance_qualified': closed['exact_acceptance_qualified']})
    first, last = trajectory[0], trajectory[-1]
    assert first['reported_EOL_Ha'] == model['EOL_Ha'] and first['orbital_residual'] == model['orbital_residual']
    assert {t['label']: t['energy_Ha'] for t in first['targets']} == {t['label']: t['energy_Ha'] for t in model['targets']}
    by_label = {t['label']: t for t in last['targets']}
    for row in model['rows']:
        t = by_label[row['label']]
        row.update(calculated_cm1=t['excitation_cm1'], error_cm1=t['nominal_gap_cm1'],
                   error_percent=100 * t['nominal_gap_cm1'] / row['reference_cm1'],
                   goal_pass=abs(t['nominal_gap_cm1'] / row['reference_cm1']) < .01,
                   source_energy_Ha=t['energy_Ha'], source_column=t['column'], exact_J_local_root=t['selected_level'][2],
                   physical_reference_overlap_squared=t['tracking']['principal_overlap_squared'][0],
                   **{k: t[k] for k in ('delta_upper_from_initial_Ha', 'delta_lower_from_initial_Ha', 'transition_shift_from_initial_cm1')})
    assert all(not r['goal_pass'] for r in model['rows'])
    asset_prefix = contract['evidence_prefix'].removeprefix('dist/') + '/' + branch + '/'
    history = scientific['search_memory']['history_size']
    counting_note = ('共同主线保留 209 点；两分支各 33 条记录，共 66 条。两路 local 0–11 保存身份相同，'
                     'local 0 是共同 208 点；首次分叉 local 12。全部轮次 277 条记录对应 262 个保存点身份，'
                     '不是 273 个不同物理点。两个 global 240 属于不同分支，不能合并为唯一末态。')
    model.update(id=contract['model_ids'][branch],
                 title=f'原生29轨道MCDHF · {branch}：共同208点续32步 · global240（本分支）',
                 saved_run=f'{root.name}/{branch}/accepted_0032', fork_branch=branch, history_size=history,
                 accepted_steps=32, global_initial_step=208, global_accepted_steps=240,
                 ground=by_label['6s_1/2'], targets=last['targets'], multiplets=last['multiplets'],
                 EOL_Ha=last['reported_EOL_Ha'], initial_EOL_Ha=first['reported_EOL_Ha'],
                 EOL_delta_Ha=last['reported_EOL_Ha'] - first['reported_EOL_Ha'],
                 nominal_gap_RMS_cm1=last['nominal_four_gap_RMS_cm1'], initial_nominal_gap_RMS_cm1=first['nominal_four_gap_RMS_cm1'],
                 initial_orbital_residual=first['orbital_residual'], orbital_residual=last['orbital_residual'],
                 energy_indicator_Ha=last['energy_indicator_Ha'], point_manifest_sha256=last['source_manifest_sha256'],
                 point_arrays_sha256=last['source_arrays_sha256'], elapsed_uncontrolled_s=receipt['elapsed_uncontrolled_seconds'],
                 peak_RSS_MiB=actual['peak_RSS_MiB'], runtime_controls=2219, completion_metadata_checks=15,
                 completed_calls=actual['completed_calls'], source=str((run / 'actual/result.json').relative_to(REPO)),
                 sha256=sha(run / 'actual/result.json'), source_asset=asset_prefix + 'actual-result.json',
                 completion_asset=asset_prefix + 'completion.json', trajectory_asset=asset_prefix + 'trajectory.json',
                 stationarity_note=f'仅改变搜索记忆容量为 {history}；起始历史同为空。32 个更新后原驻值两门仍 FAIL。' + counting_note,
                 scope=model['scope'] + ' ' + counting_note,
                 boundary_note='原 f043/DC/29轨道/1046目标CSF/3451度量与五UID等权0.2保持；不按NIST或最小gap选点。'
                 '40容量在32步内未填满，实际保留pair数未导出。40 EOL更低但残差更大；两路无单一优胜或受控速度结论。'
                 '每个上、下态同点相减，未拼旧Breit/QED/反冲。' + counting_note)
    evidence = site / contract['evidence_prefix'] / branch
    evidence.mkdir(parents=True, exist_ok=False)
    assets = []
    for name in ('completion.json', 'actual/result.json', 'root_execution.json', 'contract.json', 'execution_contract.json'):
        target = evidence / name.replace('/', '-')
        shutil.copyfile(run / name, target)
        assets.append({'path': str((run / name).relative_to(REPO)), 'asset': str(target.relative_to(site / 'dist')), 'sha256': sha(target)})
    own_trajectory = HERE / f'{branch}-trajectory.json'
    write(own_trajectory, {'model_id': model['id'], 'fork_branch': branch, 'common_baseline_global': 208,
                          'terminal_global_branch_step': 240, 'trajectory': trajectory,
                          'stationarity_passed': False, 'experimental_certification': False})
    shutil.copyfile(own_trajectory, evidence / 'trajectory.json')
    assets.append({'path': str(own_trajectory.relative_to(REPO)), 'asset': asset_prefix + 'trajectory.json', 'sha256': sha(own_trajectory)})
    saved_run = {'model_id': model['id'], 'fork_branch': branch, 'terminal_iteration': 32,
                 'global_initial_step': 208, 'global_terminal_step': 240, 'trajectory': trajectory,
                 'receipt': {k: receipt[k] for k in ('started_utc', 'finished_utc', 'exit_code', 'elapsed_uncontrolled_seconds', 'timing_scope')}}
    return model, saved_run, assets, identities


def main():
    contract = load(HERE / 'contract.json')
    site = Path(contract['site'])
    assert contract['status'] == 'prospective_before_local_site_edits' and contract['local_only']
    assert sha(HERE / 'prechange_manifest.json') == contract['prechange_manifest_sha256']
    for name, item in load(HERE / 'prechange_manifest.json')['files'].items():
        assert sha(site / name) == item['sha256'], name
    for name, expected in contract['inputs_sha256'].items():
        assert sha(REPO / name) == expected, name
    assert sha(HERE / 'original_overlay.json') == contract['old_overlay_sha256']
    original = load(HERE / 'original_overlay.json')
    pair = load(REPO / contract['run_root'] / 'completion.json')
    assert pair['same_baseline_identity_check_passed'] and pair['comparison']['common_accepted_horizon'] == 32
    assert pair['comparison']['first_observed_numerical_divergence']['accepted_index'] == 12
    results = [append_branch(branch, original, contract, site) for branch in contract['branches']]
    shared_indices = [i for i in range(33) if results[0][3][i] == results[1][3][i]]
    assert shared_indices == list(range(12))
    old_identities = set()
    for old_run in original['runs']:
        for point in old_run['trajectory']:
            path = REPO / point['source_manifest']
            assert sha(path) == point['source_manifest_sha256']
            old_identities.add(point_identity(load(path)))
    assert len(old_identities) == 209 and results[0][3][0] in old_identities
    fork_identities = set(results[0][3] + results[1][3])
    assert len(fork_identities) == 54 and len(fork_identities & old_identities) == 1
    assert len(fork_identities | old_identities) == 262
    evidence = site / contract['evidence_prefix']
    common_assets = []
    for name in ('completion.json', 'root_pair_execution.json', 'RESULTS.md', 'completion_execution.json'):
        source = REPO / contract['run_root'] / name
        shutil.copyfile(source, evidence / name)
        common_assets.append({'path': str(source.relative_to(REPO)), 'asset': str((evidence / name).relative_to(site / 'dist')), 'sha256': sha(source)})
    overlay = copy.deepcopy(original)
    overlay['models'] = [v[0] for v in results] + overlay['models']
    overlay['runs'] = [v[1] for v in results] + overlay['runs']
    new_assets = [asset for v in results for asset in v[2]] + common_assets
    overlay['source_files'] = new_assets + overlay['source_files']
    overlay.update(total_models=21, total_rows=84, previous_v7_models_preserved=19, previous_v7_rows_preserved=76,
                   common_lineage_distinct_points=209, fork_record_count=66, per_run_point_records=277,
                   distinct_saved_point_identities=262, fork_shared_local_indices=list(range(12)),
                   fork_distinct_saved_point_identities=54, fork_identities_overlapping_common_lineage=1,
                   distinct_global_points_scope='Retained common lineage only; branch global indices are not unique point identities.',
                   global_accepted_updates_scope='Common lineage208; each separate fork adds32, not a single merged64-update trajectory.',
                   fork_global240_is_not_unique_terminal=True,
                   endpoint_selection='Original predeclared endpoints retained; both independent fork32/global240 terminals from literal208, no best-gap or single-winner selection.')
    assert overlay['models'][2:] == original['models'] and overlay['runs'][2:] == original['runs']
    assert overlay['source_files'][16:] == original['source_files']
    write(site / contract['overlay_path'], overlay)
    write(HERE / 'prepared.json', {'all_saved_JSON_preparation_checks_passed': True,
        'contract_sha256': sha(HERE / 'contract.json'), 'builder_sha256': sha(__file__),
        'models': 21, 'rows': 84, 'old_models_preserved': 19, 'old_rows_preserved': 76,
        'fork_records': 66, 'all_run_records': 277, 'common_lineage_points': 209,
        'distinct_saved_point_identities': 262, 'fork_shared_local_indices': shared_indices,
        'overlay_sha256': sha(site / contract['overlay_path']),
        'new_assets_sha256': {str(p.relative_to(site)): sha(p) for p in sorted(evidence.rglob('*')) if p.is_file()},
        'science_calls': 0, 'physical_array_replay': False, 'publication_performed': False})
    print(json.dumps({'models': 21, 'rows': 84, 'all_run_records': 277, 'saved_distinct_identities': 262,
                      'shared_fork_local_indices': shared_indices, 'science_calls': 0, 'publication_performed': False}))


if __name__ == '__main__':
    main()
