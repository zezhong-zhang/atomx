"""Independent saved-scalar/fork identity and preserved-site audit, zero physics."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    c = load(HERE / 'contract.json')
    site = Path(c['site'])
    before, overlay = load(HERE / 'original_overlay.json'), load(site / c['overlay_path'])
    preimage = load(HERE / 'prechange_manifest.json')['files']
    changed = [name for name, value in preimage.items() if sha(site / name) != value['sha256']]
    assert changed == [c['overlay_path']]
    assert overlay['models'][2:] == before['models'] and overlay['runs'][2:] == before['runs']
    assert overlay['source_files'][16:] == before['source_files']
    old = load(HERE / 'old_models.json')
    assert len(old) == 19 and sum(len(m['rows']) for m in old) == 76
    combined = overlay['models'][:2] + old
    assert len(combined) == 21 and len({m.get('id', m['saved_run']) for m in combined}) == 21
    assert sum(len(m['rows']) for m in combined) == 84
    assert len(list((site / 'dist/ions').glob('*.gz'))) == 6019
    assert len(overlay['source_files']) == 36 and all(sha(site / 'dist' / a['asset']) == a['sha256'] for a in overlay['source_files'])
    maximum_line = maximum_rms = maximum_moment = 0.
    identities, fork_ids, total_records = set(), {}, 0
    for run in overlay['runs']:
        for point in run['trajectory']:
            total_records += 1
            path = REPO / point['source_manifest']
            assert sha(path) == point['source_manifest_sha256']
            manifest = load(path)
            identity = tuple(manifest[key] for key in ('source_map_sha256', 'metric_sha256', 'geometry_sha256',
                  'objective_sha256', 'state_sha256', 'owner_input_sha256', 'owner_H_sha256'))
            identities.add(identity)
            if 'fork_branch' not in run:
                continue
            branch = run['fork_branch']
            fork_ids.setdefault(branch, []).append(identity)
            assert point['fork_branch'] == branch and point['saved_point_identity'] == list(identity)
            assert point['global_accepted_step'] == 208 + point['iteration']
            assert not point['stationarity_passed'] and not point['experimental_certification']
            assert point['exact_stored_H_acceptance_qualified'] is (None if point['iteration'] == 0 else True)
            targets = {t['label']: t for t in point['targets']}
            original = {t['label']: t for t in manifest['targets']}
            lower = targets['6s_1/2']['energy_Ha']
            for label, target in targets.items():
                assert target['UID'] == original[label]['UID'] and target['energy_Ha'] == original[label]['energy_Ha']
                assert target['selected_level'] == original[label]['selected_level']
                maximum_line = max(maximum_line, abs((target['energy_Ha'] - lower) * overlay['Hartree_cm1'] - target['excitation_cm1']))
                assert target['NIST_uncertainty_cm1'] is None and target['state_average_weight'] == .2
            rms = math.sqrt(sum(t['nominal_gap_cm1']**2 for label, t in targets.items() if label != '6s_1/2') / 4)
            maximum_rms = max(maximum_rms, abs(rms - point['nominal_four_gap_RMS_cm1']))
            for moment in point['multiplets']:
                lo, hi, weights = (('5d_3/2', '5d_5/2', (4, 6)) if moment['label'] == '5d'
                                   else ('6p_1/2', '6p_3/2', (2, 4)))
                center = (weights[0] * targets[lo]['excitation_cm1'] + weights[1] * targets[hi]['excitation_cm1']) / sum(weights)
                split = targets[hi]['excitation_cm1'] - targets[lo]['excitation_cm1']
                maximum_moment = max(maximum_moment, abs(center - moment['centroid_cm1']), abs(split - moment['splitting_cm1']))
    assert total_records == 277 and len(identities) == 262
    assert len(fork_ids['history10']) == len(fork_ids['history40']) == 33
    shared = [i for i in range(33) if fork_ids['history10'][i] == fork_ids['history40'][i]]
    assert shared == list(range(12)) and len(set(fork_ids['history10'] + fork_ids['history40'])) == 54
    assert maximum_line <= 1e-8 and maximum_rms <= 1e-9 and maximum_moment <= 1e-9
    for model, run in zip(overlay['models'][:2], overlay['runs'][:2], strict=True):
        branch = model['fork_branch']
        assert model['id'] == run['model_id'] == c['model_ids'][branch]
        assert model['history_size'] == int(branch.removeprefix('history'))
        assert model['accepted_steps'] == 32 and model['global_initial_step'] == 208 and model['global_accepted_steps'] == 240
        assert model['rci_csf_count'] == 1046 and model['radial_orbital_count'] == 29 and model['active_coordinate_count'] == 3451
        assert not model['full_parent_computed'] and model['state_average_weights'] == [.2] * 5
        for key in ('source_map', 'metric', 'geometry', 'objective'):
            assert model[key + '_sha256'] == c[key + '_sha256']
        last = run['trajectory'][-1]
        assert model['targets'] == last['targets'] and model['multiplets'] == last['multiplets']
        assert model['EOL_Ha'] == last['reported_EOL_Ha'] and model['ground'] == next(t for t in last['targets'] if t['label'] == '6s_1/2')
        assert not model['stationarity_passed'] and not model['experimental_gate_passed']
        assert model['orbital_residual'] > 1e-7 and model['energy_indicator_Ha'] > 1e-8
        assert all(not row['goal_pass'] for row in model['rows'])
        assert model['completion_metadata_checks'] == 15 and model['runtime_controls'] == 2219
        assert '262' in model['scope'] and '0–11' in model['stationarity_note'] and '分支' in model['title']
        by_label = {t['label']: t for t in last['targets']}
        for row in model['rows']:
            target = by_label[row['label']]
            assert row['calculated_cm1'] == target['excitation_cm1'] and row['error_cm1'] == target['nominal_gap_cm1']
    assert overlay['fork_global240_is_not_unique_terminal'] and overlay['distinct_global_points'] == 209
    assert overlay['distinct_saved_point_identities'] == 262 and overlay['per_run_point_records'] == 277
    report = {'all_saved_data_and_preservation_checks_passed': True, 'models': 21, 'rows': 84,
              'old_models_preserved': 19, 'old_rows_preserved': 76, 'unchanged_gzip': 6019,
              'unchanged_page_and_hosting': True, 'fork_records': 66, 'all_run_records': 277,
              'all_five_state_records': 1385, 'common_lineage_points': 209,
              'distinct_saved_point_identities': 262, 'shared_fork_local_indices': shared,
              'first_diverging_local_index': 12, 'two_different_branch_terminal_global240': True,
              'max_transition_reconstruction_error_cm1': maximum_line, 'max_RMS_reconstruction_error_cm1': maximum_rms,
              'max_moment_reconstruction_error_cm1': maximum_moment,
              'overlay_sha256': sha(site / c['overlay_path']), 'page_sha256': sha(site / 'dist/index.html'),
              'contract_sha256': sha(HERE / 'contract.json'), 'verifier_sha256': sha(__file__),
              'new_science_calls': 0, 'physical_array_replay': False, 'publication_performed': False}
    (HERE / 'data_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
