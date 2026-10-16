"""Append the frozen latest diagnostic without changing any spectrum row."""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SITE = Path('/home/zzhang/.local/share/atomx-nist-atlas/site')
DIST = SITE / 'dist'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    prefix = 'docs/yb_mcdhf_rci_2026_10_11/'
    sources = {
        'stage003.json': prefix + 'results_stage003.json',
        'mixed-probe-result.json': prefix + 'mixed_ci_stationarity_probe/result.json',
        'mixed-probe-proof.json': prefix + 'mixed_ci_stationarity_probe/card_evidence.json',
        'mixed-probe-independent-verification.json': prefix + 'mixed_ci_stationarity_probe/independent_verification.json',
    }
    overlay_path = DIST / 'updates/yb-mcdhf-stage002.json'
    overlay = json.loads(overlay_path.read_text())
    prepared = json.loads((HERE / 'prepared.json').read_text())
    contract = {
        'type': 'prospective presentation-only latest-diagnostic addendum',
        'parent_contract_sha256': sha(HERE / 'contract.json'),
        'previous_prepared_sha256': sha(HERE / 'prepared.json'),
        'previous_overlay_sha256': sha(overlay_path),
        'inputs_sha256': {path: sha(ROOT / path) for path in sources.values()},
        'hypothesis': 'The completed mixed-CI probe invalidates an unqualified not-yet-evaluated status for the saved 531 model.',
        'fixed_conditions': 'The eight stage002 spectra, NIST corpus, historical models, nucleus, thresholds and site hosting remain identical.',
        'distinguishing_outcome': 'The diagnostic residuals and failed gap outcome render as a separate source-bound paragraph; no extra spectrum is added.',
        'acceptance': 'Exact proof copies, source-chain checks, unchanged model rows and existing browser/data gates all pass.',
        'stop_condition': 'Stop if an input SHA, fixed spectrum or original NIST corpus changes. No scientific calculation or deployment.',
    }
    assert not (HERE / 'contract_addendum_stage003.json').exists()
    write(HERE / 'contract_addendum_stage003.json', contract)
    shutil.copyfile(HERE / 'prepared.json', HERE / 'prepared_stage002.json')
    card = json.loads((ROOT / sources['mixed-probe-proof.json']).read_text())
    result = json.loads((ROOT / sources['mixed-probe-result.json']).read_text())
    stage = json.loads((ROOT / sources['stage003.json']).read_text())
    assert stage['inputs']['mixed_CI_probe']['sha256'] == sha(ROOT / sources['mixed-probe-result.json'])
    assert stage['inputs']['mixed_CI_independent_replay']['sha256'] == sha(ROOT / sources['mixed-probe-independent-verification.json'])
    assert card['inputs'][sources['mixed-probe-result.json']] == sha(ROOT / sources['mixed-probe-result.json'])
    assert result['baseline']['stationarity_passed'] is False and result['one_step']['stationarity_passed'] is False
    assert card['card_summary']['four_nominal_NIST_gaps_improved'] is False
    spectra_before = {m['id']: json.dumps([m['rows'], m['multiplets']], sort_keys=True) for m in overlay['models']}
    overlay['latest_stationarity_probe'] = {
        'scope': card['scope'],
        'operator': 'DC EOL only; not a new static-Breit spectrum',
        'CSFs': 531,
        'baseline_residual': card['stationarity']['baseline_orbital_residual'],
        'one_step_residual': card['stationarity']['accepted_orbital_residual'],
        'orbital_tolerance': card['stationarity']['orbital_tolerance'],
        'baseline_stationarity_passed': False,
        'one_step_stationarity_passed': False,
        'EOL_delta_Ha': card['card_summary']['EOL_delta_Ha'],
        'four_NIST_gaps_improved': False,
        'full_expanded_MCDHF_completed': False,
        'static_Breit_spectra_changed': False,
        'result_sha256': sha(ROOT / sources['mixed-probe-result.json']),
        'result_asset': 'evidence/yb-mcdhf-stage002/mixed-probe-result.json',
        'stage003_sha256': sha(ROOT / sources['stage003.json']),
    }
    for model in overlay['models']:
        if model['id'].startswith('stage_4f5p5s_'):
            model['stationarity_note'] = '固定005轨道的531-CSF混合CI已测DC EOL驻值：原metric残差0.0367607 > 1e−7；一步后0.0249933仍未过门。不是已收敛扩展MCDHF，DC一步不修改这组固定轨道谱。'
            model['mixed_CI_stationarity_probe_passed'] = False
    assert spectra_before == {m['id']: json.dumps([m['rows'], m['multiplets']], sort_keys=True) for m in overlay['models']}
    for name, source in sources.items():
        asset = 'evidence/yb-mcdhf-stage002/' + name
        shutil.copyfile(ROOT / source, DIST / asset)
        overlay['source_files'].append({'path': source, 'asset': asset, 'sha256': sha(ROOT / source)})
        prepared['new_assets_sha256']['dist/' + asset] = sha(DIST / asset)
    write(overlay_path, overlay)
    prepared['new_assets_sha256']['dist/updates/yb-mcdhf-stage002.json'] = sha(overlay_path)
    prepared['addendum_contract_sha256'] = sha(HERE / 'contract_addendum_stage003.json')
    prepared['addendum_inputs_sha256'] = contract['inputs_sha256']
    prepared['addendum_script_sha256'] = sha(Path(__file__))
    prepared['spectrum_rows_changed_by_addendum'] = False
    write(HERE / 'prepared.json', prepared)
    print('Latest frozen mixed-CI status appended; eight spectra unchanged')


if __name__ == '__main__':
    main()
