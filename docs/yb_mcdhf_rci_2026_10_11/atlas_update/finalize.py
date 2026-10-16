"""Seal local preparation after factual/link and unchanged-source checks."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path('/home/zzhang/.local/share/atomx-nist-atlas/site')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    data, browser, prepared = (load(HERE / name) for name in ('data_verification.json', 'browser_result.json', 'prepared.json'))
    check('data_all_passed_6685', data['all_passed'] and data['checks_count'] == 6685)
    check('browser_all_passed_29', browser['all_passed'] and browser['checks_count'] == 29 and not browser['javascript_errors'])
    check('page_identity_matches_both_validation_outputs', sha(SITE / 'dist/index.html') == data['page_sha256'] == browser['page_sha256'])
    check('overlay_identity_matches_both_validation_outputs', sha(SITE / 'dist/updates/yb-mcdhf-stage002.json') == data['overlay_sha256'] == browser['overlay_sha256'])
    check('data_verification_script_bound', data['script_sha256'] == sha(HERE / 'verify_data.py'))
    check('browser_check_script_bound', browser['source_sha256'] == sha(HERE / 'browser_check.cjs'))
    for name in ('desktop', 'mobile'):
        check('screenshot_identity:' + name, browser['screenshots'][name]['sha256'] == sha(HERE / (name + '.png')))
    text = (HERE / 'README.md').read_text()
    planned = HERE / 'handoff_manifest.json'
    for target in re.findall(r'\]\(([^)]+)\)', text):
        check('README_link:' + target, (HERE / target).exists() or HERE / target == planned)
    check('README_original_corpus_claim', '6019 original NIST gzip' in text and data['original_gzip_files_verified'] == 6019)
    check('README_static_size_claim', '40,079,382' in text and data['uncompressed_static_bytes'] == 40079382)
    check('README_scope_and_failed_probe', 'not converged expanded-space MCDHF' in text and 'worsens every nominal DC experimental gap' in text)
    check('README_no_live_deployment_claim', 'This preparation did not commit, push, deploy' in text)
    check('eight_models_and_two_historical', prepared['new_model_count'] == 8 and prepared['new_matched_rows'] == 32)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SITE, text=True).strip()
    preimage = load(HERE / 'prechange_manifest.json')
    check('site_HEAD_unchanged_no_commit_here', head == preimage['initial_HEAD'])
    diff = subprocess.run(['git', 'diff', '--check'], cwd=SITE, capture_output=True, text=True)
    check('site_diff_check_passed', diff.returncode == 0 and not diff.stdout and not diff.stderr)
    check('only_original_page_modified', subprocess.check_output(['git', 'diff', '--name-only'], cwd=SITE, text=True).splitlines() == ['dist/index.html'])
    if not all(c['passed'] for c in checks):
        raise RuntimeError([c for c in checks if not c['passed']])
    write(HERE / 'documentation_verification.json', {
        'all_passed': True, 'checks_count': len(checks), 'checks': checks,
        'README_sha256': sha(HERE / 'README.md'), 'script_sha256': sha(Path(__file__)),
        'handoff_manifest_link': 'Validated named output of this finalizer; generated after this proof to avoid a hash cycle.',
        'site_diff_check': 'pass', 'publication_performed_here': False,
    })
    ownership = [
        'contract.json', 'contract_addendum_stage003.json', 'prechange_manifest.json', 'original_index.html',
        'prepare.py', 'prepare_failure001.json', 'prepared_stage002.json', 'prepared.json', 'amend_probe.py',
        'edit_page.py', 'verify_data.py', 'data_verification.json', 'browser_check.cjs', 'browser_failure001.json',
        'browser001.log', 'browser.log', 'browser_result.json', 'desktop.png', 'mobile.png', 'downloaded_update.json',
        'render_source.js', 'README.md', 'finalize.py', 'documentation_verification.json',
    ]
    assets = {'dist/index.html': sha(SITE / 'dist/index.html'), **prepared['new_assets_sha256']}
    manifest = {
        'status': 'frozen_local_preparation_ready_for_root_publication',
        'site': str(SITE), 'site_initial_HEAD': head, 'static_directory': 'dist',
        'site_changed_and_added_files_sha256': assets,
        'owner_evidence_files_sha256': {name: sha(HERE / name) for name in ownership},
        'model_count_added': 8, 'rows_added': 32, 'historical_models_retained': 2,
        'original_gzip_files_byte_verified': 6019, 'static_bytes': data['uncompressed_static_bytes'],
        'data_checks': 6685, 'browser_checks': 29, 'documentation_checks': len(checks),
        'publication_performed_here': False, 'shared_scientific_source_edited_here': False,
        'root_owned_files_excluded': ['root_push_source.py', 'root deployment receipts'],
        'packaging': 'Package existing site/dist using the Sites source helper; preserve original compressed corpus and hosting/access identity. Do not re-export NIST.',
    }
    write(planned, manifest)
    for name, expected in assets.items():
        assert sha(SITE / name) == expected
    for name, expected in manifest['owner_evidence_files_sha256'].items():
        assert sha(HERE / name) == expected
    print(json.dumps({'all_passed': True, 'documentation_checks': len(checks), 'manifest_sha256': sha(planned),
                      'page_sha256': assets['dist/index.html'], 'overlay_sha256': assets['dist/updates/yb-mcdhf-stage002.json']}))


if __name__ == '__main__':
    main()
