"""Verify archive hashes, then execute affected tests and physical replays in it."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    archive_path = HERE / 'Atomx_有限CCSD_SD_Breit源码与验证.zip'
    with tempfile.TemporaryDirectory(prefix='atomx-sd-ccsd-verify-') as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read('BUNDLE_MANIFEST.json'))
            assert sum(v['bytes'] for v in manifest['files'].values()) < 512 * 1024**2
            assert set(archive.namelist()) == set(manifest['files']) | {'BUNDLE_MANIFEST.json'}
            for name, entry in manifest['files'].items():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                data = archive.read(name)
                assert len(data) == entry['bytes']
                assert hashlib.sha256(data).hexdigest() == entry['sha256']
            archive.extractall(extracted)
        final = json.loads((HERE / 'source_identity_chain.json').read_text())['final']
        source = {name: entry['sha256'] for name, entry in manifest['files'].items()
                  if name.startswith('src/atomx/')}
        assert source == final and len(source) == 336
        environment = os.environ.copy()
        environment.update(PYTHONPATH='src:.', OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
        python = str((ROOT / '.venv/bin/python').absolute())
        test_command = [python, '-m', 'pytest', '-q', *manifest['tests']]
        with (HERE / 'bundle_tests.log').open('w') as log:
            tests = subprocess.run(test_command, cwd=extracted, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=180)
        test_text = (HERE / 'bundle_tests.log').read_text()
        assert tests.returncode == 0 and re.search(r'\b119 passed\b', test_text), test_text[-2500:]
        replay_command = [python, 'docs/yb_sd_breit_completion_2026_10_09/replay_saved.py',
                          '--output', str(HERE / 'bundle_replay.json')]
        with (HERE / 'bundle_replay.log').open('w') as log:
            replay = subprocess.run(replay_command, cwd=extracted, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=60)
        assert replay.returncode == 0, (HERE / 'bundle_replay.log').read_text()[-2500:]
        replay_result = json.loads((HERE / 'bundle_replay.json').read_text())
        assert replay_result['passed'] and replay_result['source_sha256'] == source
        assert Path(replay_result['imported_atomx']).is_relative_to(extracted)
        result = dict(passed=True, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                      files=len(manifest['files']), source_modules=len(source),
                      source_matches_final=True, affected_tests_passed=119,
                      test_command=test_command, test_returncode=tests.returncode,
                      replay_command=replay_command, replay_returncode=replay.returncode,
                      imported_from_extracted_archive=True,
                      replay_max_energy_error_Ha=max(v[m]['energy_error_Ha']
                          for v in replay_result['five_sector_replay'].values() for m in ('DC', 'DCB')),
                      CCSD_replay_energy_error_Ha=replay_result['CCSD_energy_error_Ha'],
                      limits='Uses installed native dependencies; saved physical matrices replayed, not another full Yb radial assembly')
        (HERE / 'bundle_validation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if not k.endswith('command')}, indent=2))


if __name__ == '__main__':
    main()
