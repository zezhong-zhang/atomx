"""Verify the delivered bytes and run native regression from extracted source."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE/'Atomx_原生约化核SD源码与验证.zip'
DESTINATION = HERE/'extracted_bundle'
DESTINATION.mkdir(exist_ok=False)
with zipfile.ZipFile(ARCHIVE) as archive:
    manifest = json.loads(archive.read('BUNDLE_MANIFEST.json'))
    assert set(archive.namelist()) == set(manifest['files']) | {'BUNDLE_MANIFEST.json', 'REPLAY.txt'}
    for name, record in manifest['files'].items():
        payload = archive.read(name)
        assert len(payload) == record['bytes']
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
    for name in archive.namelist():
        destination = DESTINATION/name
        assert destination.resolve().is_relative_to(DESTINATION.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(name))
root = DESTINATION/'atomx'
source = json.loads((root/'docs/atomic_reduced_sd_2026_10_09/final_source_sha256.json').read_text())
assert len(source) == manifest['source_modules'] == 339
for name, expected in source.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
angular = json.loads((root/'docs/atomic_reduced_sd_2026_10_09/final_pci/pci_angular_validation.json').read_text())
with tarfile.open(DESTINATION/'external/pCI_pristine_pinned.tar.gz') as reference:
    assert b'GNU GENERAL PUBLIC LICENSE' in reference.extractfile('LICENSE.md').read()
    for name, expected in angular['source_sha256'].items():
        assert hashlib.sha256(reference.extractfile(name).read()).hexdigest() == expected
environment = dict(os.environ, PYTHONPATH=str(root/'src'), NUMBA_CACHE_DIR=str(DESTINATION/'cache'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
import_path = subprocess.check_output([sys.executable, '-c', 'import atomx;print(atomx.__file__)'],
                                     cwd=root, env=environment, text=True).strip()
assert Path(import_path).resolve() == (root/'src/atomx/__init__.py').resolve()
command = [sys.executable, '-m', 'pytest', '-q'] + ['tests/'+name for name in manifest['tests']]
with (HERE/'bundle_tests.log').open('w') as log:
    result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                            timeout=180)
record = dict(archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              archive_bytes=ARCHIVE.stat().st_size, files_verified=len(manifest['files']),
              source_modules=len(source), source_import_path=import_path,
              original_pci_sources_verified=True, test_command=command,
              exit_code=result.returncode, passed=result.returncode == 0,
              limitations='56 selected native tests; excludes unfinished large-pool SD and valence accuracy gates')
(HERE/'bundle_validation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
assert record['passed']
