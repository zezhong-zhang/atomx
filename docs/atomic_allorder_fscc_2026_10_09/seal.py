"""Seal final local scientific evidence after extracted-source validation."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    validation = json.loads((HERE/'bundle_validation.json').read_text())
    archive = HERE/'Atomx_全阶SD_FSCCSDT源码与验证.zip'
    assert validation['passed'] and validation['independent_import']
    assert validation['zip_sha256'] == digest(archive)
    assert '64 passed' in validation['test_result']
    assert validation['frozen_integral_replay']['energy_error_Ha'] == 0.0
    identity = json.loads((HERE/'source_identity_chain.json').read_text())
    assert all(digest(ROOT/name) == sha for name, sha in identity['final'].items())
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('BUNDLE_MANIFEST.json'))
        for name, sha in identity['final'].items():
            assert hashlib.sha256(bundle.read('atomx/'+name)).hexdigest() == sha
        for name, record in manifest['files'].items():
            data = bundle.read(name)
            assert len(data) == record['bytes']
            assert hashlib.sha256(data).hexdigest() == record['sha256']
    excluded = {'artifact_manifest.json', 'delivery_artifacts.json',
                'delivery_receipt_report.json', 'delivery_receipt_source.json'}
    records = {}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in excluded:
            continue
        records[str(path.relative_to(ROOT))] = dict(
            sha256=digest(path), bytes=path.stat().st_size)
    (HERE/'artifact_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(dict(sealed_files=len(records),
                         sealed_bytes=sum(r['bytes'] for r in records.values()),
                         source_modules=len(identity['final']),
                         validated_package_sha256=validation['zip_sha256'])))


if __name__ == '__main__':
    main()
