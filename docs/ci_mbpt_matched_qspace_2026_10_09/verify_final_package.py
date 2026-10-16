"""Final package bytes; numerical payload identical to independently run copy."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
old=json.loads((HERE/'bundle_validation.json').read_text())
tested=Path(old['extracted_to'])
assert old['passed']
source=json.loads((HERE/'final_source_sha256.json').read_text())
consumer=('tests/test_ci_effective_qspace.py',
          'docs/ci_mbpt_matched_qspace_2026_10_09/probe_yb_qspace.py',
          'docs/ci_mbpt_matched_qspace_2026_10_09/run_yb_qspace.py',
          'docs/ci_mbpt_matched_qspace_2026_10_09/analyze_qspace.py')
with zipfile.ZipFile(HERE/'Atomx_有限Q空间源码与验证.zip') as z:
    manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))['files']
    for name,meta in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256']
    for name,value in source.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==value
        assert hashlib.sha256((tested/name).read_bytes()).hexdigest()==value
    for name in consumer:
        assert ast.dump(ast.parse(z.read(name)),include_attributes=False)==ast.dump(ast.parse((tested/name).read_bytes()),include_attributes=False)
    report=dict(passed=True,final_files_verified=len(manifest),source_modules=len(source),
                final_archive_sha256=hashlib.sha256((HERE/'Atomx_有限Q空间源码与验证.zip').read_bytes()).hexdigest(),
                compared_to=tested.as_posix(),native_source_byte_identical_to_53_test_and_five_state_extracted_replay=True,
                test_and_replay_AST_identical=True,
                post_verification_changes='Test formatting, current navigation, report text and validation records; no numerical implementation, inputs, tolerances or consumer semantics changed')
(HERE/'final_package_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
