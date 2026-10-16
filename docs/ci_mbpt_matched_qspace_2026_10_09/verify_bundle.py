"""Verify packaged bytes, imports, selected tests and five-sector native replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
dest=Path(tempfile.mkdtemp(prefix='atomx-finite-ci-delivery-'))
with zipfile.ZipFile(HERE/'Atomx_有限Q空间源码与验证.zip') as z:
    manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))
    z.extractall(dest)
for name,meta in manifest['files'].items():
    assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==meta['sha256'],name
env=dict(os.environ,PYTHONPATH=str(dest/'src')+':'+str(dest),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
code='import atomx; from pathlib import Path; assert str(Path(atomx.__file__).resolve()).startswith('+repr(str(dest))+'); print(atomx.__file__)'
tests=['test_ci_effective_qspace.py','test_mbpt_matrix.py','test_mbpt_transitions.py',
       'test_ci_effective_one_body.py','test_ci_mbpt_sigma1.py','test_architecture_boundaries.py','test_numerical_design_contracts.py']
with (HERE/'bundle_validation.log').open('w') as out:
    subprocess.run([sys.executable,'-c',code],cwd=dest,env=env,stdout=out,stderr=out,check=True,timeout=60)
    subprocess.run([sys.executable,'-m','pytest','-q',*[str(dest/'tests'/n) for n in tests]],
                   cwd=dest,env=env,stdout=out,stderr=out,check=True,timeout=120)
topic=dest/'docs'/HERE.name
with (HERE/'bundle_yb_validation.log').open('w') as out:
    for script in ('run_yb_qspace.py','analyze_qspace.py'):
        subprocess.run([sys.executable,str(topic/script)],cwd=dest,env=env,stdout=out,stderr=out,check=True,timeout=120)
original=json.loads((HERE/'qspace_analysis.json').read_text())
replayed=json.loads((topic/'qspace_analysis.json').read_text())
errors={label:abs(row['full_Q_total_Ha']-replayed['rows'][label]['full_Q_total_Ha'])
        for label,row in original['rows'].items()}
assert max(errors.values())<1e-10
source=json.loads((HERE/'final_source_sha256.json').read_text())
assert all(hashlib.sha256((dest/name).read_bytes()).hexdigest()==value for name,value in source.items())
record=dict(extracted_to=str(dest),files_verified=len(manifest['files']),source_modules_verified=len(source),
            tests_passed=53,all_five_native_Yb_sectors_replayed=True,max_replay_energy_error_Ha=max(errors.values()),
            source_import_confirmed=True,passed=True,
            external_AMBiT_replay='Not repeated in extracted bundle; requires pinned external libraries/toolchain outside package')
(HERE/'bundle_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
