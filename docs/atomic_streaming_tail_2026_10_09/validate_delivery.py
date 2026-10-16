"""Verify shipped bytes, import extracted source and exercise the actual backend."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import zipfile

from build_delivery import TESTS, ZIPNAME
from verify_evidence import sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')


def main():
    archive_path = HERE/ZIPNAME
    with tempfile.TemporaryDirectory(prefix='atomx_streaming_bundle_') as temp:
        dest = Path(temp)
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read('BUNDLE_MANIFEST.json'))
            for name, record in manifest['files'].items():
                data = archive.read(name)
                if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                    raise RuntimeError('bundle byte mismatch '+name)
            archive.extractall(dest)
        with tarfile.open(dest/'external/EXP-T-atomx-source.tar.xz', 'r:xz') as tar:
            tar.extractall(dest/'external')
        old = dest/'atomx/docs/atomic_allorder_fscc_2026_10_09'
        source = json.loads((old/'backend_source_archive_identity.json').read_text())
        for name, digest in source['files'].items():
            assert sha(dest/'external/EXP-T-atomx'/name) == digest, name
        final = json.loads((dest/'atomx/docs/atomic_streaming_tail_2026_10_09/final_source_sha256.json').read_text())
        assert all(sha(dest/'atomx'/name) == digest for name, digest in final.items())
        exe = dest/'external/backend-linux-x86_64/expt.x'
        exe.chmod(0o755)
        env = dict(os.environ, PYTHONPATH=str(dest/'atomx/src'),
                   ATOMX_EXPT_EXECUTABLE=str(exe), OPENBLAS_NUM_THREADS='1')
        libs = BASE/'toolchain/usr/lib/x86_64-linux-gnu'
        env['LD_LIBRARY_PATH'] = f'{libs}:{libs}/openblas-serial:'+env.get('LD_LIBRARY_PATH', '')
        python = str(ROOT/'.venv/bin/python')
        check = subprocess.check_output([python, '-c', 'import atomx; print(atomx.__file__)'],
                                        cwd=dest/'atomx', env=env, text=True).strip()
        assert Path(check).is_relative_to(dest), 'imported original working tree'
        with (HERE/'bundle_tests.log').open('w') as log:
            tested = subprocess.run([python, '-m', 'pytest', '-q', *['tests/'+t for t in TESTS]],
                                    cwd=dest/'atomx', env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
        assert tested.returncode == 0, 'extracted test suite failed'
        script = '''import json, os, numpy as np
from pathlib import Path
import atomx
from atomx.methods.fscc import compute, load_integrals, read_heff
old = Path("docs/atomic_allorder_fscc_2026_10_09")
records = []
for name, model, active, tile in [("be_ccsdt_tile4_final","CCSDT",tuple(range(2,10)),4),
                                  ("yb_o22_r2","CCSD",None,16)]:
    a = load_integrals(old/name)
    if active is None:
        keys = {(6,-1),(6,1),(6,-2),(5,2),(5,-3)}
        active = tuple(i for i,(n,k,m) in enumerate(a.labels) if (n,k) in keys and i>=a.noccupied)
    saved = json.loads((old/name/"result.json").read_text())
    r = compute(a,executable=os.environ["ATOMX_EXPT_EXECUTABLE"],directory="fresh_"+name,
                model=model,sector="0h1p",active_indices=active,tile_size=tile,threads=2,
                tolerance=1e-12,residual_bound_tolerance_Ha=1e-9,memory_bytes=32*1024**3)
    error = float(np.max(abs(np.asarray(r.sector_energies_Ha)-saved["sector_energies_Ha"])))
    h = read_heff(Path("fresh_"+name)/"scratch/HEFF")["0h1p",1]
    h0 = read_heff(old/name/"scratch/HEFF")["0h1p",1]
    matrix_error = float(np.max(abs(h-h0)))
    assert error<1e-10 and matrix_error<1e-10
    records.append(dict(case=name,energy_error_Ha=error,signed_heff_error_Ha=matrix_error,
                        residual_bound_Ha=r.equation_residual_bound_Ha,passed=True))
print(json.dumps(dict(imported_atomx=atomx.__file__,records=records,passed=True)))
'''
        replay = json.loads(subprocess.check_output([python, '-c', script], cwd=dest/'atomx',
                                                   env=env, text=True, timeout=300))
        assert Path(replay['imported_atomx']).is_relative_to(dest)
        record = dict(passed=True, zip_sha256=sha(archive_path), files_verified=len(manifest['files']),
                      native_source_modules=len(final), external_source_files_verified=len(source['files']),
                      test_result=(HERE/'bundle_tests.log').read_text().strip(), independent_import=True,
                      frozen_integral_replay=replay,
                      runtime_scope='Extracted Atomx source and backend binary; original local GNU runtime/Python dependencies, runtime not shipped')
        (HERE/'bundle_validation.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
