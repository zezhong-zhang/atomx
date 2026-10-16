"""Validate shipped bytes and run the electronic suite from extracted source."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
TESTS = ['test_spinor_integrals_fscc.py', 'test_finite_ccsd.py', 'test_ci_sd_space.py',
         'test_ci_effective_qspace.py', 'test_ci_effective_one_body.py', 'test_mbpt_matrix.py',
         'test_architecture_boundaries.py', 'test_architecture_scientific_fidelity.py']


def main():
    archive_path = HERE/'Atomx_全阶SD_FSCCSDT源码与验证.zip'
    with tempfile.TemporaryDirectory(prefix='atomx_fscc_bundle_') as temp:
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
        topic = dest/'atomx/docs/atomic_allorder_fscc_2026_10_09'
        source = json.loads((topic/'backend_source_archive_identity.json').read_text())
        for name, digest in source['files'].items():
            if hashlib.sha256((dest/'external/EXP-T-atomx'/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('nested backend source mismatch '+name)
        exe = dest/'external/backend-linux-x86_64/expt.x'
        exe.chmod(0o755)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(dest/'atomx/src')
        env['ATOMX_EXPT_EXECUTABLE'] = str(exe)
        env['OPENBLAS_NUM_THREADS'] = '1'
        libs = BASE/'toolchain/usr/lib/x86_64-linux-gnu'
        env['LD_LIBRARY_PATH'] = f'{libs}:{libs}/openblas-serial:'+env.get('LD_LIBRARY_PATH', '')
        python = str(ROOT/'.venv/bin/python')
        check = subprocess.check_output([python, '-c', 'import atomx; print(atomx.__file__)'], cwd=dest/'atomx', env=env, text=True).strip()
        if not Path(check).is_relative_to(dest):
            raise RuntimeError('test imported original working tree')
        with (HERE/'bundle_tests.log').open('w') as log:
            result = subprocess.run([python, '-m', 'pytest', '-q', *['tests/'+t for t in TESTS]],
                cwd=dest/'atomx', env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError('extracted source test gate failed')
        # Exact signed-input electronic replay, independent of orbital rebuilds.
        script = '''import json,numpy as np
from pathlib import Path
import atomx
from atomx.methods.fscc import load_integrals,compute
topic=Path("docs/atomic_allorder_fscc_2026_10_09")
a=load_integrals(topic/"be_ccsdt_tile4_final")
old=json.loads((topic/"be_ccsdt_tile4_final/result.json").read_text())
import os
r=compute(a,executable=os.environ["ATOMX_EXPT_EXECUTABLE"],directory="fresh_bundle_replay",model="CCSDT",sector="0h1p",active_indices=range(2,10),tile_size=4,tolerance=1e-12,residual_bound_tolerance_Ha=1e-9)
err=float(np.max(abs(np.array(r.sector_energies_Ha)-old["sector_energies_Ha"])))
assert err<1e-10
print(json.dumps(dict(imported_atomx=atomx.__file__,energy_error_Ha=err,passed=True)))
'''
        replay = json.loads(subprocess.check_output([python, '-c', script], cwd=dest/'atomx', env=env, text=True, timeout=120))
        if not Path(replay['imported_atomx']).is_relative_to(dest):
            raise RuntimeError('replay imported original source')
        record = dict(passed=True, zip_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            files_verified=len(manifest['files']), native_source_modules=manifest['native_source_modules'],
            external_source_files_verified=len(source['files']), test_result=(HERE/'bundle_tests.log').read_text().strip(),
            independent_import=True, frozen_integral_replay=replay,
            runtime_scope='Extracted Atomx/backend binary with retained local GNU Fortran runtime; runtime itself not shipped')
        (HERE/'bundle_validation.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
