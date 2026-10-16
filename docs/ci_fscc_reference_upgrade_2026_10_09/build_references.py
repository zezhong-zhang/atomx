"""Task-local reference builds: pinned source, isolated compiler and logs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
PREFIX = BASE/'toolchain/usr'
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = str(PREFIX/'lib/x86_64-linux-gnu')+':'+str(PREFIX/'lib/x86_64-linux-gnu/openblas-serial')+':'+env.get('LD_LIBRARY_PATH', '')
env['OMP_NUM_THREADS'] = '2'
cmake = ['uv', 'run', '--no-project', '--with', 'cmake==3.31.6', 'cmake']
common = ['-DCMAKE_Fortran_COMPILER='+str(BASE/'gfortran'),
          '-DCMAKE_PREFIX_PATH='+str(PREFIX),
          '-DCMAKE_LIBRARY_PATH='+str(PREFIX/'lib/x86_64-linux-gnu/openblas-serial'),
          '-DCMAKE_BUILD_TYPE=Release']

def build(name, options, workers):
    start = time.monotonic()
    commands = [cmake+['-S', str(BASE/name), '-B', str(BASE/(name+'-build'))]+common+options,
                cmake+['--build', str(BASE/(name+'-build')), '--parallel', str(workers)]]
    record = dict(code=name, commands=commands, workers=workers, stages=[])
    with (OUT/(name+'_build.log')).open('w') as log:
        for command in commands:
            remaining = 900-(time.monotonic()-start)
            try:
                result = subprocess.run(command, cwd=BASE/name, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, timeout=max(.1, remaining))
                record['stages'].append(result.returncode)
                if result.returncode:
                    break
            except subprocess.TimeoutExpired:
                record['stages'].append('timeout_900s')
                break
    record['elapsed_seconds'] = time.monotonic()-start
    record['success'] = record['stages'] == [0, 0]
    (OUT/(name+'_build.json')).write_text(json.dumps(record, indent=2))
    print(name, record['success'], round(record['elapsed_seconds'], 2), flush=True)
    return record

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build, 'pCI', ['-DUSE_DP_INTEGRALS=ON'], 2),
                   pool.submit(build, 'AMBiT', ['-DUSE_OPENBLAS=ON', '-DUSE_OPENMP=ON', '-DBUILD_TESTING=OFF'], 2)]
        records = [f.result() for f in futures]
    records.append(build('EXP-T', ['-DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch',
                                  '-DBUILD_TESTING=OFF', '-DNOFORTRAN=OFF', '-DTARGET=HASWELL'], 4))
    (OUT/'reference_builds.json').write_text(json.dumps(records, indent=2))
