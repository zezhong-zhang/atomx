"""Bounded external-reference execution with preserved input and stdout."""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

OUT = Path(__file__).resolve().parent
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = str(BASE/'toolchain/usr/lib/x86_64-linux-gnu')+':'+str(BASE/'toolchain/usr/lib/x86_64-linux-gnu/openblas-serial')+':'+env.get('LD_LIBRARY_PATH', '')
env['OMP_NUM_THREADS'] = '2'
env['OPENBLAS_NUM_THREADS'] = '1'

def limit():
    resource.setrlimit(resource.RLIMIT_AS, (4096*1024**2, 4096*1024**2))

def run(name, command, directory, run_env=env):
    started = time.monotonic()
    record = dict(name=name, command=command, cwd=str(directory), seconds_cap=120, address_space_MiB=4096)
    with (OUT/(name+'.log')).open('w') as log:
        try:
            p = subprocess.run(command, cwd=directory, env=run_env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=120, preexec_fn=limit)
            record['returncode'] = p.returncode
        except subprocess.TimeoutExpired:
            record['returncode'] = 'timeout'
    record['seconds'] = time.monotonic()-started
    (OUT/(name+'.json')).write_text(json.dumps(record, indent=2))
    print(name, record['returncode'], round(record['seconds'], 2), flush=True)
    return record

if __name__ == '__main__':
    records = []
    d=OUT/'ambit_be_smoke'; d.mkdir(exist_ok=True)
    shutil.copy2(OUT/'BeII.input', d/'BeII.input')
    exe = BASE/'AMBiT-build/src/ambit'
    records.append(run('AMBiT_BeII', [str(exe), 'BeII.input'], d))
    # Check the official hole example's declared space without production
    # diagonalisation. This does not demonstrate Yb accuracy.
    d=OUT/'ambit_mg_sizes'; d.mkdir(exist_ok=True)
    shutil.copy2(BASE/'AMBiT/doc/Examples/MgI/mg.input', d/'mg.input')
    records.append(run('AMBiT_MgI_sizes', [str(exe), '--check-sizes', 'mg.input'], d))
    build = BASE/'EXP-T-build'
    if (build/'expt.x').exists():
        records.append(run('EXPT_help', [str(build/'expt.x'), '--help'], OUT))
        d=OUT/'expt_vibrot'; d.mkdir(exist_ok=True)
        shutil.copytree(BASE/'EXP-T/test/vibrot_levels', d, dirs_exist_ok=True)
        shutil.copy2(BASE/'EXP-T/test/minitest.py', OUT/'minitest.py')
        e=dict(env);e['PATH']=str(build)+':'+env['PATH']
        records.append(run('EXPT_vibrot', ['/home/zzhang/code/atomx/.venv/bin/python', 'test.py'], d, e))
    (OUT/'reference_smoke.json').write_text(json.dumps(records, indent=2))
