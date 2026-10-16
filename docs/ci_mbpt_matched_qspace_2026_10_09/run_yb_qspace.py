"""Sequential per-sector caps; record failures before stopping."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
started=time.monotonic()
env=dict(os.environ,PYTHONPATH='src:.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for label in ('6s_1/2','5d_3/2','5d_5/2','6p_1/2','6p_3/2'):
    path=HERE/(label.replace('/','_')+'_qspace.log')
    t=time.monotonic()
    with path.open('w') as out:
        try:
            r=subprocess.run([sys.executable,str(HERE/'probe_yb_qspace.py'),label],
                             cwd=ROOT,env=env,stdout=out,stderr=out,timeout=300)
            code=r.returncode
        except subprocess.TimeoutExpired:
            code='timeout'
    record=dict(label=label,exit_code=code,seconds=time.monotonic()-t)
    (HERE/(label.replace('/','_')+'_execution.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if code!=0: raise RuntimeError('Stopped on failed sector: '+label)
print(json.dumps(dict(all_sectors_pass=True,seconds=time.monotonic()-started)),flush=True)
