"""Sequential bounded execution; retain logs and all nonzero outcomes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
labels=sys.argv[1:] or ['6s_1/2','5d_3/2','5d_5/2','6p_1/2','6p_3/2']
contracts=json.loads((HERE/'contract.json').read_text())
records=[]
for label in labels:
    stem=label.replace('/','_')
    cmd=[sys.executable,str(HERE/'probe_yb.py'),label]
    start=time.monotonic()
    outcome={"label":label,"command":cmd,"wall_cap_seconds":contracts['resources']['wall_seconds_per_sector']}
    with (HERE/(stem+'_execution.log')).open('w') as log:
        try:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=contracts['resources']['wall_seconds_per_sector'],
                env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
            outcome['exit_code']=result.returncode
        except subprocess.TimeoutExpired:
            outcome.update(exit_code=None,timeout=True)
    outcome['elapsed_seconds']=time.monotonic()-start
    records.append(outcome)
    (HERE/'execution_records.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(outcome),flush=True)
    if outcome['exit_code'] != 0:
        raise SystemExit('Sector failed; retained evidence, no model shrink or gate change.')
