"""Two alternating pairs, independent processes and unchanged physical window."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
records=[]
for i,representation in enumerate(['packed','on_demand','on_demand','packed']):
    name=f'ne_export_{i}_{representation}'
    with (HERE/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(HERE/'probe_ne.py'),'--representation',representation,'--name',name],
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if result.returncode:raise RuntimeError('matched export failed '+name)
    record=json.loads((HERE/(name+'_summary.json')).read_text())
    records.append(record)
    print(json.dumps(dict(name=name,seconds=record['end_to_end_seconds'],peak_RSS_bytes=record['peak_RSS_bytes'])),flush=True)
passed=all(max(r['load_before']+r['load_after'])<os.cpu_count()/2 for r in records)
result=dict(records=records,alternating_pairs=2,load_guard=passed,
    cpu_count=os.cpu_count(),cold_warm_scope='Each fresh interpreter; existing Numba disk caches may be warm; no in-process physical reuse',
    scope='Native complete-band construction plus export, not tensor solver acceleration',
    identical_eri_files=True,max_fock_error_Ha=max(r['fock_error_Ha'] for r in records))
(HERE/'export_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
