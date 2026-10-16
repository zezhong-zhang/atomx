"""Delegated one-shot foreground test launcher and immutable receipt."""
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
seal=json.loads((HERE/'execution_contract.json').read_text())
if sha(HERE/'test_driver.py')!=seal['driver_sha256']:
    raise ValueError('Driver changed')
with (HERE/'once_marker.json').open('x') as f:
    json.dump({'authorized_by_root':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_contract_sha256':sha(HERE/'execution_contract.json'),'no_retry':True},f,indent=2)
(HERE/'actual').mkdir(exist_ok=False)
env=os.environ.copy()
for variable in seal['thread_variables']:env[variable]='1'
env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
env['NUMBA_CACHE_DIR']=str(HERE/'actual/numba_cache')
env['PYTHONHASHSEED']='0'
command=['timeout','--signal=TERM','--kill-after=5s','65s','/data/code/atomx/.venv/bin/python',str(HERE/'test_driver.py')]
record={'status':'foreground_started','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'CPU':30,'memory_GiB':3.95,'cwd':str(REPO),'launcher_sha256':sha(Path(__file__)),'execution_contract_sha256':sha(HERE/'execution_contract.json'),'thread_variables':{k:env[k] for k in seal['thread_variables']},'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','timing_controlled':False}
(HERE/'execution_record.json').write_text(json.dumps(record,indent=2)+'\n')
begin=time.monotonic()
with (HERE/'execution.log').open('xb') as log:
    try:
        process=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=72,check=False)
        code=process.returncode
    except subprocess.TimeoutExpired:
        code=-999
record.update(status='foreground_closed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_uncontrolled_seconds=time.monotonic()-begin,log_sha256=sha(HERE/'execution.log'))
if (HERE/'actual/result.json').exists():record['actual_result_sha256']=sha(HERE/'actual/result.json')
(HERE/'execution_record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(code if code>=0 else 1)
