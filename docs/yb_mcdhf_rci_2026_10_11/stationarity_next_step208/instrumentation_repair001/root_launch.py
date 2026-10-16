"""Exact one foreground same208 directional discriminator with durable receipt."""
import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parent
R=next(parent for parent in B.parents if (parent / "pyproject.toml").is_file())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    seal=json.loads((B/'execution_contract.json').read_text())
    for name,digest in seal['root_owned_frozen_files'].items(): assert sha(B/name)==digest,name
    assert sha(Path(__file__))==seal['launcher_sha256']
    assert not (B/'actual').exists()
    assert os.getloadavg()[0]<=seal['maximum_system_load1'] and seal['CPU'] in os.sched_getaffinity(0)
    cache=B/'numba_cache';cache.mkdir(exist_ok=False)
    with (B/'root_once_marker.json').open('x') as f:
        f.write(json.dumps({'seal_sha256':sha(B/'execution_contract.json'),'launcher_sha256':sha(Path(__file__))}));f.flush();os.fsync(f.fileno())
    env=dict(os.environ)
    for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[n]='1'
    env['PYTHONHASHSEED']='0';env['NUMBA_CACHE_DIR']=str(cache)
    command=['timeout','--signal=TERM','--kill-after=5s','315s','taskset','-c','30','/data/code/atomx/.venv/bin/python',str(B/'run.py'),'--execute']
    started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
    with (B/'execution.log').open('xb') as log:
        p=subprocess.run(command,env=env,cwd=str(R),stdout=log,stderr=subprocess.STDOUT)
    receipt={'status':'foreground_closed','exit_code':p.returncode,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_uncontrolled_seconds':time.monotonic()-t,'command':command,'seal_sha256':sha(B/'execution_contract.json'),'driver_sha256':sha(B/'run.py'),'contract_sha256':sha(B/'contract.json'),'launcher_sha256':sha(Path(__file__)),'log_sha256':sha(B/'execution.log'),'outer_wall_seconds':315,'kill_after_seconds':5,'partial_outputs_preserved':True,'no_retry':True,'timing_claim_valid':False,'actual_iteration_trigger':seal['actual_iteration_trigger']}
    (B/'root_execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'exit_code':p.returncode,'elapsed_s':receipt['elapsed_uncontrolled_seconds'],'receipt_sha256':sha(B/'root_execution.json')}))
    return p.returncode
if __name__=='__main__':raise SystemExit(main())
