"""Foreground root admission of immutable saved-array algebra once."""
import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from run import authorize
B=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    binding_path=B/'execution_binding.json'
    contract,inputs,binding=authorize(binding_path)
    assert sha(Path(__file__))==binding['launcher_sha256']
    assert not (B/'actual').exists()
    with (B/'root_once_marker.json').open('x') as f:
        f.write(json.dumps({'binding_sha256':sha(binding_path),'launcher_sha256':sha(Path(__file__))}));f.flush();os.fsync(f.fileno())
    command=['timeout','--signal=TERM','--kill-after=5s','135s','/data/code/atomx/.venv/bin/python',str(B/'run.py'),'--execute','--binding',str(binding_path)]
    env=dict(os.environ)
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): env[n]='1'
    started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
    with (B/'execution.log').open('xb') as log:
        r=subprocess.run(command,env=env,cwd=str(next(p for p in B.parents if (p/'pyproject.toml').exists())),stdout=log,stderr=subprocess.STDOUT)
    receipt={'status':'foreground_closed','exit_code':r.returncode,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_uncontrolled_seconds':time.monotonic()-t,'command':command,'binding_sha256':sha(binding_path),'driver_sha256':sha(B/'run.py'),'launcher_sha256':sha(Path(__file__)),'contract_sha256':sha(B/'contract.json'),'log_sha256':sha(B/'execution.log'),'outer_wall_seconds':135,'kill_after_seconds':5,'no_retry':True,'original_failed_execution_preserved':True,'new_physical_producer_calls':0,'timing_claim_valid':False}
    (B/'root_execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'exit_code':r.returncode,'elapsed_s':receipt['elapsed_uncontrolled_seconds'],'receipt_sha256':sha(B/'root_execution.json')}))
    return r.returncode
if __name__=='__main__': raise SystemExit(main())
