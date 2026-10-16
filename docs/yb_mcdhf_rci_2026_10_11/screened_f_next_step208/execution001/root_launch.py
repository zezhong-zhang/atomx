"""One admitted native screened-f foreground with immutable root receipt."""
import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parent
R=next(parent for parent in B.parents if (parent / 'pyproject.toml').is_file())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=json.loads((B/'execution_binding.json').read_text())
 for n,d in s['root_owned_frozen_files'].items(): assert sha(B/n)==d,n
 assert sha(Path(__file__))==s['launcher_sha256']
 assert not (B/'actual').exists()
 assert 31 in os.sched_getaffinity(0) and os.getloadavg()[0]<=s['resources']['maximum_system_load1']
 with (B/'root_once_marker.json').open('x') as f:
  json.dump({'binding_sha256':sha(B/'execution_binding.json'),'launcher_sha256':sha(Path(__file__))},f);f.flush();os.fsync(f.fileno())
 e=dict(os.environ)
 for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):e[n]='1'
 e['PYTHONHASHSEED']='0'
 command=['timeout','--signal=TERM','--kill-after=5s','915s','taskset','-c','31','/data/code/atomx/.venv/bin/python',str(B/'run.py'),'--execute','--binding',str(B/'execution_binding.json')]
 started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
 with (B/'execution.log').open('xb') as log:p=subprocess.run(command,env=e,cwd=str(R),stdout=log,stderr=subprocess.STDOUT)
 r={'status':'foreground_closed','exit_code':p.returncode,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_uncontrolled_s':time.monotonic()-t,'command':command,'binding_sha256':sha(B/'execution_binding.json'),'driver_sha256':sha(B/'run.py'),'contract_sha256':sha(B/'contract.json'),'launcher_sha256':sha(Path(__file__)),'log_sha256':sha(B/'execution.log'),'outer_wall_seconds':915,'kill_after_seconds':5,'partial_outputs_preserved':True,'no_retry':True,'timing_claim_valid':False,'actual_iteration_trigger':s['actual_iteration_trigger']}
 with (B/'root_execution.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({'status':r['status'],'exit_code':p.returncode,'elapsed_s':r['elapsed_uncontrolled_s'],'receipt_sha256':sha(B/'root_execution.json')}))
 return p.returncode
if __name__=='__main__':raise SystemExit(main())
