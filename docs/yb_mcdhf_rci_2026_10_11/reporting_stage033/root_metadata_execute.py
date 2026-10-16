"""Root single bounded metadata integration; no card/API/science."""
import datetime, hashlib, json, os, pathlib, resource, subprocess, time
R=pathlib.Path('/data/code/atomx');B=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((B/'contract.json').read_text());h=json.loads((B/'handoff.json').read_text())
assert sha(B/'contract.json')=='27d414836aeada8d4f5b056d10dff7c75fa292e646aa6f2955d13da9a1860d30'
assert sha(B/'handoff.json')=='ece8f9b02c443b6275237fb66d479e3aade81203d11aa9961c3da67ebc0d6ee6'
for name,digest in h['files'].items():assert sha(B/name)==digest,name
for name,digest in c['inputs'].items():assert sha(R/name)==digest,name
prep=json.loads((B/'preparation.json').read_text());assert prep['all_passed'] and len(prep['checks'])==125 and all(x['passed'] for x in prep['checks'])
registry=R/'docs/atomx_hourly_progress_2026_10_10/progress.json';assert sha(registry)==c['registry_before_sha256']
assert not any((B/name).exists() for name in c['exclusive_outputs']);assert not (B.parent/'results_stage033.json').exists()
def once(name,v):
    with (B/name).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
seal=h['root_binding_required'];seal['root_review']={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_contract_handoff_read_full':True,'all_inputs_hash_verified':True,'old_science_states_failures_preserved':True,'hour28four_probes_and_Yb_zero_separate':True,'no_card_timer_API_science_calls':True,'root_wrapper_sha256':sha(pathlib.Path(__file__))};once('root_completion_binding.json',seal)
assert 30 in os.sched_getaffinity(0)
def limits():
    os.sched_setaffinity(0,{30});resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
command=['timeout','--kill-after=5s','60s',*h['command']]
env=os.environ.copy()
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
started=time.monotonic()
with (B/'root_stdout.log').open('x') as out,(B/'root_stderr.log').open('x') as err:proc=subprocess.run(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,preexec_fn=limits,timeout=70)
receipt={'status':'foreground_closed','exit_code':proc.returncode,'elapsed_uncontrolled_s':time.monotonic()-started,'root_seal_sha256':sha(B/'root_completion_binding.json'),'contract_sha256':sha(B/'contract.json'),'handoff_sha256':sha(B/'handoff.json'),'registry_after_sha256':sha(registry),'stdout_sha256':sha(B/'root_stdout.log'),'stderr_sha256':sha(B/'root_stderr.log'),'command':command,'scientific_NPZ_API_card_calls':0,'input_unchanged':all(sha(R/name)==digest for name,digest in c['inputs'].items())}
for name in ('integration_proof.json','current_card_evidence.json'): 
    if (B/name).exists():receipt[name+'_sha256']=sha(B/name)
once('root_execution.json',receipt);print(json.dumps(receipt,indent=2));raise SystemExit(proc.returncode)
