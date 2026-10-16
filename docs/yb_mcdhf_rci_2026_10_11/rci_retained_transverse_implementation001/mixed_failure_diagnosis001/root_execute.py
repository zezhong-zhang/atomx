"""Root-owned single foreground saved-array diagnostic admission."""
import datetime, hashlib, json, os, pathlib, resource, signal, subprocess, time
R = pathlib.Path('/data/code/atomx')
P = pathlib.Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
h = json.loads((P/'handoff.json').read_text())
assert sha(P/'handoff.json') == 'f9dcb7b66ab590e342f306c17915220a15e8bc312801df85295c65a168be192e'
for name, digest in h['files'].items(): assert sha(P/name) == digest, name
c = json.loads((P/'contract.json').read_text())
for row in c['inputs']: assert sha(R/row['path']) == row['sha256'], row['path']
assert 27 in os.sched_getaffinity(0)
assert max(os.getloadavg()) <= 32
source = {str(p.relative_to(R)):sha(p) for p in sorted((R/'src/atomx').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.nbc','.nbi')}
assert len(source) == 365
binding = dict(execution_authorized_by_root=True, driver_sha256=sha(P/'run.py'), root_wrapper_sha256=sha(pathlib.Path(__file__)), handoff_sha256=sha(P/'handoff.json'), contract_sha256=sha(P/'contract.json'), input_roles=c['inputs'], current_shared_complete_source_files=source, resources=c['resources'], CPU_affinity=27, hypothesis=c['hypotheses'], fixed=c['fixed'], original_two_failures_preserved=True, optimizer_authorized=False, all_predeclared_scales=True, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (P/'execution_binding.json').open('x') as f: json.dump(binding,f,indent=2);f.write('\n')
env=os.environ.copy()
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
cmd=['/data/code/atomx/.venv/bin/python','-I',str(P/'run.py'),'--execute']
def limits():
    os.sched_setaffinity(0,{27})
    resource.setrlimit(resource.RLIMIT_AS,(c['resources']['AS_bytes'],c['resources']['AS_bytes']))
start=time.monotonic(); timeout=False
with (P/'foreground.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
    try: code=child.wait(timeout=c['resources']['outer_wall_s'])
    except subprocess.TimeoutExpired:
        timeout=True;os.killpg(child.pid,signal.SIGTERM)
        try: code=child.wait(timeout=5)
        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);code=child.wait()
after={str(p.relative_to(R)):sha(p) for p in sorted((R/'src/atomx').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.nbc','.nbi')}
receipt=dict(status='foreground_closed', exit_code=code, timed_out=timeout, elapsed_uncontrolled_s=time.monotonic()-start, root_binding_sha256=sha(P/'execution_binding.json'), log_sha256=sha(P/'foreground.log'), complete_source_unchanged=after==source, input_unchanged=all(sha(R/x['path'])==x['sha256'] for x in c['inputs']), command=cmd, NumPy_only_saved_consumer=True, new_physical_producers=0, original_suite_passed=False, optimizer_authorized=False)
with (P/'root_execution.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
raise SystemExit(code if code else 1 if timeout or not receipt['complete_source_unchanged'] or not receipt['input_unchanged'] else 0)
