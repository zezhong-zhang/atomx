"""Root-reviewed, single foreground admission for the matched owner control."""
import datetime
import hashlib
import json
import os
import pathlib
import resource
import signal
import subprocess
import time

R = pathlib.Path('/data/code/atomx')
P = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def shared_source():
    return {str(p.relative_to(R)): sha(p) for p in sorted((R / 'src/atomx').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.suffix not in ('.pyc', '.nbc', '.nbi')}


assert sha(P / 'executor_handoff.json') == 'e2a0af68d3ed23fab0714eccca318b98642c8804b8efb83615bc066e7c7f5827'
handoff = json.loads((P / 'executor_handoff.json').read_text())
roles = []
for name, digest in handoff['files'].items():
    assert sha(P / name) == digest, name
    roles.append({'path': str((P / name).relative_to(R)), 'sha256': digest})
contract = json.loads((P / 'executor_contract.json').read_text())
assert sha(P / 'source_recipe.json') == contract['source_recipe_sha256']
recipe = json.loads((P / 'source_recipe.json').read_text())
for epoch, key in (('current', 'current365'), ('preowner', 'preowner_matched365')):
    assert len(recipe[key]) == 365
    for name, digest in recipe[key].items():
        path = P / epoch / 'source' / name.removeprefix('src/')
        assert sha(path) == digest, str(path)
        roles.append({'path': str(path.relative_to(R)), 'sha256': digest})
for row in contract['literal_inputs']:
    assert sha(R / row['path']) == row['sha256'], row['path']
    roles.append(row)
source = shared_source()
assert len(source) == 365
assert 27 in os.sched_getaffinity(0)
assert max(os.getloadavg()) <= 32
assert not (P / 'actual').exists()
binding = {
    'execution_authorized_by_root': True,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'driver_sha256': sha(P / 'run.py'),
    'root_wrapper_sha256': sha(pathlib.Path(__file__)),
    'handoff_sha256': sha(P / 'executor_handoff.json'),
    'contract_sha256': sha(P / 'executor_contract.json'),
    'source_recipe_sha256': sha(P / 'source_recipe.json'),
    'input_roles': roles,
    'current_shared_complete_source_files': source,
    'resources': contract['resources'],
    'CPU_affinity': 27,
    'hypothesis': 'The owner increment preserves full DC H and signed g at the identical baseline with all other radial and numerical source bytes matched.',
    'distinguishing_outcomes': 'Exact equality isolates this baseline owner increment; any mismatch retains failure and stops interpretation. Neither outcome upgrades old-source or large-step FD gates.',
    'acceptance': contract['acceptance'],
    'stops': contract['stops'],
    'original_two_failures_preserved': True,
    'optimizer_authorized': False,
}
write_once(P / 'execution_binding.json', binding)
env = os.environ.copy()
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    env[name] = '1'
command = ['/data/code/atomx/.venv/bin/python', '-I', str(P / 'run.py'), '--execute']


def limits():
    os.sched_setaffinity(0, {27})
    resource.setrlimit(resource.RLIMIT_AS, (contract['resources']['AS_bytes'], contract['resources']['AS_bytes']))


start = time.monotonic()
timeout = False
with (P / 'foreground.log').open('x') as log:
    child = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, preexec_fn=limits)
    try:
        code = child.wait(timeout=contract['resources']['outer_wall_s'])
    except subprocess.TimeoutExpired:
        timeout = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
receipt = {
    'status': 'foreground_closed', 'exit_code': code, 'timed_out': timeout,
    'elapsed_uncontrolled_s': time.monotonic() - start,
    'binding_sha256': sha(P / 'execution_binding.json'),
    'log_sha256': sha(P / 'foreground.log'),
    'complete_source_unchanged': shared_source() == source,
    'all_bound_inputs_unchanged': all(sha(R / x['path']) == x['sha256'] for x in roles),
    'command': command, 'original_mixed_suite_passed': False, 'optimizer_authorized': False,
}
write_once(P / 'root_execution.json', receipt)
print(json.dumps(receipt, indent=2))
raise SystemExit(code if code else int(timeout or not receipt['complete_source_unchanged'] or not receipt['all_bound_inputs_unchanged']))
