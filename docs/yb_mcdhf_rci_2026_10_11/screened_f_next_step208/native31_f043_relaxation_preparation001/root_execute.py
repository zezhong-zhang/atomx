"""Single foreground native31 admission with explicit reviewed file hashes."""
import argparse
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


parser = argparse.ArgumentParser()
for field in ('handoff', 'driver', 'contract', 'preparation'):
    parser.add_argument('--' + field + '-sha256', required=True)
args = parser.parse_args()
for field, filename in (('handoff', 'handoff.json'), ('driver', 'run.py'),
                        ('contract', 'contract.json'), ('preparation', 'preparation.json')):
    assert sha(P / filename) == getattr(args, field + '_sha256'), filename
handoff = json.loads((P / 'handoff.json').read_text())
for name, digest in handoff['files_sha256'].items():
    assert sha(R / name) == digest, name
contract = json.loads((P / 'contract.json').read_text())
roles = list(contract['input_roles'].values())
for row in roles:
    assert sha(R / row['path']) == row['sha256'], row['path']
identity = json.loads((R / contract['input_roles']['source_identity']['path']).read_text())
source_root = R / contract['source']['archive']
frozen = identity['all_atomx_python_sha256']
assert len(frozen) == 356 and identity['source_map_sha256'] == contract['source']['map_sha256']
for name, digest in frozen.items():
    assert sha(source_root / name.removeprefix('src/')) == digest, name
source = shared_source()
assert len(source) == 365
assert 31 in os.sched_getaffinity(0) and os.getloadavg()[0] <= 32
assert not (P / 'actual').exists()
caps = contract['resources']
binding = dict(
    execution_authorized_by_root=True,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    driver_sha256=args.driver_sha256, contract_sha256=args.contract_sha256,
    preparation_sha256=args.preparation_sha256, handoff_sha256=args.handoff_sha256,
    root_wrapper_sha256=sha(pathlib.Path(__file__)),
    source_map_sha256=contract['source']['map_sha256'],
    frozen_source_files=frozen, current_shared_complete_source_files=source,
    input_roles=roles, CPU=31, AS_GiB=3.95, inner_seconds=600, outer_seconds=615,
    max_H=65, max_CI=65, max_gradient_from_weights=51,
    max_optimizer=1, max_new_accepted=16,
    hypothesis=contract['hypothesis'], fixed=contract['fixed'], gates=contract['gates'],
    public_optimizer=contract['public_optimizer'], stops=contract['stop'],
    resources=caps, current_source_adopted=False,
    original_current_regression_failures_preserved=True,
)
write_once(P / 'execution_binding.json', binding)
env = os.environ.copy()
for name in caps['thread_environment']:
    env[name] = '1'
command = ['/data/code/atomx/.venv/bin/python', '-I', str(P / 'run.py'),
           '--execution-binding', str(P / 'execution_binding.json')]


def limits():
    # Keep inherited affinity until the driver's exact CPU31 admission.
    limit = int(caps['AS_GiB'] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


start = time.monotonic()
timeout = False
with (P / 'foreground.log').open('x') as log:
    child = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, preexec_fn=limits)
    try:
        code = child.wait(timeout=caps['outer_seconds'])
    except subprocess.TimeoutExpired:
        timeout = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
receipt = dict(
    status='foreground_closed', exit_code=code, timed_out=timeout,
    elapsed_uncontrolled_s=time.monotonic() - start,
    binding_sha256=sha(P / 'execution_binding.json'), log_sha256=sha(P / 'foreground.log'),
    complete_shared_source_unchanged=shared_source() == source,
    frozen_source_unchanged=all(sha(source_root / name.removeprefix('src/')) == digest
                              for name, digest in frozen.items()),
    input_unchanged=all(sha(R / x['path']) == x['sha256'] for x in roles),
    command=command, current_source_adopted=False, controlled_timing=False,
)
write_once(P / 'root_execution.json', receipt)
print(json.dumps(receipt, indent=2))
raise SystemExit(code if code else int(timeout or not receipt['complete_shared_source_unchanged']
                                      or not receipt['frozen_source_unchanged'] or not receipt['input_unchanged']))
