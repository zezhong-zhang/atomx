"""Root one-attempt foreground execution after entire-source review."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'AGENTS.md').is_file())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


binding = load(HERE / 'execution_binding.json')
contract = load(HERE / 'contract.json')
inputs = {ROOT / row['path']: row['sha256'] for row in contract['input_roles'].values()}
identity = load(ROOT / contract['input_roles']['source_identity']['path'])
source = ROOT / contract['source_archive']
inputs.update({source / name.removeprefix('src/'): digest
               for name, digest in identity['all_atomx_python_sha256'].items()})
inputs.update({HERE / 'run.py': binding['driver_sha256'],
               HERE / 'contract.json': binding['preparation_contract_sha256'],
               HERE / 'preparation.json': binding['preparation_sha256'],
               HERE / 'handoff.json': binding['handoff_sha256'],
               Path(__file__).resolve(): binding['guardian_sha256']})
binding_sha = sha(HERE / 'execution_binding.json')
for path, digest in inputs.items():
    if sha(path) != digest:
        raise ValueError('Changed reviewed execution input: ' + str(path))
if (HERE / 'actual').exists():
    raise FileExistsError('One attempt only; preserve existing actual outputs')
environment = os.environ.copy()
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    environment[name] = '1'
started = time.monotonic()
with (HERE / 'execution.log').open('xb') as log:
    process = subprocess.Popen(
        ['/data/code/atomx/.venv/bin/python', '-I', str(HERE / 'run.py'),
         '--execution-binding', str(HERE / 'execution_binding.json')],
        env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stopped = False
    try:
        code = process.wait(timeout=contract['resources']['outer_seconds'])
    except subprocess.TimeoutExpired:
        stopped = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
unchanged = (sha(HERE / 'execution_binding.json') == binding_sha
             and all(sha(path) == digest for path, digest in inputs.items()))
result, failure = HERE / 'actual/result.json', HERE / 'actual/failure.json'
receipt = dict(status='foreground_closed', exit_code=code, wall_stop=stopped,
               elapsed_uncontrolled_s=time.monotonic() - started, controlled_timing=False,
               source_and_inputs_unchanged=unchanged, execution_binding_sha256=binding_sha,
               guardian_sha256=binding['guardian_sha256'], verified_file_count=len(inputs),
               result_sha256=sha(result) if result.exists() else None,
               failure_sha256=sha(failure) if failure.exists() else None,
               execution_log_sha256=sha(HERE / 'execution.log'),
               stationarity_or_experimental_precision_certified=False)
(HERE / 'root_execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
raise SystemExit(code if unchanged else 70)
