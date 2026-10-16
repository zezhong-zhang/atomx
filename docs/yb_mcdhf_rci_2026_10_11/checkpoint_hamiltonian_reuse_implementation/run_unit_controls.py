"""Bounded source-bound unit controls; not a 531-CSF physics/timing run."""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    revision = sys.argv[1] if len(sys.argv) == 2 else ''
    if revision not in ('', '002', '003'):
        raise ValueError('only declared unit revisions are supported')
    contract_path = HERE / f'unit_contract{revision}.json'
    contract = json.loads(contract_path.read_text())
    if contract.get('runner_sha256') not in (None, sha(__file__)):
        raise RuntimeError('unit runner changed')
    inputs = contract['source_identity']['all_atomx_python_sha256']
    for name, expected in inputs.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'unit source changed: {name}')
    for name, expected in contract['tests_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'unit test changed: {name}')
    available = sorted(os.sched_getaffinity(0))
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    preflight = dict(available_cpu_count=len(available), selected_cpu=available[0],
                     load_average=os.getloadavg(), mem_available_kib=int(memory['MemAvailable'].split()[0]),
                     timing_claims=False)
    if preflight['mem_available_kib'] < 6 * 1024**2:
        raise RuntimeError('unit controls require at least 6 GiB host free memory')
    (HERE / f'unit_preflight{revision}.json').write_text(json.dumps(preflight, indent=2) + '\n')
    limits = contract['resources']

    def restrict():
        os.sched_setaffinity(0, {available[0]})
        cap = limits['max_address_space_bytes']
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CPU, (limits['max_wall_seconds'], limits['max_wall_seconds']))

    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONPATH'] = str(ROOT / 'src') + ':' + str(ROOT)
    command = [str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', '--durations=10',
               *contract['tests_sha256']]
    started = time.monotonic()
    returncode, stopped = None, None
    log_path = HERE / f'unit_controls{revision}.log'
    with log_path.open('w') as output:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, preexec_fn=restrict,
                                    timeout=limits['max_wall_seconds'], check=False)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            stopped = 'hard_wall_time_limit'
    proof = dict(command=command, contract_sha256=sha(contract_path), runner_sha256=sha(__file__),
                 exit_code=returncode, stopping_condition=stopped,
                 observed_elapsed_seconds=time.monotonic() - started,
                 child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                 log_sha256=sha(log_path),
                 source_unchanged=all(sha(ROOT / name) == expected for name, expected in inputs.items()),
                 tests_unchanged=all(sha(ROOT / name) == expected
                                     for name, expected in contract['tests_sha256'].items()),
                 scope='small native unit controls; no large531 solve or controlled timing claim')
    proof['all_passed'] = returncode == 0 and stopped is None and proof['source_unchanged'] and proof['tests_unchanged']
    (HERE / f'unit_execution{revision}.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))
    return 0 if proof['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
