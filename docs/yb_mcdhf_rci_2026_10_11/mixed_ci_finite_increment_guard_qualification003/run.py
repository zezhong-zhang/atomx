"""Root-sealed one-H successor; preparation performs no physical work."""
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
PRIOR = BASE.parent / 'mixed_ci_finite_increment_implementation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_inputs(contract):
    for name, expected in contract['inputs_sha256'].items():
        if sha(REPO / name) != expected:
            raise ValueError(f'bound input changed: {name}')
    identity = json.loads((PRIOR / 'candidate_source_identity.json').read_text())
    if identity['source_map_sha256'] != contract['source_map_sha256']:
        raise ValueError('candidate source map changed')
    for name, expected in identity['all_atomx_python_sha256'].items():
        if (sha(PRIOR / 'source' / name.removeprefix('src/')) != expected
                or sha(REPO / name) != expected):
            raise ValueError(f'candidate source changed: {name}')
    return identity


def main():
    contract = json.loads((BASE / 'contract.json').read_text())
    execution = json.loads((BASE / 'execution_contract.json').read_text())
    if execution.get('guard_qualification_execution_authorized') is not True:
        raise ValueError('root explicit numerical execution seal is required')
    required = {'contract_sha256': BASE / 'contract.json',
                'runner_sha256': BASE / 'run.py',
                'test_sha256': BASE / 'test_saved_guards.py',
                'preparation_verification_sha256': BASE / 'preparation_verification.json'}
    for key, path in required.items():
        if execution.get(key) != sha(path):
            raise ValueError(f'root seal mismatch: {key}')
    if not json.loads((BASE / 'preparation_verification.json').read_text())['all_passed']:
        raise ValueError('readonly preparation proof failed')
    identity = verify_inputs(contract)
    mapping_audit = json.loads((BASE.parent / 'mixed_ci_one_body_mapping_audit/actual/result.json').read_text())
    if mapping_audit.get('all_passed') is not True:
        raise ValueError('independent zero-builder representation audit must pass before003')
    if any(os.environ.get(key) != '1' for key in
           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')):
        raise ValueError('explicit single-thread execution is required')
    affinity, load = sorted(os.sched_getaffinity(0)), os.getloadavg()
    if len(affinity) != 1 or load[0] > contract['host_admission']['maximum_system_load1']:
        raise ValueError('host/CPU admission failed')
    cap = int(contract['resources']['address_space_GiB'] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    def stop(signum, frame):
        raise TimeoutError('prospective successor120s wall cap exhausted')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(contract['resources']['wall_time_seconds'])
    output = BASE / 'actual'
    output.mkdir(exist_ok=False)
    os.environ['ATOMX_MIXED_GUARD_OUTPUT'] = str(output)
    sys.path.insert(0, str(PRIOR / 'source'))
    start = time.monotonic()
    result = dict(status='started', guard_qualification_passed=False,
        contract_sha256=sha(BASE / 'contract.json'), execution_sha256=sha(BASE / 'execution_contract.json'),
        source_map_sha256=identity['source_map_sha256'], load_before=load, affinity=affinity,
        original_coarse_qualification_status='failed; immutable',
        scope='saved medium-step derivatives plus genuine-owner unsupported/legacy guards')
    write_json(output / 'execution_progress.json', result)
    code = 1
    try:
        import pytest
        code = int(pytest.main(['-q', str(BASE / 'test_saved_guards.py'), '--tb=short',
                               '-x', '-o', 'addopts=']))
        calls = json.loads((output / 'calls.json').read_text())
        result.update(pytest_exit_code=code, actual_calls=calls)
        if code == 0 and (calls['started'] != contract['calls']
                          or calls['completed'] != contract['calls']):
            raise ValueError('exact successor call accounting failed')
        verify_inputs(contract)
        result.update(pytest_exit_code=code, guard_qualification_passed=code == 0,
                      status='passed' if code == 0 else 'failed', actual_calls=calls)
    except BaseException as error:
        result.update(status='failed', failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds=time.monotonic()-start, load_after=os.getloadavg(),
                      max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output / 'result.json', result)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
