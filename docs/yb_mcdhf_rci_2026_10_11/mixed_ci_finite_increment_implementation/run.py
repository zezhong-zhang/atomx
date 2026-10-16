"""Root-sealed bounded finite qualification; no numerical work in preparation."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((BASE/'qualification_contract.json').read_text())
    execution = json.loads((BASE/'execution_contract.json').read_text())
    if execution.get('finite_qualification_execution_authorized') is not True:
        raise RuntimeError('root numerical execution seal is required')
    if execution['qualification_contract_sha256'] != digest(BASE/'qualification_contract.json'):
        raise RuntimeError('root contract binding changed')
    for name, expected in contract['prepared_files_sha256'].items():
        if digest(REPO/name) != expected:
            raise RuntimeError(f'prepared input changed: {name}')
    if execution['preparation_verification_sha256'] != digest(BASE/'preparation_verification.json'):
        raise RuntimeError('root readonly preparation proof binding changed')
    if json.loads((BASE/'preparation_verification.json').read_text())['all_passed'] is not True:
        raise RuntimeError('readonly preparation guards failed')
    old_source = BASE.parent/'checkpoint_hamiltonian_reuse_implementation/source002'
    old_identity = json.loads((old_source.parent/'source_identity002.json').read_text())
    for name, expected in old_identity['all_atomx_python_sha256'].items():
        if digest(old_source/name.removeprefix('src/')) != expected:
            raise RuntimeError(f'frozen original354 comparator source changed: {name}')
    identity = json.loads((BASE/'candidate_source_identity.json').read_text())
    for name, expected in identity['all_atomx_python_sha256'].items():
        if digest(BASE/'source'/name.removeprefix('src/')) != expected or digest(REPO/name) != expected:
            raise RuntimeError(f'candidate archived/current source changed: {name}')
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1' or os.environ.get('OMP_NUM_THREADS') != '1':
        raise RuntimeError('qualification requires explicit single-thread environment')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError('qualification requires one actual CPU affinity')
    load = os.getloadavg()
    if load[0] > contract['host_admission']['maximum_system_load1']:
        raise RuntimeError('qualification host load admission failed')
    cap = int(contract['resources']['address_space_GiB']*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    start = time.monotonic()
    def stop(signum, frame):
        raise TimeoutError('prospective 300-second finite qualification budget exhausted')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(contract['resources']['wall_time_seconds'])
    output = BASE/'actual'
    output.mkdir(exist_ok=False)
    os.environ['ATOMX_MIXED_INCREMENT_OUTPUT'] = str(output)
    sys.path.insert(0, str(BASE/'source'))
    result = dict(status='started', contract_sha256=digest(BASE/'qualification_contract.json'),
                  execution_sha256=digest(BASE/'execution_contract.json'), affinity=affinity, load_before=load,
                  source_map_sha256=identity['source_map_sha256'], numerical_qualification_passed=False)
    (output/'execution_progress.json').write_text(json.dumps(result, indent=2)+'\n')
    try:
        import pytest
        code = pytest.main(['-q', str(REPO/'tests/test_mixed_ci_finite_increment.py'),
                            '--tb=short', '-x', '-o', 'addopts='])
        result.update(pytest_exit_code=int(code), numerical_qualification_passed=code == 0,
                      status='passed' if code == 0 else 'failed')
    except BaseException as error:
        result.update(status='failed', failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds=time.monotonic()-start, load_after=os.getloadavg(),
                      max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    raise SystemExit(result['pytest_exit_code'])


if __name__ == '__main__':
    main()
