"""Root-owned, once-only bounded qualification of the shared RCI owner."""
import hashlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = '/data/code/atomx/.venv/bin/python'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_map():
    paths = sorted(p for p in (REPO / 'src/atomx').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts
                   and p.suffix not in ('.pyc', '.nbc', '.nbi'))
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def verify(binding):
    assert source_map() == binding['source_files'], 'shared source epoch changed'
    for name, digest in binding['input_files'].items():
        assert sha(REPO / name) == digest, name
    assert sha(HERE / 'root_launch.py') == binding['driver_sha256']


def child():
    import runpy
    import signal
    binding = json.loads((HERE / 'execution_binding.json').read_text())
    verify(binding)
    assert sys.executable == PYTHON
    assert sorted(os.sched_getaffinity(0)) == [1]
    assert all(os.environ[k] == '1' for k in THREAD_KEYS)
    cap = binding['AS_bytes']
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.alarm(binding['inner_wall_s'])
    put('child_admission.json', dict(affinity=[1], AS_bytes=cap,
        thread_environment={k: os.environ[k] for k in THREAD_KEYS},
        source_files=len(binding['source_files']), interpreter=sys.executable,
        controlled_timing=False, scientific_scope='primary shared owner unit qualification'))
    sys.argv = ['pytest', '-q', 'tests/test_ci_hamiltonian_snapshot.py']
    runpy.run_module('pytest', run_name='__main__')


def parent():
    binding = json.loads((HERE / 'execution_binding.json').read_text())
    verify(binding)
    mem = dict(line.split(':', 1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].strip().split()[0]) * 1024
    assert available >= 4 * 1024**3
    assert os.getloadavg()[0] <= 32
    assert 1 in os.sched_getaffinity(0)
    with (HERE / 'once_marker.json').open('x') as stream:
        json.dump(dict(binding_sha256=sha(HERE / 'execution_binding.json'),
                       started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), stream)
    env = os.environ.copy()
    env.update({k: '1' for k in THREAD_KEYS})
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONPATH'] = str(REPO / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['NUMBA_CACHE_DIR'] = str(HERE / 'numba_cache')
    started = time.monotonic()
    command = ['/usr/bin/timeout', '--kill-after=5s', '315s', '/usr/bin/taskset',
               '-c', '1', PYTHON, str(HERE / 'root_launch.py'), '--child']
    with (HERE / 'foreground.log').open('xb') as log:
        outcome = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    after = source_map()
    same = after == binding['source_files']
    inputs_same = all(sha(REPO / name) == digest for name, digest in binding['input_files'].items())
    put('root_execution.json', dict(status='foreground_closed', exit_code=outcome.returncode,
        binding_sha256=sha(HERE / 'execution_binding.json'), command=command,
        elapsed_uncontrolled_s=elapsed, source_unchanged=same,
        input_files_unchanged=inputs_same, controlled_timing=False,
        log_sha256=sha(HERE / 'foreground.log'),
        resource_admission=dict(MemAvailable_bytes=available, load1=os.getloadavg()[0])))
    print(json.dumps(dict(status='foreground_closed', exit_code=outcome.returncode,
                          elapsed_uncontrolled_s=elapsed, source_unchanged=same)))
    if outcome.returncode or not same or not inputs_same:
        raise SystemExit(outcome.returncode or 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--child']:
        child()
    elif not sys.argv[1:]:
        parent()
    else:
        raise SystemExit('unsupported argument')
