"""Once-only affected regression with fixed source and test file identities."""
import datetime
import hashlib
import json
import os
import resource
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    seal = json.loads((HERE / 'execution_contract.json').read_text())
    assert seal['root_authorized_apply']
    for path, expected in seal['test_file_sha256'].items():
        assert sha(REPO / path) == expected, path
    for path, change in seal['changes'].items():
        assert sha(REPO / path) == change['proposed_sha256'], path
    host = {'load': os.getloadavg(), 'affinity': sorted(os.sched_getaffinity(0))}
    assert 30 in host['affinity']
    with (HERE / 'once_marker.json').open('x') as file:
        json.dump({'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'execution_contract_sha256': sha(HERE / 'execution_contract.json'),
                   'driver_sha256': sha(__file__), 'exactly_one_test_launch': True}, file, indent=2)
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    cache = HERE / 'numba_cache'
    cache.mkdir(exist_ok=False)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', NUMBA_CACHE_DIR=str(cache))
    command = ['timeout', '--signal=TERM', '--kill-after=5s', '180s',
               '/data/code/atomx/.venv/bin/python', '-m', 'pytest', '-q', '-x',
               '--junitxml=' + str(HERE / 'pytest.xml'), *seal['tests']]

    def limits():
        os.sched_setaffinity(0, {30})
        cap = int(3.95 * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    started = time.monotonic()
    with (HERE / 'pytest.log').open('wb') as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, preexec_fn=limits, check=False)
    suites = ET.parse(HERE / 'pytest.xml').getroot() if (HERE / 'pytest.xml').exists() else []
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'errors', 'failures', 'skipped')}
    receipt = {'status': 'foreground_closed', 'exit_code': result.returncode,
               'started_utc': json.loads((HERE / 'once_marker.json').read_text())['started_utc'],
               'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'elapsed_uncontrolled_seconds': time.monotonic() - started,
               'execution_contract_sha256': sha(HERE / 'execution_contract.json'),
               'driver_sha256': sha(__file__), 'log_sha256': sha(HERE / 'pytest.log'),
               'test_counts': counts, 'host_admission': host, 'resources': seal['resources'],
               'scope': 'Affected regression, no Yb optimization or speedup certificate.'}
    (HERE / 'root_test_execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'exit_code', 'elapsed_uncontrolled_seconds', 'test_counts')}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
