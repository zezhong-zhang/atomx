"""Selected historical FAC/operator regressions, serialized with timing work."""

import fcntl
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TESTS = ['tests/test_fac_ci_operator.py', 'tests/test_ci_analytic_slater_condon.py',
         'tests/test_ci_grid_kernel.py']


def hashes():
    paths = [ROOT/p for p in TESTS] + [ROOT/'tests/data/fac_ci_operator.json',
                                     ROOT/'tests/data/fac_ci_spinors.npz']
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    before = hashes()
    with (HERE.parents[1]/'timing.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.monotonic()
        with (HERE/'selected_tests.log').open('w') as stream:
            child = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=600, check=False)
        elapsed = time.monotonic()-start
    after = hashes()
    (HERE/'selected_tests.json').write_text(json.dumps(dict(
        command=[sys.executable, '-m', 'pytest', '-q', *TESTS],
        exit_code=child.returncode, before=before, after=after,
        source_and_historical_fixtures_unchanged=before == after,
        elapsed_seconds=elapsed, performance_measurement=False,
        thresholds='Existing archived fixture tolerances unchanged; fresh campaign parent gates reported separately',
        log='selected_tests.log'), indent=2)+'\n')
    if child.returncode or before != after:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
