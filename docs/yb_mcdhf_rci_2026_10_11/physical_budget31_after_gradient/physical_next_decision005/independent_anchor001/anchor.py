"""Run only the small five-UID regression and retain its actual measurements."""

import hashlib
import json
import os
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/code/atomx").resolve()
TEST = "tests/test_mcdhf_state_resolved_derivative.py"
SOURCES = [
    TEST,
    "src/atomx/methods/csf/variational.py",
    "src/atomx/physics/csf/radial_density.py",
    "src/atomx/physics/csf/matrix.py",
    "src/atomx/angular/csf_hamiltonian.py",
    "src/atomx/numerics/bspline/orbital_optimization.py",
    "src/atomx/numerics/bspline/convol.py",
    "src/atomx/numerics/bspline/dirac_assembly.py",
]


def hashes():
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES
    }


class Capture:
    measurement = None

    def __init__(self):
        self.outcomes = []

    def pytest_runtest_call(self, item):
        self.measurement = item.funcargs["measurements"]

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.outcomes.append(dict(test=report.nodeid, outcome=report.outcome))


if __name__ == "__main__":
    if (HERE / "result.json").exists():
        raise RuntimeError("preserved evidence; no automatic repeated anchor")
    assert os.sched_getaffinity(0) == {26}
    threads = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    assert all(os.environ.get(key) == "1" for key in threads)
    available = next(
        int(x.split()[1]) * 1024
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    assert os.getloadavg()[0] <= 16 and available >= 10 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    signal.alarm(180)
    before = hashes()
    import pytest

    capture = Capture()
    status = pytest.main(["-q", TEST], plugins=[capture])
    after = hashes()
    result = dict(
        status="PASS" if status == 0 and before == after else "FAIL",
        pytest_exit_code=int(status),
        outcomes=capture.outcomes,
        measurement=capture.measurement,
        source_sha256=before,
        source_unchanged=before == after,
        peakRSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        resources=dict(
            CPU=26, threads=1, AS_GiB=6, inner_seconds=180, outer_seconds=195
        ),
        prior_existing_targeted_tests="30 PASS in1.23s, CPU26/thread1/6GiB:polar+gradient_reuse+stable_energy",
        prototype_scope="Earlier symmetric-basis prototype passed; this independent RKB point is freshly measured, no predicted output copied",
    )
    (HERE / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    signal.alarm(0)
    sys.exit(0 if result["status"] == "PASS" else 1)
