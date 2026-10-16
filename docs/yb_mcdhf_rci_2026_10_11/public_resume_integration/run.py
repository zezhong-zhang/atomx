"""Execute the prospectively bounded affected public-optimizer regression."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, payload):
    (HERE / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sources():
    return {str(path.relative_to(ROOT)): sha(path)
            for path in sorted((ROOT / "src/atomx").rglob("*.py"))}


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    for name, expected in contract["test_inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Original regression input changed: " + name)
    new_test = "tests/test_orbital_optimizer_resume.py"
    before = sources()
    tests = [*contract["test_inputs_sha256"], new_test]
    cache = tempfile.mkdtemp(prefix="atomx-public-resume-numba-")
    environment = os.environ.copy()
    environment.update({name: "1" for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    environment["NUMBA_CACHE_DIR"] = cache
    cpus = sorted(os.sched_getaffinity(0))
    admission = {"load": os.getloadavg(), "available_affinity_CPUs": cpus,
                 "load1_per_CPU": os.getloadavg()[0] / len(cpus)}
    if admission["load1_per_CPU"] > .8:
        save("admission_failure.json", admission)
        raise RuntimeError("Regression host resource admission failed")
    command = [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", *tests]
    binding = {"contract_sha256": sha(HERE / "contract.json"), "runner_sha256": sha(Path(__file__)),
               "all_atomx_python_sha256": before,
               "source_map_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
               "test_inputs_sha256": {name: sha(ROOT / name) for name in tests},
               "resource_admission": admission, "isolated_numba_cache": cache,
               "command": command, "created_at": datetime.now(timezone.utc).isoformat()}
    save("execution_binding.json", binding)
    started = time.monotonic()
    with (HERE / "pytest.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        os.sched_setaffinity(process.pid, {cpus[0]})
        try:
            code = process.wait(timeout=contract["resources"]["wall_budget_s"])
            timeout = False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code, timeout = process.returncode, True
    source_unchanged = sources() == before
    tests_unchanged = all(sha(ROOT / name) == expected for name, expected in binding["test_inputs_sha256"].items())
    outcome = {"passed": code == 0 and not timeout and source_unchanged and tests_unchanged,
               "returncode": code, "timeout": timeout, "source_unchanged_during_run": source_unchanged,
               "original_and_new_test_inputs_unchanged": tests_unchanged,
               "seconds_uncontrolled": time.monotonic() - started,
               "execution_binding_sha256": sha(HERE / "execution_binding.json"),
               "pytest_log_sha256": sha(HERE / "pytest.log"), "performance_claim": False,
               "experimental_or_full_CV_certification": False}
    save("verification.json", outcome)
    print(json.dumps(outcome, indent=2), flush=True)
    print((HERE / "pytest.log").read_text()[-5000:], flush=True)
    if not outcome["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
