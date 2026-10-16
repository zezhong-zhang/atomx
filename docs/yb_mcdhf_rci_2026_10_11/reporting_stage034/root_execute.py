"""Single foreground root admission for JSON-only report integration."""
import hashlib
import json
import os
import resource
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    assert Path(path).suffix != ".npz"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    assert sha(HERE / "contract.json") == "190491d0b5e2c8060800828ef21f94d0f9017ae3cdbc398e4b85a2ba4b19bada"
    assert sha(HERE / "refresh.py") == contract["driver_sha256"]
    assert all(sha(REPO / name) == value for name, value in contract["inputs"].items())
    assert not any((REPO / name).exists() for name in contract["outputs"])
    seal = dict(authorized=True, driver_sha256=contract["driver_sha256"],
        contract_sha256=sha(HERE / "contract.json"), root_wrapper_sha256=sha(__file__),
        registry_before_sha256=contract["registry_before_sha256"], science_calls=0)
    binding = HERE / "execution_binding.json"
    once(binding, seal)
    caps = contract["resources"]

    def limit():
        os.sched_setaffinity(0, {caps["CPU"]})
        resource.setrlimit(resource.RLIMIT_AS, (caps["AS_bytes"], caps["AS_bytes"]))
        signal.alarm(caps["inner_seconds"])

    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    command = ["/data/code/atomx/.venv/bin/python", "-I", str(HERE / "refresh.py"), "--root-binding", str(binding)]
    start, timed_out = time.monotonic(), False
    with (HERE / "foreground.log").open("x") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, preexec_fn=limit,
            start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=caps["outer_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    receipt = dict(status="foreground_closed", exit_code=code, timed_out=timed_out,
        elapsed_uncontrolled_s=time.monotonic() - start, command=command,
        binding_sha256=sha(binding), log_sha256=sha(HERE / "foreground.log"),
        frozen_input_source_unchanged=all(sha(REPO / name) == value for name, value in contract["inputs"].items()),
        new_science_NPZ_API_card_timer_calls=0)
    once(HERE / "root_execution.json", receipt)
    print(json.dumps(receipt, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
