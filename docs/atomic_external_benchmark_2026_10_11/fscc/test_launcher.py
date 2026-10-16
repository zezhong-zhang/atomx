"""Private launcher's owned-child watchdog and non-polling completion gates."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def launcher():
    spec = importlib.util.spec_from_file_location(
        "finite_fscc_private_launcher", Path(__file__).with_name("benchmark.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blocking_backend_run


def test_blocking_child_return_and_actual_stdout(tmp_path):
    target = tmp_path / "child.log"
    with target.open("w") as log:
        result = launcher()(
            [sys.executable, "-c", "print('owned child'); raise SystemExit(7)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=2.0,
        )
    assert result.returncode == 7
    assert target.read_text() == "owned child\n"


def test_watchdog_reaps_only_its_timed_out_child(tmp_path):
    pid_file = tmp_path / "owned.pid"
    command = [
        sys.executable,
        "-c",
        "import os,pathlib,time; pathlib.Path('owned.pid').write_text(str(os.getpid())); time.sleep(10)",
    ]
    with (
        (tmp_path / "child.log").open("w") as log,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        launcher()(
            command,
            cwd=tmp_path,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=0.25,
        )
    assert pid_file.exists()
    assert not Path("/proc", pid_file.read_text()).exists()
