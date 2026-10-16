"""Independent complete output and admission/statistical controls."""
import importlib.util
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_saved_input_replays_native_signed_panel():
    adapter = load("atomx_action")
    actual = adapter.load_context(HERE / "fresh_inputs001")()
    independent = np.loadtxt(HERE / "fresh_inputs001/ambit_sigma_output.txt")
    assert actual.shape == independent.shape == (27, 6)
    assert np.array_equal(actual[:, :3], independent[:, :3])
    np.testing.assert_allclose(actual[:, 3:], independent[:, 3:], atol=1e-11, rtol=1e-10)
    assert np.max(np.abs(actual[actual[:, 0] != actual[:, 1], 5])) > 1e-7


def test_statistics_preserve_slow_and_failed_environment_observations():
    driver = load("benchmark")
    pairs = [dict(controlled_timing=True, scientific_gate=True, left=4., right=2.),
             dict(controlled_timing=False, scientific_gate=True, left=100., right=2.),
             dict(controlled_timing=True, scientific_gate=True, left=6., right=2.)]
    stats = driver.pair_statistics(pairs, "left", "right")
    assert stats["pairs"] == 3
    assert stats["observational_ratios"] == [2., 50., 3.]
    assert stats["left_median_seconds"] == 6.
    assert stats["speedup"] is None
    pairs[1]["controlled_timing"] = True
    assert driver.pair_statistics(pairs, "left", "right")["speedup"] == 3.
    pairs[2]["scientific_gate"] = False
    assert driver.pair_statistics(pairs, "left", "right")["speedup"] is None
    assert driver.pair_statistics(pairs[:2], "left", "right")["speedup"] is None


def test_original_resource_and_load_thresholds():
    driver = load("benchmark")
    original = dict(load=[24., 20., 19.], affinity=list(range(32)),
                    available_bytes=32 * 1024**3, disk_bytes=8 * 1024**3)
    assert all(driver.preflight(original, 8).values())
    original["load"][0] = 24.0001
    assert not driver.preflight(original, 8)["cpu_load"]
    original["available_bytes"] -= 1
    assert not driver.preflight(original, 8)["memory"]
    original["disk_bytes"] -= 1
    assert not driver.preflight(original, 8)["disk"]


def test_child_timer_keeps_program_exit_and_actual_resident_memory(tmp_path):
    output = tmp_path / "timer.json"
    started = time.perf_counter()
    run = subprocess.run(
        [str(HERE / "child_timer"), str(output), sys.executable, "-c",
         "import time; a=bytearray(8*1024**2); time.sleep(.02); raise SystemExit(7)"],
        check=False,
    )
    elapsed = time.perf_counter() - started
    actual = json.loads(output.read_text())
    assert run.returncode == 7
    assert .02 <= actual["seconds"] <= elapsed
    assert actual["max_rss_bytes"] >= 8 * 1024**2
    assert actual["user_seconds"] >= 0
    assert actual["system_seconds"] >= 0
    assert actual["pid"] > 0
