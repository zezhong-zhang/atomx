"""Numerical reproducibility and bounds-safety gates.

Two defects motivated these tests, both of which shipped and neither of
which any existing test could see:

* an out-of-bounds read in a numba convolution kernel that produced a
  *nondeterministic* NaN (~30 % of fresh processes) in the README's own
  ``Atom(Z=4, configs=[...])`` example and in ``Atom(Z=10, numerics=
  "bspline")``;
* value drift between fresh interpreter processes, which a digest over
  several subprocesses detects.

``test_panel_is_deterministic_across_processes`` is cheap enough for every
CI run.  ``test_panel_has_no_out_of_bounds_reads`` is marked ``slow``
because numba bounds checking forces a full recompile; CI runs it in a
dedicated job (see ``.github/workflows/test.yml``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_determinism.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(ROOT), timeout=1800,
    )


def test_panel_is_deterministic_across_processes():
    """Identical input must give bit-identical output in fresh processes."""
    proc = _run("--repeat", "3", "--panel", "quick")
    assert proc.returncode == 0, (
        f"determinism gate failed (exit {proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.slow
def test_panel_has_no_out_of_bounds_reads():
    """Run the panel with NUMBA_BOUNDSCHECK=1 — catches OOB kernel reads.

    With bounds checking off such a read returns heap garbage: harmless when
    the value is finite (it is multiplied by a structurally zero rkb entry)
    and a poisoned ``nan`` integral otherwise, which is exactly how the
    nondeterministic-NaN bug stayed invisible.
    """
    proc = _run("--repeat", "1", "--panel", "full", "--boundscheck")
    assert proc.returncode == 0, (
        f"bounds-check gate failed (exit {proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
