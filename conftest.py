"""Pytest root conftest for the worktree.

The atomx venv has its editable install pinned to the main repo's
``src/`` directory (via a ``.pth`` file), which means running tests
from a git worktree would otherwise import the main tree rather than
the worktree tree.  Prepend this worktree's ``src`` to ``sys.path`` so
tests exercise the code here.

Numerical-thread hygiene (critical review F1, HYP follow-up): on a
loaded box (load average > 50, many concurrent CI/agent runs) an
unpinned OpenBLAS spawns one thread per core and the resulting
thrash makes a single 549x549 ``eigh`` exceed 60 s — the default
suite then never finishes even though every test is individually
fast.  CI pins ``OPENBLAS_NUM_THREADS=1`` in the workflow; the local
run must pin the same way or the gate means nothing.  Set the
variables here, before any test module imports numpy, so a bare
``pytest`` behaves like CI.  Explicit environment settings from the
caller always win (setdefault).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

_WORKTREE_SRC = Path(__file__).resolve().parent / "src"
if str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))


def pytest_configure(config):
    """Apply thread limits even if numpy/scipy were imported before us.

    The env vars above cover the common path (conftest loads before any
    test module).  If a plugin already pulled numpy in, threadpoolctl
    clamps the loaded BLAS pools directly — belt and suspenders, same
    limit CI uses.  Import errors are tolerated so the suite still runs
    on a bare interpreter.
    """
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass
