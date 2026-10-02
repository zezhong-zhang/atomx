"""Process-parallel independent atom solves (efficiency tier 1).

The measured parallelism landscape (2026-10-02, 32-core box):

* BLAS threading is NOT a lever: OPENBLAS 1 -> 8 threads on the Fe
  closed-shell DHF solve went 1.13 s -> 1.27 s (negative).  The SCF
  eigensolves are small (N ~ 220 dense, ~4.5 ms each) — thread
  oversubscription only adds overhead.
* Banded eigensolvers are NOT available to the HF Fock problem: the
  measured matrix bandwidth is 166 of N = 219 (35% dense) — the
  non-local exchange fills the B-spline band (pure Galerkin width
  ~18).  Recorded as dead end #10.
* The REAL lever for multi-configuration workloads (XFEL cascades,
  Z panels): independent solves in a process pool.  Near-linear on
  cores; workers pin BLAS threads to 1 to avoid oversubscription.

This module is that lever: :func:`solve_parallel` runs a list of
independent ``Atom`` solves in a fork-based process pool with
per-worker thread pinning, returning results in job order.  Fork
inherits the parent's warm numba/LAPACK state, so per-worker JIT
recompilation is not paid.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _pin_threads() -> None:
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def summarize(atom, result) -> dict:
    """Default picklable per-job summary (results are not picklable)."""
    return {
        "Z": atom.Z,
        "total_energy": float(atom.total_energy),
        "converged": bool(getattr(result, "converged", True)),
        "method": str(getattr(result, "method_name", type(result).__name__)),
    }


def _run_job(job: dict) -> Any:
    """Worker entry: pin threads, construct the Atom, solve, reduce."""
    _pin_threads()
    from atomx import Atom

    job = dict(job)
    reduce_fn = job.pop("reduce", summarize)
    solve_kwargs = dict(job.pop("solve_kwargs", {}) or {})
    atom = Atom(**job)
    result = atom.solve(**solve_kwargs)
    return reduce_fn(atom, result)


def solve_parallel(
    jobs: Sequence[dict],
    *,
    n_workers: int | None = None,
    max_workers: int | None = None,
) -> list:
    """Solve many independent ``Atom(**job)`` configurations in parallel.

    Parameters
    ----------
    jobs : sequence of dict
        ``Atom`` constructor keyword dicts.  An optional ``"solve_kwargs"``
        key is popped and forwarded to ``Atom.solve``.  The dicts are
        consumed (popped) in the workers, so pass copies if reusing them.
    n_workers : int, optional
        Process-pool size.  Defaults to ``min(len(jobs), os.cpu_count())``.
    max_workers : int, optional
        Deprecated alias of ``n_workers`` (kept for call-site symmetry
        with :class:`ProcessPoolExecutor`).

    A job dict may carry ``"reduce"``: a top-level
    ``callable(atom, result) -> picklable`` that defines the returned
    payload (default: :func:`summarize` — solved ``Atom`` objects and
    raw results contain unpicklable attributes and do not survive the
    process boundary).

    Returns
    -------
    list
        One payload per job, in job order.
    """
    if not jobs:
        return []
    if n_workers is None:
        n_workers = max_workers
    if n_workers is None:
        n_workers = min(len(jobs), os.cpu_count() or 1)
    n_workers = max(1, min(int(n_workers), len(jobs)))

    # fork context: children inherit warm numba caches and imported
    # modules; each child pins its BLAS threads (measured: unpinned
    # pools oversubscribe and lose to the serial baseline).
    executor = ProcessPoolExecutor(max_workers=n_workers)
    try:
        return list(executor.map(_run_job, [dict(j) for j in jobs]))
    finally:
        executor.shutdown(wait=True)


__all__ = ["solve_parallel", "summarize"]
