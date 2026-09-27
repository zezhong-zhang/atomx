#!/usr/bin/env python3
"""Keep a bounded all-element DHF worker pool running until pending jobs drain."""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "output/nist_spectra/full_dhf.sqlite"
WORKER = ROOT / "scripts/generate_full_dhf_nist.py"
LOG_DIR = ROOT / "output/nist_spectra/worker_logs"
GIB = 1024 ** 3


def memory_bytes() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        field, _, raw = line.partition(":")
        if field in ("MemAvailable", "SwapFree"):
            values[field] = int(raw.strip().split()[0]) * 1024
    return values["MemAvailable"], values["SwapFree"]


def queue_state(ledger: Path, min_z: int, max_z: int) -> tuple[int, int, set[int]]:
    with sqlite3.connect(ledger, timeout=60) as db:
        pending = db.execute("""SELECT COUNT(*) FROM config_jobs
            WHERE z BETWEEN ? AND ? AND status='pending'""",
            (min_z, max_z)).fetchone()[0]
        retry_queued = db.execute("""SELECT COUNT(*) FROM config_jobs
            WHERE z BETWEEN ? AND ? AND status IN
            ('retry_pending','repair_pending')""",
            (min_z, max_z)).fetchone()[0]
        running = {pid for (pid,) in db.execute("""SELECT worker_pid
            FROM config_jobs WHERE z BETWEEN ? AND ? AND status='running'
              AND worker_pid IS NOT NULL""", (min_z, max_z))}
    return pending, retry_queued, running


def available_slots(max_running: int, running_pids: set[int],
                    child_pids: set[int], memory_available: int,
                    swap_free: int, min_available: int,
                    min_swap_free: int) -> int:
    """Count child processes between jobs without double-counting their claims."""
    active = len(running_pids | child_pids)
    if memory_available < min_available or swap_free < min_swap_free:
        return 0
    return max(0, max_running - active)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--min-z", type=int, default=21)
    parser.add_argument("--max-z", type=int, default=118)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--jobs-per-worker", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-available-gib", type=int, default=16)
    parser.add_argument("--min-swap-free-gib", type=int, default=4)
    parser.add_argument("--once", action="store_true",
                        help="print capacity without starting workers")
    args = parser.parse_args()
    if (not 1 <= args.min_z <= args.max_z <= 118 or
            args.max_running < 1 or args.jobs_per_worker < 1 or
            args.poll_seconds < 1 or args.min_available_gib < 1 or
            args.min_swap_free_gib < 0):
        parser.error("invalid worker-pool limits")
    if not args.ledger.is_file():
        parser.error(f"ledger not found: {args.ledger}")
    children: dict[int, subprocess.Popen] = {}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for pid, child in list(children.items()):
            result = child.poll()
            if result is not None:
                print(f"worker {pid} exited {result}", flush=True)
                del children[pid]
        pending, retry_queued, running = queue_state(
            args.ledger, args.min_z, args.max_z)
        memory_available, swap_free = memory_bytes()
        slots = available_slots(
            args.max_running, running, set(children), memory_available,
            swap_free, args.min_available_gib * GIB,
            args.min_swap_free_gib * GIB)
        print(f"pending={pending} retry_queued={retry_queued} "
              f"tracked_running={len(running)} "
              f"own_children={len(children)} available_GiB={memory_available/GIB:.1f} "
              f"swap_free_GiB={swap_free/GIB:.1f} slots={slots}", flush=True)
        if args.once:
            return
        for _ in range(min(slots, pending)):
            env = dict(os.environ)
            env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
            command = [sys.executable, str(WORKER), "--worker-only",
                       "--output", str(args.ledger),
                       "--min-z", str(args.min_z), "--max-z", str(args.max_z),
                       "--max-calculations", str(args.jobs_per_worker),
                       "--refresh-site-every", "0"]
            log_path = LOG_DIR / f"pool_{time.time_ns()}_{os.getpid()}.log"
            with log_path.open("ab") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            children[child.pid] = child
            print(f"started worker {child.pid}; log={log_path}", flush=True)
        if not pending and not retry_queued and not running and not children:
            print("active queue drained", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
