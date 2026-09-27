#!/usr/bin/env python
"""Retry new H–Ca solver failures once while the main DHF batch runs."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_full_dhf_nist import (  # noqa: E402
    OUTPUT, process_start_ticks, run, update_lines)


def stage_failed_jobs(path=OUTPUT, *, min_z: int = 1,
                      max_z: int = 20, ground_only: bool = False) -> int:
    """Move each newly failed configuration to one isolated repair attempt."""
    with sqlite3.connect(path, timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("""CREATE TABLE IF NOT EXISTS config_repair_attempts (
            z INTEGER NOT NULL, charge INTEGER NOT NULL,
            configuration TEXT NOT NULL, PRIMARY KEY(z,charge,configuration))""")
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("""SELECT j.z,j.charge,j.configuration
            FROM config_jobs j LEFT JOIN config_repair_attempts a
            ON a.z=j.z AND a.charge=j.charge
               AND a.configuration=j.configuration
            WHERE j.z BETWEEN ? AND ? AND j.status='failed'
              AND (?=0 OR j.is_ground=1) AND a.z IS NULL""",
            (min_z, max_z, int(ground_only))).fetchall()
        species = set()
        for z, charge, configuration in rows:
            db.execute("""INSERT INTO config_repair_attempts VALUES (?,?,?)""",
                       (z, charge, configuration))
            db.execute("""UPDATE config_jobs SET status='repair_pending'
                WHERE z=? AND charge=? AND configuration=? AND status='failed'""",
                       (z, charge, configuration))
            species.add((z, charge))
        for z, charge in species:
            update_lines(db, z, charge)
        return len(rows)


def active_count(path=OUTPUT, *, min_z: int = 1,
                 max_z: int = 20) -> int:
    with sqlite3.connect(path, timeout=60) as db:
        return db.execute("""SELECT COUNT(*) FROM config_jobs
            WHERE z BETWEEN ? AND ? AND status IN
            ('pending','retry_pending','repair_pending','running')""",
            (min_z, max_z)).fetchone()[0]


def repair_count(path=OUTPUT, *, min_z: int = 1,
                 max_z: int = 20) -> int:
    with sqlite3.connect(path, timeout=60) as db:
        return db.execute("""SELECT COUNT(*) FROM config_jobs
            WHERE z BETWEEN ? AND ? AND status='repair_pending'""",
            (min_z, max_z)).fetchone()[0]


def running_ground_count(path=OUTPUT, *, min_z: int = 1,
                         max_z: int = 20) -> int:
    with sqlite3.connect(path, timeout=60) as db:
        return db.execute("""SELECT COUNT(*) FROM config_jobs j
            INNER JOIN config_repair_attempts a USING(z,charge,configuration)
            WHERE j.z BETWEEN ? AND ? AND j.is_ground=1
              AND j.status='running'""",
            (min_z, max_z)).fetchone()[0]


def recover_abandoned_claims(path=OUTPUT, *, min_z: int = 1,
                             max_z: int = 20) -> int:
    """Requeue tracked jobs only after their exact worker process is gone."""
    with sqlite3.connect(path, timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        columns = {row[1] for row in db.execute("PRAGMA table_info(config_jobs)")}
        if not {"worker_pid", "worker_start_ticks", "claimed_from_status"} <= columns:
            return 0
        candidates = db.execute("""SELECT z,charge,configuration,worker_pid,
            worker_start_ticks,claimed_from_status FROM config_jobs
            WHERE z BETWEEN ? AND ? AND status='running'
              AND worker_pid IS NOT NULL""", (min_z, max_z)).fetchall()
        abandoned = []
        for row in candidates:
            pid, recorded_tick = row[3], row[4]
            actual_tick = process_start_ticks(pid)
            if actual_tick == recorded_tick and actual_tick is not None:
                continue
            if Path(f"/proc/{pid}").exists() and actual_tick is None:
                continue
            abandoned.append(row)
        if not abandoned:
            return 0
        db.execute("BEGIN IMMEDIATE")
        repaired = 0
        species = set()
        allowed = {"pending", "retry_pending", "repair_pending"}
        for z, charge, configuration, pid, ticks, previous in abandoned:
            queue = previous if previous in allowed else "pending"
            cursor = db.execute("""UPDATE config_jobs SET status=?,
                worker_pid=NULL,worker_start_ticks=NULL,claimed_from_status=NULL
                WHERE z=? AND charge=? AND configuration=? AND status='running'
                  AND worker_pid=? AND worker_start_ticks IS ?""",
                (queue, z, charge, configuration, pid, ticks))
            if cursor.rowcount:
                repaired += 1
                species.add((z, charge))
        for z, charge in species:
            update_lines(db, z, charge)
        db.commit()
        return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--refresh-seconds", type=int, default=300)
    parser.add_argument("--min-z", type=int, default=1)
    parser.add_argument("--max-z", type=int, default=20)
    parser.add_argument("--ground-only", action="store_true",
                        help="repair only ground configurations")
    parser.add_argument("--refresh-only", action="store_true",
                        help="refresh the comparison site without claiming jobs")
    parser.add_argument("--max-running-ground", type=int, default=0,
                        help="defer repairs while this many ground jobs run; 0 disables")
    args = parser.parse_args()
    if (args.poll_seconds < 1 or args.refresh_seconds < 1
            or not 1 <= args.min_z <= args.max_z <= 118
            or args.max_running_ground < 0):
        parser.error("poll and refresh intervals must be positive")
    next_refresh = (time.monotonic() if args.refresh_only
                    else time.monotonic() + args.refresh_seconds)
    while True:
        if not args.refresh_only:
            abandoned = recover_abandoned_claims(min_z=args.min_z,
                                                 max_z=args.max_z)
            if abandoned:
                print(f"requeued {abandoned} abandoned Z={args.min_z}–"
                      f"{args.max_z} claims", flush=True)
            staged = stage_failed_jobs(min_z=args.min_z, max_z=args.max_z,
                                       ground_only=args.ground_only)
            if staged:
                print(f"staged {staged} new solver failures in "
                      f"Z={args.min_z}–{args.max_z}", flush=True)
            if repair_count(min_z=args.min_z, max_z=args.max_z) and (
                    args.max_running_ground == 0 or
                    running_ground_count(min_z=args.min_z, max_z=args.max_z)
                    < args.max_running_ground):
                run(OUTPUT, min_z=args.min_z, max_z=args.max_z,
                    recover_running=False, retry_only=True,
                    queue_status="repair_pending", refresh_site_every=25)
        remaining = active_count(min_z=args.min_z, max_z=args.max_z)
        if remaining == 0:
            break
        if time.monotonic() >= next_refresh:
            built = subprocess.run(
                [sys.executable, str(ROOT / "scripts/compare_theory_nist.py")],
                cwd=ROOT, check=False, capture_output=True, text=True)
            print(f"periodic site refresh exit={built.returncode}; "
                  f"{remaining} scoped jobs active", flush=True)
            if built.returncode:
                print(built.stderr[-1000:], file=sys.stderr, flush=True)
            next_refresh = time.monotonic() + args.refresh_seconds
        time.sleep(args.poll_seconds)
    built = subprocess.run(
        [sys.executable, str(ROOT / "scripts/compare_theory_nist.py")],
        cwd=ROOT, check=False, capture_output=True, text=True)
    print(f"Z={args.min_z}–{args.max_z} queue drained; "
          f"final site refresh exit={built.returncode}",
          flush=True)
    if built.returncode:
        print(built.stderr[-1000:], file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
