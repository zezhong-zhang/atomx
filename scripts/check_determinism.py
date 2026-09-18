#!/usr/bin/env python
"""Numerical reproducibility / bounds-safety gate for atomx.

Two failure modes motivated this script, both of which shipped:

1. An out-of-bounds read inside a numba kernel
   (``convol._convol_case2_mixed`` indexed a 112-row density at row 113,
   supplied by ``num_space = max(nsp, nsq)``).  With bounds checking off the
   read returns whatever the neighbouring heap holds: usually a finite float
   — which is then multiplied by a structurally zero ``rkb`` row, so the
   integral silently survives — and occasionally ``nan``/``inf``, which
   poisons the whole integral.  The result was a *nondeterministic* NaN:
   identical code and inputs, ~30 % of fresh processes dead, 70 % fine.
   ``NUMBA_BOUNDSCHECK=1`` turns that into a deterministic ``IndexError``.

2. Silent cross-run drift in the SCF/CI numbers, which a value-identical
   digest across fresh interpreter processes detects.

Usage
-----
    python scripts/check_determinism.py                 # run the panel, print values
    python scripts/check_determinism.py --repeat 8      # 8 fresh processes, compare digests
    python scripts/check_determinism.py --repeat 4 --boundscheck
    python scripts/check_determinism.py --panel quick

Exit status is non-zero on any digest mismatch, non-finite result, or
(numba) bounds-check violation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Panels are dicts label -> zero-argument callable returning a float.
PANELS = ("quick", "full")


def _panel(name: str):
    from atomx import Atom

    def ne_bspline() -> float:
        atom = Atom(Z=10, numerics="bspline")
        atom.solve()
        return float(atom.total_energy)

    def ne_shooting() -> float:
        atom = Atom(Z=10, numerics="shooting")
        atom.solve()
        return float(atom.total_energy)

    def be_mcdhf_ci() -> float:
        atom = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
        atom.solve()
        return float(atom.total_energy)

    def fe17_dfs_local_exchange() -> float:
        atom = Atom(Z=26, charge=16, config="1s2 2s2 2p6",
                    numerics="bspline", exchange="gu")
        atom.solve()
        return float(atom.total_energy)

    def ne_bspline_breit_qed() -> float:
        atom = Atom(Z=10, numerics="bspline", breit=True, qed=True)
        atom.solve()
        return float(atom.total_energy)

    quick = {
        # bspline DHF: F^k/G^k cell integrals + _density_sym/_convol_case2_mixed
        "Ne-bspline-dhf": ne_bspline,
        # MCDHF + CI: _SLATER_MEMO id-keyed integral cache, CI eigensolve
        "Be-mcdhf-ci": be_mcdhf_ci,
        # shooting engine: SLAG6/Y^k path (independent of the cell algorithm)
        "Ne-shooting-dhf": ne_shooting,
    }
    full = dict(quick)
    full.update({
        # DFS local exchange: _convol_case1_mixed + accumulated densities
        "Fe17-bspline-gu": fe17_dfs_local_exchange,
        # post-SCF correction stack on the bspline grid
        "Ne-bspline-breit-qed": ne_bspline_breit_qed,
    })
    return quick if name == "quick" else full


def _emit(panel: str) -> int:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    for label, fn in _panel(panel).items():
        value = fn()
        if value != value or value in (float("inf"), float("-inf")):
            print(f"NON-FINITE {label} = {value!r}", file=sys.stderr)
            return 3
        # The marker keeps the digest robust against third-party chatter on
        # stdout (matplotlib prints a font-cache notice on a cold cache).
        print(f"PANEL {label} {value!r}")
    return 0


def _spawn(panel: str, boundscheck: bool, extra_env: dict | None = None) -> tuple[int, str, str]:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="atomx-mpl-"))
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if boundscheck:
        # Bounds checking is a *compile-time* numba flag: it must go in the
        # environment before atomx is imported, and the cached artifacts must
        # not be shared with normal runs (a boundschecked kernel served from
        # the default cache would also slow every later run).
        env["NUMBA_BOUNDSCHECK"] = "1"
        env["NUMBA_FULL_TRACEBACKS"] = "1"
        env["NUMBA_CACHE_DIR"] = str(
            Path(tempfile.gettempdir()) / "atomx-numba-boundscheck")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--emit", "--panel", panel],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _digest(text: str) -> str:
    """Digest of the panel lines only, ignoring unrelated stdout chatter."""
    lines = [ln for ln in text.splitlines() if ln.startswith("PANEL ")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, default="quick")
    parser.add_argument("--repeat", type=int, default=0,
                        help="run this many fresh interpreter processes and "
                             "compare digests (0 = single in-process run)")
    parser.add_argument("--boundscheck", action="store_true",
                        help="run children with NUMBA_BOUNDSCHECK=1")
    parser.add_argument("--emit", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.emit:
        return _emit(args.panel)

    if args.repeat <= 0:
        return _emit(args.panel)

    digests: set[str] = set()
    for run in range(args.repeat):
        code, out, err = _spawn(args.panel, args.boundscheck)
        if code != 0:
            print(f"run {run + 1}/{args.repeat} FAILED (exit {code})",
                  file=sys.stderr)
            tail = "\n".join((out + err).strip().splitlines()[-25:])
            print(tail, file=sys.stderr)
            return 1
        digests.add(_digest(out))
    if len(digests) != 1:
        print(f"NON-DETERMINISTIC: {len(digests)} distinct results over "
              f"{args.repeat} fresh processes", file=sys.stderr)
        first = _spawn(args.panel, args.boundscheck)[1]
        print("example output:\n" + first, file=sys.stderr)
        return 2
    detail = " (boundscheck on)" if args.boundscheck else ""
    print(f"OK: {args.repeat} fresh processes, digest {digests.pop()}{detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
