"""Worked example: Ne climbing the standard physics ladder.

Demonstrates the new ``Atom.climb`` / ``Sampling`` API introduced for
HYP-13.  Each rung adds one well-defined physical effect and the
``show()`` table prints the energy delta against the previous rung
along with a one-line description of what that delta represents.

Run with::

    .venv/bin/python docs/gallery/ladder_example.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# When run from a git worktree whose atomx source isn't installed,
# prefer the local src/ over any venv install.
_WORKTREE_SRC = Path(__file__).resolve().parents[2] / "src"
if _WORKTREE_SRC.is_dir() and str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))

from atomx import Atom, Ladder, Sampling                     # noqa: E402
from atomx.methods import DiracFock                          # noqa: E402


def main() -> None:
    print("Ne (Z=10) — climbing the standard physics ladder")
    print()

    ne = Atom(Z=10)

    t0 = time.perf_counter()
    run = ne.climb(through=DiracFock(breit=True, qed=True),
                   sampling=Sampling.quick)
    t = time.perf_counter() - t0

    run.show(unit="eV")
    print()
    print(f"  total wall time: {t:.2f} s  (Sampling.quick)")
    print()
    print("  successful rungs:", len(run.results))
    if run.failed:
        print("  failed rungs:")
        for o in run.failed:
            print(f"    - {o.name}: {o.error}")

    # Same atom, same sampling, but stop at non-relativistic HF — useful
    # when iterating on configuration before paying for the relativistic
    # rungs.
    print()
    print("Stopping early at HartreeFockSlater for a quick sanity check:")
    print()
    from atomx.methods import HartreeFockSlater
    ne.climb(through=HartreeFockSlater(),
             sampling=Sampling.quick).show(unit="eV")


if __name__ == "__main__":
    main()
