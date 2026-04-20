"""Side-by-side: HF vs Slater vs FAC/Gu vs DHFS shooting.

Prints total E, run time, and 1s eigenvalue for neutral Ne using each
of atomx's local-exchange flavours. Run with::

    .venv/bin/python docs/gallery/local_exchange_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# When run from a git worktree whose atomx source isn't installed,
# prefer the local src/ over any venv install. No-op otherwise.
_WORKTREE_SRC = Path(__file__).resolve().parents[2] / "src"
if _WORKTREE_SRC.is_dir() and str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402


def run(flavour: str) -> None:
    t0 = time.perf_counter()
    a = Atom(Z=10, config="1s2 2s2 2p6", exchange=flavour)
    a.solve()
    wall = time.perf_counter() - t0
    e_total = a.total_energy
    eps_1s_raw = a.orbital_energies.get("1s1/2") or a.orbital_energies.get("1s")
    eps_1s = (eps_1s_raw * HARTREE_EV) if eps_1s_raw is not None else 0.0
    print(
        f"  {flavour:>9}: E_total = {e_total*HARTREE_EV:>12.3f} eV "
        f"(= {e_total:>11.4f} Ha)   "
        f"ε_1s = {eps_1s:>10.2f} eV   "
        f"t = {wall:>5.2f} s"
    )


if __name__ == "__main__":
    print("Neon (Z=10, 1s² 2s² 2p⁶):")
    for flavour in ("hf", "shooting", "fac"):
        run(flavour)

    print()
    print("Reference: sfac default (SE+VP on, Breit off):  -3500.282 eV")
    print("Reference: sfac Coulomb-only (Task 0 gate):     -3501.114 eV")
