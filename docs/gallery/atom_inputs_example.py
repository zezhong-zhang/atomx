"""The canonical `Atom` surface: ten arguments, one per physical question.

Solves the same neon atom three ways (plain values, self-describing objects,
and a plasma-screened ion), prints what the constructor resolved, and then
shows five physically impossible combinations with the error each raises.

Run with::

    .venv/bin/python docs/gallery/atom_inputs_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# When run from a git worktree whose atomx source isn't installed,
# prefer the local src/ over any venv install. No-op otherwise.
_WORKTREE_SRC = Path(__file__).resolve().parents[2] / "src"
if _WORKTREE_SRC.is_dir() and str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))

from atomx import (  # noqa: E402
    Atom,
    Basis,
    Corrections,
    Correlation,
    Environment,
    Exchange,
)
from atomx.constants import HARTREE_EV  # noqa: E402


def solve(label: str, atom: Atom) -> Atom:
    atom.solve()
    print(f"{label}")
    print(f"  E_total  = {atom.total_energy * HARTREE_EV:>12.3f} eV "
          f"(= {atom.total_energy:>11.6f} Ha)")
    print(f"  resolved : {atom.resolved_inputs}")
    return atom


if __name__ == "__main__":
    # 1. Plain values: the common case needs no objects at all.
    solve("Neutral neon, plain values (shooting engine, Breit + QED on):",
          Atom(10))

    # 2. The same physics with every question answered by an object.  Each
    #    object carries only the knobs its own physics reads.
    solve("Neon, everything spelled out (B-spline, refined K-shell grid):",
          Atom(10,
               config="1s2 2s2 2p6",
               charge=0,
               nuclear="fermi",
               exchange="hf",
               corrections=Corrections(breit=True, qed=True,
                                       qed_backend="auto"),
               correlation=Correlation(averaging="ci"),
               engine="bspline",
               basis=Basis(ns=None, ks=9, r_max=75.0,
                           grid_hi=0.02, grid_he=0.10),
               environment=None))

    # 3. A plasma is a physical environment, not a label: the Debye length
    #    lambda_D = sqrt(kT / 4 pi n_e) screens the nucleus inside the SCF.
    #    Ne^2+ is used because nuclear screening only makes sense for an ion
    #    (a neutral atom has no net charge left to screen).
    vacuum = solve("Ne^2+, same ion without the plasma (the reference):",
                   Atom(10, config="1s2 2s2", charge=2))
    dense = solve("Ne^2+ in a dense plasma (Debye screening, wired):",
                  Atom(10, config="1s2 2s2", charge=2,
                       environment=Environment(plasma_ne=1e22,
                                               plasma_Te=100.0,
                                               screening="debye")))
    shift = (dense.orbital_energies["1s1/2"]
             - vacuum.orbital_energies["1s1/2"])
    print(f"  every orbital sits {shift:+.4f} Ha higher than in vacuum "
          f"(~Z/lambda_D = {10 / 14.05:.4f} Ha): the ionisation potential "
          f"depression.")

    print()
    print("Rejected combinations (each names the argument and the way out):")
    for label, build in (
        ("a retired keyword",
         lambda: Atom(10, numerics="bspline")),
        ("alpha_x outside Exchange(model='slater')",
         lambda: Atom(10, exchange=Exchange(model="hf", alpha_x=2 / 3))),
        ("grid_* with the shooting engine",
         lambda: Atom(10, engine="shooting", basis=Basis(grid_hi=0.02))),
        ("plasma_Te without plasma_ne",
         lambda: Atom(10, environment=Environment(plasma_Te=10.0))),
        ("slater_method on the B-spline engine",
         lambda: Atom(10, engine="bspline",
                      exchange=Exchange(model="slater",
                                        slater_method="power_series"))),
    ):
        try:
            build()
        except (TypeError, ValueError) as err:
            print(f"\n  {label}:\n    {err}")
