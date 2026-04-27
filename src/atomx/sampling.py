"""Numerical-discretization tiers for AtomX calculations.

A :class:`Sampling` object collects every "how do we discretize this?"
knob into one place — basis size, radial grid, SCF tolerance, max
iterations.  It is *orthogonal* to the physics rung: the same
``Sampling`` can drive ``HartreeFock`` or ``DiracFock(breit=True, qed=True)``.

Four named tiers cover most use cases:

============= =============================================================
Tier          What it guarantees
============= =============================================================
``quick``     Fast scoping pass.  ε_1s within ~0.5 eV of ``standard`` for
              Z ≤ 92, ~10× faster.  Use to iterate on configuration, not
              for publication numbers.
``standard``  Default.  Matches DBSR_HF / Fortran DHFS to ~3·10⁻⁷ relative
              on the Z = 2..92 calibration panel (neutral atoms).  The
              B-spline backend uses the auto-selected DBSR_HF physics
              grid (``ns=None``).
``high``      Total-energy convergence to 10⁻⁷ on Z = 2..92.  ~3× slower
              than ``standard``.
``publication`` Tight tolerance for results that must be defended in
              writing.  ~10× slower than ``standard``.
============= =============================================================

For full control, drop into the constructor::

    Sampling(ns=200, ks=9, n_grid=2000, tol=1e-7, max_iter=80)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Sampling:
    """Numerical-discretization configuration for an atomic calculation.

    Attributes
    ----------
    ns : int or None
        Number of B-spline functions.  ``None`` (default) means
        "auto-select from the DBSR_HF physics grid" — the recommended
        choice and what ``Atom(Z).solve()`` does today.
    ks : int
        B-spline polynomial order.  Default 9, matching DBSR_HF.
    n_grid : int
        Number of radial grid points for the ODE-shooting backend
        (``Atom(exchange='shooting' | 'fac')`` and the perturbative
        Breit-Pauli rung).
    r_max : float
        Outer radius of the radial box, in Bohr.
    tol : float
        SCF convergence tolerance — the maximum allowed change in any
        orbital eigenvalue between successive iterations, in Hartree.
    max_iter : int
        SCF iteration cap.  Lanthanides / actinides automatically get
        an internal floor of 120 in the current B-spline solver (see
        ``atomx/atom.py``); this is the cap used everywhere else.
    starting_potential : {None, "tfda", "gsz"}
        Approximate potential for the SCF initial guess.  ``None``
        (default) lets the solver pick — TFDA is auto-selected for
        Z ≥ 20 in the current code.
    label : str
        Free-form name carried through for logging / debugging.
        Set automatically on the named tiers.
    """

    ns: Optional[int] = None
    ks: int = 9
    n_grid: int = 1000
    r_max: float = 75.0
    tol: float = 1e-5
    max_iter: int = 40
    starting_potential: Optional[str] = None
    label: str = "custom"

    def __post_init__(self):
        if self.starting_potential not in (None, "tfda", "gsz"):
            raise ValueError(
                f"starting_potential must be None, 'tfda', or 'gsz' "
                f"(got {self.starting_potential!r})"
            )
        if self.ks < 4:
            raise ValueError(f"ks must be >= 4 (got {self.ks})")
        if self.n_grid < 50:
            raise ValueError(f"n_grid must be >= 50 (got {self.n_grid})")
        if self.tol <= 0 or self.max_iter < 1:
            raise ValueError(
                f"tol > 0 and max_iter >= 1 required "
                f"(got tol={self.tol}, max_iter={self.max_iter})"
            )

    def with_(self, **changes) -> "Sampling":
        """Return a copy with the listed fields updated.

        ``label`` is reset to ``"custom"`` unless explicitly overridden.
        """
        if "label" not in changes:
            changes["label"] = "custom"
        return replace(self, **changes)

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return (
            f"Sampling({self.label}: ns={self.ns}, ks={self.ks}, "
            f"n_grid={self.n_grid}, tol={self.tol:g})"
        )


# ---------------------------------------------------------------------------
# Named tiers
# ---------------------------------------------------------------------------

#: Fast scoping pass.  See :mod:`atomx.sampling` module docstring.
quick = Sampling(
    ns=60, ks=9, n_grid=400, r_max=60.0,
    tol=1e-4, max_iter=40,
    label="quick",
)

#: Default tier — ``ns=None`` means auto-select the DBSR_HF physics grid.
standard = Sampling(
    ns=None, ks=9, n_grid=1000, r_max=75.0,
    tol=1e-5, max_iter=40,
    label="standard",
)

#: Tight tolerance for total-energy convergence to ~10⁻⁷.
high = Sampling(
    ns=200, ks=9, n_grid=2000, r_max=75.0,
    tol=1e-7, max_iter=80,
    label="high",
)

#: Publication-grade tolerance.  Slow.
publication = Sampling(
    ns=300, ks=9, n_grid=4000, r_max=100.0,
    tol=1e-9, max_iter=200,
    label="publication",
)


# Expose the tiers as class-level attributes so ``Sampling.standard``
# is the natural call site:
#
#     ne = Atom(Z=10, sampling=Sampling.standard).solve()
#
# ``Sampling`` is frozen; assigning to its class is a one-time bootstrap
# and cannot be re-done by callers.
Sampling.quick = quick  # type: ignore[attr-defined]
Sampling.standard = standard  # type: ignore[attr-defined]
Sampling.high = high  # type: ignore[attr-defined]
Sampling.publication = publication  # type: ignore[attr-defined]


__all__ = ["Sampling", "quick", "standard", "high", "publication"]
