"""Numerical-discretization tiers for AtomX calculations.

A :class:`Sampling` object collects every "how do we discretize this?"
knob into one place — basis size, radial grid, SCF tolerance, max
iterations.  It is *orthogonal* to the physics rung: the same
``Sampling`` can drive ``HartreeFockSlater`` or ``DiracFock(breit=True, qed=True)``.

Four named tiers provide progressively finer discretization settings:
``quick``, ``standard``, ``high``, and ``publication``. They are input presets,
not error bounds. Check radial box/grid/basis refinement separately from SCF
stationarity and from the completeness of the chosen physical model.

For full control, drop into the constructor::

    Sampling(ns=200, ks=9, n_grid=2000, tol=1e-7, max_iter=80)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import math
from atomx.numerics.shooting.adaptive import RadialRefinement


@dataclass(frozen=True)
class Sampling:
    """Numerical-discretization configuration for an atomic calculation.

    Attributes
    ----------
    ns : int or None
        Number of B-spline functions.  ``None`` (default) means
        "auto-select from the DBSR_HF physics grid" — the recommended
        choice for the B-spline engine
        (``Atom(10, engine='bspline').solve()``; the bare default now
        resolves to the shooting engine).
    ks : int
        B-spline polynomial order.  Default 9, matching DBSR_HF.
    n_grid : int
        Number of radial grid points for the ODE-shooting backend
        (``Atom(10, engine='shooting')`` and the perturbative
        Breit-Pauli rung).
    r_max : float
        Outer radius of the radial box, in Bohr.
    tol : float
        Solver stopping tolerance. Direct DF combines relative energy
        change with orbital-update and occupied-stationarity checks;
        this is not a bound on total-energy or experimental error.
    max_iter : int
        SCF iteration cap. Direct Hamiltonian DF applies it separately
        to initialization and coherent radial sweeps. Legacy fixed-ensemble
        B-spline calculations with f shells retain a floor of 120.
    starting_potential : {None, "tfda", "gsz"}
        Approximate potential for the SCF initial guess.  ``None``
        (default) lets the solver pick — TFDA is auto-selected for
        Z ≥ 20 in the current code.
    label : str
        Free-form name carried through for logging / debugging.
        Set automatically on the named tiers.
    radial_refinement : RadialRefinement or None
        Explicit grid/box convergence checks for single-configuration
        shooting. With a policy, ``n_grid`` is the starting size and
        additional full SCF solves compare total, orbital and excitation
        energies. Automatic boxes can grow; explicitly set boxes stay fixed.
        The default ``None`` performs one calculation at the selected size.
    orbital_tolerance : float or None
        Independent dimensionless L2 orbital-update threshold for iterative DF.
        Supported by HF shooting and iterative B-spline DF, including
        Hamiltonian averaging. Other methods reject it. ``None`` retains
        each solver's existing stopping criteria. This is not an orbital
        error bound; radial refinement must be checked separately.
    """

    ns: Optional[int] = None
    ks: int = 9
    n_grid: int = 1000
    r_max: float = 75.0
    tol: float = 1e-5
    max_iter: int = 40
    starting_potential: Optional[str] = None
    label: str = "custom"
    radial_refinement: Optional[RadialRefinement] = None
    orbital_tolerance: Optional[float] = None

    def __post_init__(self):
        if self.orbital_tolerance is not None and (
                not math.isfinite(self.orbital_tolerance) or self.orbital_tolerance <= 0):
            raise ValueError('orbital_tolerance must be finite and positive')
        if self.radial_refinement is not None and not isinstance(self.radial_refinement, RadialRefinement):
            raise TypeError('radial_refinement must be RadialRefinement or None')
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
#     ne = Atom(10, basis=Basis(sampling=Sampling.standard)).solve()
#
# ``Sampling`` is frozen; assigning to its class is a one-time bootstrap
# and cannot be re-done by callers.
Sampling.quick = quick  # type: ignore[attr-defined]
Sampling.standard = standard  # type: ignore[attr-defined]
Sampling.high = high  # type: ignore[attr-defined]
Sampling.publication = publication  # type: ignore[attr-defined]


__all__ = ["Sampling", "quick", "standard", "high", "publication"]
