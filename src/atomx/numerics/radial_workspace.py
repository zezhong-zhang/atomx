"""Radial representation evaluation shared by operators and observables.

Every Dirac-based correction (Uehling VP, Wichmann–Kroll VP, the
self-energy backends, Breit) reduces to integrals of the form

    ∫ f(P(r), Q(r), r) dr  ≈  Σ_i w_i · f(P_i, Q_i, r_i)

The physics modules historically unpacked a B-spline workspace dict
(``gr``/``grw`` quadrature nodes + ``bsp_p``/``bsp_q`` basis tables) and
evaluated the coefficient vectors on the quadrature grid.  The ODE
shooting engine produces the SAME ingredients directly: the radial grid
``r``, the SLAG6 integration weights (the index Jacobian times rule weights),
and per-orbital ``(P, Q)`` arrays already tabulated on the grid.

This module is the minimal seam that lets one physics implementation
serve both flavours:

* :func:`make_grid_workspace` wraps ``(r, w)`` arrays as a workspace;
* :func:`is_grid_workspace` distinguishes the two flavours;
* :func:`workspace_arrays` returns the radial nodes and weights;
* :func:`eval_orbital_pair` returns ``(P(r), Q(r))`` on the workspace
  grid — a B-spline expansion evaluation for the B-spline flavour, a
  no-op for the grid flavour.

The returned arrays keep each flavour's native shape (2D ``(nv,
n_gauss)`` for B-spline, 1D for grids); every consumer combines them
element-wise and contracts with ``np.sum(w * ...)``, so the shape never
leaks.
"""

from __future__ import annotations

import numpy as np

#: Marker key/value identifying a raw radial-grid workspace.
_GRID_KIND = "radial_grid"


def make_grid_workspace(r, w):
    """Wrap a radial grid + integration weights as a correction workspace.

    Parameters
    ----------
    r : ndarray
        Radial nodes in Bohr, strictly positive and increasing.  Callers
        holding a grid that starts at ``r = 0`` (e.g. Salvat's SGRID)
        must slice off the origin point first — every physical density
        ``P² + Q²`` vanishes there, but potentials like ``-Z/r`` and the
        Uehling kernel do not, and ``0 · inf`` would poison the sums.
    w : ndarray
        Integration weights on ``r``: ``∫ f dr ≈ sum(f(r) * w)``.
        For shooting, pass SLAG6 weights times the grid Jacobian, not the
        Jacobian alone. Compute endpoint weights before slicing the origin.
    """
    r = np.asarray(r, dtype=float)
    w = np.asarray(w, dtype=float)
    if r.ndim != 1 or r.shape != w.shape:
        raise ValueError(
            f"grid workspace needs matching 1D r/w arrays "
            f"(got shapes {r.shape} and {w.shape})")
    if r.size and r[0] <= 0.0:
        raise ValueError(
            "grid workspace radial nodes must be strictly positive "
            "(slice off the r = 0 origin point)")
    return {"kind": _GRID_KIND, "gr": r, "grw": w}


def is_grid_workspace(data) -> bool:
    """True iff ``data`` is a raw radial-grid workspace (not B-spline)."""
    return isinstance(data, dict) and data.get("kind") == _GRID_KIND


def workspace_arrays(data):
    """Radial nodes and integration weights of either workspace flavour.

    Returns ``(gr, grw)`` with ``∫ f dr ≈ np.sum(grw * f)``.  Shapes are
    flavour-native: 2D ``(nv, n_gauss)`` for B-spline, 1D for grids.
    """
    return data["gr"], data["grw"]


def eval_orbital_pair(P, Q, data):
    """Orbital components tabulated on the workspace's radial nodes.

    B-spline flavour: ``P``/``Q`` are expansion coefficient vectors and
    are evaluated through the (kinetic-balance-aware) basis tables.
    Grid flavour: ``P``/``Q`` are already the tabulated values — pass
    through unchanged.
    """
    if is_grid_workspace(data):
        return np.asarray(P, dtype=float), np.asarray(Q, dtype=float)

    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

    nv = data["nv"]
    ns = data["ns"]
    ksp = data.get("ksp", data["ks"])
    ksq = data.get("ksq", data["ks"])
    nsp = data.get("nsp", ns)
    nsq = data.get("nsq", ns)
    bsp_p = data["bsp_p"]
    bsp_q = data.get("bsp_q", bsp_p)
    fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
    fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
    return fP, fQ


__all__ = [
    "make_grid_workspace",
    "is_grid_workspace",
    "workspace_arrays",
    "eval_orbital_pair",
]
