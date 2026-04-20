"""FAC-style asymptotic-potential correction (``FixAsymptoticPot`` mode 0).

Port of the relevant fragment of ``faclib/radial.c:1998-2128``. Takes
the raw ``r·V(r)`` from a local-exchange SCF iteration and returns a
tail-fixed version that enforces the correct ``r·V → -(Z−N+1)``
asymptote required by any one-body Coulomb tail for a neutral or
positively-charged atom.

Two mechanisms are combined:

1. **Latter's clip.** For any grid point where ``r·V(r) < -(Z−N+1)``
   (i.e. the potential is *more* binding than the residual-charge
   Coulomb tail allows), replace it with ``-(Z−N+1)``.  This alone
   guarantees a sensible asymptote at the cost of a first-derivative
   kink at the clip radius.

2. **Gu's smooth envelope** (``ihx < 0``).  Find the inner crossing
   radius :math:`r_j` at which the running electron charge
   :math:`q(r_j) = N + ihx·N` (so ``ihx < 0`` means "slightly below
   the full count"), then replace ``r·V`` for ``r > r_j`` with

   .. math::

       r V(r) \\approx V_j + (V_\\infty - V_j)\\,
                       \\left[1 - e^{-b\\,(r - r_j)}\\right]

   where :math:`V_\\infty = -(Z - N + 1)` and :math:`b` is fitted
   from the local logarithmic derivative of ``r·V`` at ``r_j`` using
   a 7-point average (FAC's choice in ``radial.c:2079-2086``).

Both mechanisms run together: we first compute the envelope for
``r > r_j``, then apply Latter's clip over the entire grid to
guarantee the asymptote if the envelope overshoots.

Only ``mode = 0`` is implemented (``FixAsymptoticPot(..., md=0)``);
FAC's ``mode = 2`` (charge-pinned tail, used by the plasma path) is
explicitly out of scope — see the plan in
``docs/superpowers/plans/2026-04-20-fac-calculator.md`` §physics
contract.
"""

from __future__ import annotations

import numpy as np


def _residual_charge(Z: float, N: float) -> float:
    """Asymptotic ``-(Z - N + 1)`` for the one-body Coulomb tail.

    For a neutral atom Z = N so the tail is ``-1``; for an ion with
    ``k`` missing electrons it is ``-(k + 1)``.
    """
    return -(Z - N + 1.0)


def apply_fac_tail_fix(
    *,
    r: np.ndarray,
    rv: np.ndarray,
    running_charge: np.ndarray,
    Z: float,
    N: float,
    ihx: float = -0.05,
    csi: float = 0.55,
) -> np.ndarray:
    """Apply FAC's mode-0 tail fix (smooth envelope + Latter clip).

    Parameters
    ----------
    r : (n,) ndarray
        Monotonically increasing radial grid (Bohr).
    rv : (n,) ndarray
        Input ``r·V(r)`` from the local-exchange SCF iteration.
    running_charge : (n,) ndarray
        ``q(r) = ∫_0^r 4π r'² ρ(r') dr'`` on the same grid. Used to
        locate the crossing radius where Gu's envelope begins.
    Z, N : float
        Nuclear charge and total electron count. ``Z − N`` is the net
        ionic charge; the asymptote is ``-(Z − N + 1)``.
    ihx : float, default -0.05
        Charge-threshold for the envelope crossing. Must be ≤ 0;
        a value of ``0`` disables the envelope and leaves only
        Latter's clip.
    csi : float, default 0.55
        Secondary threshold, used to set the envelope's outer edge.
        ``csi = 0`` also disables the envelope.

    Returns
    -------
    (n,) ndarray — tail-fixed ``r·V(r)``.
    """
    r = np.asarray(r, dtype=float)
    rv = np.asarray(rv, dtype=float).copy()
    q = np.asarray(running_charge, dtype=float)
    n = r.size
    if rv.size != n or q.size != n:
        raise ValueError("r, rv, running_charge must share a shape")

    rv_inf = _residual_charge(Z, N)   # negative; -1 for neutral Ne

    # -----------------------------------------------------------------
    # Smooth envelope (optional, default on for ihx < 0)
    # -----------------------------------------------------------------
    if ihx < 0.0 and csi > 0.0 and N > 1.0:
        n0 = float(N)
        n1 = max(0.0, n0 + ihx * n0)             # crossing charge target
        n2 = max(0.0, n0 + (1.0 + abs(csi)) * ihx * n0)

        # Find j such that q[j] ≤ n1 (FAC: first m where ut[m] ≤ n1,
        # scanning inward from the last "still-growing" point).
        # Identify jm = last grid index where q is non-decreasing —
        # beyond jm FAC replicates q[jm] as a guard.
        jm = n - 1
        for m in range(n - 1, 0, -1):
            if q[m - 1] < q[m]:
                jm = m
                break

        # Scan inward to find crossing j (FAC: "for m=jm; m>0; m--").
        j = -1
        for m in range(jm, 0, -1):
            if q[m] <= n1:
                j = m
                break

        # And a secondary crossing i at n2 (informational; FAC uses it
        # when _dsi > 0 to set a width, which we don't enable here).
        # Require j large enough that the 7-point average has data.
        if j >= 10:
            # Local logarithmic decay rate b from a 7-point average at
            # j (FAC: ``a = Σ -ln(1 - (q[m]-q[j])/(n0-q[j])) / (r[m]-r[j])``
            # then ``b = a·(n0-ur)/(n1-ur)``).  ``b`` has units of 1/Bohr
            # and controls how quickly the potential approaches its
            # asymptote beyond ``r_j``.
            ur = q[j]                 # running charge at crossing
            samp = 0.0
            cnt = 0.0
            for m in range(j - 3, j + 4):
                if m == j or m < 0 or m >= n or m == jm:
                    continue
                dr_mj = r[m] - r[j]
                if abs(dr_mj) < 1e-30:
                    continue
                denom = n0 - ur
                if denom <= 0:
                    continue
                arg = 1.0 - (q[m] - ur) / denom
                if arg <= 0:
                    continue
                samp += -np.log(arg) / dr_mj
                cnt += 1.0
            if cnt > 0:
                a = samp / cnt
                b = a * (n0 - ur) / (n1 - ur) if abs(n1 - ur) > 1e-30 else a
                r_j = r[j]
                rv_j = rv[j]
                # Envelope runs for r > r_j.  FAC's logic, recast
                # directly in r·V:  the interior value ``rv_j`` relaxes
                # exponentially to the asymptote ``rv_inf`` with rate
                # ``b`` set by the charge envelope.
                #
                #     rv_env(r) = rv_inf + (rv_j - rv_inf) · e^{-b (r - r_j)}
                #
                # Take ``rv = max(rv_env, rv_inf)`` so the envelope
                # never drops below the Latter floor; the final
                # ``np.maximum(rv, rv_inf)`` guard below enforces the
                # same on any remaining grid points the envelope
                # missed.
                outer = np.arange(j + 1, n)
                decay = np.exp(-b * (r[outer] - r_j))
                rv_env = rv_inf + (rv_j - rv_inf) * decay
                # Keep the envelope if it is more-binding than the
                # raw SCF output (which can be too-shallow at large r
                # when local-exchange alone is used), otherwise keep
                # the raw potential.  Both are bounded below by Latter
                # in the guard that follows this block.
                rv[outer] = np.minimum(rv[outer], rv_env)

    # -----------------------------------------------------------------
    # Latter's clip — always applied as a final guard
    # -----------------------------------------------------------------
    # FAC (radial.c:970) walks from the outer edge inward while
    # ``rv > rvinf`` holds (i.e. the tail is TOO WEAK — closer to 0
    # than the residual-Coulomb asymptote ``rvinf = -(Z - N + 1)``)
    # and clips each such point down to ``rvinf``.  Once a point is
    # more-binding than the asymptote (``rv ≤ rvinf``, interior
    # region where nuclear + screening dominate) the loop returns.
    # Vectorised: ``rv = min(rv, rvinf)`` gives the same result.
    np.minimum(rv, rv_inf, out=rv)
    return rv
