"""Welton self-energy estimator.

Welton's heuristic (T. A. Welton, Phys. Rev. 74, 1157 (1948); see also
Bethe & Salpeter §123, Jackson *Classical Electrodynamics* §15.8) pictures
the electron as smeared over a Compton-wavelength-scale radiative
fluctuation.  Averaging the nuclear potential over that smearing yields
the well-known closed-form estimate

    ΔE_a = (α³ Z / (3π)) · ln(1/(Zα)²) · ⟨a | ∇²V_nuc(r) | a⟩

For a point nucleus ``V_nuc = -Z/r`` ⇒ ``∇²V_nuc = 4π Z δ³(r)``, giving

    ΔE_a = (4 α³ Z² / 3) · ln(1/(Zα)²) · |ψ_a(0)|²

The operator ∇²V_nuc is localized at the nucleus — therefore Welton's
estimate is non-zero only for orbitals with appreciable amplitude at the
origin, i.e. the large component of j = 1/2 orbitals (κ = −1 and κ = +1).
Higher-l orbitals receive zero contribution here.

Accuracy: typically good to 20–30 % against exact Mohr values for
hydrogenic systems.  Its main value is that it works for **every** j = 1/2
orbital regardless of (n, l), unlike the Mohr table which only covers
1s and 2s/2p explicitly.  Use it for:

*   Sanity-checking the Mohr / Shabaev backends.
*   Estimating SE of n ≥ 3 s-orbitals where Mohr falls back to hydrogenic
    n^-3 scaling anyway.
*   Rapid Z-dependence scans.

Not recommended for precision calculations — prefer ``"shabaev"`` (when
implemented) for those.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend


def _density_at_origin(P, Q, bsp_data) -> float:
    """Approximate |ψ(0)|² for an orbital on the radial grid.

    Uses the innermost (positive-r) grid point as a proxy for
    ``|P(r)/r|² + |Q(r)/r|²`` extrapolated to ``r → 0``.  For κ = ±1
    orbitals (γ ≈ 1) this is a well-behaved finite limit.  For |κ| ≥ 2
    the large component vanishes at the origin and the proxy correctly
    returns a tiny value.

    The 3D density at the origin is taken to be

        |ψ(0)|² = [ |R_P(0)|² + |R_Q(0)|² ] / (4π)

    where ``R_P = P/r``, ``R_Q = Q/r`` are the radial wavefunctions and the
    ``1/(4π)`` factor comes from the ``|Y_00|²`` angular piece.
    """
    from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays

    gr, _ = workspace_arrays(bsp_data)
    fP, fQ = eval_orbital_pair(P, Q, bsp_data)

    # Flatten to grid order (B-spline arrays are (nv, n_gauss); grid
    # workspaces are already 1D) and take the innermost r > 0 node.
    # Using P/r at that single point gives the dominant-r^γ
    # extrapolation of the Dirac radial wavefunction to r → 0 for
    # γ ≈ 1 orbitals (κ = ±1), which is what Welton needs.
    r_flat = np.ravel(gr)
    i0 = int(np.argmax(r_flat > 0.0))
    r0 = float(r_flat[i0])
    P0 = float(np.ravel(fP)[i0])
    Q0 = float(np.ravel(fQ)[i0])
    RP0_sq = (P0 / r0) ** 2
    RQ0_sq = (Q0 / r0) ** 2

    # |ψ(0)|² = (|R_P|² + |R_Q|²) / (4π)
    return (RP0_sq + RQ0_sq) / (4.0 * np.pi)


@register_backend
class WeltonBackend(SEBackend):
    """Welton (1948) closed-form self-energy heuristic for j = 1/2 orbitals.

    Covers every (n, κ=±1) orbital — extends Mohr's n ≤ 2 coverage to
    arbitrary principal quantum number, at the cost of ~25 % accuracy.
    """

    name = "welton"
    description = (
        "Welton (1948) closed-form estimate: "
        "ΔE_a = (4α³Z²/3) ln(1/(Zα)²) ⟨a|∇²V_nuc|a⟩."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        aZ = ALPHA * Z
        # ln(1/(Zα)²) is only meaningful for Zα < 1 — clamp for heavy Z to
        # keep the estimate tame near the "Dirac plunge" at Z ≈ 137.
        ln_factor = 0.0 if aZ >= 1.0 else np.log(1.0 / (aZ * aZ))

        # ΔE_a = (1/6) × ⟨(δr)²⟩ × ⟨∇²V_nuc⟩_a
        #      = (1/6) × (2α³/π) ln(1/(Zα)²) × 4π Z |ψ_a(0)|²
        #      = (4 α³ Z / 3) × ln(1/(Zα)²) × |ψ_a(0)|²
        # (the Z³ scaling of hydrogenic |ψ(0)|² is already captured
        # numerically by _density_at_origin from the actual orbital).
        pref = (4.0 / 3.0) * ALPHA ** 3 * Z * ln_factor

        per_orbital: dict = {}
        E_SE = 0.0
        for P, Q, occ, _kappa, label in orbitals:
            if bsp_data is None or np.max(np.abs(P)) < 1e-10:
                per_orbital[label] = 0.0
                continue

            psi0_sq = _density_at_origin(P, Q, bsp_data)
            E_a = pref * psi0_sq

            # The Welton heuristic is leading-order only for κ = ±1
            # (j = 1/2).  Higher-|κ| orbitals have ψ(0) that vanishes
            # formally; keep the tiny numerical contribution for
            # completeness but don't scale it up.
            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(E_SE=E_SE, per_orbital=per_orbital, method=self.name)
