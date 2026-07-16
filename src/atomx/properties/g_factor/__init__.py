"""Bound-electron g-factor (gyromagnetic ratio).

The bound-electron g-factor is one of the most precisely measured
atomic quantities; combined with theory it gives the most accurate
determination of the electron mass.

This module provides three layers:

1. **Closed-form Dirac (Breit-Margolis) g-factor for hydrogenic
   bound states** — exact relativistic formula in a point-Coulomb
   potential.
2. **SCF orbital g-factor** — for many-electron atoms, computed
   from the relativistic radial integral of the converged Dirac
   orbital.
3. **Leading-order corrections** — Schwinger anomalous magnetic
   moment, leading-order Zα QED, and finite-nuclear-size
   contributions, summed.

For high-precision (>10⁻⁸) work the higher-order Grotch / Pachucki /
Yerokhin corrections are needed; this module only supplies the
leading terms.

References
----------
- B. Margolis, *Phys. Rev.* **156**, 23 (1967).
- T. Beier, *Phys. Rep.* **339**, 79 (2000) — review of bound-
  electron g-factors and QED corrections.
- J. Cheng, W. J. Childs, *Phys. Rev. A* **31**, 2775 (1985).
- V. A. Yerokhin, P. Indelicato, V. M. Shabaev, *Phys. Rev. A*
  **69**, 052503 (2004) — two-loop QED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from atomx.constants import ALPHA
from atomx.numerics.bspline.evaluate import eval_P, eval_Q


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GFactorResult:
    """Bound-electron g-factor breakdown."""

    g_total: float
    g_dirac: float = 0.0
    g_qed: float = 0.0
    g_fns: float = 0.0
    label: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hydrogenic closed-form (Breit-Margolis)
# ---------------------------------------------------------------------------


def g_factor_dirac_hydrogenic(Z: int, n: int, kappa: int) -> float:
    """Closed-form Dirac g-factor for a hydrogenic bound state.

    Margolis (1967) Eq. 23, simplified for point Coulomb:

        g_{nκ} = (κ / [j(j+1)]) · [κ·ε − 1/2]

    with the Sommerfeld energy ε(n,κ) = [1 + (Zα)² / (n_r + γ)²]^{-1/2}
    where n_r = n − |κ| and γ = √(κ² − (Zα)²).

    For κ = -1 (s_{1/2}), this reduces to the Breit (1928) formula
    g = (2/3)·(1 + 2√(1 − (Zα)²)).
    """
    j2 = 2 * abs(kappa) - 1
    j = j2 / 2.0
    if abs(kappa) > n or kappa == 0:
        raise ValueError(f"Forbidden quantum numbers (n={n}, κ={kappa}).")

    Zalpha = Z * ALPHA
    gamma = np.sqrt(kappa * kappa - Zalpha * Zalpha)
    n_r = n - abs(kappa)
    eps = 1.0 / np.sqrt(1.0 + (Zalpha / (n_r + gamma)) ** 2)
    return (kappa / (j * (j + 1.0))) * (kappa * eps - 0.5)


# ---------------------------------------------------------------------------
# Leading QED corrections
# ---------------------------------------------------------------------------


def g_factor_qed(Z: int, kappa: int) -> float:
    """Leading-order QED corrections: Schwinger + first-order bound state.

    Δg_QED = α/π · [1 + (Zα)²/6]

    (s-state Grotch leading-order; higher orders not included.)
    """
    Zalpha = Z * ALPHA
    a_e = ALPHA / np.pi
    return a_e * (1.0 + (Zalpha ** 2) / 6.0)


# ---------------------------------------------------------------------------
# Finite nuclear size (leading Glazov-Shabaev)
# ---------------------------------------------------------------------------


def g_factor_finite_nuclear_size(
    Z: int,
    kappa: int,
    R_rms_fm: float,
) -> float:
    """Leading FNS correction (s₁/₂ only).

    Δg_FNS ≈ -(8/3)·(Zα)⁴·(R/a₀)² · g_NR  with g_NR ≈ 2.

    Higher-Z finite-nuclear-size enhancement F(Zα) is not included.
    Returns 0 for κ ≠ -1.
    """
    if kappa != -1:
        return 0.0
    a0_fm = 5.29177210903e4
    R_a0 = R_rms_fm / a0_fm
    Zalpha = Z * ALPHA
    return -(8.0 / 3.0) * (Zalpha ** 4) * (R_a0 ** 2) * 2.0


# ---------------------------------------------------------------------------
# SCF orbital g-factor (B-spline radial integral)
# ---------------------------------------------------------------------------


def g_factor_orbital(scf_result, label: str) -> float:
    """Per-orbital Dirac g-factor, computed from the SCF radial spinor.

    Margolis (1967) Eq. 23 in the operator form:

        g_κ = (κ/(j(j+1))) · [ κ · ⟨P² − Q²⟩ − 1/2 ]

    where ⟨·⟩ is the radial integral on the SCF orbital and the
    spinor is normalised to ∫(P² + Q²) dr = 1.  The ⟨P² − Q²⟩
    factor is the Dirac analogue of the Sommerfeld energy ε in the
    closed-form hydrogenic expression and reduces to it for pure
    point-Coulomb orbitals.

    For point-Coulomb (no screening) this returns the Breit-Margolis
    closed-form value; for SCF orbitals it captures leading "wave-
    function shielding" corrections.  Many-body / polarisation
    contributions beyond the single-orbital level are NOT included
    here — see :func:`atomx.methods.mbpt.mbpt2_energy` for the
    correlation infrastructure on which a polarised g-factor would
    build.
    """
    matched = [s for s in scf_result.subshells if s.label == label]
    if not matched:
        raise KeyError(
            f"Subshell {label!r} not found; available: "
            f"{[s.label for s in scf_result.subshells]}"
        )
    sh = matched[0]
    kappa = sh.kappa
    j = (2 * abs(kappa) - 1) / 2.0

    orb_list = scf_result.orbitals_raw.get(kappa, [])
    sh_kappa = [s for s in scf_result.subshells if s.kappa == kappa]
    idx = sh_kappa.index(sh)
    if idx >= len(orb_list):
        raise ValueError(f"No raw orbital stored for {label!r}.")
    P, Q, _ = orb_list[idx]

    bsp_data = scf_result._bsp_data
    grw = bsp_data["grw"]
    P_grid = eval_P(P, bsp_data)
    Q_grid = eval_Q(Q, bsp_data)
    norm = float(np.sum(grw * (P_grid ** 2 + Q_grid ** 2)))
    pq_diff = float(np.sum(grw * (P_grid ** 2 - Q_grid ** 2))) / max(norm, 1e-30)

    # Margolis form: matches the closed-form hydrogenic value at Q² ~ Zα/2 ≪ 1.
    return (kappa / (j * (j + 1.0))) * (kappa * pq_diff - 0.5)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def g_factor(
    scf_result,
    label: str,
    *,
    R_rms_fm: Optional[float] = None,
    include_qed: bool = True,
    include_fns: bool = True,
) -> GFactorResult:
    """Total bound-electron g-factor for a subshell.

    Parameters
    ----------
    scf_result
        Converged SCFResult.
    label
        Subshell label (e.g. ``"1s1/2"``).
    R_rms_fm
        Nuclear RMS charge radius in fm (required if
        ``include_fns=True``).
    include_qed
        Add Schwinger + leading-order bound-state QED.
    include_fns
        Add leading finite-nuclear-size correction (s₁/₂ only).

    Returns
    -------
    GFactorResult
    """
    matched = [s for s in scf_result.subshells if s.label == label]
    if not matched:
        raise KeyError(f"Subshell {label!r} not found.")
    sh = matched[0]
    Z = scf_result.Z

    g_dirac = g_factor_orbital(scf_result, label)
    g_qed_val = g_factor_qed(Z, sh.kappa) if include_qed else 0.0

    g_fns_val = 0.0
    if include_fns:
        if R_rms_fm is None:
            raise ValueError(
                "R_rms_fm required when include_fns=True.  Use "
                "atomx.data.nuclear_structure for isotope radii."
            )
        g_fns_val = g_factor_finite_nuclear_size(Z, sh.kappa, R_rms_fm)

    g_total = g_dirac + g_qed_val + g_fns_val
    return GFactorResult(
        g_total=g_total,
        g_dirac=g_dirac,
        g_qed=g_qed_val,
        g_fns=g_fns_val,
        label=label,
        provenance={
            "Z": Z,
            "kappa": sh.kappa,
            "R_rms_fm": R_rms_fm,
            "include_qed": include_qed,
            "include_fns": include_fns,
        },
    )


__all__ = [
    "GFactorResult",
    "g_factor",
    "g_factor_orbital",
    "g_factor_dirac_hydrogenic",
    "g_factor_qed",
    "g_factor_finite_nuclear_size",
]

