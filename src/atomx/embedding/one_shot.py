"""One-shot first-principles spectral product (design E3 v1).

Pipeline (every stage inside the two jointly-verified repos, no borrowed
tables anywhere):

    Dirac-Fock SCF (Ni2+ 3d8)            [E1.5: Cowan-verified 0.04%]
      -> free-ion F2/F4, zeta_3d
      -> explicit screening convention   [#26: dd x0.65, zeta x0.4531]
      -> impurity solve (E1 Lehmann)
      -> spectral function A(w) + occupation
      -> impurity-greens@1 payload with full provenance

The L2 CT-QMC cross-benchmark (external TRIQS reference set) is explicitly
open and tracked in the design doc; every quantity produced here already
carries its own anchors (sum rules, Racah ladder, removal-edge identity,
Cowan free-ion values).
"""
from __future__ import annotations

import numpy as np

from atomx.embedding.impurity import (
    ImpuritySpec,
    occupation,
    solve_atomic_limit,
    spectral_function,
)
from atomx.embedding.io import impurity_greens_payload
from atomx.embedding.native_params import dirac_fock_3d_params

# benchmark-consistent screening convention (hyperx ledger #26)
SCREENING_DD = 0.6498
SCREENING_ZETA = 0.4531


def one_shot_spectral(
    *,
    n_electrons: int = 8,
    omega_eV: np.ndarray | None = None,
    eta_eV: float = 0.1,
    temperature_K: float = 300.0,
    screening_dd: float = SCREENING_DD,
    screening_zeta: float = SCREENING_ZETA,
):
    """Dirac-Fock -> screening -> impurity spectrum -> payload."""
    df = dirac_fock_3d_params()
    spec = ImpuritySpec(
        n_electrons=n_electrons,
        slater_eV={
            "F0": 0.409231746031746,  # baricenter convention constant
            "F2": screening_dd * df["F2_eV"],
            "F4": screening_dd * df["F4_eV"],
        },
        zeta_d_eV=screening_zeta * df["zeta_3d_eV"],
        temperature_K=temperature_K,
    )
    sol = solve_atomic_limit(spec)
    omega = omega_eV if omega_eV is not None else np.linspace(-15, 5, 400)
    A, G = spectral_function(sol, omega, eta=eta_eV)
    payload = impurity_greens_payload(
        species="Ni", n_electrons=n_electrons, omega_eV=omega, G=G,
        occupation=occupation(sol), temperature_K=temperature_K,
        backend="atomx:one-shot(DF->screen->Lehmann)", eta_eV=eta_eV,
    )
    payload["provenance"]["df_free_ion_eV"] = {
        "F2": df["F2_eV"], "F4": df["F4_eV"], "zeta_3d": df["zeta_3d_eV"],
    }
    payload["provenance"]["screening"] = {
        "dd": screening_dd, "zeta": screening_zeta,
        "convention": "benchmark-consistent (hyperx ledger #26)",
    }
    return {
        "payload": payload,
        "spectral_function": A,
        "occupation": occupation(sol),
        "evals_N": sol[f"N={n_electrons}"]["evals"],
        "params_used": {
            "F2": screening_dd * df["F2_eV"],
            "F4": screening_dd * df["F4_eV"],
            "zeta_3d": screening_zeta * df["zeta_3d_eV"],
        },
    }
