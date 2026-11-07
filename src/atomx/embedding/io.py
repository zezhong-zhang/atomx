"""Payloads for the embedding module (contract schema ``impurity-greens@1``).

The schema name is reserved on the hyperx interaction-contract side
(docs/plans/Probe_Matter_Interaction_Contract_S7_S8.md). Rules follow the
contract: JSON-serializable, provenance mandatory, no silent defaults.
"""
from __future__ import annotations

import numpy as np

IMPURITY_GREENS_SCHEMA = "impurity-greens@1"


def impurity_greens_payload(
    *,
    species: str,
    n_electrons: int,
    omega_eV: np.ndarray,
    G: np.ndarray,
    occupation: np.ndarray,
    temperature_K: float,
    backend: str,
    eta_eV: float,
) -> dict:
    """Emit an ``impurity-greens@1`` payload from a solver result.

    G has shape (10 spin-orbitals, len(omega)); grids serialize as lists.
    """
    return {
        "schema": IMPURITY_GREENS_SCHEMA,
        "species": species,
        "n_electrons": n_electrons,
        "omega_eV": np.asarray(omega_eV).tolist(),
        "G_orb": [
            {
                "re": np.real(g).tolist(),
                "im": np.imag(g).tolist(),
            }
            for g in np.asarray(G)
        ],
        "occupation_per_spinorbital": np.asarray(occupation).tolist(),
        "provenance": {
            "model": backend,
            "level": "atomic-limit(Hubbard-I)",
            "temperature_K": temperature_K,
            "eta_eV": eta_eV,
            "backend_note": (
                "reference d-shell assembly; CI-native backend tracked as E1.5"
            ),
        },
    }
