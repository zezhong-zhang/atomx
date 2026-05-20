"""R-matrix close-coupling scattering.

Phase 1 of HYP-172 ships the single-channel Wigner R-matrix driver
:func:`compute_single_channel` on a B-spline internal-region basis with
a symmetric Bloch operator (``b = 0``) and surface matching against the
Riccati-Bessel functions. The multi-channel close-coupling driver
:func:`compute` remains a scaffold and raises ``NotImplementedError``
until Phase 2 lands.

References
----------
- P. G. Burke, *R-Matrix Theory of Atomic Collisions* (Springer, 2011).
- K. A. Berrington, W. B. Eissner, P. H. Norrington, *Comput. Phys.
  Commun.* **92**, 290 (1995) (RMATRX-II / DARC).
- O. Zatsarinny and C. Froese Fischer, *Comput. Phys. Commun.* **202**,
  287 (2016) (DBSR_HF; same B-spline conventions reused here).
"""

from __future__ import annotations

from atomx.scattering.r_matrix.coupled_channels import (
    CCResult,
    compute_multichannel,
    extract_resonances,
    r_matrix_excitation_cross_section,
)
from atomx.scattering.r_matrix.solver import RMatrixResult, compute_single_channel
from atomx.scattering.r_matrix.target_states import (
    TargetState,
    heplus_target_states,
)


def compute(*args, **kwargs) -> CCResult:
    """Close-coupling R-matrix solve — aliases :func:`compute_multichannel`.

    Use :func:`compute_single_channel` for the single-open-channel case.
    """
    return compute_multichannel(*args, **kwargs)


__all__ = [
    "CCResult",
    "RMatrixResult",
    "TargetState",
    "compute",
    "compute_multichannel",
    "compute_single_channel",
    "extract_resonances",
    "heplus_target_states",
    "r_matrix_excitation_cross_section",
]
