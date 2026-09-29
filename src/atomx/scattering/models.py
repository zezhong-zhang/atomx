"""Composable model choices for scattering and property calculations.

These dataclasses keep physics choices (potential model, local exchange)
separate from numerical choices (B-spline vs shooting continuum).  Public
process APIs can accept one object instead of growing many process-specific
keyword flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class PotentialModel:
    """Local one-electron potential used for continuum / distorted waves.

    ``kind="bare"`` is the nuclear Coulomb field.  ``kind="static_hartree"``
    builds the residual-core V^N/V^{N-1} potential from an atomic result and
    can optionally include FAC/Gu or Slater local exchange.
    """

    kind: str = "bare"
    include_exchange: bool = False
    exchange: str = "fac"
    alpha_x: float = 2.0 / 3.0
    fac_ihx: float = -1.0
    fac_hxs: float = 1.0
    fac_csi: float = 0.55
    fac_self_interaction_correction: bool = True
    latter_correction: bool = True
    n_radial: int = 4000
    r_max: Optional[float] = None

    @classmethod
    def bare(cls) -> "PotentialModel":
        return cls(kind="bare")

    @classmethod
    def static_hartree(cls, **kwargs) -> "PotentialModel":
        return cls(kind="static_hartree", **kwargs)

    @classmethod
    def static_hartree_exchange(cls, **kwargs) -> "PotentialModel":
        return cls(kind="static_hartree", include_exchange=True, **kwargs)

    def build_callable(
        self,
        scf_result,
        *,
        Z: Optional[float] = None,
        exclude_label: Optional[str] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build a numerical potential callable for radial ODE solvers."""
        kind = self.kind.lower()
        if Z is None:
            Z = float(scf_result.Z)
        if kind == "bare":
            from atomx.potentials.static_field import bare_nuclear_potential

            return bare_nuclear_potential(float(Z))
        if kind in {"static_hartree", "hartree", "vn", "vn-1"}:
            from atomx.potentials.static_field import static_hartree_potential

            return static_hartree_potential(
                scf_result,
                exclude_label=exclude_label,
                n_radial=self.n_radial,
                r_max=self.r_max,
                include_exchange=self.include_exchange,
                exchange=self.exchange,
                alpha_x=self.alpha_x,
                fac_ihx=self.fac_ihx,
                fac_hxs=self.fac_hxs,
                fac_csi=self.fac_csi,
                fac_self_interaction_correction=self.fac_self_interaction_correction,
                latter_correction=self.latter_correction,
            )
        raise ValueError(f"unknown potential model kind {self.kind!r}")


@dataclass(frozen=True)
class ContinuumModel:
    """Continuum-wave numerical model plus its potential choice."""

    method: str = "bspline"
    potential: Optional[PotentialModel] = None
    shooting_n_grid: int = 2000
    shooting_r_max: Optional[float] = None

    @classmethod
    def bspline(cls) -> "ContinuumModel":
        return cls(method="bspline")

    @classmethod
    def shooting(
        cls,
        *,
        potential: Optional[PotentialModel] = None,
        n_grid: int = 2000,
        r_max: Optional[float] = None,
    ) -> "ContinuumModel":
        return cls(
            method="shooting",
            potential=potential,
            shooting_n_grid=n_grid,
            shooting_r_max=r_max,
        )

    @classmethod
    def fac_residual_core(
        cls,
        *,
        n_grid: int = 2000,
        r_max: Optional[float] = None,
    ) -> "ContinuumModel":
        """FAC-like shooting continuum in V^{N-1} + Gu local exchange."""
        return cls.shooting(
            potential=PotentialModel.static_hartree_exchange(exchange="fac"),
            n_grid=n_grid,
            r_max=r_max,
        )


__all__ = ["PotentialModel", "ContinuumModel"]
