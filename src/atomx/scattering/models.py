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
    ``screening="mean"`` scales the entire Hartree density by ``(N-1)/N``;
    ``"full"`` retains it, with optional removal of a named orbital electron
    through ``build_callable(exclude_label=...)``. Mean screening and named
    removal are mutually exclusive.
    ``kind="frozen_scf"`` requires a saved local bound SCF field.
    ``kind="fac_frozen"`` uses that field for Gu results and an explicit
    HF-bound/Gu-local-continuum hybrid for non-Gu results.
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
    screening: str = "full"

    @classmethod
    def bare(cls) -> "PotentialModel":
        return cls(kind="bare")

    @classmethod
    def static_hartree(cls, **kwargs) -> "PotentialModel":
        return cls(kind="static_hartree", **kwargs)

    @classmethod
    def static_hartree_exchange(cls, **kwargs) -> "PotentialModel":
        return cls(kind="static_hartree", include_exchange=True, **kwargs)

    @classmethod
    def frozen_scf(cls) -> "PotentialModel":
        """Reuse a saved local SCF field, requiring a compatible bound result."""
        return cls(kind="frozen_scf", include_exchange=True, exchange="gu")

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

            if self.include_exchange:
                raise ValueError("a bare Coulomb field cannot include local exchange")
            return bare_nuclear_potential(float(Z))
        if kind in {"frozen_scf", "fac_frozen"}:
            from atomx.potentials.frozen_field import frozen_scf_potential

            if exclude_label is not None or self.screening != "full":
                raise ValueError(
                    "a frozen SCF field cannot remove an electron or rescale density"
                )
            if float(Z) != float(scf_result.Z):
                raise ValueError("a frozen SCF field cannot change nuclear charge")
            snapshot = getattr(scf_result, "_local_central_field", None)
            spec = getattr(scf_result, "calculation_spec", None)
            gu_reference = spec is not None and spec.physics.get("exchange") == "gu"
            if kind == "frozen_scf" or snapshot is not None or gu_reference:
                if (
                    snapshot is not None
                    and kind == "fac_frozen"
                    and (
                        self.fac_ihx != snapshot.fac_ihx
                        or self.fac_hxs != snapshot.fac_hxs
                        or self.fac_csi != snapshot.fac_csi
                        or not self.fac_self_interaction_correction
                        or not self.latter_correction
                        or not self.include_exchange
                        or self.exchange.lower() not in {"fac", "gu"}
                    )
                ):
                    raise ValueError(
                        "requested Gu parameters differ from the saved SCF field; use frozen_scf or an explicit rebuilt potential"
                    )
                return frozen_scf_potential(scf_result)
            # HF has no local SCF field. This is the explicitly documented
            # HF-bound/Gu-continuum hybrid, constructed from its full density.
            kind = "static_hartree"
        if kind in {"static_hartree", "hartree", "vn", "vn-1"}:
            from atomx.potentials.static_field import static_hartree_potential

            potential = static_hartree_potential(
                scf_result,
                exclude_label=exclude_label,
                screening=self.screening,
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
            potential.origin = "density_reconstruction"
            if self.kind.lower() == "fac_frozen":
                spec = getattr(scf_result, "calculation_spec", None)
                bound_exchange = (
                    spec.physics.get("exchange", "unknown")
                    if spec is not None
                    else "unknown"
                )
                potential.origin = f"{bound_exchange}_bound_gu_continuum_hybrid"
            return potential
        raise ValueError(f"unknown potential model kind {self.kind!r}")


@dataclass(frozen=True)
class ContinuumModel:
    """Continuum-wave numerical model plus its potential choice."""

    method: str = "bspline"
    potential: Optional[PotentialModel] = None
    shooting_n_grid: int = 2000
    shooting_r_max: Optional[float] = None
    orbital_removal: str = "capture"

    def __post_init__(self):
        if self.orbital_removal not in {"capture", "none"}:
            raise ValueError("orbital_removal must be 'capture' or 'none'")

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
        orbital_removal: str = "capture",
    ) -> "ContinuumModel":
        return cls(
            method="shooting",
            potential=potential,
            shooting_n_grid=n_grid,
            shooting_r_max=r_max,
            orbital_removal=orbital_removal,
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

    @classmethod
    def fac_frozen_core(cls, *, n_grid: int = 2000, r_max: Optional[float] = None):
        """Continuum in the initial FAC/Gu field, retaining its full density.

        Reuses a Gu bound result's saved local field without refitting SIC.
        HF bound orbitals use Gu exchange/SIC on their full initial density,
        an explicit hybrid model. The residual ion is not relaxed.
        """
        return cls.shooting(
            potential=PotentialModel(
                kind="fac_frozen", include_exchange=True, exchange="fac"
            ),
            n_grid=n_grid,
            r_max=r_max,
            orbital_removal="none",
        )


__all__ = ["PotentialModel", "ContinuumModel"]
