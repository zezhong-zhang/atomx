"""A solved local central field, without density reconstruction or SIC refits."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FrozenLocalField:
    """Snapshot of the electronic screening charge on the bound SCF mesh.

    Interpolate ``r * V_electronic`` rather than the full Coulomb potential.
    Keep the analytic nuclear field and the prescribed exterior charge.
    Linear interpolation reproduces every bound quadrature value exactly;
    it introduces no new tail fit or exchange model.
    """

    radius: np.ndarray
    screening_charge: np.ndarray
    Z: float
    electron_count: float
    residual_charge: float
    nuclear: Callable | None = None
    exchange: str = "gu"
    fac_ihx: float = -1.0
    fac_hxs: float = 1.0
    fac_csi: float = 0.55

    def __post_init__(self):
        radius = np.array(self.radius, dtype=float, copy=True).ravel()
        charge = np.array(self.screening_charge, dtype=float, copy=True).ravel()
        if (
            radius.size < 2
            or radius.shape != charge.shape
            or not np.isfinite(radius).all()
            or not np.isfinite(charge).all()
            or np.any(radius <= 0)
            or np.any(np.diff(radius) <= 0)
        ):
            raise ValueError(
                "frozen field needs a finite increasing positive radial table"
            )
        if not np.isfinite([self.Z, self.electron_count, self.residual_charge]).all():
            raise ValueError("frozen field charges must be finite")
        radius.setflags(write=False)
        charge.setflags(write=False)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "screening_charge", charge)

    def build_callable(self):
        """Return V(r) in Hartree, preserving the bound field's nucleus."""
        radius, charge = self.radius, self.screening_charge
        exterior = self.Z - self.residual_charge
        nuclear = self.nuclear
        nuclear_cache = None

        def field(r):
            nonlocal nuclear_cache
            r = np.asarray(r, dtype=float)
            if not np.isfinite(r).all() or np.any(r <= 0):
                raise ValueError(
                    "frozen field evaluation radii must be finite and positive"
                )
            screening = np.interp(r, radius, charge, right=exterior)
            # A finite electronic field at the origin: do not turn the first
            # tabulated screening value into another 1/r singularity.
            electronic = np.where(r < radius[0], charge[0] / radius[0], screening / r)
            if nuclear is None:
                nucleus = -self.Z / r
            elif nuclear_cache is not None and np.array_equal(nuclear_cache[0], r):
                nucleus = nuclear_cache[1]
            else:
                nucleus = np.vectorize(nuclear, otypes=[float])(r)
                nuclear_cache = np.array(r, copy=True), nucleus
            return nucleus + electronic

        field.residual_charge = self.residual_charge
        field.screening = "full"
        field.origin = "bound_scf_snapshot"
        return field


def frozen_scf_potential(result):
    """Use a stored local field; nonlocal HF results cannot supply one."""
    if not result.converged:
        raise ValueError("a frozen SCF field requires a converged bound calculation")
    snapshot = getattr(result, "_local_central_field", None)
    if snapshot is None:
        raise ValueError(
            "no saved local SCF field; solve with Gu exchange or choose an explicit continuum potential"
        )
    signature = tuple((sh.n, sh.kappa, float(sh.occupation)) for sh in result.subshells)
    if (
        signature != getattr(result, "_local_central_field_subshells", None)
        or float(result.Z) != snapshot.Z
    ):
        raise ValueError(
            "bound state occupations or nuclear charge differ from its frozen field"
        )
    return snapshot.build_callable()
