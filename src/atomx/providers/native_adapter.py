"""Native atomx provider (ADR-0001 D1, §3).

The native relativistic solvers (``SCFResult``, ``MCDHFResult``,
``ShootingResult``) already satisfy
:class:`~atomx.scattering.protocols.AtomicResultLike`.  This adapter wraps
one of them into an :class:`ElectronicStructurePayload`, so the existing
atomx path becomes the *reference provider* and every current benchmark
keeps guarding regressions throughout the epic.

A native result carries orthonormal Dirac orbitals on a radial grid, so
the payload exposes :data:`Capability.RADIAL_ORBITALS` and leaves
``overlap = None`` (orthonormal ⇒ ``S = I``).
"""

from __future__ import annotations

from typing import Any, Optional

from atomx import __version__ as _atomx_version
from atomx.providers.contract import (
    Capability,
    ElectronicStructurePayload,
    Provenance,
    SpinBlock,
)
from atomx.state_protocols import AtomicResultLike

__all__ = ["NativeProvider", "from_atomic_result"]


def _looks_relativistic(result: AtomicResultLike) -> bool:
    """Use declared physics; retain a nonzero-Q heuristic for legacy inputs."""
    spec = getattr(result, "calculation_spec", None)
    equation = (spec.physics.get("equation") if spec is not None else None)
    equation = equation or getattr(result, "_equation", None)
    if equation is not None:
        if equation not in ("dirac", "schrodinger"):
            raise ValueError(f"unsupported native equation {equation!r}")
        return equation == "dirac"
    import numpy as np

    for orbs in getattr(result, "orbitals_raw", {}).values():
        for entry in orbs:
            Q = entry[1] if len(entry) > 1 else None
            if Q is not None and np.any(np.asarray(Q) != 0):
                return True
    return False


class NativeProvider:
    """Wrap an atomx :class:`AtomicResultLike` as a provider.

    Parameters
    ----------
    result
        A converged native atomx result (``SCFResult`` / ``MCDHFResult`` /
        ``ShootingResult`` or any duck-typed ``AtomicResultLike``).
    method
        Method label for provenance (e.g. ``"DHF"``, ``"MCDHF"``).
    theory_level
        Free-form theory-level label; defaults to ``method``.
    core_hole, final_state
        Optional spectroscopy context (I2).
    """

    def __init__(
        self,
        result: AtomicResultLike,
        *,
        method: str = "atomx-native",
        theory_level: Optional[str] = None,
        core_hole: str = "",
        final_state: str = "",
    ) -> None:
        if not isinstance(result, AtomicResultLike):
            raise TypeError(
                "NativeProvider needs an AtomicResultLike (Z, subshells, "
                "orbital_energies, orbitals_raw); got "
                f"{type(result).__name__}"
            )
        self.result = result
        self._method = method
        self._theory_level = theory_level
        self._core_hole = core_hole
        self._final_state = final_state

    # -- provider Protocol --------------------------------------------------

    def capabilities(self):
        return self.build_payload().capabilities()

    def supports(self, *requested: Capability) -> bool:
        return self.build_payload().supports(*requested)

    def build_payload(self) -> ElectronicStructurePayload:
        r = self.result
        relativistic = _looks_relativistic(r)
        method = self._method
        if method == "atomx-native":
            method = getattr(r, "method_name", "atomx-native") or "atomx-native"
        spec = getattr(r, "calculation_spec", None)
        equation_declared = (spec is not None and spec.physics.get("equation") is not None
                             or getattr(r, "_equation", None) is not None)
        basis = ("atomx-bspline" if getattr(r, "_bsp_data", None)
                 else "atomx-radial-grid" if getattr(r, "_grid", None) is not None
                 else "atomx-native-unspecified")
        metadata = {"native_result_type": type(r).__name__,
                    "equation_source": "declared" if equation_declared else "legacy_Q_heuristic"}
        if spec is not None:
            metadata["calculation_spec"] = spec.to_dict()
        prov = Provenance(
            provider="atomx-native",
            code_version=f"atomx {_atomx_version}",
            method=method,
            basis=basis,
            theory_level=self._theory_level or method,
            approximations=(["Physical model and operator scope recorded in calculation_spec"]
                            if spec is not None else ["Legacy physical-model provenance unavailable"]),
            core_hole=self._core_hole,
            final_state=self._final_state,
            energy_zero="orbital_energies",
            is_ml=False,
        )
        payload = ElectronicStructurePayload(
            Z=int(r.Z),
            provenance=prov,
            subshells=list(r.subshells),
            orbital_energies=dict(r.orbital_energies),
            orbitals_raw=dict(getattr(r, "orbitals_raw", {})),
            # Native Dirac orbitals are orthonormal ⇒ overlap is identity,
            # represented as absent (the documented "None ⇒ orthonormal").
            overlap=None,
            spin=SpinBlock(collinear=True, soc=relativistic, coupling="jj")
            if relativistic
            else None,
            metadata=metadata,
        )
        return payload.validate()


def from_atomic_result(result: AtomicResultLike, **kwargs: Any):
    """Convenience: ``NativeProvider(result, **kwargs).build_payload()``."""
    return NativeProvider(result, **kwargs).build_payload()
