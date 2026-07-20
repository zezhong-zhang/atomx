"""PySCF molecular adapter (ADR-0001 D5, stage G1 / HYP-904).

Translates a converged PySCF mean-field object (``scf.RHF`` / ``scf.ROHF``)
into an :class:`ElectronicStructurePayload`: AO/MO coefficients, AO overlap
``S``, the one-particle density matrix, per-MO energies/occupations, the
(optional) two-electron repulsion integrals, and a stamped
:class:`Provenance`.

PySCF is an **optional** dependency (``pip install "atomx[pyscf]"``).  It is
imported lazily so ``import atomx.providers`` never requires it; calling the
adapter without PySCF installed raises a clear :class:`ImportError`.

Scope note (HYP-904 foundation): restricted references (RHF/ROHF, 2-D
``mo_coeff``) are supported.  Unrestricted (UHF) is rejected fail-closed
with an explicit message rather than silently averaging spin channels.

PySCF works natively in Hartree atomic units, so the energy coercion at the
contract boundary is an identity here; it still runs so the unit invariant
(I1) is enforced uniformly across back-ends.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from atomx.providers.contract import (
    Capability,
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    ProviderContractError,
    Provenance,
    TwoElectronIntegrals,
)

__all__ = ["PySCFProvider", "from_scf", "MOLabel"]


class MOLabel:
    """Lightweight ``SubshellLike`` for a molecular orbital.

    ``kappa = 0`` is a sentinel meaning "not a radial Dirac spinor" — a
    molecular MO has no single ``(n, kappa)``.  Downstream radial kernels
    that require :data:`Capability.RADIAL_ORBITALS` will fail-closed rather
    than misread this as a spinor.
    """

    __slots__ = ("label", "kappa", "occupation")

    def __init__(self, label: str, occupation: float) -> None:
        self.label = label
        self.kappa = 0
        self.occupation = float(occupation)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MOLabel({self.label!r}, occ={self.occupation})"


def _require_pyscf():
    try:
        import pyscf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised without pyscf
        raise ImportError(
            "PySCF is required for atomx.providers.pyscf_adapter but is not "
            "installed.\nInstall it with:  pip install \"atomx[pyscf]\"  "
            "(or  pip install pyscf)"
        ) from exc
    return pyscf


class PySCFProvider:
    """Wrap a converged PySCF restricted mean-field object as a provider.

    Parameters
    ----------
    mf
        A PySCF SCF object (``pyscf.scf.RHF``/``ROHF``) after ``kernel()``.
    with_eri
        Also expose the AO two-electron integrals ``(ij|kl)``.  Off by
        default because the dense tensor is ``O(n_ao^4)``.
    core_hole, final_state
        Optional spectroscopy context stamped into provenance (I2).
    """

    def __init__(
        self,
        mf: Any,
        *,
        with_eri: bool = False,
        core_hole: str = "",
        final_state: str = "",
    ) -> None:
        _require_pyscf()
        if not hasattr(mf, "mol") or not hasattr(mf, "mo_coeff"):
            raise TypeError(
                "PySCFProvider expects a PySCF mean-field object with a "
                "converged .mo_coeff and .mol"
            )
        if getattr(mf, "mo_coeff", None) is None:
            raise ProviderContractError(
                "PySCF mean-field is not solved yet — call mf.kernel() first"
            )
        mo_coeff = np.asarray(mf.mo_coeff)
        if mo_coeff.ndim != 2:
            raise ProviderContractError(
                "only restricted references (RHF/ROHF, 2-D mo_coeff) are "
                "supported in this foundation; got mo_coeff with "
                f"ndim={mo_coeff.ndim} (UHF/unrestricted not yet supported)"
            )
        self.mf = mf
        self.with_eri = with_eri
        self._core_hole = core_hole
        self._final_state = final_state

    # -- provider Protocol --------------------------------------------------

    def capabilities(self):
        return self.build_payload().capabilities()

    def supports(self, *requested: Capability) -> bool:
        return self.build_payload().supports(*requested)

    def build_payload(self) -> ElectronicStructurePayload:
        pyscf = _require_pyscf()
        mf = self.mf
        mol = mf.mol

        Z = int(sum(mol.atom_charges()))
        try:
            ao_labels: Optional[List[str]] = [
                str(x) for x in mol.ao_labels()
            ]
        except Exception:  # pragma: no cover - defensive
            ao_labels = None

        C = np.asarray(mf.mo_coeff, dtype=float)
        mo_energy = np.asarray(mf.mo_energy, dtype=float)
        mo_occ = np.asarray(mf.mo_occ, dtype=float)
        S = np.asarray(mf.get_ovlp(), dtype=float)
        dm = np.asarray(mf.make_rdm1(), dtype=float)
        if dm.ndim == 3:  # ROHF returns (2, nao, nao)
            dm = dm.sum(axis=0)

        orbitals = OrbitalCoefficients(
            coefficients=C,
            energies=mo_energy,
            occupations=mo_occ,
            basis_labels=ao_labels,
            spin="restricted",
        )
        overlap = OverlapBlock(matrix=S, basis_labels=ao_labels)
        density = DensityMatrix(matrix=dm, basis="ao")

        two_electron = None
        if self.with_eri:
            eri = np.asarray(mol.intor("int2e"), dtype=float)
            two_electron = TwoElectronIntegrals(
                eri=eri, basis="ao", notation="chemist"
            )

        # AtomicResultLike core (MO-labelled) so the MUST caps are met.
        subshells = [
            MOLabel(f"MO{i}", occ) for i, occ in enumerate(mo_occ)
        ]
        orbital_energies = {
            f"MO{i}": float(e) for i, e in enumerate(mo_energy)
        }

        basis_id = mol.basis
        if not isinstance(basis_id, str):
            basis_id = str(basis_id)
        prov = Provenance(
            provider="pyscf",
            code_version=f"pyscf {pyscf.__version__}",
            method=type(mf).__name__,
            basis=basis_id,
            theory_level=type(mf).__name__,
            approximations=["nonrel-gto"],
            core_hole=self._core_hole,
            final_state=self._final_state,
            energy_zero="mo_energy",
            is_ml=False,
        )

        payload = ElectronicStructurePayload(
            Z=Z,
            provenance=prov,
            subshells=subshells,
            orbital_energies=orbital_energies,
            orbitals_raw={},  # no radial grid; use the MO block instead
            overlap=overlap,
            orbitals=orbitals,
            density=density,
            two_electron=two_electron,
            metadata={
                "n_ao": int(C.shape[0]),
                "n_mo": int(C.shape[1]),
                "n_electrons": int(mol.nelectron),
                "molecule": mol.atom if isinstance(mol.atom, str) else None,
            },
        )
        return payload.validate()


def from_scf(mf: Any, **kwargs: Any) -> ElectronicStructurePayload:
    """Convenience: ``PySCFProvider(mf, **kwargs).build_payload()``."""
    return PySCFProvider(mf, **kwargs).build_payload()
