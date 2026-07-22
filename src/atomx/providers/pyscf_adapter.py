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

from types import MethodType
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
    ensure_hartree,
)

__all__ = [
    "PySCFProvider",
    "build_core_hole_delta_scf_payload",
    "from_core_hole_delta_scf",
    "from_scf",
    "MOLabel",
]


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
        energy_zero: str = "mo_energy",
        metadata: Optional[dict[str, Any]] = None,
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
        self._energy_zero = energy_zero
        self._metadata = dict(metadata or {})

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
            energy_zero=self._energy_zero,
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
                **self._metadata,
            },
        )
        return payload.validate()


def from_scf(mf: Any, **kwargs: Any) -> ElectronicStructurePayload:
    """Convenience: ``PySCFProvider(mf, **kwargs).build_payload()``."""
    return PySCFProvider(mf, **kwargs).build_payload()


def _ao_label_matches_shell(ao_label: str, atom_symbol: str, shell: str) -> bool:
    parts = ao_label.split()
    if len(parts) < 3:
        return False
    return parts[1].lower() == atom_symbol.lower() and parts[2].lower().startswith(
        shell.lower()
    )


def _resolve_core_orbital_index(mf: Any, core_orbital: int | str) -> tuple[int, str]:
    C = np.asarray(mf.mo_coeff, dtype=float)
    if isinstance(core_orbital, int):
        if core_orbital < 0 or core_orbital >= C.shape[1]:
            raise ProviderContractError(
                f"core orbital index {core_orbital} outside 0..{C.shape[1] - 1}"
            )
        return core_orbital, f"MO{core_orbital}"

    label = str(core_orbital).strip()
    if label.upper().startswith("MO") and label[2:].isdigit():
        index = int(label[2:])
        if index < 0 or index >= C.shape[1]:
            raise ProviderContractError(
                f"core orbital label {label!r} outside available MO range"
            )
        return index, label

    parts = label.split()
    if len(parts) != 2:
        raise ProviderContractError(
            "core_orbital must be an MO index, 'MO<n>', or an atom/shell label "
            "such as 'O 1s'"
        )
    atom_symbol, shell = parts
    S = np.asarray(mf.get_ovlp(), dtype=float)
    SC = S @ C
    try:
        ao_labels = [str(x) for x in mf.mol.ao_labels()]
    except Exception as exc:  # pragma: no cover - defensive
        raise ProviderContractError("cannot resolve AO labels for core orbital") from exc
    mask = np.array(
        [_ao_label_matches_shell(x, atom_symbol, shell) for x in ao_labels],
        dtype=bool,
    )
    if not np.any(mask):
        raise ProviderContractError(f"no AO labels match core orbital {label!r}")
    populations = np.einsum("mi,mi->i", C[mask, :], SC[mask, :])
    index = int(np.argmax(populations))
    return index, label


def _core_hole_occ(mo_energy: np.ndarray, n_electrons: int, spin: int, core_index: int):
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2
    if n_alpha < n_beta or n_alpha + n_beta != n_electrons:
        raise ProviderContractError(
            "core-hole electron/spin counts do not form a valid ROHF state"
        )
    if n_alpha - n_beta != 1:
        raise ProviderContractError("core-hole ΔSCF currently supports one open shell")
    order = list(np.argsort(np.asarray(mo_energy, dtype=float)))
    occ = np.zeros_like(mo_energy, dtype=float)
    occ[core_index] = 1.0
    doubly = [i for i in order if i != core_index][:n_beta]
    occ[doubly] = 2.0
    if float(np.sum(occ)) != float(n_electrons):
        raise ProviderContractError("failed to construct core-hole occupations")
    return occ


def build_core_hole_delta_scf_payload(
    ground_mf: Any,
    core_orbital: int | str,
    *,
    with_eri: bool = False,
    max_cycle: Optional[int] = None,
) -> ElectronicStructurePayload:
    """Run a restricted PySCF ΔSCF core-hole calculation and return a payload.

    ``core_orbital`` is explicit spectroscopy identity: either a numeric MO,
    ``"MO<n>"``, or an atom/shell label such as ``"O 1s"``.  The resulting
    payload carries ``Capability.CORE_HOLE_SPEC`` through provenance.
    """

    _require_pyscf()
    from pyscf import scf

    ground_payload = PySCFProvider(ground_mf).build_payload()
    if getattr(ground_mf, "e_tot", None) is None:
        raise ProviderContractError("ground-state mean field has no total energy")
    core_index, core_label = _resolve_core_orbital_index(ground_mf, core_orbital)
    ground_occ = np.asarray(ground_mf.mo_occ, dtype=float)
    if ground_occ[core_index] < 1.0:
        raise ProviderContractError(
            f"selected core orbital {core_label!r} is not occupied in the ground state"
        )

    mol = ground_mf.mol.copy()
    mol.charge = int(getattr(mol, "charge", 0)) + 1
    mol.spin = int(getattr(mol, "spin", 0)) + 1
    mol.build(False, False)
    mf = scf.ROHF(mol)
    mf.conv_tol = getattr(ground_mf, "conv_tol", mf.conv_tol)
    if max_cycle is not None:
        mf.max_cycle = int(max_cycle)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        del mo_coeff
        energies = np.asarray(self.mo_energy if mo_energy is None else mo_energy)
        return _core_hole_occ(energies, self.mol.nelectron, self.mol.spin, core_index)

    mf.get_occ = MethodType(get_occ, mf)
    target_occ = _core_hole_occ(
        np.asarray(ground_mf.mo_energy, dtype=float),
        mol.nelectron,
        mol.spin,
        core_index,
    )
    dm0 = mf.make_rdm1(np.asarray(ground_mf.mo_coeff, dtype=float), target_occ)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise ProviderContractError("core-hole ΔSCF PySCF calculation did not converge")

    delta_e = ensure_hartree(float(mf.e_tot) - float(ground_mf.e_tot))
    metadata = {
        "ground_state_total_energy_hartree": ensure_hartree(float(ground_mf.e_tot)),
        "core_hole_total_energy_hartree": ensure_hartree(float(mf.e_tot)),
        "delta_scf_energy_hartree": delta_e,
        "core_hole": {
            "orbital_label": core_label,
            "occupation_before": float(ground_occ[core_index]),
            "occupation_after": float(mf.mo_occ[core_index]),
            "occupation_delta": float(mf.mo_occ[core_index] - ground_occ[core_index]),
            "initial_state_energy_hartree": delta_e,
        },
    }
    payload = PySCFProvider(
        mf,
        with_eri=with_eri,
        core_hole=core_label,
        energy_zero="delta_scf_total_energy",
        metadata=metadata,
    ).build_payload()
    payload.require(Capability.CORE_HOLE_SPEC)
    residual = payload.orthonormality_residual()
    if residual is not None and residual > 1e-7:
        raise ProviderContractError(
            f"core-hole MO coefficients fail C^T S C = I; residual={residual:.3e}"
        )
    expected_nelec = ground_payload.metadata["n_electrons"] - 1
    electron_count = payload.electron_count()
    if electron_count is None or abs(electron_count - expected_nelec) > 1e-6:
        raise ProviderContractError(
            "core-hole density electron count mismatch: "
            f"got {electron_count}, expected {expected_nelec}"
        )
    return payload


def from_core_hole_delta_scf(
    ground_mf: Any,
    core_orbital: int | str,
    **kwargs: Any,
) -> ElectronicStructurePayload:
    """Convenience wrapper for :func:`build_core_hole_delta_scf_payload`."""

    return build_core_hole_delta_scf_payload(ground_mf, core_orbital, **kwargs)
