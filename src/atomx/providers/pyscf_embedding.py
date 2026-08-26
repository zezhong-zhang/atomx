"""PySCF cluster / point-charge embedding and local-orbital bridging (HYP-909).

Stage G2 of ADR-0001: extend the PySCF adapter from isolated molecules to
**embedded clusters** that model a condensed-matter local environment, and
provide a *local-orbital bridge* that keeps the central atom's core-orbital
identity traceable across environments (isolated atom → molecule → embedded
cluster) so environmental chemical shifts are well-defined numbers.

Physics model
-------------
* **Quantum region** — a finite cluster of real atoms (central atom plus
  optional coordination shells) treated with PySCF RHF/ROHF in a GTO basis.
* **Environment** — the surrounding ionic lattice represented as an array of
  fractional/formal point charges folded into the one-electron Hamiltonian
  via :func:`pyscf.qmmm.mm_charge` (Madelung potential).  Boundary sites may
  carry Evjen fractional charges (faces 1/2, edges 1/4, corners 1/8) so the
  truncated lattice stays neutral.
* **Local-orbital bridge** — the central atom's core orbital is identified
  in every environment by its Mulliken population on that atom's named AO
  shell (e.g. ``Mg 1s``).  Matching is fail-closed: the matched MO must
  dominate the AO subspace and be unambiguous against the runner-up.
* **Energy zero** — two declared, traceable conventions:
  ``"mo_energy"`` (orbital-energy difference between environments, I4) and
  ``"delta_scf_total_energy"`` (core binding energy
  ``E(N-1 core hole) − E(N)``; identical point-charge layouts cancel
  exactly in the difference, so embedded ΔSCF is the preferred
  environment-to-environment comparator).

PySCF remains an optional dependency; every public entry point imports it
lazily and raises a clear :class:`ImportError` when missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MethodType
from typing import Any, Optional, Sequence

import numpy as np

from atomx.providers.contract import (
    Capability,
    ElectronicStructurePayload,
    ProviderContractError,
    ensure_hartree,
)
from atomx.providers.pyscf_adapter import (
    CORE_HOLE_OVERLAP_THRESHOLD,
    PySCFProvider,
    _core_hole_metadata,
    _mom_core_hole_occ,
    _require_converged_mean_field,
    _require_pyscf,
)

__all__ = [
    "EmbeddingModel",
    "CoreOrbitalIdentity",
    "CoreOrbitalBridge",
    "EmbeddingRecord",
    "ClusterConvergenceStudy",
    "rocksalt_embedding",
    "build_embedded_scf",
    "core_orbital_identity",
    "bridge_core_orbitals",
    "embedded_core_hole_delta_scf",
    "study_cluster_convergence",
]

#: Default dominance floor for the Mulliken population of the tracked MO on
#: the named atom-shell AO subspace (I2: the core-hole identity must be real).
CORE_IDENTITY_DOMINANCE_THRESHOLD = 0.8

#: Minimum lead of the best-matching MO population over the runner-up;
#: below this the orbital match is ambiguous and the bridge fails closed.
CORE_IDENTITY_AMBIGUITY_MARGIN = 0.05


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingModel:
    """A quantum cluster embedded in a point-charge environment.

    Attributes
    ----------
    quantum_atoms
        PySCF atom-spec string for the quantum region, e.g.
        ``"Mg 0 0 0; O 2.106 0 0; ..."`` (Angstrom).
    central_atom_index
        Index (into ``quantum_atoms``) of the atom whose core orbital is
        tracked across environments.
    core_shell
        AO shell label of the tracked core orbital, e.g. ``"1s"``.
    mm_coords
        ``(n_mm, 3)`` point-charge positions in Angstrom.
    mm_charges
        ``(n_mm,)`` point charges in elementary-charge units.
    label
        Human-readable environment label used in records/provenance.
    approximations
        Extra approximation tags stamped into provenance (I5).
    """

    quantum_atoms: str
    central_atom_index: int
    core_shell: str
    mm_coords: np.ndarray
    mm_charges: np.ndarray
    label: str = ""
    approximations: tuple[str, ...] = ("point-charge-embedding",)

    def __post_init__(self) -> None:
        coords = np.asarray(self.mm_coords, dtype=float)
        charges = np.asarray(self.mm_charges, dtype=float)
        if coords.size == 0:
            coords = coords.reshape(0, 3)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ProviderContractError(
                f"mm_coords must have shape (n, 3), got {coords.shape}"
            )
        if charges.ndim != 1 or charges.shape[0] != coords.shape[0]:
            raise ProviderContractError(
                "mm_charges must be 1-D with one entry per point charge: "
                f"coords {coords.shape}, charges {charges.shape}"
            )
        if not (np.all(np.isfinite(coords)) and np.all(np.isfinite(charges))):
            raise ProviderContractError(
                "mm_coords/mm_charges must contain only finite values"
            )
        n_atoms = self.n_quantum_atoms
        if not 0 <= self.central_atom_index < n_atoms:
            raise ProviderContractError(
                f"central_atom_index {self.central_atom_index} outside "
                f"0..{n_atoms - 1}"
            )
        if not self.core_shell.strip():
            raise ProviderContractError("core_shell must be non-empty")
        object.__setattr__(self, "mm_coords", coords)
        object.__setattr__(self, "mm_charges", charges)

    @property
    def n_quantum_atoms(self) -> int:
        return len([a for a in self.quantum_atoms.split(";") if a.strip()])

    @property
    def n_mm_charges(self) -> int:
        return int(self.mm_charges.shape[0])

    @property
    def net_mm_charge(self) -> float:
        return float(np.sum(self.mm_charges))

    def scaled_mm_charges(self, factor: float) -> "EmbeddingModel":
        """Return a copy with MM charges scaled by ``factor`` (sensitivity).

        The scale is recorded in the label/approximations so the energy-zero
        provenance of a charge-sensitivity sweep stays traceable (I4/I5).
        """
        if not np.isfinite(factor) or factor < 0.0:
            raise ProviderContractError(
                f"charge scale factor must be a non-negative finite number, "
                f"got {factor!r}"
            )
        return replace(
            self,
            mm_charges=self.mm_charges * float(factor),
            label=f"{self.label}|scale={factor:g}",
            approximations=self.approximations
            + (f"mm-charge-scale={factor:g}",),
        )

    def build_mol(self, basis: str):
        """Build the PySCF ``Mole`` for the quantum region (Angstrom)."""
        _require_pyscf()
        from pyscf import gto

        if not isinstance(basis, str) or not basis.strip():
            raise ProviderContractError("basis must be a non-empty string")
        try:
            mol = gto.M(
                atom=self.quantum_atoms, basis=basis, unit="Angstrom", verbose=0
            )
        except RuntimeError as exc:
            raise ProviderContractError(
                "failed to build the quantum-region Mole — the restricted "
                "embedding requires a closed-shell quantum cluster (check "
                "quantum_shells / central species for an odd electron "
                f"count): {exc}"
            ) from exc
        if mol.nelectron % 2 != 0:
            raise ProviderContractError(
                "quantum region has an odd electron count "
                f"({mol.nelectron}); restricted embedding requires a "
                "closed-shell quantum cluster — adjust quantum_shells or the "
                "central species"
            )
        return mol

    def metadata(self) -> dict[str, Any]:
        """Traceable record of the embedding (goes into payload metadata)."""
        return {
            "embedding_label": self.label,
            "embedding_central_atom_index": int(self.central_atom_index),
            "embedding_core_shell": self.core_shell,
            "embedding_n_quantum_atoms": self.n_quantum_atoms,
            "embedding_n_mm_charges": self.n_mm_charges,
            "embedding_net_mm_charge": self.net_mm_charge,
            "embedding_coord_unit": "angstrom",
        }


def rocksalt_embedding(
    cation: str,
    anion: str,
    q_cation: float,
    q_anion: float,
    nearest_neighbor_angstrom: float,
    *,
    quantum_shells: int = 0,
    mm_shells: int = 2,
    evjen: bool = True,
    core_shell: str = "1s",
) -> EmbeddingModel:
    """Build a rock-salt (B1) embedded-cluster model centered on a cation.

    Lattice sites sit at ``(i, j, k) * nearest_neighbor_angstrom`` with even
    parity ``i+j+k`` carrying the cation (origin = central cation).  Sites
    with Chebyshev radius ``<= quantum_shells`` are quantum atoms; sites out
    to ``mm_shells`` become point charges with formal charges
    ``q_cation``/``q_anion``.  With ``evjen=True`` the boundary shell carries
    fractional charges (face 1/2, edge 1/4, corner 1/8) so the truncated
    lattice is exactly neutral; the net MM charge (``-q_cation``, balancing
    the quantum cube) is recorded in the model metadata either way.
    """
    for name, value in (
        ("q_cation", q_cation),
        ("q_anion", q_anion),
        ("nearest_neighbor_angstrom", nearest_neighbor_angstrom),
    ):
        if not np.isfinite(value):
            raise ProviderContractError(f"{name} must be finite, got {value!r}")
    if nearest_neighbor_angstrom <= 0.0:
        raise ProviderContractError("nearest_neighbor_angstrom must be positive")
    if quantum_shells < 0 or mm_shells <= quantum_shells:
        raise ProviderContractError(
            "require 0 <= quantum_shells < mm_shells, got "
            f"quantum_shells={quantum_shells}, mm_shells={mm_shells}"
        )
    d = float(nearest_neighbor_angstrom)
    qm_sites: list[tuple[int, int, int]] = []
    mm_sites: list[tuple[np.ndarray, float]] = []
    for i in range(-mm_shells, mm_shells + 1):
        for j in range(-mm_shells, mm_shells + 1):
            for k in range(-mm_shells, mm_shells + 1):
                radius = max(abs(i), abs(j), abs(k))
                charge = q_cation if (i + j + k) % 2 == 0 else q_anion
                if radius <= quantum_shells:
                    qm_sites.append((i, j, k))
                else:
                    if evjen and radius == mm_shells:
                        n_face = sum(
                            1 for c in (i, j, k) if abs(c) == mm_shells
                        )
                        charge *= 0.5**n_face
                    mm_sites.append((np.array([i, j, k], dtype=float) * d, charge))

    def _site_sort_key(site: tuple[int, int, int]) -> tuple[int, int]:
        return (max(abs(c) for c in site), site[0] ** 2 + site[1] ** 2 + site[2] ** 2)

    qm_sites = sorted(qm_sites, key=_site_sort_key)
    atoms = "; ".join(
        f"{cation if (i + j + k) % 2 == 0 else anion} "
        f"{i * d:.10f} {j * d:.10f} {k * d:.10f}"
        for i, j, k in qm_sites
    )
    mm_coords = (
        np.stack([c for c, _ in mm_sites])
        if mm_sites
        else np.zeros((0, 3), dtype=float)
    )
    mm_charges = (
        np.array([q for _, q in mm_sites], dtype=float)
        if mm_sites
        else np.zeros(0, dtype=float)
    )
    approximations = ["point-charge-embedding", "rocksalt-cluster"]
    if evjen:
        approximations.append("evjen-boundary-charges")
    return EmbeddingModel(
        quantum_atoms=atoms,
        central_atom_index=0,
        core_shell=core_shell,
        mm_coords=mm_coords,
        mm_charges=mm_charges,
        label=(
            f"{cation}{anion}:qm{quantum_shells}/mm{mm_shells}"
            f"{'/evjen' if evjen else ''}"
        ),
        approximations=tuple(approximations),
    )


# ---------------------------------------------------------------------------
# Embedded SCF
# ---------------------------------------------------------------------------


def build_embedded_scf(
    embedding: EmbeddingModel,
    basis: str,
    *,
    conv_tol: Optional[float] = None,
    max_cycle: Optional[int] = None,
):
    """Run a converged, point-charge-embedded RHF on the quantum region.

    Returns the PySCF mean-field object decorated with
    :func:`pyscf.qmmm.mm_charge`.  Non-convergence is fail-closed.
    """
    _require_pyscf()
    from pyscf import qmmm, scf

    mol = embedding.build_mol(basis)
    mf = scf.RHF(mol)
    if conv_tol is not None:
        mf.conv_tol = float(conv_tol)
    if max_cycle is not None:
        mf.max_cycle = int(max_cycle)
    if embedding.n_mm_charges:
        mf = qmmm.mm_charge(mf, embedding.mm_coords, embedding.mm_charges)
    mf.kernel()
    _require_converged_mean_field(mf, context=f"embedded SCF ({embedding.label})")
    return mf


# ---------------------------------------------------------------------------
# Local-orbital bridge: core-orbital identity across environments
# ---------------------------------------------------------------------------


def _ao_mask_for_atom_shell(mol: Any, atom_index: int, shell: str) -> np.ndarray:
    """Boolean AO mask selecting one atom's named shell (e.g. ``"1s"``)."""
    try:
        ao_labels = [str(x) for x in mol.ao_labels()]
    except Exception as exc:  # pragma: no cover - defensive
        raise ProviderContractError("cannot resolve AO labels for orbital match") from exc
    mask = []
    for label in ao_labels:
        parts = label.split()
        mask.append(
            len(parts) >= 3
            and parts[0] == str(atom_index)
            and parts[2].lower().startswith(shell.lower())
        )
    mask_arr = np.array(mask, dtype=bool)
    if not np.any(mask_arr):
        raise ProviderContractError(
            f"no AO labels match atom {atom_index} shell {shell!r}"
        )
    return mask_arr


@dataclass(frozen=True)
class CoreOrbitalIdentity:
    """Traceable identity of a core orbital in one environment (I2/I4)."""

    label: str                    # e.g. "Mg 1s"
    environment: str              # e.g. "isolated-atom" | embedding label
    mo_index: int
    energy_hartree: float
    occupation: float
    dominance: float              # Mulliken population on the named AO shell
    runner_up: float              # second-best population (ambiguity audit)
    energy_zero: str = "mo_energy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_hartree", ensure_hartree(self.energy_hartree))
        if not self.label.strip():
            raise ProviderContractError("core orbital label must be non-empty")


def core_orbital_identity(
    mf: Any,
    atom_index: int,
    shell: str = "1s",
    *,
    environment: str = "",
    dominance_threshold: float = CORE_IDENTITY_DOMINANCE_THRESHOLD,
    ambiguity_margin: float = CORE_IDENTITY_AMBIGUITY_MARGIN,
) -> CoreOrbitalIdentity:
    """Identify ``shell`` of ``atom_index`` among the MOs of a converged mf.

    The match uses the Mulliken population
    ``sum_{mu in shell} C[mu, i] * (S C)[mu, i]`` of each MO on the named
    atom-shell AO subspace.  It fails closed unless the best MO dominates
    the subspace (``>= dominance_threshold``) and leads the runner-up by
    ``ambiguity_margin`` — a blurred or contested match would make the
    chemical-shift provenance physically ambiguous.
    """
    _require_pyscf()
    _require_converged_mean_field(mf, context="orbital-identity")
    C = np.asarray(mf.mo_coeff, dtype=float)
    if C.ndim != 2:
        raise ProviderContractError(
            "core orbital identity requires a restricted (2-D mo_coeff) "
            "mean field"
        )
    mo_energy = np.asarray(mf.mo_energy, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)
    S = np.asarray(mf.get_ovlp(), dtype=float)
    if mo_energy.shape != (C.shape[1],):
        raise ProviderContractError("mo_energy shape mismatches mo_coeff")
    mask = _ao_mask_for_atom_shell(mf.mol, atom_index, shell)
    SC = S @ C
    populations = np.einsum("mi,mi->i", C[mask, :], SC[mask, :])
    order = np.argsort(populations)[::-1]
    best = int(order[0])
    dominance = float(populations[best])
    runner_up = float(populations[order[1]]) if C.shape[1] > 1 else 0.0
    symbol = str(mf.mol.atom_symbol(atom_index))
    label = f"{symbol} {shell}"
    if dominance < dominance_threshold:
        raise ProviderContractError(
            f"core orbital identity {label!r} lost: best MO{best} population "
            f"{dominance:.4f} < dominance threshold {dominance_threshold:.4f} "
            f"(environment={environment or 'unspecified'})"
        )
    if dominance - runner_up < ambiguity_margin:
        raise ProviderContractError(
            f"core orbital identity {label!r} is ambiguous: MO{best} "
            f"population {dominance:.4f} vs runner-up {runner_up:.4f} "
            f"< margin {ambiguity_margin:.4f} "
            f"(environment={environment or 'unspecified'})"
        )
    return CoreOrbitalIdentity(
        label=label,
        environment=environment,
        mo_index=best,
        energy_hartree=float(mo_energy[best]),
        occupation=float(mo_occ[best]),
        dominance=dominance,
        runner_up=runner_up,
    )


@dataclass(frozen=True)
class CoreOrbitalBridge:
    """The traceable link of one core orbital between two environments."""

    label: str
    reference: CoreOrbitalIdentity
    target: CoreOrbitalIdentity
    delta_epsilon_hartree: float   # target − reference, energy_zero below
    energy_zero: str = "mo_energy"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "delta_epsilon_hartree", ensure_hartree(self.delta_epsilon_hartree)
        )


def bridge_core_orbitals(
    reference: CoreOrbitalIdentity,
    target: CoreOrbitalIdentity,
) -> CoreOrbitalBridge:
    """Bridge the same named core orbital across two environments.

    Fails closed when the two identities name different orbitals — cross-
    environment comparison is only meaningful for the same ``(atom, shell)``.
    The returned ``delta_epsilon_hartree`` is the orbital-energy chemical
    shift ``ε(target) − ε(reference)`` with the declared ``"mo_energy"``
    zero; for the total-energy comparator use the ΔSCF binding energies from
    :func:`embedded_core_hole_delta_scf` / :func:`study_cluster_convergence`.
    """
    if reference.label != target.label:
        raise ProviderContractError(
            "cannot bridge different core orbitals: "
            f"{reference.label!r} vs {target.label!r}"
        )
    if reference.energy_zero != target.energy_zero:
        raise ProviderContractError(
            "cannot bridge identities with different energy zeros: "
            f"{reference.energy_zero!r} vs {target.energy_zero!r}"
        )
    return CoreOrbitalBridge(
        label=reference.label,
        reference=reference,
        target=target,
        delta_epsilon_hartree=target.energy_hartree - reference.energy_hartree,
        energy_zero=reference.energy_zero,
    )


# ---------------------------------------------------------------------------
# Embedded ΔSCF core hole (traceable energy zero)
# ---------------------------------------------------------------------------


def embedded_core_hole_delta_scf(
    embedding: EmbeddingModel,
    basis: str,
    *,
    with_eri: bool = False,
    conv_tol: Optional[float] = None,
    max_cycle: Optional[int] = None,
) -> tuple[ElectronicStructurePayload, CoreOrbitalIdentity, float]:
    """ΔSCF core-hole calculation on the embedded cluster.

    Applies the *same* point-charge layout to the ground-state RHF and the
    MOM-tracked ROHF core-hole state, so the QM/MM electrostatic terms are
    identical in both total energies and cancel exactly in
    ``delta_scf_binding_energy = E(N−1) − E(N)`` — the traceable,
    environment-comparable energy zero (I4).

    Returns ``(core_hole_payload, ground_identity, binding_energy_hartree)``.
    """
    _require_pyscf()
    from pyscf import qmmm, scf

    ground_mf = build_embedded_scf(
        embedding, basis, conv_tol=conv_tol, max_cycle=max_cycle
    )
    identity = core_orbital_identity(
        ground_mf,
        embedding.central_atom_index,
        embedding.core_shell,
        environment=embedding.label,
    )
    ground_occ = np.asarray(ground_mf.mo_occ, dtype=float)
    if ground_occ[identity.mo_index] < 1.0:
        raise ProviderContractError(
            f"tracked core orbital {identity.label!r} is not occupied in the "
            "embedded ground state"
        )
    ground_coeff = np.asarray(ground_mf.mo_coeff, dtype=float)
    S = np.asarray(ground_mf.get_ovlp(), dtype=float)
    ref_ao = ground_coeff[:, identity.mo_index].copy()

    mol = ground_mf.mol.copy()
    mol.charge = int(getattr(mol, "charge", 0)) + 1
    mol.spin = int(getattr(mol, "spin", 0)) + 1
    mol.build(False, False)
    mf = scf.ROHF(mol)
    if conv_tol is not None:
        mf.conv_tol = float(conv_tol)
    if max_cycle is not None:
        mf.max_cycle = int(max_cycle)
    if embedding.n_mm_charges:
        mf = qmmm.mm_charge(mf, embedding.mm_coords, embedding.mm_charges)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        energies = np.asarray(self.mo_energy if mo_energy is None else mo_energy)
        coeff = np.asarray(self.mo_coeff if mo_coeff is None else mo_coeff)
        occ, _, _ = _mom_core_hole_occ(
            coeff,
            energies,
            S,
            ref_ao,
            self.mol.nelectron,
            self.mol.spin,
        )
        return occ

    target_occ, _, _ = _mom_core_hole_occ(
        ground_coeff,
        np.asarray(ground_mf.mo_energy, dtype=float),
        S,
        ref_ao,
        mol.nelectron,
        mol.spin,
    )
    mf.get_occ = MethodType(get_occ, mf)
    dm0 = mf.make_rdm1(ground_coeff, target_occ)
    mf.kernel(dm0=dm0)
    _require_converged_mean_field(
        mf, context=f"embedded core-hole ΔSCF ({embedding.label})"
    )

    _, final_index, final_overlap = _mom_core_hole_occ(
        np.asarray(mf.mo_coeff, dtype=float),
        np.asarray(mf.mo_energy, dtype=float),
        S,
        ref_ao,
        mol.nelectron,
        mol.spin,
    )
    single_occupied = np.flatnonzero(np.isclose(np.asarray(mf.mo_occ, dtype=float), 1.0))
    if single_occupied.shape != (1,) or int(single_occupied[0]) != final_index:
        raise ProviderContractError(
            "embedded core-hole ΔSCF converged with inconsistent MOM "
            f"occupation: tracked MO{final_index}, single-occupied "
            f"columns={single_occupied.tolist()}"
        )
    if final_overlap < CORE_HOLE_OVERLAP_THRESHOLD:
        raise ProviderContractError(
            "embedded core-hole ΔSCF lost the named orbital character: "
            f"max_overlap={final_overlap:.6f}, "
            f"threshold={CORE_HOLE_OVERLAP_THRESHOLD:.6f}"
        )

    binding = ensure_hartree(float(mf.e_tot) - float(ground_mf.e_tot))
    metadata = {
        **embedding.metadata(),
        "ground_state_total_energy_hartree": ensure_hartree(float(ground_mf.e_tot)),
        "core_hole_total_energy_hartree": ensure_hartree(float(mf.e_tot)),
        "delta_scf_binding_energy_hartree": binding,
        "core_hole": _core_hole_metadata(
            identity.label,
            ground_occ,
            np.asarray(mf.mo_occ, dtype=float),
            identity.mo_index,
            final_index,
            final_overlap,
            binding,
        ),
    }
    payload = PySCFProvider(
        mf,
        with_eri=with_eri,
        core_hole=identity.label,
        energy_zero="delta_scf_total_energy",
        metadata=metadata,
    ).build_payload()
    payload.provenance.approximations = [
        *payload.provenance.approximations,
        *embedding.approximations,
    ]
    payload.require(Capability.CORE_HOLE_SPEC)
    residual = payload.orthonormality_residual()
    if residual is not None and residual > 1e-7:
        raise ProviderContractError(
            "embedded core-hole MO coefficients fail C^T S C = I; "
            f"residual={residual:.3e}"
        )
    expected_nelec = ground_mf.mol.nelectron - 1
    electron_count = payload.electron_count()
    if electron_count is None or abs(electron_count - expected_nelec) > 1e-6:
        raise ProviderContractError(
            "embedded core-hole density electron count mismatch: "
            f"got {electron_count}, expected {expected_nelec}"
        )
    return payload, identity, binding


# ---------------------------------------------------------------------------
# Condensed-matter local-environment convergence study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingRecord:
    """One row of the convergence study (traceable per config)."""

    label: str
    quantum_shells: int
    mm_shells: int
    evjen: bool
    charge_scale: float
    n_quantum_atoms: int
    n_mm_charges: int
    net_mm_charge: float
    identity: CoreOrbitalIdentity
    total_energy_hartree: float
    binding_energy_hartree: float
    orbital_shift_hartree: float    # ε − ε(reference), zero "mo_energy"
    binding_shift_hartree: float    # E_bind − E_bind(reference), ΔSCF zero

    def __post_init__(self) -> None:
        for name_ in (
            "total_energy_hartree",
            "binding_energy_hartree",
            "orbital_shift_hartree",
            "binding_shift_hartree",
        ):
            object.__setattr__(self, name_, ensure_hartree(getattr(self, name_)))


@dataclass(frozen=True)
class ClusterConvergenceStudy:
    """Cluster-size / embedding-charge sensitivity of one local environment.

    ``reference_identity``/``reference_binding_hartree`` are the isolated
    central atom's values; every record's shifts are taken against them with
    the declared energy zeros, so the full table is auditable (I4).
    """

    lattice: str
    basis: str
    core_shell: str
    nearest_neighbor_angstrom: float
    reference_identity: CoreOrbitalIdentity
    reference_binding_hartree: float
    records: tuple[EmbeddingRecord, ...] = field(default_factory=tuple)

    def to_table(self) -> list[dict[str, Any]]:
        """Flat dicts (one per record) for CSV export / reporting."""
        table = []
        for rec in self.records:
            table.append(
                {
                    "label": rec.label,
                    "quantum_shells": rec.quantum_shells,
                    "mm_shells": rec.mm_shells,
                    "evjen": rec.evjen,
                    "charge_scale": rec.charge_scale,
                    "n_quantum_atoms": rec.n_quantum_atoms,
                    "n_mm_charges": rec.n_mm_charges,
                    "net_mm_charge": rec.net_mm_charge,
                    "core_mo_index": rec.identity.mo_index,
                    "core_dominance": rec.identity.dominance,
                    "core_epsilon_hartree": rec.identity.energy_hartree,
                    "binding_energy_hartree": rec.binding_energy_hartree,
                    "orbital_shift_hartree": rec.orbital_shift_hartree,
                    "binding_shift_hartree": rec.binding_shift_hartree,
                }
            )
        return table


def study_cluster_convergence(
    *,
    cation: str,
    anion: str,
    q_cation: float,
    q_anion: float,
    nearest_neighbor_angstrom: float,
    basis: str,
    core_shell: str = "1s",
    quantum_shells: Sequence[int] = (0, 1),
    mm_shells: Sequence[int] = (1, 2, 3),
    evjen: bool = True,
    charge_scales: Sequence[float] = (1.0,),
    conv_tol: Optional[float] = None,
    max_cycle: Optional[int] = None,
) -> ClusterConvergenceStudy:
    """Convergence study of a rock-salt local environment (acceptance, HYP-909).

    Sweeps ``quantum_shells`` (cluster size) × ``mm_shells`` (embedding
    radius) × ``charge_scales`` (embedding-charge sensitivity).  The
    reference is the isolated central cation; each config reports the
    core-orbital identity, orbital-energy shift and ΔSCF binding-energy
    shift against that reference, all with declared energy zeros.
    """
    _require_pyscf()
    from pyscf import gto, scf

    from atomx.providers.pyscf_adapter import build_core_hole_delta_scf_payload

    for q in quantum_shells:
        if q < 0:
            raise ProviderContractError(f"quantum_shells entries must be >= 0, got {q}")
    mms = sorted(set(int(m) for m in mm_shells))
    if not mms or mms[0] < 1:
        raise ProviderContractError("mm_shells entries must be >= 1")

    # Isolated-atom reference (same basis, same ΔSCF protocol → one zero).
    ref_mol = gto.M(atom=f"{cation} 0 0 0", basis=basis, verbose=0)
    ref_mf = scf.RHF(ref_mol)
    if conv_tol is not None:
        ref_mf.conv_tol = float(conv_tol)
    if max_cycle is not None:
        ref_mf.max_cycle = int(max_cycle)
    ref_mf.kernel()
    _require_converged_mean_field(ref_mf, context="isolated-atom reference")
    ref_identity = core_orbital_identity(
        ref_mf, 0, core_shell, environment="isolated-atom"
    )
    ref_payload = build_core_hole_delta_scf_payload(
        ref_mf, ref_identity.mo_index, max_cycle=max_cycle
    )
    ref_binding = ensure_hartree(
        ref_payload.metadata["delta_scf_energy_hartree"]
    )

    records: list[EmbeddingRecord] = []
    for q_shells in quantum_shells:
        for m_shells in mms:
            if m_shells <= q_shells:
                continue
            model = rocksalt_embedding(
                cation,
                anion,
                q_cation,
                q_anion,
                nearest_neighbor_angstrom,
                quantum_shells=q_shells,
                mm_shells=m_shells,
                evjen=evjen,
                core_shell=core_shell,
            )
            for scale in charge_scales:
                scaled = model.scaled_mm_charges(scale) if scale != 1.0 else model
                payload, identity, binding = embedded_core_hole_delta_scf(
                    scaled,
                    basis,
                    conv_tol=conv_tol,
                    max_cycle=max_cycle,
                )
                ground_energy = payload.metadata[
                    "ground_state_total_energy_hartree"
                ]
                records.append(
                    EmbeddingRecord(
                        label=scaled.label,
                        quantum_shells=q_shells,
                        mm_shells=m_shells,
                        evjen=evjen,
                        charge_scale=float(scale),
                        n_quantum_atoms=scaled.n_quantum_atoms,
                        n_mm_charges=scaled.n_mm_charges,
                        net_mm_charge=scaled.net_mm_charge,
                        identity=identity,
                        total_energy_hartree=ground_energy,
                        binding_energy_hartree=binding,
                        orbital_shift_hartree=(
                            identity.energy_hartree - ref_identity.energy_hartree
                        ),
                        binding_shift_hartree=binding - ref_binding,
                    )
                )
    if not records:
        raise ProviderContractError(
            "convergence sweep produced no configurations — check "
            "quantum_shells/mm_shells/charge_scales"
        )
    return ClusterConvergenceStudy(
        lattice=f"rocksalt-{cation}{anion}",
        basis=basis,
        core_shell=core_shell,
        nearest_neighbor_angstrom=float(nearest_neighbor_angstrom),
        reference_identity=ref_identity,
        reference_binding_hartree=ref_binding,
        records=tuple(records),
    )
