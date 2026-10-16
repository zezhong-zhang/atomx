"""Complete native closed-core VN virtual-band construction for SD/FSCC."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from atomx.methods.mbpt import build_virtual_basis


@dataclass(frozen=True)
class AtomicVirtualSpace:
    orbitals: object
    occupied_keys: tuple
    virtual_keys: tuple
    identity: object

    def __post_init__(self):
        copied = {}
        for key, values in self.orbitals.items():
            copied[key] = tuple(
                np.frombuffer(
                    np.asarray(x).tobytes(), dtype=np.asarray(x).dtype
                ).reshape(np.asarray(x).shape)
                for x in values
            )
        object.__setattr__(self, "orbitals", MappingProxyType(copied))
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))


def build_atomic_space(core, *, virtual_lmax, virtual_energy_max_Ha):
    """Retain every requested bound/pseudo-continuum VN complement root.

    No per-kappa count selection, frozen partition, amplitude selection or
    negative-energy spinor substitution occurs. The consuming method checks
    the common metric and full memory bounds before correlation allocation.
    """
    if not core.converged or any(
        s.occupation != 2 * abs(s.kappa) for s in core.subshells
    ):
        raise ValueError("atomic reference must be converged and closed shell")
    if (
        isinstance(virtual_lmax, bool)
        or not isinstance(virtual_lmax, int)
        or virtual_lmax < 0
        or isinstance(virtual_energy_max_Ha, bool)
        or not np.isrealobj(virtual_energy_max_Ha)
        or not np.isscalar(virtual_energy_max_Ha)
        or not np.isfinite(virtual_energy_max_Ha)
    ):
        raise ValueError(
            "nonnegative integer lmax and finite real energy bound required"
        )
    od = {key: values[:2] for key, values in core.orbitals_by_subshell().items()}
    occupied = tuple(od)
    virtual = []
    energies = {}
    for l in range(virtual_lmax + 1):
        for k in (-1,) if l == 0 else (l, -l - 1):
            _, states = build_virtual_basis(
                core,
                k,
                e_max=virtual_energy_max_Ha,
                potential_mode="VN",
                orthogonal_to_occupied=True,
            )
            n = max([l] + [key[0] for key in occupied if key[1] == k]) + 1
            for P, Q, eps in states:
                key = n, k
                od[key] = P, Q
                virtual.append(key)
                energies[str(key)] = eps
                n += 1
    identity = dict(
        virtual_lmax=virtual_lmax,
        virtual_energy_max_Ha=virtual_energy_max_Ha,
        virtual_orbital_energies_Ha=energies,
        positive_energy_projector="native Dirac VN complement, complete retained band",
        virtual_space_source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    )
    return AtomicVirtualSpace(od, occupied, tuple(virtual), identity)
