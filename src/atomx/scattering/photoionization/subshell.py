"""Occupation-weighted subshell E1 photoionization on a shooting continuum."""

from dataclasses import asdict, dataclass

import numpy as np

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.scattering.models import ContinuumModel


@dataclass
class SubshellPIResult:
    photon_energies_eV: np.ndarray
    sigma_by_orbital_a0_sq: dict
    provenance: dict

    @property
    def sigma_total_a0_sq(self):
        return sum(
            self.sigma_by_orbital_a0_sq.values(), np.zeros_like(self.photon_energies_eV)
        )

    @property
    def sigma_total_cm2(self):
        return self.sigma_total_a0_sq * (BOHR_RADIUS_M * 100) ** 2


def subshell_photoionization(
    result, photon_energies_eV, *, continuum_model, orbitals=None, thresholds_eV=None
):
    """E1 cross sections per atom, with each occupation included once.

    Supply an explicit shooting continuum model: frozen-initial and
    residual-core fields are distinct approximations. Thresholds default
    to negative orbital eigenvalues; supplied configuration/level energy
    differences replace only the continuum kinetic energy, retaining the
    initial bound functions. Provenance records every threshold and model.
    """
    import copy

    from atomx.scattering.radiative_recombination import (
        _rr_potential,
        _shooting_photoionization_cross_section,
    )

    energies = np.asarray(photon_energies_eV, dtype=float)
    if energies.ndim != 1 or not np.isfinite(energies).all() or np.any(energies <= 0):
        raise ValueError(
            "photon_energies_eV must be a positive finite one-dimensional array"
        )
    if (
        not isinstance(continuum_model, ContinuumModel)
        or continuum_model.method != "shooting"
    ):
        raise ValueError(
            "subshell_photoionization requires an explicit shooting continuum model"
        )
    selected = {s.label: s for s in result.subshells if s.occupation > 0}
    if orbitals is not None:
        names = [orbitals] if isinstance(orbitals, str) else list(orbitals)
        if not set(names).issubset(selected):
            raise ValueError("requested orbital is not occupied in the initial state")
        selected = {name: selected[name] for name in names}
    thresholds = {
        label: -result.orbital_energies[label] * HARTREE_EV for label in selected
    }
    if thresholds_eV is not None:
        if not set(thresholds_eV).issubset(selected):
            raise ValueError("threshold supplied for an unselected orbital")
        thresholds.update(thresholds_eV)
    if any(not np.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError("all ionization thresholds must be positive and finite")
    view = copy.copy(result)
    view.orbital_energies = dict(result.orbital_energies)
    bound = result.orbitals_by_subshell()
    cross_sections = {}
    potential_origins = {}
    for label, sh in selected.items():
        view.orbital_energies[label] = -float(thresholds[label]) / HARTREE_EV
        P, Q, _ = bound[sh.n, sh.kappa]
        prepared_potential = _rr_potential(view, sh, continuum_model)
        potential, _ = prepared_potential
        potential_origins[label] = getattr(
            potential, "origin", "explicit_continuum_model"
        )
        cross_sections[label] = sh.occupation * _shooting_photoionization_cross_section(
            view,
            sh,
            P,
            Q,
            energies / HARTREE_EV,
            continuum_model,
            potential=prepared_potential,
        )
    return SubshellPIResult(
        energies.copy(),
        cross_sections,
        dict(
            thresholds_eV=thresholds,
            threshold_source="supplied" if thresholds_eV is not None else "orbital",
            continuum_model=asdict(continuum_model),
            continuum_potential_origins=potential_origins,
            multipoles=["E1"],
            occupation_weighted=True,
        ),
    )
