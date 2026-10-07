"""Configuration-average, per-vacancy E1 fluorescence on initial orbitals."""

import numpy as np

from atomx.angular.recoupling import twoj_of_kappa
from atomx.config.subshell import l_letter
from atomx.constants import HARTREE_EV
from atomx.numerics.radial_workspace import make_grid_workspace
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import einstein_A_eL


def fluorescence_rates(result, hole, *, photon_energies_eV=None):
    """Return per-vacancy and total-hole rates in s^-1 for an n-l hole.

    ``hole`` is (n,l). The ensemble must have equal jj occupation fractions
    within each n-l shell. Optional energies map source (n,l) to positive
    photon energies; callers can obtain these from frozen configuration,
    relaxed delta-SCF, or resolved levels. Without them this explicitly
    uses initial orbital eigenvalue differences, which omit configuration
    energy/relaxation shifts. Only E1 length, long-wavelength is included.
    """
    from atomx.scattering.xray import AtomicScattering

    model = AtomicScattering.from_result(result)
    # Grid workspace over the model's quadrature so the dipole radial
    # integral comes from the shared kernel (radial_eL static limit).
    workspace = make_grid_workspace(model.r, model.weights)
    h_n, h_l = hole
    targets = [a for a in model.orbitals if (a.n, a.l) == (h_n, h_l)]
    if not targets:
        raise ValueError("hole orbital is absent from the initial state")
    degeneracy = 2 * (2 * h_l + 1)
    vacancies = degeneracy - sum(a.occupation for a in targets)
    if vacancies <= 0:
        raise ValueError("the requested shell has no vacancy")
    photon_energies_eV = (
        None if photon_energies_eV is None else dict(photon_energies_eV)
    )
    if photon_energies_eV is not None and any(
        not np.isfinite(e) or e <= 0 for e in photon_energies_eV.values()
    ):
        raise ValueError("photon energies must be positive and finite")
    rates, lines = {}, []
    for a in model.orbitals:
        source = (a.n, a.l)
        if source == (h_n, h_l) or a.occupation <= 0:
            continue
        for h in targets:
            angular = angular_eL(a.kappa, h.kappa, 1)
            if not angular:
                continue
            a_label = f"{a.n}{l_letter(a.l)}{twoj_of_kappa(a.kappa)}/2"
            h_label = f"{h.n}{l_letter(h.l)}{twoj_of_kappa(h.kappa)}/2"
            if photon_energies_eV is None:
                energy = (
                    result.orbital_energies[a_label] - result.orbital_energies[h_label]
                ) * HARTREE_EV
            else:
                if source not in photon_energies_eV:
                    continue
                energy = photon_energies_eV[source]
            if energy <= 0:
                continue
            radial = radial_eL(a.P, a.Q, h.P, h.Q, workspace, 1)
            rate = (
                a.occupation
                / degeneracy
                * einstein_A_eL(
                    energy / HARTREE_EV,
                    (angular * radial) ** 2,
                    1,
                    2 * abs(a.kappa) - 1,
                )
            )
            rates[source] = rates.get(source, 0.0) + float(rate)
            lines.append(
                dict(
                    source=a_label,
                    hole=h_label,
                    photon_eV=float(energy),
                    per_vacancy_per_s=float(rate),
                )
            )
    total = sum(rates.values())
    return dict(
        per_vacancy_per_s=total,
        total_per_s=vacancies * total,
        vacancies=float(vacancies),
        per_source_per_s=rates,
        lines=lines,
        energy_source="supplied" if photon_energies_eV is not None else "orbital",
        multipoles=["E1"],
        radial_field="initial",
        radial_operator="long_wavelength_length",
    )
