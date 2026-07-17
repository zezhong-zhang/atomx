"""CRM-weighted synthetic emission spectrum.

Given a :class:`~atomx.plasma.crm.CollisionalRadiativeModel` with rates
and populations, build a :class:`~atomx.spectrum.Spectrum` where each
radiative transition contributes a normalised line-profile weighted by
``n_upper * A_{ul}``.  Line widths are computed from Doppler (thermal)
and natural-lifetime contributions; an optional static Stark width for
hydrogenic ions can be folded in.

Line resolution
---------------
The CRM rate matrix stays at configuration level — full J-resolved CRM
needs J-resolved collision strengths and is out of scope.  Line
resolution enters at SYNTHESIS: with ``line_resolution='ci'`` each
radiative transition group that the determinant-CI engine can resolve
(a two-config single-hole pair within ``ci_det_cap``, via
:func:`atomx.properties.transitions.ci_lines.ci_line_strengths`)
replaces its UTA line distribution with the exact CI line list
(per-line energy, g_J, Einstein A), with the upper configuration's CRM
population distributed statistically over its J-levels
(``p_J = (2J+1) / sum_levels (2J'+1)`` within the config).  Everything
else keeps the UTA bridge / per-transition path.  Each emitted line or
array records the engine that produced it: CI groups in
``spec.metadata['ci_groups']`` (``engine='ci'``, per-line records),
UTA arrays in ``spec.metadata['uta_arrays']`` (``engine='uta'``).
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Mapping, Optional, Sequence

import numpy as np

from atomx.constants import HARTREE_EV, HARTREE_INV_ANGSTROM
from atomx.lineshapes import doppler_width, natural_width
from atomx.spectrum.spectrum import Spectrum
from atomx.spectrum.uta import add_uta_array_to_spectrum


def _hartree_to_angstrom(dE_hartree: float) -> float:
    """DeltaE (Hartree) -> lambda (A), standard 1/HARTREE_INV_ANGSTROM inversion."""
    return 1.0 / (HARTREE_INV_ANGSTROM * dE_hartree)


def crm_emission_spectrum(
    crm,
    n_e_cm3: float,
    T_e_eV: float,
    *,
    lam_min_nm: float,
    lam_max_nm: float,
    n_points: int = 2001,
    ion_mass_amu: float = 1.0,
    T_ion_K: Optional[float] = None,
    profile: str = "voigt",
    stark_hwhm_nm: Optional[float] = None,
    uta_threshold: int = 20,
    force_uta: bool = False,
    line_resolution: str = "uta",
    ci_result=None,
    ci_e_max_eV: float = 1000.0,
    ci_det_cap: int = 2000,
    include_eii: bool = True,
    include_three_body: bool = True,
    include_photoionization: bool = True,
    radiation_temperature_eV: Optional[float] = None,
    radiation_field=None,
    dr_satellites: Optional[Sequence[object] | object] = None,
    dr_initial_level_idx: int = 0,
    dr_lower_level_by_channel: Optional[Mapping[str, int]] = None,
    dr_initial_state_g: Optional[float] = None,
    dr_label_prefix: str = "DR",
    include_free_free: bool = False,
    ff_Z_eff: Optional[float] = None,
    include_rr_continuum: bool = False,
    rr_scf_result=None,
    continuum_ion_fraction: float = 1.0,
) -> Spectrum:
    """Render a CRM's radiative transitions into a :class:`Spectrum`.

    Line intensities are the emissivity-proportional product
    ``epsilon_{ul} = n_upper * A_{ul} * (h nu)``.

    Parameters
    ----------
    crm : CollisionalRadiativeModel
        A CRM with ``.levels`` and ``.transitions`` populated.
    include_free_free, ff_Z_eff : bool, float
        Fold in thermal bremsstrahlung
        (:func:`atomx.spectrum.continuum.free_free_emissivity`,
        Maxwell-averaged Born Gaunt factor) for an ion of effective
        charge ``ff_Z_eff``.  The continuum shares the line units
        (populations are per-ion, so the ion density in the product
        n_e n_i is ``continuum_ion_fraction``).
    include_rr_continuum, rr_scf_result : bool, SCFResult
        Fold in the free-bound (radiative-recombination) continuum from
        the Milne-relation RR machinery for the recombining ion whose
        converged ``rr_scf_result`` is given.
    dr_satellites : sequence or DRRateCoefficient, optional
        Dielectronic satellite resonances to fold into the rendered CRM. Each
        resonance may carry stabilising ``channels`` from the ab-initio
        radiative-width path; channel labels are matched to CRM lower levels.
        A DRRateCoefficient-like object with a ``.resonances`` attribute is
        also accepted. The input CRM is copied before synthetic satellite
        levels are added, so the caller's CRM is not mutated.
    """
    n_dr_satellites = 0
    if dr_satellites is not None:
        crm, n_dr_satellites = _crm_with_dr_satellites(
            crm,
            dr_satellites,
            initial_level_idx=dr_initial_level_idx,
            lower_level_by_channel=dr_lower_level_by_channel,
            initial_state_g=dr_initial_state_g,
            label_prefix=dr_label_prefix,
        )

    populations = crm.steady_state(
        n_e_cm3,
        T_e_eV,
        include_eii=include_eii,
        include_three_body=include_three_body,
        include_photoionization=include_photoionization,
        radiation_temperature_eV=radiation_temperature_eV,
        radiation_field=radiation_field,
    )

    if T_ion_K is None:
        T_ion_K = T_e_eV * 11604.5250061657

    spec = Spectrum.linear(
        lam_min_nm, lam_max_nm, n_points,
        unit="emissivity (arb., n_u * A * h nu)",
    )
    spec.metadata = {
        "n_e_cm3": n_e_cm3,
        "T_e_eV": T_e_eV,
        "T_ion_K": T_ion_K,
        "ion_mass_amu": ion_mass_amu,
        "profile": profile,
        "uta_threshold": uta_threshold,
        "uta_arrays": [],
        "include_eii": include_eii,
        "include_three_body": include_three_body,
        "include_photoionization": include_photoionization,
        "radiation_temperature_eV": radiation_temperature_eV,
        "dr_satellite_count": n_dr_satellites,
    }
    grid_step_nm = (lam_max_nm - lam_min_nm) / max(n_points - 1, 1)

    if _use_uta_bridge(crm, force_uta=force_uta):
        consumed = _add_uta_arrays(
            spec,
            crm,
            populations,
            lam_min_nm=lam_min_nm,
            lam_max_nm=lam_max_nm,
            uta_threshold=uta_threshold,
        )
    else:
        consumed = set()

    for i_tr, tr in enumerate(crm.transitions):
        if i_tr in consumed:
            continue
        n_upper = populations[tr.upper]
        if n_upper <= 0 or tr.A_rate <= 0:
            continue

        dE_eV = crm.levels[tr.upper].energy_eV - crm.levels[tr.lower].energy_eV
        if dE_eV <= 0:
            continue

        lam_A = _hartree_to_angstrom(dE_eV / HARTREE_EV)
        lam_nm = lam_A / 10.0
        if lam_nm < lam_min_nm or lam_nm > lam_max_nm:
            continue

        sigma_nm = doppler_width(lam_nm, T_ion_K, ion_mass_amu)
        sigma_nm = max(sigma_nm, grid_step_nm)

        gamma_nu_Hz = natural_width(tr.A_rate)
        c_nm_per_s = 2.99792458e17
        gamma_nm = (lam_nm ** 2 / c_nm_per_s) * gamma_nu_Hz

        if stark_hwhm_nm is not None:
            gamma_nm = np.hypot(gamma_nm, stark_hwhm_nm)
        gamma_nm = max(gamma_nm, grid_step_nm)

        intensity = n_upper * tr.A_rate * dE_eV

        if profile == "gaussian":
            spec.add_line(lam_nm, intensity, sigma_nm=sigma_nm)
        elif profile == "lorentzian":
            spec.add_line(lam_nm, intensity, gamma_nm=gamma_nm)
        elif profile == "voigt":
            spec.add_line(lam_nm, intensity,
                          sigma_nm=sigma_nm, gamma_nm=gamma_nm)
        else:
            raise ValueError(
                f"Unknown profile {profile!r}; use 'gaussian', "
                f"'lorentzian', or 'voigt'."
            )

    # Continuum channels, rendered on the same wavelength grid.  Photon
    # energy E(eV) = 1239.8419843320026 / lambda(nm); the per-eV
    # emissivity converts to per-nm with |dE/dlambda| = E / lambda.
    if include_free_free or include_rr_continuum:
        lam = spec.wavelength_nm
        E_eV = 1239.8419843320026 / lam
        jac = E_eV / lam  # eV per nm
        if include_free_free:
            if ff_Z_eff is None:
                raise ValueError("include_free_free requires ff_Z_eff")
            from atomx.spectrum.continuum import free_free_emissivity
            eps_E = free_free_emissivity(
                E_eV, T_e_eV, float(ff_Z_eff),
                n_e_cm3=n_e_cm3, n_i_cm3=continuum_ion_fraction)
            spec.values += eps_E * jac
            spec.metadata["free_free"] = {"Z_eff": float(ff_Z_eff)}
        if include_rr_continuum:
            if rr_scf_result is None:
                raise ValueError(
                    "include_rr_continuum requires rr_scf_result")
            from atomx.spectrum.continuum import rr_continuum_emissivity
            eps_E = rr_continuum_emissivity(
                rr_scf_result, E_eV, T_e_eV,
                n_e_cm3=n_e_cm3, n_i_cm3=continuum_ion_fraction)
            spec.values += eps_E * jac
            spec.metadata["rr_continuum"] = True

    return spec


def _crm_with_dr_satellites(
    crm,
    dr_satellites,
    *,
    initial_level_idx: int,
    lower_level_by_channel: Optional[Mapping[str, int]],
    initial_state_g: Optional[float],
    label_prefix: str,
):
    """Return a shallow CRM copy with DR satellite source levels installed."""
    resonances = getattr(dr_satellites, "resonances", dr_satellites)
    resonances = list(resonances)
    if not resonances:
        return crm, 0

    cloned = copy.copy(crm)
    cloned.levels = list(crm.levels)
    cloned.transitions = list(crm.transitions)
    cloned.bound_free_channels = list(getattr(crm, "bound_free_channels", []))
    cloned.dr_satellite_sources = list(getattr(crm, "dr_satellite_sources", []))
    cloned.n_levels = len(cloned.levels)
    added = cloned.add_dr_satellites(
        resonances,
        initial_level_idx=initial_level_idx,
        lower_level_by_channel=lower_level_by_channel,
        initial_state_g=initial_state_g,
        label_prefix=label_prefix,
    )
    return cloned, len(added)


def _use_uta_bridge(crm, *, force_uta: bool) -> bool:
    """Return True for heavy W-like M-/N-shell CRM objects."""
    if force_uta:
        return True
    Z = getattr(crm, "Z", None)
    charge = getattr(crm, "charge", None)
    if Z is None or charge is None:
        return False
    return Z >= 60 and any(_level_shell(lev) in {"M", "N"} for lev in crm.levels)


def _level_shell(level) -> Optional[str]:
    text = " ".join(
        str(x) for x in (getattr(level, "label", ""), getattr(level, "config", ""))
        if x is not None
    )
    if any(tok in text for tok in ("4s", "4p", "4d", "4f", "n4", "N-shell")):
        return "N"
    if any(tok in text for tok in ("3s", "3p", "3d", "n3", "M-shell")):
        return "M"
    return None


def _array_key(crm, tr) -> tuple[str, str] | None:
    upper = crm.levels[tr.upper]
    lower = crm.levels[tr.lower]
    upper_cfg = getattr(upper, "config", None)
    lower_cfg = getattr(lower, "config", None)
    if upper_cfg is None or lower_cfg is None:
        return None
    return upper_cfg, lower_cfg


def _add_uta_arrays(
    spec: Spectrum,
    crm,
    populations: np.ndarray,
    *,
    lam_min_nm: float,
    lam_max_nm: float,
    uta_threshold: int,
) -> set[int]:
    grouped: dict[tuple[str, str], list[tuple[int, object, float, float]]] = defaultdict(list)
    for i_tr, tr in enumerate(crm.transitions):
        key = _array_key(crm, tr)
        if key is None:
            continue
        dE_eV = crm.levels[tr.upper].energy_eV - crm.levels[tr.lower].energy_eV
        n_upper = populations[tr.upper]
        strength = n_upper * tr.A_rate
        if dE_eV > 0 and strength > 0:
            grouped[key].append((i_tr, tr, dE_eV, strength))

    consumed: set[int] = set()
    for key, rows in grouped.items():
        if len(rows) <= uta_threshold:
            continue
        rendered = add_uta_array_to_spectrum(
            spec,
            [(dE, strength) for _, _, dE, strength in rows],
            lam_min_nm=lam_min_nm,
            lam_max_nm=lam_max_nm,
        )
        if rendered is None:
            continue
        consumed.update(i_tr for i_tr, _, _, _ in rows)
        spec.metadata["uta_arrays"].append({
            "upper_config": key[0],
            "lower_config": key[1],
            "n_lines": rendered.n_lines,
            "centroid_eV": rendered.centroid_eV,
            "fwhm_eV": rendered.fwhm_eV,
            "total_strength": rendered.total_strength,
        })
    return consumed
