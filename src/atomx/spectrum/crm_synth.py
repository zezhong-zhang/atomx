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

from atomx.constants import BOLTZMANN_EV, HARTREE_EV, HARTREE_INV_ANGSTROM, HC_EV_NM
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
    if line_resolution not in {"uta", "ci"}:
        raise ValueError("line_resolution must be 'uta' or 'ci'")
    if profile not in {"gaussian", "lorentzian", "voigt"}:
        raise ValueError("profile must be 'gaussian', 'lorentzian', or 'voigt'")
    if stark_hwhm_nm is not None and (not np.isfinite(stark_hwhm_nm) or stark_hwhm_nm < 0):
        raise ValueError("stark_hwhm_nm must be finite and nonnegative")
    if line_resolution == "ci":
        ci_source = getattr(ci_result, "_result", ci_result)
        if ci_source is None or not hasattr(ci_source, "subshells"):
            raise ValueError("line_resolution='ci' requires a solved closed-shell ci_result")

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
        T_ion_K = T_e_eV / BOLTZMANN_EV

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
        "ci_groups": [],
        "ci_fallbacks": [],
        "line_resolution": line_resolution,
        "include_eii": include_eii,
        "include_three_body": include_three_body,
        "include_photoionization": include_photoionization,
        "radiation_temperature_eV": radiation_temperature_eV,
        "dr_satellite_count": n_dr_satellites,
    }
    total_decay = np.zeros(len(crm.levels))
    for i, level in enumerate(crm.levels):
        total_decay[i] += max(float(getattr(level, "autoionization_rate_s", 0.0)), 0.0)
    for tr in crm.transitions:
        if tr.A_rate > 0:
            total_decay[tr.upper] += tr.A_rate
    # Autoionizing synthetic levels carry the nonradiative contribution.
    for upper, rate in getattr(crm, "autoionization_rates", {}).items():
        total_decay[int(upper)] += float(rate)

    consumed: set[int] = set()
    if line_resolution == "ci":
        # J-resolved CRMs (including from_atom) already carry individual
        # transition rates.  Their levels have no configuration labels: a
        # statistical redistribution would discard their solved populations.
        for i_tr, tr in enumerate(crm.transitions):
            if tr.A_rate > 0 and _array_key(crm, tr) is None:
                spec.metadata["ci_fallbacks"].append({
                    "transition_index": i_tr,
                    "reason": "missing configuration labels; retained CRM transition",
                })
        consumed = _add_ci_groups(
            spec, crm, populations, ci_source,
            ci_e_max_eV=ci_e_max_eV, ci_det_cap=ci_det_cap,
            lam_min_nm=lam_min_nm, lam_max_nm=lam_max_nm,
            T_ion_K=T_ion_K, ion_mass_amu=ion_mass_amu,
            profile=profile, stark_hwhm_nm=stark_hwhm_nm)
    if _use_uta_bridge(crm, force_uta=force_uta):
        consumed |= _add_uta_arrays(
            spec, crm, populations,
            lam_min_nm=lam_min_nm, lam_max_nm=lam_max_nm,
            uta_threshold=uta_threshold, excluded=consumed)

    for i_tr, tr in enumerate(crm.transitions):
        if i_tr in consumed:
            continue
        n_upper = populations[tr.upper]
        if n_upper <= 0 or tr.A_rate <= 0:
            continue
        dE_eV = crm.levels[tr.upper].energy_eV - crm.levels[tr.lower].energy_eV
        if dE_eV <= 0:
            continue
        lam_nm = _hartree_to_angstrom(dE_eV / HARTREE_EV) / 10.0
        if lam_nm < lam_min_nm or lam_nm > lam_max_nm:
            continue
        _render_line(spec, lam_nm, n_upper * tr.A_rate * dE_eV,
                     total_decay[tr.upper] + total_decay[tr.lower],
                     T_ion_K, ion_mass_amu, profile, stark_hwhm_nm)

    # Continuum channels, rendered on the same wavelength grid.  Photon
    # energy E(eV) = HC_EV_NM / lambda(nm); the per-eV emissivity
    # converts to per-nm with |dE/dlambda| = E / lambda.
    if include_free_free or include_rr_continuum:
        lam = spec.wavelength_nm
        E_eV = HC_EV_NM / lam
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
    excluded: set[int] | None = None,
) -> set[int]:
    grouped: dict[tuple[str, str], list[tuple[int, object, float, float]]] = defaultdict(list)
    for i_tr, tr in enumerate(crm.transitions):
        if excluded and i_tr in excluded:
            continue
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


def _render_line(spec, lam_nm, intensity, decay_rate, T_ion_K,
                 ion_mass_amu, profile, stark_hwhm_nm):
    sigma = doppler_width(lam_nm, T_ion_K, ion_mass_amu)
    gamma = lam_nm**2 / 2.99792458e17 * natural_width(decay_rate)
    if stark_hwhm_nm is not None:
        # This option models an additional Lorentzian HWHM; a true
        # quasi-static microfield distribution needs its full kernel.
        gamma += stark_hwhm_nm
    if profile == "gaussian":
        spec.add_line(lam_nm, intensity, sigma_nm=sigma)
    elif profile == "lorentzian":
        spec.add_line(lam_nm, intensity, gamma_nm=gamma)
    else:
        spec.add_line(lam_nm, intensity, sigma_nm=sigma, gamma_nm=gamma)


def _add_ci_groups(spec, crm, populations, ci_result, *, ci_e_max_eV,
                   ci_det_cap, lam_min_nm, lam_max_nm, T_ion_K,
                   ion_mass_amu, profile, stark_hwhm_nm):
    from atomx.properties.transitions.ci_lines import CISizeError, ci_line_strengths

    grouped = defaultdict(list)
    for i, tr in enumerate(crm.transitions):
        key = _array_key(crm, tr)
        if key is not None and tr.A_rate > 0:
            grouped[key].append(i)
    consumed = set()
    resolved = []
    ci_decay = defaultdict(float)
    for (upper_cfg, lower_cfg), indices in grouped.items():
        try:
            line_set = ci_line_strengths(
                ci_result, upper_cfg, lower_cfg, e_max_eV=ci_e_max_eV,
                det_cap=ci_det_cap)
        except (ValueError, NotImplementedError, CISizeError) as exc:
            spec.metadata["ci_fallbacks"].append({"upper_config": upper_cfg,
                "lower_config": lower_cfg, "reason": str(exc)})
            continue
        upper_levels = [level for level in line_set.levels_i
                        if level.config == upper_cfg]
        if not upper_levels:
            spec.metadata["ci_fallbacks"].append({"upper_config": upper_cfg,
                "lower_config": lower_cfg, "reason": "No upper CI levels"})
            continue
        upper_population = sum(float(populations[i]) for i, level in enumerate(crm.levels)
                               if getattr(level, "config", None) == upper_cfg)
        statistical_sum = sum(level.two_J+1 for level in upper_levels)
        for line in line_set.lines:
            ci_decay[(line.upper.config, line.upper.index)] += line.A_per_s
        resolved.append((upper_cfg, lower_cfg, indices, line_set,
                         upper_population, statistical_sum))

    # Collect every supported branch before rendering. A CI level's
    # lifetime is the sum over all its radiative exits, not just the pair
    # currently being written into the spectrum.
    for upper_cfg, lower_cfg, indices, line_set, upper_population, statistical_sum in resolved:
        line_records = []
        for line in line_set.lines:
            if line.upper.config != upper_cfg or line.lower.config != lower_cfg:
                continue
            lam_nm = line.wavelength_angstrom/10
            if not lam_min_nm <= lam_nm <= lam_max_nm:
                continue
            population = upper_population*(line.upper.two_J+1)/statistical_sum
            intensity = population*line.A_per_s*line.omega_au*HARTREE_EV
            _render_line(spec, lam_nm, intensity,
                         ci_decay[(line.upper.config, line.upper.index)]
                         + ci_decay[(line.lower.config, line.lower.index)],
                         T_ion_K, ion_mass_amu, profile, stark_hwhm_nm)
            line_records.append({"wavelength_nm": lam_nm,
                "upper_J": line.upper.two_J/2,
                "lower_J": line.lower.two_J/2,
                "A_per_s": line.A_per_s, "population": population,
                "intensity": intensity})
        spec.metadata["ci_groups"].append({"engine": "ci",
            "upper_config": upper_cfg, "lower_config": lower_cfg,
            "n_lines": len(line_records), "lines": line_records})
        consumed.update(indices)
    return consumed
