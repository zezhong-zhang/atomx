#!/usr/bin/env python3
"""Audit every archived NIST isotope line against mass-aware atomx estimates.

The point-Dirac column is restricted to classified one-electron inter-n lines.
The DHF column transfers the computed leading normal mass shift to the named
isotope; a single-configuration DHF energy does not resolve multiplet lines.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from atomx.constants import HARTREE_CM1
from atomx.physics.dirac import hydrogenic_dirac_energy
from scripts.generate_full_dhf_nist import expand_nist_config, configuration_has_single_level


def number(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw.strip().strip('[]()'))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def shifted_energy(job: tuple | None, target_mass: float) -> float | None:
    if job is None:
        return None
    status, energy, nms, original_mass = job
    if status != 'done' or energy is None or nms is None or not original_mass:
        return None
    return energy + nms * (original_mass / target_mass - 1)


def collect(archive: Path, ledger: Path, isotopes: Path) -> tuple[list[dict], dict]:
    # Import here to avoid a circular import: the overview imports collect().
    from scripts.compare_theory_nist import calculated_any_nm

    masses = json.loads(isotopes.read_text())['all_isotopes']
    jobs: dict[tuple[int, int, str], tuple] = {}
    if ledger.exists():
        with sqlite3.connect(ledger) as db:
            jobs = {(z, charge, config): (status, energy, nms, mass)
                    for z, charge, config, status, energy, nms, mass in db.execute(
                        'SELECT z,charge,configuration,status,energy_ha,nms_ha,'
                        'nuclear_mass_me FROM config_jobs')}
    records = []
    counts: Counter[str] = Counter()
    with sqlite3.connect(archive) as db:
        db.row_factory = sqlite3.Row
        source = db.execute('''SELECT r.spectrum,r.row_index,r.record_json,
                                     s.z,s.mass_number,s.charge,s.url,s.sha256
                              FROM nist_isotope_records r
                              JOIN nist_isotope_sources s USING(kind,spectrum)
                              WHERE r.kind='lines' AND s.status='imported'
                              ORDER BY s.z,s.mass_number,s.charge,r.row_index''')
        for item in source:
            record = json.loads(item['record_json'])
            z, charge, a = item['z'], item['charge'], item['mass_number']
            mass_data = masses.get(f'{z}:{a}')
            mass = mass_data['nuclear_mass_me'] if mass_data else None
            lower = expand_nist_config(record.get('conf_i', ''), z - charge)
            upper = expand_nist_config(record.get('conf_k', ''), z - charge)
            observed = number(record.get('obs_wl_vac(nm)'))
            ritz = number(record.get('ritz_wl_vac(nm)'))
            dhf_nm = None
            dirac_nm = None
            if mass and lower and upper:
                low_energy = shifted_energy(jobs.get((z, charge, lower)), mass)
                high_energy = shifted_energy(jobs.get((z, charge, upper)), mass)
                if low_energy is not None and high_energy is not None:
                    delta = (high_energy - low_energy) * HARTREE_CM1
                    if delta > 1e-3:
                        dhf_nm = 1e7 / delta
                if charge == z - 1:
                    point_nm = calculated_any_nm(
                        z, lower, record.get('J_i', ''),
                        upper, record.get('J_k', ''))
                    if point_nm is not None:
                        dirac_nm = point_nm * (1 + 1 / mass)
            unique = (lower is not None and upper is not None and
                      configuration_has_single_level(lower) and
                      configuration_has_single_level(upper))
            if observed is None:
                status = 'ritz_only' if ritz is not None else 'no_wavelength'
            elif mass is None:
                status = 'missing_isotope_mass'
            elif lower is None or upper is None:
                status = 'unclassified_configuration'
            elif dhf_nm is not None and unique:
                status = 'dhf_unique_level'
            elif dhf_nm is not None:
                status = 'dhf_configuration_average'
            elif dirac_nm is not None:
                status = 'dirac_mass_baseline_only'
            else:
                status = 'no_resolved_theory_pair'
            counts[status] += 1
            records.append({
                'spectrum': item['spectrum'], 'z': z, 'charge': charge,
                'mass_number': a, 'row_index': item['row_index'],
                'source_url': item['url'], 'source_sha256': item['sha256'],
                'observed_nm': observed,
                'observed_unc_nm': number(record.get('unc_obs_wl')),
                'ritz_nm': ritz,
                'ritz_unc_nm': number(record.get('unc_ritz_wl')),
                'lower_configuration_raw': record.get('conf_i', ''),
                'lower_configuration': lower,
                'lower_term': record.get('term_i', ''),
                'lower_j': record.get('J_i', ''),
                'upper_configuration_raw': record.get('conf_k', ''),
                'upper_configuration': upper,
                'upper_term': record.get('term_k', ''),
                'upper_j': record.get('J_k', ''),
                'lower_level_id': record.get('ID_i', ''),
                'upper_level_id': record.get('ID_k', ''),
                'nuclear_mass_me': mass,
                'dhf_mass_transferred_nm': dhf_nm,
                'dhf_energy_error_percent':
                    100 * (observed / dhf_nm - 1)
                    if observed is not None and dhf_nm else None,
                'dirac_mass_nm': dirac_nm,
                'dirac_energy_error_percent':
                    100 * (observed / dirac_nm - 1)
                    if observed is not None and dirac_nm else None,
                'status': status,
            })
    return records, {'line_rows': len(records), 'statuses': dict(counts),
                     'observed_rows': sum(r['observed_nm'] is not None for r in records),
                     'model_note': 'DHF mass transfer uses only the leading normal mass shift; specific mass shift, field shift, relativistic recoil and isotope-dependent radiative shifts are absent. Multiplet averages are unscored.'}


def collect_levels(archive: Path, ledger: Path, isotopes: Path) -> tuple[list[dict], dict]:
    from scripts.compare_theory_nist import kappa_of

    masses = json.loads(isotopes.read_text())['all_isotopes']
    jobs: dict[tuple[int, int, str], tuple] = {}
    if ledger.exists():
        with sqlite3.connect(ledger) as db:
            jobs = {(z, charge, config): (status, energy, nms, mass)
                    for z, charge, config, status, energy, nms, mass in db.execute(
                        'SELECT z,charge,configuration,status,energy_ha,nms_ha,'
                        'nuclear_mass_me FROM config_jobs')}
    sources: dict[str, list[tuple[sqlite3.Row, dict]]] = {}
    with sqlite3.connect(archive) as db:
        db.row_factory = sqlite3.Row
        for item in db.execute('''SELECT r.spectrum,r.row_index,r.record_json,
                                         s.z,s.mass_number,s.charge,s.url,s.sha256
                                  FROM nist_isotope_records r
                                  JOIN nist_isotope_sources s USING(kind,spectrum)
                                  WHERE r.kind='levels' AND s.status='imported'
                                  ORDER BY s.z,s.mass_number,s.charge,r.row_index'''):
            sources.setdefault(item['spectrum'], []).append((item, json.loads(item['record_json'])))
    result = []
    counts: Counter[str] = Counter()
    for spectrum, group in sources.items():
        numeric = [(number(rec.get('Level (cm-1)')), item, rec)
                   for item, rec in group]
        ground = min((row for row in numeric if row[0] is not None),
                     key=lambda row: (row[0], row[1]['row_index']), default=None)
        for item, record in group:
            z, charge, a = item['z'], item['charge'], item['mass_number']
            mass_data = masses.get(f'{z}:{a}')
            mass = mass_data['nuclear_mass_me'] if mass_data else None
            config = expand_nist_config(record.get('Configuration', ''), z - charge)
            ground_config = (expand_nist_config(
                ground[2].get('Configuration', ''), z - charge) if ground else None)
            measured = number(record.get('Level (cm-1)'))
            prefix = record.get('Prefix', '')
            suffix = record.get('Suffix', '')
            reference_kind = ('theoretical' if '(' in prefix else
                              'semiempirical' if '[' in prefix else
                              'offset' if '+' in suffix else 'evaluated')
            dhf_cm1 = None
            dirac_cm1 = None
            if mass and config and ground_config:
                state_energy = shifted_energy(jobs.get((z, charge, config)), mass)
                base_energy = shifted_energy(jobs.get((z, charge, ground_config)), mass)
                if state_energy is not None and base_energy is not None:
                    dhf_cm1 = (state_energy - base_energy) * HARTREE_CM1
                if charge == z - 1:
                    state = kappa_of(config, record.get('J', ''))
                    base = kappa_of(ground_config, ground[2].get('J', ''))
                    if state and base and state[0] != base[0]:
                        dirac_cm1 = ((hydrogenic_dirac_energy(*state, z)
                                      - hydrogenic_dirac_energy(*base, z))
                                     * HARTREE_CM1 / (1 + 1 / mass))
            unique = bool(config and ground_config and
                          configuration_has_single_level(config) and
                          configuration_has_single_level(ground_config))
            if ground is not None and item['row_index'] == ground[1]['row_index']:
                status = 'ground_reference'
            elif measured is None:
                status = 'no_numeric_energy'
            elif mass is None:
                status = 'missing_isotope_mass'
            elif config is None or ground_config is None:
                status = 'unclassified_configuration'
            elif dhf_cm1 is None:
                status = 'no_dhf_configuration_pair'
            elif reference_kind != 'evaluated':
                status = f'{reference_kind}_reference'
            elif unique:
                status = 'dhf_unique_level'
            else:
                status = 'dhf_configuration_average'
            counts[status] += 1
            result.append({
                'spectrum': spectrum, 'z': z, 'charge': charge,
                'mass_number': a, 'row_index': item['row_index'],
                'source_url': item['url'], 'source_sha256': item['sha256'],
                'level_id': record.get('Level ID', ''),
                'configuration_raw': record.get('Configuration', ''),
                'configuration': config, 'term': record.get('Term', ''),
                'j': record.get('J', ''),
                'energy_cm1': measured,
                'energy_prefix': prefix, 'energy_suffix': suffix,
                'uncertainty_cm1': number(record.get('Uncertainty (cm-1)')),
                'reference': record.get('Reference', ''),
                'reference_kind': reference_kind,
                'ground_level_id': ground[2].get('Level ID', '') if ground else '',
                'nuclear_mass_me': mass,
                'dhf_mass_transferred_cm1': dhf_cm1,
                'dhf_residual_cm1': dhf_cm1 - measured
                    if dhf_cm1 is not None and measured is not None else None,
                'dirac_mass_cm1': dirac_cm1,
                'dirac_residual_cm1': dirac_cm1 - measured
                    if dirac_cm1 is not None and measured is not None else None,
                'status': status,
            })
    return result, {'level_rows': len(result), 'statuses': dict(counts),
                    'model_note': 'Each excitation is relative to the lowest numeric level in the same NIST isotope spectrum. DHF is a configuration average except where both configurations have one possible level. Bracketed, parenthesized and offset references are not strict experimental scores.'}


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
