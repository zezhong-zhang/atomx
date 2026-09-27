#!/usr/bin/env python
"""Match cached atomx DHF/MCDHF states to NIST observed transitions.

Only complete, configuration-validated (2J, parity) blocks are paired. A line
is scored only when both classified endpoints resolve to paired NIST levels.
"""

from __future__ import annotations

import collections
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from verify_nist_levels import (  # noqa: E402
    CASES, CM1, Cache, _nist_block, _owner, choose_model,
    ci_model_physical_error, compare_ci_rung, compare_config_rung,
)
import nist_asd as nist  # noqa: E402


def paired_levels(case, ion, model, payload) -> dict[int, float]:
    """Return NIST row index → atomx excitation energy in cm⁻¹.

    This reproduces the benchmark block pairing and checks its paired count.
    """
    energies = payload["energies"]
    if not energies:
        return {}
    if physical_error := ci_model_physical_error(case, energies):
        raise ValueError(physical_error)
    ground = min(energies)
    blocks = collections.defaultdict(list)
    dominant = payload.get("dominant", [])
    for i, (energy, j, parity) in enumerate(zip(
            energies, payload["two_j"], payload["parity"], strict=True)):
        label = dominant[i] if i < len(dominant) else ""
        blocks[(int(j), int(parity))].append((energy, _owner(label, model)))
    for block in blocks.values():
        block.sort()
    result = {}
    for key, calculated in blocks.items():
        observed = _nist_block(ion, model, *key)
        if len(calculated) == len(observed):
            selected = calculated
        elif case.mode == "block-min" and calculated and observed:
            k = min(len(calculated), len(observed))
            if len(calculated) > k and (calculated[k][0] - ground) * CM1 <= observed[k-1].energy_cm1:
                continue
            selected = calculated[:k]
        else:
            continue
        owners = [_owner(lv.configuration, model, nist_style=True)
                  for lv in observed[:len(selected)]]
        if [owner for _, owner in selected] != owners:
            continue
        for (energy, _), lv in zip(selected, observed, strict=True):
            result[lv.index] = (energy - ground) * CM1
    expected = compare_ci_rung(case, ion, "MCDHF", model, payload, 0).n_levels
    if len(result) != expected:
        raise ValueError(f"state pairing disagrees with benchmark for {case.label}")
    return result


def configuration_average_levels(case, ion, model, energies) -> dict[int, float]:
    """Assign DHF configuration energy to each uniquely owned NIST level.

    These are configuration averages, so terms and fine-structure levels of
    the same configuration share an atomx energy. They are explicitly labeled
    DHF in every result and never presented as J-resolved predictions.
    """
    ground = energies[case.ground]
    result = {}
    for level in ion.levels:
        if level.energy_cm1 is None or len(level.j_values) != 1:
            continue
        owners = [config for config in model
                  if nist.nist_config_matches(level.configuration, config)]
        if len(owners) == 1:
            result[level.index] = (energies[owners[0]] - ground) * CM1
    expected = compare_config_rung(case, ion, "DHF", model, energies, 0).n_levels
    if len(result) != expected:
        raise ValueError(f"configuration pairing disagrees with benchmark for {case.label}")
    return result


def correction_branch_warning(plain: dict, corrected: dict) -> bool:
    """Flag large changes in uniquely labeled excitation energies.

    This checks orbital/state tracking, not the physical size of the total
    Breit/QED correction. Ambiguous dominant labels are not paired.
    """
    def unique_excitations(payload):
        states = collections.defaultdict(list)
        ground = min(payload["energies"])
        for energy, j, parity, label in zip(payload["energies"], payload["two_j"],
                                             payload["parity"], payload["dominant"],
                                             strict=True):
            states[(label, j, parity)].append(energy - ground)
        return {key: values[0] for key, values in states.items() if len(values) == 1}

    a, b = unique_excitations(plain), unique_excitations(corrected)
    return any(abs(a[key] - b[key]) > 0.5 for key in a.keys() & b.keys())


def cached_models(case, ion, model, cache):
    """Yield every complete cached rung with its NIST-level energy mapping."""
    for breit, qed, name in [(False, False, "DHF"),
                              (True, True, "DHF+B+QED")]:
        keys = [f"scf|{case.z}|{config}|{int(breit)}|{int(qed)}"
                for config in model]
        if all(key in cache.data for key in keys):
            energies = {config: float(cache.data[key])
                        for config, key in zip(model, keys, strict=True)}
            yield name, configuration_average_levels(case, ion, model, energies), "SCF cached"
    for breit, qed, name in [(False, False, "MCDHF"),
                              (True, True, "MCDHF+B+QED")]:
        key = f"ci|{case.z}|{','.join(model)}|{int(breit)}|{int(qed)}"
        if key in cache.data:
            payload = cache.data[key]
            if payload.get("converged") is False:
                continue
            if ci_model_physical_error(case, payload["energies"]):
                continue
            status = "converged" if payload.get("converged") is True else "legacy cache; convergence unrecorded"
            plain = cache.data.get(f"ci|{case.z}|{','.join(model)}|0|0")
            corrected = cache.data.get(f"ci|{case.z}|{','.join(model)}|1|1")
            if plain and corrected and correction_branch_warning(plain, corrected):
                # A correction this large signals a different state branch,
                # not a Breit/QED displacement of the same physical state.
                # Neither branch is a validated line prediction until the
                # state correspondence has been repaired.
                continue
            yield name, paired_levels(case, ion, model, payload), status


def uniquely_resolved_configuration(ion, configuration: str) -> bool:
    """Whether NIST gives one physical level for this configuration."""
    return sum(lv.energy_cm1 is not None and
               nist.nist_config_matches(configuration, lv.configuration)
               for lv in ion.levels) == 1


def endpoint(ion, levels: dict[int, float], configuration: str, term: str,
             j_text: str, energy_cm1: float | None):
    """Resolve a NIST line endpoint by classification and level energy."""
    if not configuration or not term or not j_text or energy_cm1 is None:
        return None
    candidates = [lv for lv in ion.levels if lv.index in levels
                  and lv.term == term and lv.j_text == j_text
                  and nist.nist_config_matches(configuration, lv.configuration)
                  and lv.energy_cm1 is not None
                  and abs(lv.energy_cm1 - energy_cm1) <= 5]
    return candidates[0] if len(candidates) == 1 else None


def collect(db_path: Path, cache_path: Path) -> tuple[list[dict], list[dict]]:
    """Collect all available atomx DHF/MCDHF observed line pairs."""
    cache = Cache(cache_path)
    database = sqlite3.connect(db_path)
    database.row_factory = sqlite3.Row
    line_rows, level_rows = [], []
    try:
        for case in CASES:
            ion = nist.load(case.label, with_lines=False)
            if ion.charge == ion.z - 1:
                continue  # the separate hydrogenic comparison covers these ions
            model, _, _ = choose_model(case, ion)
            for name, matched, cache_status in cached_models(case, ion, model, cache):
                largest_residual = max((abs(matched[lv.index] - lv.energy_cm1)
                                        for lv in ion.levels if lv.index in matched
                                        and lv.energy_cm1 is not None), default=0)
                if largest_residual > 100_000:
                    cache_status += "; level residual >100,000 cm⁻¹"
                for lv in ion.levels:
                    if lv.index in matched:
                        level_rows.append(dict(z=ion.z, charge=ion.charge,
                                               row_index=lv.index, model=name,
                                               cache_status=cache_status,
                                               calculated_energy_cm1=matched[lv.index],
                                               residual_cm1=matched[lv.index]-lv.energy_cm1))
                for line in database.execute("""SELECT * FROM lines WHERE z=? AND charge=?
                                               AND observed_nm IS NOT NULL AND medium='vacuum'""",
                                             (ion.z, ion.charge)):
                    lower = endpoint(ion, matched, line["lower_configuration"],
                                     line["lower_term"], line["lower_j"],
                                     line["lower_energy_cm1"])
                    upper = endpoint(ion, matched, line["upper_configuration"],
                                     line["upper_term"], line["upper_j"],
                                     line["upper_energy_cm1"])
                    if lower is None or upper is None:
                        continue
                    if name.startswith("DHF") and not (
                            uniquely_resolved_configuration(ion, lower.configuration)
                            and uniquely_resolved_configuration(ion, upper.configuration)):
                        # DHF here supplies a configuration average; assigning
                        # it to each term/J component creates fake exact-line
                        # comparisons and distorts the periodic RMS.
                        continue
                    gap = matched[upper.index] - matched[lower.index]
                    if gap <= 0 or not math.isfinite(gap):
                        continue
                    calculated_nm = 1e7 / gap
                    line_rows.append(dict(z=ion.z, charge=ion.charge,
                        symbol=case.label.split()[0], ion=case.label,
                        nist_row_index=line["row_index"], model=name,
                        cache_status=cache_status,
                        lower=lower.configuration, lower_term=lower.term,
                        lower_j=lower.j_text, upper=upper.configuration,
                        upper_term=upper.term, upper_j=upper.j_text,
                        observed_nm=line["observed_nm"],
                        observed_unc_nm=line["observed_unc_nm"],
                        nist_medium="vacuum",
                        ritz_nm=line["ritz_nm"], calculated_nm=calculated_nm,
                        energy_error_percent=100*(line["observed_nm"]/calculated_nm-1),
                        lower_level_row=lower.index, upper_level_row=upper.index))
    finally:
        database.close()
    return line_rows, level_rows


def cached_panel(cache_path: Path) -> list[dict]:
    """Current cached DHF/MCDHF scores for the level chart."""
    cache = Cache(cache_path)
    rows = []
    for case in CASES:
        ion = nist.load(case.label, with_lines=False)
        if ion.charge == ion.z - 1:
            continue
        model, _, _ = choose_model(case, ion)
        for breit, qed, name in [(False, False, "DHF"),
                                  (True, True, "DHF+B+QED"),
                                  (False, False, "MCDHF"),
                                  (True, True, "MCDHF+B+QED")]:
            if name.startswith("MCDHF"):
                key = f"ci|{case.z}|{','.join(model)}|{int(breit)}|{int(qed)}"
                if key not in cache.data:
                    continue
                if cache.data[key].get("converged") is False:
                    continue
                result = compare_ci_rung(case, ion, name, model, cache.data[key], 0)
            else:
                keys = [f"scf|{case.z}|{config}|{int(breit)}|{int(qed)}"
                        for config in model]
                if not all(key in cache.data for key in keys):
                    continue
                energies = {config: float(cache.data[key])
                            for config, key in zip(model, keys, strict=True)}
                result = compare_config_rung(case, ion, name, model, energies, 0)
            stats = result.stats
            if result.n_levels:
                rows.append(dict(system=case.label, z=case.z, rung=name,
                                 n_levels=result.n_levels,
                                 n_excluded_blocks=result.n_excluded_blocks,
                                 rms_cm1=stats["rms_cm1"], rms_eV=stats["rms_ev"]))
    return rows


def rejected_ci_panels(cache_path: Path) -> list[dict]:
    """List cached CI panels rejected by a physical bound-state check."""
    cache = Cache(cache_path)
    rows = []
    for case in CASES:
        if case.ground != "1s2":
            continue
        ion = nist.load(case.label, with_lines=False)
        model, _, _ = choose_model(case, ion)
        for breit, qed, rung in ((False, False, "MCDHF"),
                                  (True, True, "MCDHF+B+QED")):
            key = f"ci|{case.z}|{','.join(model)}|{int(breit)}|{int(qed)}"
            payload = cache.data.get(key)
            if payload is None:
                continue
            reason = ci_model_physical_error(case, payload["energies"])
            if reason:
                rows.append({"system": case.label, "rung": rung,
                             "reason": reason})
    return rows


def add_model_audit(path: Path, line_rows: list[dict], level_rows: list[dict]) -> None:
    """Store all rungs; prefer requested DHF+B+QED in the main row audit."""
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS line_model_audit(
          z INTEGER, charge INTEGER, row_index INTEGER, model TEXT,
          calculated_nm REAL, observed_energy_error_percent REAL,
          lower_level_row INTEGER, upper_level_row INTEGER,
          PRIMARY KEY(z,charge,row_index,model));
        CREATE TABLE IF NOT EXISTS level_model_audit(
          z INTEGER, charge INTEGER, row_index INTEGER, model TEXT,
          calculated_energy_cm1 REAL, residual_cm1 REAL,
          PRIMARY KEY(z,charge,row_index,model));
        """)
        db.executemany("INSERT OR REPLACE INTO line_model_audit VALUES (?,?,?,?,?,?,?,?)",
                       [(r["z"],r["charge"],r["nist_row_index"],r["model"],
                         r["calculated_nm"],r["energy_error_percent"],
                         r["lower_level_row"],r["upper_level_row"]) for r in line_rows])
        db.executemany("INSERT OR REPLACE INTO level_model_audit VALUES (?,?,?,?,?,?)",
                       [(r["z"],r["charge"],r["row_index"],r["model"],
                         r["calculated_energy_cm1"],r["residual_cm1"]) for r in level_rows])
        for z, charge in {(r["z"], r["charge"]) for r in level_rows}:
            db.execute("""UPDATE line_audit SET status='no_validated_state_pair'
                WHERE z=? AND charge=? AND status='no_atomx_line_calculation'
                """, (z, charge))
            db.execute("""UPDATE level_audit SET status='no_validated_level_pair'
                WHERE z=? AND charge=? AND status='no_atomx_level_calculation'""",
                       (z, charge))
        priority = {"DHF": 0, "MCDHF": 1,
                    "MCDHF+B+QED": 2, "DHF+B+QED": 3,
                    "DHF+B+QED+NMS": 4}
        for r in sorted(line_rows, key=lambda item: priority[item["model"]]):
            db.execute("""UPDATE line_audit SET status='compared_observed',
                    calculated_nm=?,observed_energy_error_percent=?,
                    ritz_energy_error_percent=?,model=?
                    WHERE z=? AND charge=? AND row_index=?""",
                    (r["calculated_nm"],r["energy_error_percent"],
                     100*(r["ritz_nm"]/r["calculated_nm"]-1) if r["ritz_nm"] else None,
                     r["model"],r["z"],r["charge"],r["nist_row_index"]))
        for r in sorted(level_rows, key=lambda item: priority[item["model"]]):
            db.execute("""UPDATE level_audit SET status='compared_nist_level',
                    calculated_energy_cm1=?,residual_cm1=?,model=?
                    WHERE z=? AND charge=? AND row_index=?""",
                    (r["calculated_energy_cm1"],r["residual_cm1"],r["model"],
                     r["z"],r["charge"],r["row_index"]))
