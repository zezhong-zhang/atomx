"""NIST state matching and configuration centroids on caller-owned records.

No solver dispatch, repository paths, network access, or archive mutation.
Configuration averages and evaluated complete manifolds remain distinct
from resolved spectral lines. Unmatched and incomplete records are retained.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from math import comb
import math
import sqlite3
from atomx.config import ground_state_config
from atomx.config.parsing import parse_nr_shells
from atomx.reference_data.nist import _filled, configuration_map

def expand_nist_config(text: str, electrons: int) -> str | None:
    """Restore only closed inner subshells omitted by NIST's notation."""
    if not text.strip() or not 1 <= electrons <= 118:
        return None
    shown = configuration_map(text)
    if not shown or any(q <= 0 or q > _filled(*key)
                        for key, q in shown.items()):
        return None
    deficit = electrons - sum(shown.values())
    if deficit < 0:
        return None
    full = dict(shown)
    reference = configuration_map(ground_state_config(electrons))
    for key in sorted(reference):
        occupancy = reference[key]
        if key not in full and occupancy == _filled(*key) and occupancy <= deficit:
            full[key] = occupancy
            deficit -= occupancy
    if deficit:
        return None
    letters = "spdfghiklmnoqrtuvwxyz"
    config = " ".join(f"{n}{letters[l]}{q}" for (n, l), q in sorted(full.items()))
    try:
        parsed = parse_nr_shells(config)
    except ValueError:
        return None
    return config if sum(q for _, _, q in parsed) == electrons else None


@lru_cache(maxsize=None)
def configuration_has_single_level(configuration: str) -> bool:
    """Whether a nonrelativistic configuration has only one term/J level.

    A closed-shell core plus one s electron has J=1/2.  Any open p, d, or
    higher shell, or multiple open shells, produces a multiplet.  NIST may
    list only one member of a multiplet, so archive row count alone cannot
    establish physical uniqueness for a configuration-average DHF energy.
    """
    try:
        open_shells = [(ell, occupation) for n, ell, occupation
                       in parse_nr_shells(configuration)
                       if occupation != _filled(n, ell)]
    except ValueError:
        return False
    return (not open_shells or
            len(open_shells) == 1 and open_shells[0] == (0, 1))


def refractive_index_air(vacuum_nm: float) -> float:
    """Peck–Reeder standard-air index used for the NIST air comparison."""
    wavelength_um = vacuum_nm / 1000.0
    inv_square = wavelength_um ** -2
    return (1 + 8.06051e-5 + 2.480990e-2 / (132.274 - inv_square)
            + 1.74557e-4 / (39.32957 - inv_square))


def observed_vacuum_nm(wavelength_nm: float, medium: str) -> float | None:
    if medium == "vacuum":
        return wavelength_nm
    if medium != "air":
        return None
    vacuum = wavelength_nm * 1.00028
    for _ in range(5):
        vacuum = wavelength_nm * refractive_index_air(vacuum)
    return vacuum


def weight_from_j(j_text: str | None) -> int | None:
    try:
        j = Fraction((j_text or "").strip())
    except (ValueError, ZeroDivisionError):
        return None
    g = 2 * j + 1
    return int(g) if j >= 0 and g.denominator == 1 else None


def expected_weight(configuration: str) -> int:
    weight = 1
    for _n, ell, occupation in parse_nr_shells(configuration):
        weight *= comb(4 * ell + 2, occupation)
    return weight


def level_centroids(source: sqlite3.Connection, jobs: set[tuple[int, int, str]],
                    max_z: int) -> dict[tuple[int, int, str], dict]:
    groups: dict[tuple[int, int, str], dict] = defaultdict(lambda: dict(
        level_rows=0, weighted_energy=0.0, weight_sum=0,
        missing_energy_or_j=0, non_evaluated=0, g_disagreements=0))
    columns = {r[1] for r in source.execute('PRAGMA table_info(levels)')}
    configuration = ("COALESCE(NULLIF(configuration_nist,''),configuration)"
                     if 'configuration_nist' in columns else 'configuration')
    bound_filter = " AND lower(COALESCE(term,'')) NOT LIKE '%limit%'" if 'term' in columns else ''
    for z, charge, config, j_text, energy, kind, g_raw in source.execute(
            f"""SELECT z,charge,{configuration},j_text,energy_cm1,energy_kind,
                      statistical_weight_raw FROM levels WHERE z<=?{bound_filter}""",
            (max_z,)):
        if not config:
            continue
        full = expand_nist_config(config, z - charge)
        key = (z, charge, full)
        if full is None or key not in jobs:
            continue
        group = groups[key]
        group["level_rows"] += 1
        g = weight_from_j(j_text)
        if g is None or energy is None or not math.isfinite(energy):
            group["missing_energy_or_j"] += 1
            continue
        group["weight_sum"] += g
        group["weighted_energy"] += g * energy
        if kind != "evaluated":
            group["non_evaluated"] += 1
        try:
            if g_raw and Fraction(g_raw.strip()) != g:
                group["g_disagreements"] += 1
        except (ValueError, ZeroDivisionError):
            pass
    result = {}
    for key in jobs:
        group = groups.get(key)
        expected = expected_weight(key[2])
        complete = (group is not None and group["level_rows"] > 0
                    and group["missing_energy_or_j"] == 0
                    and group["weight_sum"] == expected)
        status = ("missing_nist_levels" if group is None else
                  "incomplete_manifold" if not complete else
                  "nonexperimental_complete" if group["non_evaluated"] else
                  "evaluated_complete")
        result[key] = {
            "status": status,
            "centroid_cm1": group["weighted_energy"] / expected if complete else None,
            "expected_weight": expected,
            "listed_weight": group["weight_sum"] if group else 0,
            "level_rows": group["level_rows"] if group else 0,
            "missing_energy_or_j": group["missing_energy_or_j"] if group else 0,
            "non_evaluated_levels": group["non_evaluated"] if group else 0,
            "nist_g_disagreements": group["g_disagreements"] if group else 0,
        }
    return result
