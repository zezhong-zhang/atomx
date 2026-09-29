"""Compare DHF+B+QED+mass ground-state ionization energies with NIST."""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path

from atomx.constants import HARTREE_CM1, HARTREE_EV
from atomx.config.parsing import parse_nr_shells

from scripts.generate_full_dhf_nist import ground_configuration_with_dhf_candidates
from atomx.reference_data.matching import configuration_has_single_level, expand_nist_config
from atomx.reference_data.matching import weight_from_j


def reference_kind(raw: str | None) -> str:
    """NIST ASD GSIE notation: measured/evaluated, semiempirical, or theory."""
    value = (raw or "").strip()
    if not value:
        return "missing"
    if value.startswith("["):
        return "semiempirical"
    if value.startswith("("):
        return "theoretical"
    return "evaluated"


def ground_centroids(reference: sqlite3.Connection,
                     grounds: dict[tuple[int, int], str | None],
                     max_z: int) -> dict[tuple[int, int], dict]:
    """Build configuration centroids only from complete assigned J manifolds.

    Derive weights from J rather than NIST's displayed g: the source audit
    found several displayed g values that violate g=2J+1 (notably S I).
    The original g is preserved and every disagreement is counted.
    """
    groups: dict[tuple[int, int], dict] = defaultdict(lambda: dict(
        level_rows=0, weighted_energy=0.0, weight_sum=0,
        unweighted_rows=0, non_evaluated_rows=0, g_disagreements=0))
    for z, charge, config, j_text, energy, kind, g_raw in reference.execute(
            """SELECT z,charge,configuration,j_text,energy_cm1,energy_kind,
                      statistical_weight_raw FROM levels WHERE z<=?""",
            (max_z,)):
        key = (z, charge)
        if not config or expand_nist_config(config, z - charge) != grounds.get(key):
            continue
        group = groups[key]
        group["level_rows"] += 1
        g = weight_from_j(j_text)
        if g is None or energy is None:
            group["unweighted_rows"] += 1
            continue
        group["weight_sum"] += g
        group["weighted_energy"] += g * energy
        if kind != "evaluated":
            group["non_evaluated_rows"] += 1
        try:
            if g_raw and Fraction(g_raw.strip()) != g:
                group["g_disagreements"] += 1
        except (ValueError, ZeroDivisionError):
            pass
    result = {}
    for key, config in grounds.items():
        if config is None:
            continue
        expected = 1
        for _n, ell, occupation in parse_nr_shells(config):
            expected *= comb(4 * ell + 2, occupation)
        group = groups[key]
        complete = (group["level_rows"] > 0 and not group["unweighted_rows"]
                    and group["weight_sum"] == expected)
        centroid = (group["weighted_energy"] / expected if complete else None)
        status = ("evaluated_complete" if complete and
                  not group["non_evaluated_rows"] else
                  "nonexperimental_complete" if complete else "incomplete")
        result[key] = dict(centroid_cm1=centroid,
                           status=status, expected_weight=expected,
                           listed_weight=group["weight_sum"],
                           level_rows=group["level_rows"],
                           nist_g_disagreements=group["g_disagreements"])
    return result


def collect(archive_path: Path, ledger_path: Path, max_z: int = 20) -> list[dict]:
    if not ledger_path.exists():
        return []
    reference = sqlite3.connect(archive_path)
    ledger = sqlite3.connect(ledger_path)
    job_rows = ledger.execute("""SELECT z,charge,configuration,status,energy_ha,
        is_ground FROM config_jobs WHERE z<=?""", (max_z,)).fetchall()
    jobs = {(z, charge, config): (status, energy)
            for z, charge, config, status, energy, _ground in job_rows}
    candidates = defaultdict(dict)
    for z, charge, config, status, energy, _ground in job_rows:
        candidates[z, charge][config] = (status, energy)
    grounds = {}
    for z, charge, config, _status, _energy, is_ground in job_rows:
        if not is_ground:
            continue
        if (z, charge) in grounds:
            raise RuntimeError(f"multiple ground jobs for Z={z}, q={charge}")
        grounds[z, charge] = config
    has_ionization = bool(reference.execute("""SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='nist_ionization'""").fetchone())
    ground_sources = {}
    for (z, charge), config in grounds.items():
        selected, provenance = ground_configuration_with_dhf_candidates(
            reference, z, charge, has_ionization=has_ionization,
            candidates=candidates[z, charge])
        if selected != config:
            if provenance.startswith(("dhf_lower_than_nist_isoelectronic_z",
                                      "dhf_lower_nist_isoelectronic_z")):
                # A worker can finish the lower competing candidate before
                # the repair sweep flips the ledger's is_ground flag.
                grounds[z, charge] = selected
            else:
                raise RuntimeError(
                    f"selected ground configuration disagrees with NIST selection "
                    f"for Z={z}, q={charge}: {config!r} versus {selected!r}")
        ground_sources[z, charge] = provenance
    centroids = ground_centroids(reference, grounds, max_z)
    source_rows = {(z, charge): (spectrum, nist_ev, raw, nist_cm1,
        uncertainty, uncertainty_cm1, binding_cm1, binding_kind, ref)
        for (z, charge, spectrum, nist_ev, raw, nist_cm1, uncertainty,
             uncertainty_cm1, binding_cm1, binding_kind, ref) in reference.execute(
            """SELECT z,charge,spectrum,ionization_eV,ionization_raw,
                      ionization_cm1,uncertainty_eV,uncertainty_cm1,
                      binding_cm1,binding_kind,reference
               FROM nist_ionization WHERE z<=? ORDER BY z,charge""",
            (max_z,))}
    species_columns = {row[1] for row in reference.execute(
        "PRAGMA table_info(species)")}
    species_labels = ({(z, charge): label for z, charge, label in
        reference.execute("SELECT z,charge,label FROM species WHERE z<=?",
                          (max_z,))} if "label" in species_columns else {})
    rows = []
    for z, charge in sorted(grounds):
        (spectrum, nist_ev, raw, nist_cm1, uncertainty, uncertainty_cm1,
         binding_cm1, binding_kind, ref) = source_rows.get(
            (z, charge), (species_labels.get((z, charge),
                           f"Z={z} q={charge}"),) + (None,) * 8)
        lower_config = grounds.get((z, charge))
        upper_config = grounds.get((z, charge + 1)) if charge < z - 1 else None
        comparison_basis = (
            "unique_ground_level"
            if lower_config and configuration_has_single_level(lower_config)
            and (charge == z - 1 or upper_config
                 and configuration_has_single_level(upper_config))
            else "configuration_average")
        lower = jobs.get((z, charge, lower_config)) if lower_config else None
        upper = (jobs.get((z, charge + 1, upper_config))
                 if upper_config else ("done", 0.0) if charge == z - 1 else None)
        predicted = predicted_cm1 = error = difference = None
        lower_centroid = centroids.get((z, charge))
        upper_centroid = (centroids.get((z, charge + 1)) if charge < z - 1
                          else dict(centroid_cm1=0.0,
                                    status="evaluated_complete",
                                    nist_g_disagreements=0))
        centroid_reference_cm1 = centroid_reference_ev = None
        centroid_residual_ev = centroid_error = None
        centroid_status = "incomplete_level_manifold"
        if (lower_centroid is not None and upper_centroid is not None
                and lower_centroid["centroid_cm1"] is not None
                and upper_centroid["centroid_cm1"] is not None
                and (nist_cm1 is not None or nist_ev is not None)):
            centroid_reference_cm1 = (
                (nist_cm1 if nist_cm1 is not None
                 else nist_ev * HARTREE_CM1 / HARTREE_EV)
                + upper_centroid["centroid_cm1"]
                - lower_centroid["centroid_cm1"])
            if centroid_reference_cm1 > 0:
                centroid_reference_ev = (centroid_reference_cm1
                                         * HARTREE_EV / HARTREE_CM1)
                centroid_status = (
                    "evaluated_complete" if reference_kind(raw) == "evaluated"
                    and lower_centroid["status"] == "evaluated_complete"
                    and upper_centroid["status"] == "evaluated_complete"
                    else "indicative_reference")
        if lower_config is None or (charge < z - 1 and upper_config is None):
            status = "ground_configuration_unavailable"
        elif lower is None or upper is None:
            status = "ground_job_missing"
        elif lower[0] in ("failed", "untrusted") or upper[0] in ("failed", "untrusted"):
            status = "ground_calculation_failed"
        elif lower[0] != "done" or upper[0] != "done":
            status = "waiting_for_ground_energies"
        else:
            predicted = (upper[1] - lower[1]) * HARTREE_EV
            predicted_cm1 = (upper[1] - lower[1]) * HARTREE_CM1
            if predicted <= 0:
                status = "inverted_ground_energies"
            elif nist_ev is None and nist_cm1 is None:
                status = "no_nist_ionization_energy"
            else:
                status = "compared"
                difference = predicted - nist_ev if nist_ev is not None else None
                error = (100 * (predicted_cm1 / nist_cm1 - 1)
                         if nist_cm1 is not None else
                         100 * difference / nist_ev)
                if centroid_reference_cm1 is not None and centroid_reference_cm1 > 0:
                    centroid_residual_ev = predicted - centroid_reference_ev
                    centroid_error = 100 * (predicted_cm1
                                            / centroid_reference_cm1 - 1)
        rows.append(dict(z=z, charge=charge, spectrum=spectrum,
            ground_configuration=lower_config,
            ionized_ground_configuration=upper_config,
            ground_configuration_source=ground_sources.get((z, charge)),
            ionized_ground_configuration_source=(
                ground_sources.get((z, charge + 1)) if charge < z - 1
                else "bare_nucleus"),
            comparison_basis=comparison_basis,
            nist_ionization_eV=nist_ev,
            nist_ionization_cm1=nist_cm1,
            nist_ionization_raw=raw,
            reference_kind=reference_kind(raw),
            nist_ground_centroid_cm1=(lower_centroid or {}).get("centroid_cm1"),
            nist_ionized_centroid_cm1=(upper_centroid or {}).get("centroid_cm1"),
            nist_ground_centroid_status=(lower_centroid or {}).get("status"),
            nist_ionized_centroid_status=(upper_centroid or {}).get("status"),
            nist_ground_weight_expected=(lower_centroid or {}).get("expected_weight"),
            nist_ground_weight_listed=(lower_centroid or {}).get("listed_weight"),
            nist_ground_g_disagreements=(lower_centroid or {}).get(
                "nist_g_disagreements"),
            nist_ionized_g_disagreements=(upper_centroid or {}).get(
                "nist_g_disagreements"),
            nist_centroid_ionization_eV=centroid_reference_ev,
            nist_centroid_ionization_cm1=centroid_reference_cm1,
            centroid_residual_eV=centroid_residual_ev,
            centroid_error_percent=centroid_error,
            centroid_status=centroid_status,
            nist_uncertainty_eV=uncertainty,
            nist_uncertainty_cm1=uncertainty_cm1,
            nist_binding_cm1=binding_cm1,
            nist_binding_kind=binding_kind,
            nist_reference=ref,
            calculated_ionization_eV=predicted,
            calculated_ionization_cm1=predicted_cm1,
            residual_eV=difference,
            energy_error_percent=error,
            status=status,
            model="DHF+B+QED+NMS"))
    reference.close()
    ledger.close()
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
