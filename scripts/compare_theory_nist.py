#!/usr/bin/env python
"""Compare atomx hydrogenic and multielectron line positions with NIST ASD.

The observed-line comparison is the experimental score. Ritz-only values are
retained as a separate reference check; they may originate from theoretical
levels and must never be counted as experimental measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from atomx.constants import HARTREE_CM1, PROTON_ELECTRON_MASS_RATIO, element_symbol  # noqa: E402
from atomx.physics.dirac import hydrogenic_dirac_energy  # noqa: E402
from plotly.offline.offline import get_plotlyjs  # noqa: E402
from compare_multielectron_nist import (  # noqa: E402
    collect as collect_multielectron, add_model_audit, cached_panel,
    rejected_ci_panels)
from compare_full_dhf_nist import collect as collect_full_dhf  # noqa: E402
from generate_full_dhf_nist import configuration_has_single_level  # noqa: E402
from compare_ionization_dhf_nist import (  # noqa: E402
    collect as collect_ionization, write_csv as write_ionization_csv)
from compare_configuration_centroids_nist import (  # noqa: E402
    collect as collect_configuration_centroids,
    write_outputs as write_configuration_centroids)
from compare_isotope_records_nist import (  # noqa: E402
    collect as collect_isotope_lines, collect_levels as collect_isotope_levels,
    write_csv as write_isotope_csv)
from audit_inverted_dhf_lines import (  # noqa: E402
    collect as collect_inverted_dhf, write_csv as write_inverted_csv)

SUBSHELL = re.compile(r"^(\d+)([spdfghiklmno])1$")
L = {letter: i for i, letter in enumerate("spdfghiklmno")}
CSV_FIELDS = ("z", "symbol", "ion", "lower", "lower_term", "upper",
              "upper_term", "lower_j", "upper_j",
              "observed_nm", "observed_unc_nm", "ritz_nm", "ritz_unc_nm",
              "nist_medium", "calculated_nm",
              "reference_nm", "reference_kind", "energy_error_percent",
              "wavelength_error_nm", "nist_row_index")


def kappa_of(configuration: str, j_text: str) -> tuple[int, int] | None:
    match = SUBSHELL.fullmatch(configuration.strip())
    if not match:
        return None
    n, letter = match.groups()
    n, ell = int(n), L[letter]
    try:
        twice_j = round(2 * float(eval_fraction(j_text)))
    except (ValueError, ZeroDivisionError):
        return None
    if twice_j == 2 * ell + 1:
        kappa = -(ell + 1)
    elif ell > 0 and twice_j == 2 * ell - 1:
        kappa = ell
    else:
        return None
    return n, kappa


def eval_fraction(value: str) -> float:
    parts = value.strip().split("/")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) / float(parts[1])
    raise ValueError(value)


def calculated_nm(z: int, lower: str, lower_j: str,
                  upper: str, upper_j: str) -> float | None:
    lo = kappa_of(lower, lower_j)
    hi = kappa_of(upper, upper_j)
    if not lo or not hi or lo[0] != 1 or lo[1] != -1:
        return None
    # Use a common, unambiguous Lyman-series window across the table.
    if not (hi[0] in range(2, 7) and hi[1] in (1, -2)):
        return None
    return calculated_any_nm(z, lower, lower_j, upper, upper_j)


def calculated_any_nm(z: int, lower: str, lower_j: str,
                      upper: str, upper_j: str) -> float | None:
    """All classified H-like transitions with a nondegenerate Dirac interval."""
    lo = kappa_of(lower, lower_j)
    hi = kappa_of(upper, upper_j)
    if not lo or not hi:
        return None
    if lo[0] == hi[0]:
        # Near-degenerate same-n intervals are controlled by the Lamb shift
        # and recoil terms absent from this point-Dirac baseline.  A tiny
        # denominator otherwise makes their percent error dominate a whole
        # element's score even when the model is outside its stated scope.
        return None
    try:
        delta_cm1 = (hydrogenic_dirac_energy(*hi, z)
                     - hydrogenic_dirac_energy(*lo, z)) * HARTREE_CM1
    except ValueError:
        return None
    return 1e7 / delta_cm1 if delta_cm1 > 1e-3 else None


def hydrogen_dhf_calibration(db_path: Path, cache_path: Path,
                             full_dhf_path: Path | None = None) -> list[dict]:
    """Compare point Dirac and cached DHF on identical observed H I lines."""
    if not cache_path.exists():
        return []
    cache = json.loads(cache_path.read_text())
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        lines = db.execute("""SELECT row_index,observed_nm,lower_configuration,
              lower_j,upper_configuration,upper_j FROM lines
              WHERE z=1 AND charge=0 AND medium='vacuum'
              AND observed_nm IS NOT NULL AND lower_configuration='1s1'
              AND upper_configuration='2p1' AND upper_j IN ('1/2','3/2')
              ORDER BY row_index""").fetchall()
    result = []
    mass_rows = {}
    if full_dhf_path is not None and full_dhf_path.exists():
        with sqlite3.connect(full_dhf_path) as ledger:
            mass_rows = dict(ledger.execute("""SELECT row_index,calculated_nm
                FROM line_jobs WHERE z=1 AND charge=0 AND status IN
                ('compared','configuration_average_only')"""))
    for line in lines:
        calc = calculated_any_nm(1, line["lower_configuration"], line["lower_j"],
                                 line["upper_configuration"], line["upper_j"])
        if calc is None:
            continue
        record = dict(row_index=line["row_index"], upper_j=line["upper_j"],
                      observed_nm=line["observed_nm"], dirac_nm=calc,
                      dhf_mass_nm=mass_rows.get(line["row_index"]))
        for suffix, label in [("0|0", "dhf_nm"), ("1|1", "dhf_bqed_nm")]:
            lower = cache.get(f"scf|1|1s1|{suffix}")
            upper = cache.get(f"scf|1|2p1|{suffix}")
            if lower is not None and upper is not None and upper > lower:
                record[label] = 1e7 / ((upper - lower) * HARTREE_CM1)
        result.append(record)
    return result


def build_record_audit(db_path: Path, output: Path) -> dict:
    """Write a status for every NIST line row, including unassessed records."""
    source = sqlite3.connect(db_path)
    source.row_factory = sqlite3.Row
    temp = output.with_suffix(".sqlite.part")
    if temp.exists():
        temp.unlink()
    audit = sqlite3.connect(temp)
    audit.executescript("""
        CREATE TABLE line_audit(
          z INTEGER NOT NULL, charge INTEGER NOT NULL, row_index INTEGER NOT NULL,
          status TEXT NOT NULL, calculated_nm REAL, observed_energy_error_percent REAL,
          ritz_energy_error_percent REAL, model TEXT,
          PRIMARY KEY(z,charge,row_index));
        CREATE INDEX audit_status ON line_audit(status,z,charge);
        CREATE TABLE level_audit(
          z INTEGER NOT NULL, charge INTEGER NOT NULL, row_index INTEGER NOT NULL,
          status TEXT NOT NULL, calculated_energy_cm1 REAL, residual_cm1 REAL,
          model TEXT, PRIMARY KEY(z,charge,row_index));
        CREATE INDEX level_audit_status ON level_audit(status,z,charge);
    """)
    counts: dict[str, int] = defaultdict(int)
    cursor = source.execute("""SELECT z,charge,row_index,observed_nm,ritz_nm,medium,
                              lower_configuration,lower_j,upper_configuration,upper_j
                              FROM lines ORDER BY z,charge,row_index""")
    batch = []
    for line in cursor:
        z, q = line["z"], line["charge"]
        observed, ritz = line["observed_nm"], line["ritz_nm"]
        calc = None
        model = None
        if q == z - 1 and kappa_of(line["lower_configuration"], line["lower_j"]) \
                and kappa_of(line["upper_configuration"], line["upper_j"]):
            calc = calculated_any_nm(z, line["lower_configuration"], line["lower_j"],
                                     line["upper_configuration"], line["upper_j"])
            model = "point-nucleus Dirac-Coulomb"
        if observed is None:
            status = "ritz_only" if calc is not None and ritz is not None else "no_observed_wavelength"
        elif line["medium"] != "vacuum":
            status = "air_or_unknown_medium"
        elif q != z - 1:
            status = "no_atomx_line_calculation"
        elif not kappa_of(line["lower_configuration"], line["lower_j"]) \
                or not kappa_of(line["upper_configuration"], line["upper_j"]):
            status = "missing_or_unsupported_classification"
        elif kappa_of(line["lower_configuration"], line["lower_j"])[0] == \
                kappa_of(line["upper_configuration"], line["upper_j"])[0]:
            status = "same_n_requires_lamb_recoil"
        elif calc is None:
            status = "dirac_degenerate_or_reversed"
        else:
            status = "compared_observed"
        observed_error = (100 * (observed / calc - 1)
                          if observed is not None and calc is not None
                          and line["medium"] == "vacuum" else None)
        ritz_error = (100 * (ritz / calc - 1)
                      if ritz is not None and calc is not None
                      and line["medium"] == "vacuum" else None)
        counts[status] += 1
        batch.append((z, q, line["row_index"], status, calc,
                      observed_error, ritz_error, model))
        if len(batch) >= 5000:
            audit.executemany("INSERT INTO line_audit VALUES (?,?,?,?,?,?,?,?)", batch)
            batch.clear()
    if batch:
        audit.executemany("INSERT INTO line_audit VALUES (?,?,?,?,?,?,?,?)", batch)
    level_counts: dict[str, int] = defaultdict(int)
    batch = []
    for level in source.execute("""SELECT z,charge,row_index,configuration,term,
                                  j_text,energy_cm1 FROM levels
                                  ORDER BY z,charge,row_index"""):
        z, q = level["z"], level["charge"]
        calculated = None
        model = None
        if level["energy_cm1"] is None:
            status = "no_nist_level_energy"
        elif q != z - 1:
            status = "no_atomx_level_calculation"
        else:
            ground_energy = hydrogenic_dirac_energy(1, -1, z)
            if "Limit" in (level["term"] or ""):
                calculated = -ground_energy * HARTREE_CM1
                status = "compared_ionization_limit"
            else:
                state = kappa_of(level["configuration"], level["j_text"])
                if state:
                    try:
                        calculated = (hydrogenic_dirac_energy(*state, z)
                                      - ground_energy) * HARTREE_CM1
                    except ValueError:
                        calculated = None
                status = "compared_nist_level" if calculated is not None else "unsupported_level_classification"
            if calculated is not None:
                model = "point-nucleus Dirac-Coulomb"
        residual = (calculated - level["energy_cm1"]
                    if calculated is not None and level["energy_cm1"] is not None
                    else None)
        level_counts[status] += 1
        batch.append((z, q, level["row_index"], status, calculated, residual, model))
        if len(batch) >= 5000:
            audit.executemany("INSERT INTO level_audit VALUES (?,?,?,?,?,?,?)", batch)
            batch.clear()
    if batch:
        audit.executemany("INSERT INTO level_audit VALUES (?,?,?,?,?,?,?)", batch)
    audit.commit()
    audit.close()
    source.close()
    temp.replace(output)
    return {"lines": dict(sorted(counts.items())),
            "levels": dict(sorted(level_counts.items()))}


def add_full_dhf_statuses(audit_path: Path, ledger_path: Path) -> None:
    """Expose every attempted observed line in the row-level atlas audit."""
    if not ledger_path.exists():
        return
    with sqlite3.connect(ledger_path) as ledger, sqlite3.connect(audit_path) as audit:
        rows = ledger.execute("""SELECT z,charge,row_index,status,calculated_nm,
            energy_error_percent,lower_configuration,upper_configuration
            FROM line_jobs""")
        audit.executemany("""UPDATE line_audit SET status='dhf_' || ?,
            calculated_nm=?,observed_energy_error_percent=?,
            model='DHF+B+QED+NMS'
            WHERE z=? AND charge=? AND row_index=?
            AND status!='compared_observed'""",
            (("configuration_average_only" if
              status == "compared" and
              (not configuration_has_single_level(lo) or
               not configuration_has_single_level(hi)) else status,
              predicted, error, z, charge, index)
             for z, charge, index, status, predicted, error, lo, hi in rows))


def load_all_observed(db_path: Path, audit_path: Path) -> list[dict]:
    """Every observed NIST line that the one-electron model can calculate."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    audit = sqlite3.connect(audit_path)
    rows = []
    try:
        for z, charge, index, predicted, error in audit.execute(
                """SELECT z,charge,row_index,calculated_nm,
                          observed_energy_error_percent FROM line_audit
                   WHERE status='compared_observed'
                     AND model='point-nucleus Dirac-Coulomb'
                   ORDER BY z,charge,row_index"""):
            line = db.execute("""SELECT observed_nm,observed_unc_nm,ritz_nm,
                                 lower_configuration,lower_term,lower_j,
                                 upper_configuration,upper_term,upper_j
                                 FROM lines WHERE z=? AND charge=? AND row_index=?""",
                              (z, charge, index)).fetchone()
            rows.append({"z": z, "charge": charge, "symbol": element_symbol(z),
                         "ion": f"{element_symbol(z)} {charge}+",
                         "lower": line["lower_configuration"],
                         "lower_term": line["lower_term"],
                         "lower_j": line["lower_j"],
                         "upper": line["upper_configuration"],
                         "upper_term": line["upper_term"],
                         "upper_j": line["upper_j"],
                         "observed_nm": line["observed_nm"],
                         "observed_unc_nm": line["observed_unc_nm"],
                         "ritz_nm": line["ritz_nm"],
                         "calculated_nm": predicted,
                         "energy_error_percent": error,
                         "nist_row_index": index})
    finally:
        audit.close()
        db.close()
    return rows


def load_lines(db_path: Path) -> list[dict]:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    query = """SELECT l.z, l.charge, l.row_index, l.observed_nm, l.observed_unc_nm,
                      l.ritz_nm, l.ritz_unc_nm,
                      l.medium, l.lower_configuration, l.lower_term, l.lower_j,
                      l.upper_configuration, l.upper_term, l.upper_j, s.label
               FROM lines AS l JOIN species AS s USING (z, charge)
               WHERE l.charge=l.z-1 AND l.lower_configuration='1s1'
                 AND l.upper_configuration IN ('2p1','3p1','4p1','5p1','6p1')
                 AND l.medium='vacuum'
               ORDER BY l.z, l.row_index"""
    rows = []
    try:
        for line in db.execute(query):
            predicted = calculated_nm(line["z"], line["lower_configuration"],
                                      line["lower_j"], line["upper_configuration"],
                                      line["upper_j"])
            if predicted is None:
                continue
            observed, ritz = line["observed_nm"], line["ritz_nm"]
            reference = observed if observed is not None else ritz
            if reference is None or reference <= 0:
                continue
            rows.append(dict(z=line["z"], symbol=element_symbol(line["z"]),
                             ion=line["label"], lower=line["lower_configuration"],
                             lower_term=line["lower_term"],
                             upper=line["upper_configuration"],
                             upper_term=line["upper_term"], lower_j=line["lower_j"],
                             upper_j=line["upper_j"], observed_nm=observed,
                             observed_unc_nm=line["observed_unc_nm"],
                             ritz_nm=ritz, ritz_unc_nm=line["ritz_unc_nm"],
                             nist_medium=line["medium"],
                             calculated_nm=predicted, reference_nm=reference,
                             reference_kind="observed" if observed is not None else "Ritz",
                             energy_error_percent=100 * (reference / predicted - 1),
                             wavelength_error_nm=predicted - reference,
                             nist_row_index=line["row_index"]))
    finally:
        db.close()
    return rows


def metrics(rows: list[dict], error_field: str = "energy_error_percent") -> dict:
    errors = [r[error_field] for r in rows]
    return {"n_lines": len(rows),
            "median_abs_energy_error_percent": statistics.median(map(abs, errors))
            if errors else None,
            "rms_energy_error_percent": math.sqrt(sum(x*x for x in errors)/len(errors))
            if errors else None,
            "mean_signed_energy_error_percent": sum(errors)/len(errors)
            if errors else None}


def load_panel(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    panel = []
    for row in rows:
        if not row.get("rms_cm1"):
            continue
        system = row["system"]
        symbol = system.split()[0]
        try:
            z = next(z for z in range(1, 119) if element_symbol(z) == symbol)
        except StopIteration:
            continue
        panel.append({"system": system, "z": z, "rung": row["rung"],
                      "n_levels": int(row["n_levels_compared"]),
                      "n_excluded_blocks": int(row["n_blocks_excluded"]),
                      "rms_cm1": float(row["rms_cm1"]),
                      "rms_eV": float(row["rms_eV"])})
    return panel


def table_position(z: int) -> tuple[int, int]:
    """Period/group-style placement, with f-block rows below the table."""
    if z == 1:
        return 1, 1
    if z == 2:
        return 1, 18
    if 3 <= z <= 10:
        return 2, z - 2 if z <= 4 else z + 8
    if 11 <= z <= 18:
        return 3, z - 10 if z <= 12 else z
    if 19 <= z <= 36:
        return 4, z - 18
    if 37 <= z <= 54:
        return 5, z - 36
    if z in (55, 56):
        return 6, z - 54
    if 57 <= z <= 71:
        return 9, z - 53
    if 72 <= z <= 86:
        return 6, z - 68
    if z in (87, 88):
        return 7, z - 86
    if 89 <= z <= 103:
        return 10, z - 85
    return 7, z - 100


def render_overview(elements: list[dict], mode: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(19, 9.8), facecolor="#07121e")
    ax.set_facecolor("#07121e")
    norm = (LogNorm(vmin=0.1, vmax=20) if mode.startswith("ionization")
            else LogNorm(vmin=0.001, vmax=100 if mode.startswith("multielectron")
                         else 2))
    cmap = plt.get_cmap("viridis")
    for element in elements:
        row, col = table_position(element["z"])
        value = element[mode]["rms_energy_error_percent"]
        color = cmap(norm(value)) if value is not None else "#233444"
        ax.add_patch(Rectangle((col - 1, 11 - row), .91, .85,
                               facecolor=color, edgecolor="#71889a", linewidth=.65))
        ax.text(col - .91, 11 - row + .62, str(element["z"]), color="white",
                fontsize=7, ha="left", va="center")
        ax.text(col - .53, 11 - row + .35, element["symbol"], color="white",
                fontsize=13, weight="bold", ha="center", va="center")
        ax.text(col - .53, 11 - row + .12,
                f"{value:.2g}%" if value is not None else "—",
                color="white" if value is not None else "#96aab8",
                fontsize=7, ha="center", va="center")
    ax.set_xlim(-.25, 18.3)
    ax.set_ylim(.7, 11.1)
    ax.set_aspect("equal")
    ax.axis("off")
    title = {"all_observed": "All classified observed wavelengths",
             "all_observed_mass": "Hydrogenic Dirac + leading isotope mass shift",
             "observed": "Observed Lyman wavelengths",
             "ritz": "Ritz Lyman wavelengths (level-derived)",
             "multielectron_bqed": "Observed wavelengths · MCDHF+B+QED",
             "multielectron_dhf": "Observed wavelengths · DHF",
             "multielectron_dhf_bqed": "Observed wavelengths · DHF+B+QED",
             "multielectron_dhf_mass": "Observed wavelengths · DHF+B+QED+normal mass shift",
             "ionization": "Unique-level NIST ionization energies · DHF+B+QED+normal mass shift",
             "ionization_centroid": "Complete NIST configuration-centroid ionization energies · DHF+B+QED+mass"}[mode]
    fig.suptitle(f"Atomx vs NIST · {title}", color="white", fontsize=22,
                 x=.12, ha="left", y=.98)
    subtitle = ("Matched unbracketed ionization energies with unique levels on both sides"
                if mode == "ionization" else
                "Complete evaluated ground-configuration level manifolds"
                if mode == "ionization_centroid" else
                "Matched transitions with classified configuration endpoints"
                if mode.startswith("multielectron") else
                "H-like classified transitions" if mode == "all_observed"
                else "H-like 1s → np, n = 2–6")
    gray_note = ("no matched unique-level value" if mode == "ionization"
                 else "no complete evaluated centroid" if mode == "ionization_centroid"
                 else "no matched record")
    fig.text(.12, .92, f"{subtitle} · RMS energy error (%) · gray = {gray_note}",
             color="#b5cbd9", fontsize=11)
    if mode == "ritz":
        fig.text(.12, .065, "Ritz values are derived from NIST energy levels and are not necessarily experimental.",
                 color="#ffbd65", fontsize=10)
    elif mode == "ionization":
        fig.text(.12, .065, "Only unbracketed NIST values with unique levels on both sides enter this map; all rows remain in the CSV.",
                 color="#ffbd65", fontsize=10)
    elif mode == "ionization_centroid":
        fig.text(.12, .065, "Weights come from 2J+1; complete evaluated level manifolds and unbracketed ionization values only.",
                 color="#ffbd65", fontsize=10)
    else:
        fig.text(.12, .065, "Only directly observed, classified vacuum wavelengths are scored.",
                 color="#ffbd65", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=.42,
                      fraction=.035, pad=.025, aspect=32)
    cb.set_label("RMS energy error (%) · logarithmic color scale", color="white")
    cb.ax.tick_params(colors="white")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def atomic_write_text(path: Path, content: str) -> None:
    """Publish complete site files while browsers and APIs are reading."""
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(content)
    part.replace(path)


def benchmark_cache_provenance(cache: Path) -> dict:
    """Report whether panel energies predate solver code used to interpret them."""
    sources = [ROOT / path for path in (
        "benchmarks/verify_nist_levels.py",
        "src/atomx/atom.py",
        "src/atomx/methods/dirac_fock.py",
        "src/atomx/methods/mcdhf/matrix.py",
        "src/atomx/methods/mcdhf/solver.py",
        "src/atomx/methods/mcdhf/variational.py",
        "src/atomx/numerics/bspline/dirac_assembly.py")]
    modified = max((path.stat().st_mtime for path in sources if path.exists()),
                   default=None)
    cached = cache.stat().st_mtime if cache.exists() else None
    stamp = lambda value: (datetime.fromtimestamp(value, timezone.utc).isoformat()
                           if value is not None else None)
    return {"cache_modified_utc": stamp(cached),
            "solver_modified_utc": stamp(modified),
            "requires_revalidation": cached is None or (
                modified is not None and cached < modified)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "output/nist_spectra/spectra.sqlite")
    parser.add_argument("--panel", type=Path,
                        default=ROOT / "output/nist_spectra/theory_comparison/panel_current.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/nist_spectra/theory_comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lines = load_lines(args.db)
    by_z: dict[int, list[dict]] = defaultdict(list)
    for row in lines:
        by_z[row["z"]].append(row)
    elements = []
    for z in range(1, 119):
        line_rows = by_z.get(z, [])
        observed = [r for r in line_rows if r["reference_kind"] == "observed"]
        ritz = [r for r in line_rows if r["ritz_nm"] is not None]
        # Compare the same exact transitions to Ritz, even when an observed
        # wavelength also exists; source classes stay separate in the UI.
        ritz_rows = [{**r, "energy_error_percent":
                      100 * (r["ritz_nm"] / r["calculated_nm"] - 1)}
                     for r in ritz]
        elements.append({"z": z, "symbol": element_symbol(z),
                         "observed": metrics(observed), "ritz": metrics(ritz_rows)})
    # The CSV is a historical benchmark snapshot. CI rows rejected by the
    # current physical-state gate must not reappear through this fallback.
    # MBPT2 is the only rung the live cache panel does not build.
    panel = [row for row in load_panel(args.panel)
             if row["rung"] == "DHF+MBPT2"]
    benchmark_cache = ROOT / "benchmarks/_nist_work/cache.json"
    cached_rows = cached_panel(benchmark_cache)
    panel.extend(cached_rows)
    rejected_panels = rejected_ci_panels(benchmark_cache)
    audit_counts = build_record_audit(args.db, args.output / "record_audit.sqlite")
    # Preserve the independent point-Dirac series before the preferred DHF
    # result is joined into the one-row-per-record audit.
    all_observed = load_all_observed(args.db, args.output / "record_audit.sqlite")
    isotope_path = ROOT / "output/nist_spectra/isotopes.json"
    isotope_choices = (json.loads(isotope_path.read_text())["selected"]
                       if isotope_path.exists() else {})
    all_observed_mass = []
    for row in all_observed:
        choice = isotope_choices.get(str(row["z"]))
        if choice is None:
            continue
        mass = choice["nuclear_mass_me"]
        predicted = row["calculated_nm"] * (1 + 1 / mass)
        all_observed_mass.append(dict(row, calculated_nm=predicted,
            energy_error_percent=100 * (row["observed_nm"] / predicted - 1),
            isotope_a=choice["a"], model="Dirac+leading mass"))
    multielectron_lines, multielectron_levels = collect_multielectron(
        args.db, ROOT / "benchmarks/_nist_work/cache.json")
    full_rows = collect_full_dhf(
        args.db, ROOT / "output/nist_spectra/full_dhf.sqlite",
        include_averages=True)
    multielectron_lines.extend(r for r in full_rows
                               if r["comparison_status"] == "compared")
    configuration_average_lines = [r for r in full_rows
                                   if r["comparison_status"] ==
                                   "configuration_average_only"]
    radial_box_lines = [r for r in full_rows
                        if r["comparison_status"] == "radial_box_unresolved"]
    unscored_dhf_lines = configuration_average_lines + radial_box_lines
    full_ledger = ROOT / "output/nist_spectra/full_dhf.sqlite"
    full_dhf_progress = {}
    if full_ledger.exists():
        with sqlite3.connect(full_ledger) as ledger:
            active_max_z = ledger.execute(
                "SELECT value FROM metadata WHERE key='active_max_z'").fetchone()
            full_dhf_progress = {
                "active_max_z": int(active_max_z[0]) if active_max_z else None,
                "configurations": dict(ledger.execute(
                    "SELECT status,COUNT(*) FROM config_jobs GROUP BY status")),
                "lines": dict(ledger.execute(
                    "SELECT status,COUNT(*) FROM line_jobs GROUP BY status")),
                "first20_configurations": dict(ledger.execute(
                    "SELECT status,COUNT(*) FROM config_jobs WHERE z<=20 GROUP BY status")),
                "first20_lines": dict(ledger.execute(
                    "SELECT status,COUNT(*) FROM line_jobs WHERE z<=20 GROUP BY status"))}
            full_dhf_progress["numerical_validations"] = [
                json.loads(value) for (value,) in ledger.execute(
                    "SELECT value FROM metadata WHERE key LIKE 'numerical_validation_%' ORDER BY key")]
            if "classification_source" in {row[1] for row in ledger.execute(
                    "PRAGMA table_info(line_jobs)")}:
                full_dhf_progress["id_recovered_lines"] = ledger.execute(
                    """SELECT COUNT(*) FROM line_jobs
                       WHERE classification_source='nist_level_id'""").fetchone()[0]
            if active_max_z:
                full_dhf_progress["scope_configurations"] = dict(ledger.execute(
                    "SELECT status,COUNT(*) FROM config_jobs WHERE z<=? GROUP BY status",
                    (int(active_max_z[0]),)))
                full_dhf_progress["scope_lines"] = dict(ledger.execute(
                    "SELECT status,COUNT(*) FROM line_jobs WHERE z<=? GROUP BY status",
                    (int(active_max_z[0]),)))
    # Older live workers can still write the previous row-count-only gate.
    # Publish the physics-based status used by the collector and row audit.
    for key, max_z in (("lines", 118), ("first20_lines", 20),
                       ("scope_lines", full_dhf_progress.get("active_max_z"))):
        if key not in full_dhf_progress or max_z is None:
            continue
        scoped = [r for r in full_rows if r["z"] <= max_z]
        for status in ("compared", "configuration_average_only"):
            full_dhf_progress[key][status] = sum(
                r["comparison_status"] == status for r in scoped)
    ionization_rows = collect_ionization(
        args.db, full_ledger, max_z=full_dhf_progress.get("active_max_z") or 20)
    isotope_lines, isotope_summary = collect_isotope_lines(
        args.db, full_ledger, isotope_path)
    isotope_levels, isotope_level_summary = collect_isotope_levels(
        args.db, full_ledger, isotope_path)
    if full_ledger.exists():
        centroid_rows, centroid_summary = collect_configuration_centroids(
            args.db, full_ledger,
            max_z=full_dhf_progress.get("active_max_z") or 20)
    else:
        centroid_rows, centroid_summary = [], {"configuration_rows": 0,
                                               "statuses": {}, "elements": []}
    inverted_rows, inverted_summary = (
        collect_inverted_dhf(full_ledger, centroid_rows,
                             max_z=full_dhf_progress.get("active_max_z") or 20)
        if full_ledger.exists() else ([], {"pairs": 0, "line_rows": 0,
                                         "statuses": {}, "pair_statuses": {}}))
    branch_manifest = ROOT / "output/nist_spectra/dhf_core_p_branch_repairs_summary.json"
    branch_repairs = json.loads(branch_manifest.read_text()) if branch_manifest.exists() else {
        "verified_repairs": 0, "rows": []}
    core_p_audit_path = ROOT / "output/nist_spectra/dhf_core_p_root_audit_summary.json"
    core_p_audit = json.loads(core_p_audit_path.read_text()) if core_p_audit_path.exists() else {
        "audited_jobs": 0, "audit_statuses": {}, "current_gsz_not_yet_audited": None}
    heavy_core_p_path = ROOT / "output/nist_spectra/dhf_heavy_core_p_root_probe_summary.json"
    heavy_core_p_probe = json.loads(heavy_core_p_path.read_text()) if heavy_core_p_path.exists() else {
        "audited_jobs": 0, "audit_statuses": {},
        "current_gsz_low_p_hole_not_yet_audited_z41_74": None}
    add_model_audit(args.output / "record_audit.sqlite",
                    multielectron_lines, multielectron_levels)
    add_full_dhf_statuses(args.output / "record_audit.sqlite", full_ledger)
    with sqlite3.connect(args.output / "record_audit.sqlite") as audit:
        audit_counts = {
            "lines": dict(audit.execute("SELECT status,COUNT(*) FROM line_audit GROUP BY status")),
            "levels": dict(audit.execute("SELECT status,COUNT(*) FROM level_audit GROUP BY status"))}
    with sqlite3.connect(args.db) as db:
        missing_config = db.execute("""SELECT COUNT(*) FROM lines
            WHERE observed_nm IS NOT NULL AND
            (lower_configuration='' OR upper_configuration='')""").fetchone()[0]
        missing_j = db.execute("""SELECT COUNT(*) FROM lines
            WHERE observed_nm IS NOT NULL AND
            (lower_j='' OR upper_j='')""").fetchone()[0]
        fully_classified = db.execute("""SELECT COUNT(*) FROM lines
            WHERE observed_nm IS NOT NULL AND lower_configuration!=''
            AND upper_configuration!='' AND lower_j!='' AND upper_j!=''""").fetchone()[0]
    all_by_z: dict[int, list[dict]] = defaultdict(list)
    for row in all_observed:
        all_by_z[row["z"]].append(row)
    for element in elements:
        element["all_observed"] = metrics(all_by_z[element["z"]])
        element["all_observed_mass"] = metrics(
            [r for r in all_observed_mass if r["z"] == element["z"]])
        element["ionization"] = metrics(
            [r for r in ionization_rows if r["z"] == element["z"]
             and r["status"] == "compared"
             and r["reference_kind"] == "evaluated"
             and r["comparison_basis"] == "unique_ground_level"])
        element["ionization_centroid"] = metrics(
            [r for r in ionization_rows if r["z"] == element["z"]
             and r["status"] == "compared"
             and r["centroid_status"] == "evaluated_complete"],
            "centroid_error_percent")
        element["ionization_all"] = metrics(
            [r for r in ionization_rows if r["z"] == element["z"]
             and r["status"] == "compared"])
        for rung, key in [("DHF+B+QED+NMS", "multielectron_dhf_mass"),
                          ("DHF", "multielectron_dhf"),
                          ("DHF+B+QED", "multielectron_dhf_bqed"),
                          ("MCDHF", "multielectron_mcdhf"),
                          ("MCDHF+B+QED", "multielectron_bqed")]:
            element[key] = metrics([r for r in multielectron_lines
                                    if r["z"] == element["z"] and r["model"] == rung])
    observed_union = ({(r["z"], r["charge"], r["nist_row_index"])
                       for r in all_observed} |
                      {(r["z"], r["charge"], r["nist_row_index"])
                       for r in multielectron_lines})
    summary = {"models": ["atomx DHF+B+QED+normal mass shift",
                          "atomx DHF", "atomx DHF+B+QED",
                          "atomx MCDHF", "atomx MCDHF+B+QED",
                          "atomx point-nucleus Dirac-Coulomb hydrogenic energy"],
               "scope": "Observed classified vacuum lines with validated atomx state pairs; hydrogenic Ritz is a separate reference",
               "observed_ions": sum(bool(e["observed"]["n_lines"]) for e in elements),
               "ritz_ions": sum(bool(e["ritz"]["n_lines"]) for e in elements),
               "observed_lines": sum(e["observed"]["n_lines"] for e in elements),
               "ritz_lines": sum(e["ritz"]["n_lines"] for e in elements),
               "panel_systems": len({r["system"] for r in panel}),
               "panel_rows": len(panel),
               "rejected_ci_panels": rejected_panels,
               "panel_cache_provenance": benchmark_cache_provenance(
                   benchmark_cache),
               "record_audit": audit_counts,
               "observed_archive_rows": sum(v for k, v in audit_counts["lines"].items()
                                            if k not in ("no_observed_wavelength", "ritz_only")),
               "observed_missing_configuration": missing_config,
               "observed_recovered_level_ids": full_dhf_progress.get(
                   "id_recovered_lines", 0),
               "observed_missing_j": missing_j,
               "observed_fully_classified": fully_classified,
               "all_observed_ions": sum(bool(v) for v in all_by_z.values()),
               "all_observed_count": len(all_observed),
               "observed_total_count": len(observed_union),
               "multielectron_lines": multielectron_lines,
               "configuration_average_lines": configuration_average_lines,
               "configuration_average_count": len(configuration_average_lines),
               "radial_box_lines": radial_box_lines,
               "radial_box_count": len(radial_box_lines),
               "unscored_dhf_lines": unscored_dhf_lines,
               "ionization_rows": ionization_rows,
               "isotope_line_summary": isotope_summary,
               "isotope_line_rows": isotope_lines,
               "isotope_level_summary": isotope_level_summary,
               "isotope_level_rows": isotope_levels,
               "configuration_centroid_summary": centroid_summary,
               "configuration_centroid_rows": [r for r in centroid_rows
                                                if r["comparison_status"] == "compared"],
               "inverted_configuration_summary": inverted_summary,
               "inverted_configuration_rows": inverted_rows,
               "dhf_branch_repairs": branch_repairs,
               "dhf_core_p_root_audit": core_p_audit,
               "dhf_heavy_core_p_probe": heavy_core_p_probe,
               "ionization_compared_count": sum(
                   r["status"] == "compared" for r in ionization_rows),
               "ionization_evaluated_count": sum(
                   r["status"] == "compared" and r["reference_kind"] == "evaluated"
                   for r in ionization_rows),
               "ionization_strict_evaluated_count": sum(
                   r["status"] == "compared" and r["reference_kind"] == "evaluated"
                   and r["comparison_basis"] == "unique_ground_level"
                   for r in ionization_rows),
               "ionization_centroid_evaluated_count": sum(
                   r["status"] == "compared"
                   and r["centroid_status"] == "evaluated_complete"
                   for r in ionization_rows),
               "ionization_semiempirical_count": sum(
                   r["status"] == "compared" and r["reference_kind"] == "semiempirical"
                   for r in ionization_rows),
               "ionization_theoretical_count": sum(
                   r["status"] == "compared" and r["reference_kind"] == "theoretical"
                   for r in ionization_rows),
               "full_dhf_progress": full_dhf_progress,
               "multielectron_level_count": len(multielectron_levels),
               "multielectron_observed_count": len({
                   (r["z"], r["charge"], r["nist_row_index"])
                   for r in multielectron_lines}),
               "hydrogen_dhf_calibration": hydrogen_dhf_calibration(
                   args.db, ROOT / "benchmarks/_nist_work/cache.json",
                   ROOT / "output/nist_spectra/full_dhf.sqlite"),
               "proton_electron_mass_ratio": PROTON_ELECTRON_MASS_RATIO,
               "elements": elements,
               "panel": panel, "lines": lines,
               "all_observed_lines": all_observed}
    summary["all_observed_mass_lines"] = all_observed_mass
    write_ionization_csv(ionization_rows, args.output / "ionization_dhf_nist.csv")
    write_isotope_csv(isotope_lines, args.output / "isotope_lines_dhf_nist.csv")
    write_isotope_csv(isotope_levels, args.output / "isotope_levels_dhf_nist.csv")
    write_inverted_csv(inverted_rows,
                       args.output / "inverted_configuration_audit.csv")
    write_configuration_centroids(
        centroid_rows, centroid_summary,
        args.output / "configuration_centroids_dhf_nist.csv",
        args.output / "configuration_centroids_summary.json")
    with (args.output / "matched_lines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(lines)
    for name, records in [("configuration_average_lines.csv", configuration_average_lines),
                          ("unscored_dhf_lines.csv", unscored_dhf_lines),
                          ("multielectron_matched_lines.csv", multielectron_lines),
                          ("multielectron_matched_levels.csv", multielectron_levels)]:
        if records:
            with (args.output / name).open("w", newline="") as handle:
                fields = list(dict.fromkeys(key for record in records for key in record))
                writer = csv.DictWriter(handle, fields)
                writer.writeheader()
                writer.writerows(records)
    atomic_write_text(args.output / "comparison.json",
                      json.dumps(summary, indent=2))
    render_overview(elements, "all_observed", args.output / "periodic_observed.png")
    render_overview(elements, "all_observed_mass",
                    args.output / "periodic_observed_mass.png")
    render_overview(elements, "observed", args.output / "periodic_lyman.png")
    render_overview(elements, "ritz", args.output / "periodic_ritz.png")
    render_overview(elements, "multielectron_bqed",
                    args.output / "periodic_multielectron.png")
    render_overview(elements, "multielectron_dhf_bqed",
                    args.output / "periodic_dhf.png")
    render_overview(elements, "multielectron_dhf_mass",
                    args.output / "periodic_dhf_mass.png")
    render_overview(elements, "ionization",
                    args.output / "periodic_ionization.png")
    render_overview(elements, "ionization_centroid",
                    args.output / "periodic_ionization_centroid.png")
    # The page previews 150 unscored estimates per element; the CSV and
    # comparison.json preserve every row.  Avoid embedding tens of thousands
    # of duplicated records in the interactive HTML.
    site_summary = {key: value for key, value in summary.items() if key not in
                    ("configuration_average_lines", "radial_box_lines",
                     "unscored_dhf_lines")}
    site_summary["unscored_counts_by_z"] = {}
    site_summary["unscored_dhf_lines"] = []
    for z in range(1, 119):
        selected = [r for r in unscored_dhf_lines if r["z"] == z]
        if not selected:
            continue
        site_summary["unscored_counts_by_z"][z] = {
            "configuration_average_only": sum(
                r["comparison_status"] == "configuration_average_only"
                for r in selected),
            "radial_box_unresolved": sum(
                r["comparison_status"] == "radial_box_unresolved"
                for r in selected)}
        site_summary["unscored_dhf_lines"].extend(sorted(
            selected, key=lambda r: (r["charge"], r["observed_nm"]))[:150])
    template = (ROOT / "scripts/theory_comparison_viewer.html").read_text()
    html = template.replace("/* PLOTLY_JS */", get_plotlyjs()).replace(
        "/* COMPARISON_DATA */", json.dumps(site_summary, separators=(",", ":")))
    atomic_write_text(args.output / "index.html", html)
    print(f"Observed: {summary['observed_lines']} matched lines in "
          f"{summary['observed_ions']} H-like ions")
    print(f"Ritz: {summary['ritz_lines']} classified lines in "
          f"{summary['ritz_ions']} H-like ions")
    print(f"Additional level panel: {summary['panel_systems']} systems, "
          f"{summary['panel_rows']} model rows")
    print(args.output / "index.html")


if __name__ == "__main__":
    # Long-running workers and supervisors may request a rebuild together.
    # Serialize the whole build because record_audit.sqlite uses a shared
    # temporary filename and the site reads several generated artifacts.
    import fcntl

    lock_path = ROOT / "output/nist_spectra/theory_comparison/.build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
