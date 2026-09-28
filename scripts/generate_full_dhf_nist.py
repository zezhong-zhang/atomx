#!/usr/bin/env python
"""Resumable DHF+Breit+QED+normal-mass calculation for NIST observed lines.

Every observed NIST row receives a status. A DHF configuration average has
one energy per configuration, so transitions within one configuration cannot
be resolved. The mass correction is leading one-body recoil; the two-body
specific mass shift is not yet present for general open-shell systems.
"""

from __future__ import annotations

import os

# Dense BLAS work on the radial basis is faster with one thread per worker;
# allowing each process to create a full machine-sized pool made the B 20d
# control take 655 s instead of 41 s on this 32-core host. Respect an explicit
# caller override, but keep a fresh CLI worker single-threaded by default.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

from atomx import Atom, Basis  # noqa: E402
from atomx.config import ground_state_config  # noqa: E402
from atomx.config.parsing import parse_nr_shells  # noqa: E402
from atomx.constants import HARTREE_CM1  # noqa: E402
from atomx.methods.dirac_fock import DiracFock  # noqa: E402
from atomx.types import Exchange  # noqa: E402
from atomx.physics.dirac import hydrogenic_dirac_energy  # noqa: E402
from nist_asd import _filled, configuration_map  # noqa: E402

ARCHIVE = ROOT / "output/nist_spectra/spectra.sqlite"
OUTPUT = ROOT / "output/nist_spectra/full_dhf.sqlite"
ISOTOPES = ROOT / "output/nist_spectra/isotopes.json"
SOLVER_VERSION = "occupied_orbital_root_guard_v2"
QED_N_VERSION = "nist_bethe_log_n200_v1"
RADIAL_GRID_VERSION = "adaptive_coulomb_box_one_electron_tail_v2"
# The Mg I 59p endpoint in the observed-line archive needs 668 splines on
# the production grid.  Keep its radial resolution instead of coarsening the
# grid (which shifted Li I 27p by about 1.3 cm^-1 in a convergence check).
MAX_BASIS_SPLINES = 700

SCHEMA = """
CREATE TABLE IF NOT EXISTS config_jobs (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, configuration TEXT NOT NULL,
 status TEXT NOT NULL, energy_ha REAL, nms_ha REAL,
 isotope_a INTEGER NOT NULL, nuclear_mass_me REAL NOT NULL,
 converged INTEGER, elapsed_seconds REAL, error TEXT,
 starting_potential TEXT,
 worker_pid INTEGER, worker_start_ticks INTEGER, claimed_from_status TEXT,
 is_ground INTEGER NOT NULL DEFAULT 0,
 r_max_bohr REAL, grid_hmax_bohr REAL,
 PRIMARY KEY(z,charge,configuration));
CREATE INDEX IF NOT EXISTS config_status ON config_jobs(status,z,charge);
CREATE TABLE IF NOT EXISTS line_jobs (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, row_index INTEGER NOT NULL,
 lower_configuration TEXT, upper_configuration TEXT,
 classification_source TEXT,
 observed_nm REAL NOT NULL, medium TEXT,
 status TEXT NOT NULL, calculated_nm REAL, calculated_vacuum_nm REAL,
 energy_error_percent REAL,
 isotope_a INTEGER NOT NULL, mass_model TEXT NOT NULL,
 PRIMARY KEY(z,charge,row_index));
CREATE INDEX IF NOT EXISTS line_status ON line_jobs(status,z,charge);
CREATE TABLE IF NOT EXISTS config_state_counts (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, configuration TEXT NOT NULL,
 n_states INTEGER NOT NULL, PRIMARY KEY(z,charge,configuration));
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


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


def ground_configuration_for_species(source: sqlite3.Connection, z: int,
                                     charge: int, *, has_ionization: bool
                                     ) -> tuple[str, str]:
    """Prefer NIST's complete ion ground shells over a neutral-sequence guess.

    The level table often abbreviates its ground as just ``5d1``. For a
    highly charged ion the neutral 69-electron filling then restores the
    wrong core. NIST's separate ground-state table supplies ``[Xe].4f14.5d``.
    An expandable level configuration stays first: NIST's ground-shell field
    and largest-component ground-configuration field can differ, while the
    level label identifies the chosen configuration for comparison.
    """
    electrons = z - charge
    level = source.execute("""SELECT configuration FROM levels
        WHERE z=? AND charge=? AND energy_cm1 IS NOT NULL
        ORDER BY ABS(energy_cm1),row_index LIMIT 1""",
        (z, charge)).fetchone()
    from_level = (expand_nist_config(level[0] or "", electrons)
                  if level else None)
    if from_level is not None:
        return from_level, "nist_levels"
    if has_ionization:
        ground = source.execute("""SELECT ground_shells FROM nist_ionization
            WHERE z=? AND charge=?""", (z, charge)).fetchone()
        from_ionization = (expand_nist_config(ground[0] or "", electrons)
                           if ground else None)
        if from_ionization is not None:
            return from_ionization, "nist_ground_shells"
        # ASD ends its direct ionic ground table before the end of the
        # periodic table. A neutral isoelectronic filling can be dramatically
        # wrong for a highly charged ion (for example 114+86: 3d8 4s2 vs
        # 3d10). Use the nearest lower-Z *ionic* member of the same electron
        # sequence as an explicitly extrapolated candidate. This is not a
        # measured ground assignment for the target ion; competing branches
        # must still be checked with their own DHF energies.
        for reference_z, shells in source.execute("""SELECT z,ground_shells
            FROM nist_ionization WHERE z<? AND z-charge=?
              AND charge>0 AND ground_shells IS NOT NULL
            ORDER BY z DESC""", (z, electrons)):
            extrapolated = expand_nist_config(shells, electrons)
            if extrapolated is not None:
                return extrapolated, f"nist_isoelectronic_z{reference_z}"
    fallback = expand_nist_config(ground_state_config(electrons), electrons)
    if fallback is None:
        raise RuntimeError(f"no ground configuration for Z={z}, q={charge}")
    return fallback, "neutral_isoelectronic_fallback"


def ground_configuration_with_dhf_candidates(
        source: sqlite3.Connection, z: int, charge: int, *,
        has_ionization: bool,
        candidates: dict[str, tuple[str, float | None]]
        ) -> tuple[str, str]:
    """Choose the lower converged DHF energy when an ionic analogue is inferred.

    A NIST ionic analogue is a useful starting configuration, not a direct
    assignment for an unlisted target ion. Compare it with the neutral
    isoelectronic fallback after *both* configurations have converged using
    the same nuclear mass and correction stack. Until then, keep the ionic
    analogue as the provisional calculation target.
    """
    preferred, provenance = ground_configuration_for_species(
        source, z, charge, has_ionization=has_ionization)
    if not provenance.startswith("nist_isoelectronic_z"):
        return preferred, provenance
    reference_job = candidates.get(preferred)
    if (reference_job is None or reference_job[0] != "done"
            or reference_job[1] is None
            or not math.isfinite(reference_job[1])):
        return preferred, provenance
    best_config, best_source, best_energy = (
        preferred, provenance, reference_job[1])
    alternatives = []
    neutral = expand_nist_config(ground_state_config(z - charge), z - charge)
    if neutral is not None and neutral != preferred:
        alternatives.append((neutral, f"dhf_lower_than_{provenance}"))
    source_z = int(provenance.removeprefix("nist_isoelectronic_z"))
    seen = {preferred}
    for reference_z, shells in source.execute("""SELECT z,ground_shells
            FROM nist_ionization WHERE z<? AND z-charge=? AND charge>0
              AND ground_shells IS NOT NULL ORDER BY z DESC LIMIT 2""",
            (z, z - charge)):
        configuration = expand_nist_config(shells, z - charge)
        if configuration is None or configuration in seen:
            continue
        seen.add(configuration)
        if len(seen) == 2:
            if configuration != neutral:
                alternatives.append((configuration,
                    f"dhf_lower_nist_isoelectronic_z{reference_z}_over_z{source_z}"))
            break
    for config, source_label in alternatives:
        job = candidates.get(config)
        if (job is not None and job[0] == "done" and job[1] is not None
                and math.isfinite(job[1]) and job[1] < best_energy - 1e-6):
            best_config, best_source, best_energy = config, source_label, job[1]
    return best_config, best_source


def level_id_configurations(source: sqlite3.Connection) -> dict[str, tuple[int, int, str]]:
    """NIST's globally unique level IDs, with the owning ion checked on use."""
    if "level_id" not in {row[1] for row in source.execute(
            "PRAGMA table_info(levels)")}:
        return {}
    return {level_id: (z, charge, config)
            for z, charge, level_id, config in source.execute(
                """SELECT z,charge,level_id,configuration FROM levels
                   WHERE level_id IS NOT NULL AND level_id<>''
                     AND configuration IS NOT NULL AND configuration<>''""")}


def line_endpoint_text(raw: str | None, level_id: str | None,
                       z: int, charge: int,
                       level_configs: dict[str, tuple[int, int, str]]) -> tuple[str, bool]:
    """Use a level's configuration only when its ID belongs to this ion."""
    if raw:
        return raw, False
    matched = level_configs.get(level_id or "")
    if matched is not None and matched[:2] == (z, charge):
        return matched[2], True
    return "", False


def classify_line_endpoints(z: int, charge: int, lower: str,
                            upper: str, medium: str | None
                            ) -> tuple[str | None, str | None, str]:
    electrons = z - charge
    lo = expand_nist_config(lower, electrons)
    hi = expand_nist_config(upper, electrons)
    if not lower or not upper:
        status = "missing_classification"
    elif lo is None or hi is None:
        status = "unsupported_configuration"
    elif lo == hi:
        status = "unresolved_configuration_average"
    elif (electrons == 1 and one_electron_n(lo) is not None
          and one_electron_n(lo) == one_electron_n(hi)):
        status = "unresolved_hydrogenic_n_manifold"
    elif medium not in ("vacuum", "air"):
        status = "unknown_wavelength_medium"
    else:
        status = "waiting_for_levels"
    return lo, hi, status


def one_electron_n(configuration: str) -> int | None:
    match = re.fullmatch(r"(\d+)[spdfghi]1", configuration)
    return int(match.group(1)) if match else None


def process_start_ticks(pid: int) -> int | None:
    """Linux process identity that remains distinct when a PID is reused."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        # The command in parentheses may contain spaces and ')'. Field 3
        # (state) starts after its final closing parenthesis; starttime is 22.
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def radial_box_unresolved(charge: int, configuration: str,
                          r_max_bohr: float = 50.0) -> bool:
    """Flag Rydberg states whose outer turning radius exceeds the 50 a0 box.

    The asymptotic charge seen by the outer electron is charge + 1.  The
    Coulomb outer turning radius is approximately 2 n**2 / (charge + 1).
    This is a conservative domain check, not a fitted accuracy cutoff.
    """
    ns = re.findall(r"\b(\d+)[spdfghiklmnoqrtuvwxyz]", configuration)
    return bool(ns) and 2 * max(map(int, ns)) ** 2 > r_max_bohr * (charge + 1)


class RadialBasisLimitError(RuntimeError):
    """The requested Rydberg box exceeds the validated production basis."""


def basis_for_job(z: int, charge: int, configuration: str) -> tuple[Basis, float, float]:
    """Contain the Coulomb outer turning point with a scalable outer grid."""
    from atomx.numerics.bspline.knots import make_physics_knots

    ns = re.findall(r"\b(\d+)[spdfghiklmnoqrtuvwxyz]", configuration)
    n_max = max(map(int, ns)) if ns else 1
    # A one-electron orbital needs a box beyond its classical outer turning
    # point to preserve the exponential tail. The analytic Dirac check found
    # n=4 and n=5 H states inaccurate in the old 50 a0 production box.
    one_electron_tail = (z - charge == 1
                         and 5 * n_max ** 2 > 50 * (charge + 1))
    if not one_electron_tail and 2 * n_max ** 2 <= 50 * (charge + 1):
        return Basis(), 50.0, 1.0
    radius = max(50.0, 5.0 * n_max ** 2 / (charge + 1))
    hmax = max(1.0, radius / (10.0 * n_max))
    _, count = make_physics_knots(ks=9, Z=z, nuclear="fermi",
                                   r_max=radius, hmax=hmax)
    if count > MAX_BASIS_SPLINES:
        raise RadialBasisLimitError(
            f"{count} splines needed for n={n_max}, charge={charge}; "
            f"production limit is {MAX_BASIS_SPLINES}")
    return Basis(r_max=radius, grid_hmax=hmax), radius, hmax


def pending_job_priority(job: tuple[int, int, str, float, int]
                         ) -> tuple[int, float, int, int, int, str]:
    """Prefer smaller Coulomb boxes within a bounded pending-job window.

    A neutral high-n orbital can occupy a worker for hours while hundreds of
    more compact ionic configurations are ready. Ground states still rank
    first; the original element order breaks ties in radial cost.
    """
    z, charge, configuration, _, is_ground = job
    n_max = max(map(int, re.findall(r"\b(\d+)[spdfghiklmnoqrtuvwxyz]", configuration)),
                default=1)
    radial_cost = n_max * n_max / (charge + 1)
    return (-is_ground, radial_cost, z, charge, len(configuration), configuration)


def one_electron_basis_valid(z: int, configuration: str,
                             energy_ha: float, mass_me: float) -> bool:
    """Sanity gate for a numerical one-electron level against exact Dirac."""
    match = re.fullmatch(r"(\d+)([spdfghi])1", configuration)
    if match is None or z > 20:
        return True
    n = int(match.group(1))
    ell = "spdfghi".index(match.group(2))
    reference = hydrogenic_dirac_energy(n, -(ell + 1), z)
    reference *= mass_me / (mass_me + 1)
    tolerance = 1e-5 if z == 1 else 0.005 * abs(reference)
    return abs(energy_ha - reference) <= tolerance


def orbital_order_valid(result) -> bool:
    """Reject a converged state whose labelled radial orbitals crossed."""
    by_kappa: dict[int, list[tuple[int, float]]] = {}
    for shell in result.subshells:
        energy = result.orbital_energies.get(shell.label)
        if energy is None or not math.isfinite(energy):
            return False
        by_kappa.setdefault(shell.kappa, []).append((shell.n, energy))
    return all(all(energy_a < energy_b for (n_a, energy_a), (n_b, energy_b)
                   in zip(sorted(rows), sorted(rows)[1:]) if n_a < n_b)
               for rows in by_kappa.values())


def orbital_node_valid(result, z: int, configuration: str) -> bool:
    """Reject displaced core-p roots in checked core-hole DHF solutions.

    A Sc IV ``3p5 4p`` GSZ start converged with its labelled 3p orbitals
    carrying two full-amplitude nodes instead of one.  Orbital energies were
    ordered, so the earlier guard accepted a false stationary branch.  Use a
    loose amplitude cutoff to ignore tiny exchange-induced tail wiggles.
    The check is intentionally limited to Z=21..74, n<=3 p cores; Sc, Kr,
    Xe, Ba, and W controls establish the branch distinction in that range.
    Heavier ions have not yet been checked. Ne I's open
    2p core can carry an exchange-induced outer lobe at this cutoff, so the
    same simple node rule must not be applied below this range.
    """
    if z < 21 or z > 74 or not has_core_hole(configuration):
        return True
    core_p_shells = [shell for shell in result.subshells
                     if shell.l == 1 and shell.n <= 3]
    if not core_p_shells:
        return True
    orbitals = result.waves.orbitals
    for shell in core_p_shells:
        wave = orbitals.get(shell.label)
        if wave is None:
            return False
        values = np.asarray(wave.P).ravel()
        if not values.size or not np.all(np.isfinite(values)):
            return False
        peak = np.max(np.abs(values))
        if peak <= 0:
            return False
        significant = values[np.abs(values) > 0.01 * peak]
        nodes = int(np.count_nonzero(np.signbit(significant[:-1]) !=
                                     np.signbit(significant[1:])))
        if nodes != shell.n - shell.l - 1:
            return False
    return True


def has_core_hole(configuration: str) -> bool:
    """Identify occupied shells with a vacancy below another occupied shell."""
    shells = sorted(parse_nr_shells(configuration))
    return any(occupation < _filled(n, ell)
               for n, ell, occupation in shells[:-1])


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


def solve_configuration(z: int, configuration: str, mass: float,
                        basis: Basis, *, retry_only: bool = False):
    """Retry a distinct SCF start only when convergence or orbital order fails."""
    starts = ([("gsz", None), (None, None)] if has_core_hole(configuration)
              else [(None, None), ("gsz", None)])
    if z < 20:
        # The screened TFDA seed recovers diffuse core-hole states for which
        # both Coulomb and GSZ starts oscillate (Li II 1s1 8s1).
        starts.append(("tfda", None))
        if retry_only:
            # B I 2s2 39d failed all three undamped starts but converged
            # with this hydrogenic seed and 0.35 orbital damping. Keep it
            # isolated to retries so ordinary states use their usual path.
            starts.insert(0, (None, 0.35))
    else:
        # Z>=20 defaults to TFDA.  Cu I 3d10 4s did not converge from TFDA
        # or GSZ without mixing, but independent damped-TFDA and genuinely
        # nuclear-only starts converged to the same total energy within 1e-7 Ha.
        # Pd I 4d10 oscillates under the 0.35 branch but reaches the
        # ordered closed-shell solution with 0.70 damping.
        starts.extend(((None, 0.35), (None, 0.70),
                       ("hydrogenic", None)))
        if retry_only:
            # An explicit retry has already exhausted the ordinary starts.
            # Try the distinct convergent branches before repeating them.
            starts = starts[2:] + starts[:2]
    failures = []
    if retry_only and z >= 20:
        # A converged local-exchange state supplies occupied orbitals when
        # direct HF starts have lost an outer shell or oscillated. The final
        # energy still comes from nonlocal DHF with Breit, QED, and nuclear
        # mass. Lr I's earlier direct starts all failed, while this route
        # converged to an ordered DHF state in 31 iterations.
        label = "slater_seed+dhf_damping=0.35"
        try:
            seed = Atom(Z=z, config=configuration, engine="bspline",
                        basis=basis).solve(
                DiracFock(exchange=Exchange.SLATER, breit=True, qed=True,
                           nuclear_mass_me=mass, starting_potential="tfda",
                           scf_damping=0.35), max_iter=120)
            if not seed.converged or not orbital_order_valid(seed):
                raise RuntimeError("Slater seed did not converge to ordered orbitals")
            result = Atom(Z=z, config=configuration, engine="bspline",
                          basis=basis).solve(
                DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                           scf_damping=0.35,
                           seed_orbitals=seed.orbitals_raw), max_iter=200)
            if not result.converged or not math.isfinite(result.total_energy):
                raise RuntimeError(f"seeded DHF did not converge: "
                                   f"residual={result.residual}")
            if not orbital_order_valid(result):
                raise RuntimeError("seeded DHF orbitals are out of energy order")
            if not orbital_node_valid(result, z, configuration):
                raise RuntimeError("seeded DHF core-p orbitals have wrong radial nodes")
            return result, label
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    for starting, damping in starts:
        label = starting or ("tfda" if z >= 20 else "hydrogenic")
        if damping is not None:
            label += f"+damping={damping:g}"
        try:
            result = Atom(Z=z, config=configuration, engine="bspline",
                          basis=basis).solve(
                DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                          starting_potential=starting, scf_damping=damping),
                # Ti I 3d4 was still approaching the same ordered SCF branch
                # at 200 damped iterations and converged at iteration 215.
                max_iter=(300 if retry_only and z >= 20 and damping == 0.70
                          else 200 if retry_only else 160 if damping is not None
                          or starting == "hydrogenic" else
                          80 if basis.r_max is not None or starting else 40))
            if not result.converged or not math.isfinite(result.total_energy):
                raise RuntimeError(f"SCF did not converge: residual={result.residual}")
            if not orbital_order_valid(result):
                raise RuntimeError("labelled same-kappa orbitals are out of energy order")
            if not orbital_node_valid(result, z, configuration):
                raise RuntimeError("labelled core-p orbitals have wrong radial nodes")
            return result, label
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    if retry_only and z >= 20:
        # V I 3d5 and Cr I 3d6 both stalled near 1e-4 when ordinary damping
        # was released. Holding the same 0.70 mix to 1e-6 converged their
        # ordered DHF+B+QED+mass solutions in 99 and 102 iterations.
        label = "tfda+damping=0.7+release=1e-6"
        try:
            result = Atom(Z=z, config=configuration, engine="bspline",
                          basis=basis).solve(
                DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                          starting_potential="tfda", scf_damping=0.70,
                          scf_damping_release_residual=1e-6), max_iter=300)
            if not result.converged or not math.isfinite(result.total_energy):
                raise RuntimeError(f"SCF did not converge: residual={result.residual}")
            if not orbital_order_valid(result):
                raise RuntimeError("labelled same-kappa orbitals are out of energy order")
            if not orbital_node_valid(result, z, configuration):
                raise RuntimeError("labelled core-p orbitals have wrong radial nodes")
            return result, label
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(failures))


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


def prepare(archive: Path, output: Path, isotopes_path: Path,
            ground_max_z: int = 118) -> dict:
    isotopes = json.loads(isotopes_path.read_text())
    source = sqlite3.connect(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as db:
        db.executescript(SCHEMA)
        existing_columns = {row[1] for row in db.execute(
            "PRAGMA table_info(config_jobs)")}
        for column, definition in (
                ("is_ground", "INTEGER NOT NULL DEFAULT 0"),
                ("r_max_bohr", "REAL"),
                ("grid_hmax_bohr", "REAL"),
                ("starting_potential", "TEXT"),
                ("worker_pid", "INTEGER"),
                ("worker_start_ticks", "INTEGER"),
                ("claimed_from_status", "TEXT")):
            if column not in existing_columns:
                db.execute(f"ALTER TABLE config_jobs ADD COLUMN {column} {definition}")
        if "classification_source" not in {row[1] for row in db.execute(
                "PRAGMA table_info(line_jobs)")}:
            db.execute("ALTER TABLE line_jobs ADD COLUMN classification_source TEXT")
        previous = db.execute("SELECT value FROM metadata WHERE key='solver_version'").fetchone()
        if previous is None or previous[0] != SOLVER_VERSION:
            # The old occupied-orbital rotation could turn a core hole into
            # a different, lower-energy configuration without reporting a
            # failure.  Keep the jobs, but require every many-electron level
            # produced by that solver to be recomputed before it is scored.
            db.execute("""UPDATE config_jobs SET status='pending',energy_ha=NULL,
                nms_ha=NULL,converged=NULL,error=NULL WHERE z-charge>=2
                AND status IN ('done','failed','running','untrusted')""")
            db.execute("""UPDATE line_jobs SET status='waiting_for_levels',
                calculated_nm=NULL,calculated_vacuum_nm=NULL,
                energy_error_percent=NULL WHERE z-charge>=2 AND status IN
                ('compared','configuration_average_only','calculation_failed',
                 'basis_validation_failed','inverted_configuration_order')""")
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('solver_version',?)",
                   (SOLVER_VERSION,))
        previous_qed = db.execute(
            "SELECT value FROM metadata WHERE key='qed_principal_n_version'").fetchone()
        qed_rebuild_species: set[tuple[int, int]] = set()
        if previous_qed is None or previous_qed[0] != QED_N_VERSION:
            # The old self-energy adapter parsed only label[0], so an n=10
            # orbital received an n=1 QED term.  n=9 also lacked the Bethe
            # logarithm. Recompute every high-n result with the NIST table.
            for z, charge, config, status in db.execute(
                    """SELECT z,charge,configuration,status FROM config_jobs
                       WHERE status IN ('done','failed','running')""").fetchall():
                ns = re.findall(r"\b(\d+)[spdfghiklmnoqrtuvwxyz]", config)
                if ns and max(map(int, ns)) >= 9:
                    db.execute("""UPDATE config_jobs SET status='pending',
                        energy_ha=NULL,nms_ha=NULL,converged=NULL,error=NULL
                        WHERE z=? AND charge=? AND configuration=?""",
                        (z, charge, config))
                    qed_rebuild_species.add((z, charge))
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('qed_principal_n_version',?)",
                   (QED_N_VERSION,))
        previous_grid = db.execute(
            "SELECT value FROM metadata WHERE key='radial_grid_version'").fetchone()
        radial_rebuild_species: set[tuple[int, int]] = set()
        if previous_grid is None or previous_grid[0] != RADIAL_GRID_VERSION:
            for z, charge, configuration in db.execute(
                    """SELECT z,charge,configuration FROM config_jobs
                       WHERE status IN ('done','failed','untrusted')""").fetchall():
                principal_n = one_electron_n(configuration)
                one_electron_tail = (
                    z - charge == 1 and principal_n is not None
                    and 5 * principal_n ** 2 > 50 * (charge + 1))
                if (radial_box_unresolved(charge, configuration)
                        or one_electron_tail):
                    db.execute("""UPDATE config_jobs SET status='pending',
                        energy_ha=NULL,nms_ha=NULL,converged=NULL,error=NULL,
                        r_max_bohr=NULL,grid_hmax_bohr=NULL
                        WHERE z=? AND charge=? AND configuration=?""",
                        (z, charge, configuration))
                    radial_rebuild_species.add((z, charge))
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('radial_grid_version',?)",
                   (RADIAL_GRID_VERSION,))
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                   ("model", "DHF; Breit=True; QED=True; Fermi nucleus; isotope-specific leading normal mass shift"))
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                   ("isotope_source_sha256", isotopes["source_sha256"]))
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                   ("recoil_limit", "normal mass shift from virial estimator only; general specific mass shift and relativistic recoil absent"))
        state_sets: dict[tuple[int, int, str], set[tuple[str, str, float]]] = {}
        for z, charge, config, term, j, energy in source.execute(
                """SELECT z,charge,configuration,term,j_text,energy_cm1
                   FROM levels WHERE energy_cm1 IS NOT NULL"""):
            full = expand_nist_config(config or "", z - charge)
            if full:
                state_sets.setdefault((z, charge, full), set()).add(
                    (term or "", j or "", energy))
        db.executemany("""INSERT OR REPLACE INTO config_state_counts
            VALUES (?,?,?,?)""",
            ((z, charge, config, len(states))
             for (z, charge, config), states in state_sets.items()))
        level_configs = level_id_configurations(source)
        line_columns = {row[1] for row in source.execute(
            "PRAGMA table_info(lines)")}
        id_columns = ("lower_level_id,upper_level_id" if
                      {"lower_level_id", "upper_level_id"} <= line_columns
                      else "NULL,NULL")
        jobs = set()
        counts = Counter()
        for (z, charge, index, lower_raw, upper_raw, lower_id, upper_id,
             observed, medium) in source.execute(
                "SELECT z,charge,row_index,lower_configuration,"
                "upper_configuration," + id_columns + ",observed_nm,medium "
                "FROM lines WHERE observed_nm IS NOT NULL "
                "ORDER BY z,charge,row_index"):
            isotope = isotopes["selected"][str(z)]
            lower, lower_from_id = line_endpoint_text(
                lower_raw, lower_id, z, charge, level_configs)
            upper, upper_from_id = line_endpoint_text(
                upper_raw, upper_id, z, charge, level_configs)
            lo, hi, status = classify_line_endpoints(
                z, charge, lower, upper, medium)
            classification_source = (
                "nist_level_id" if lower_from_id or upper_from_id
                else "line_configuration" if lower and upper else "missing")
            if status == "waiting_for_levels":
                jobs.add((z, charge, lo, isotope["a"], isotope["nuclear_mass_me"]))
                jobs.add((z, charge, hi, isotope["a"], isotope["nuclear_mass_me"]))
            db.execute("""INSERT OR IGNORE INTO line_jobs
                (z,charge,row_index,lower_configuration,upper_configuration,
                 classification_source,observed_nm,medium,status,isotope_a,mass_model)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (z, charge, index, lo, hi, classification_source,
                 observed, medium, status,
                 isotope["a"], "one-body normal mass shift"))
            counts[status] += 1
        # Every requested ion stage receives at least its NIST ground level,
        # even when ASD has no observed line with classified endpoints.
        ground_added = 0
        ground_keys = []
        ground_selection_counts = Counter()
        has_ionization = source.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='nist_ionization'""").fetchone() is not None
        for z, charge in source.execute(
                "SELECT z,charge FROM species WHERE z<=? ORDER BY z,charge",
                (ground_max_z,)):
            ground, selection = ground_configuration_for_species(
                source, z, charge, has_ionization=has_ionization)
            ground_selection_counts[selection] += 1
            ground_keys.append((z, charge, ground))
            isotope = isotopes["selected"][str(z)]
            job = (z, charge, ground, isotope["a"],
                   isotope["nuclear_mass_me"])
            if job not in jobs:
                jobs.add(job)
                ground_added += 1
        db.executemany("""INSERT OR IGNORE INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me)
            VALUES (?,?,?,'pending',?,?)""", sorted(jobs))
        db.execute("UPDATE config_jobs SET is_ground=0 WHERE z<=?",
                   (ground_max_z,))
        db.executemany("""UPDATE config_jobs SET is_ground=1
            WHERE z=? AND charge=? AND configuration=?""", ground_keys)
        for z, charge in qed_rebuild_species | radial_rebuild_species:
            update_lines(db, z, charge)
        db.commit()
    source.close()
    summary = dict(lines=sum(counts.values()), statuses=dict(counts),
                   configuration_jobs=len(jobs), ground_jobs_added=ground_added,
                   ground_selection_sources=dict(ground_selection_counts))
    summary["level_id_backfill"] = backfill_nist_level_ids(
        archive, output, isotopes_path)
    return summary


def backfill_nist_level_ids(archive: Path, output: Path,
                            isotopes_path: Path) -> dict:
    """Recover missing line endpoints through exact NIST level IDs in place.

    The source line columns stay untouched. Only lines still marked as missing
    classification are changed, so this can run while solver workers use the
    existing ledger. Newly needed configurations enter the ordinary queue.
    """
    isotopes = json.loads(isotopes_path.read_text())["selected"]
    with sqlite3.connect(archive) as source, sqlite3.connect(output,
                                                              timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("PRAGMA journal_mode=WAL")
        if "classification_source" not in {row[1] for row in db.execute(
                "PRAGMA table_info(line_jobs)")}:
            db.execute("ALTER TABLE line_jobs ADD COLUMN classification_source TEXT")
        source_columns = {row[1] for row in source.execute(
            "PRAGMA table_info(lines)")}
        if not {"lower_level_id", "upper_level_id"} <= source_columns:
            return {"recovered_lines": 0, "new_configurations": 0,
                    "statuses": {}}
        missing = set(db.execute("""SELECT z,charge,row_index FROM line_jobs
            WHERE status='missing_classification'"""))
        level_configs = level_id_configurations(source)
        updates = []
        jobs = set()
        affected = set()
        statuses = Counter()
        for (z, charge, index, lower_raw, upper_raw, lower_id, upper_id,
             medium) in source.execute("""SELECT z,charge,row_index,
                lower_configuration,upper_configuration,lower_level_id,
                upper_level_id,medium FROM lines WHERE observed_nm IS NOT NULL
                AND (COALESCE(lower_configuration,'')=''
                     OR COALESCE(upper_configuration,'')='')"""):
            if (z, charge, index) not in missing:
                continue
            lower, lower_from_id = line_endpoint_text(
                lower_raw, lower_id, z, charge, level_configs)
            upper, upper_from_id = line_endpoint_text(
                upper_raw, upper_id, z, charge, level_configs)
            if not (lower_from_id or upper_from_id):
                continue
            lo, hi, status = classify_line_endpoints(
                z, charge, lower, upper, medium)
            if status == "missing_classification":
                continue
            updates.append((lo, hi, status, "nist_level_id", z, charge, index))
            statuses[status] += 1
            affected.add((z, charge))
            if status == "waiting_for_levels":
                isotope = isotopes[str(z)]
                jobs.add((z, charge, lo, isotope["a"],
                          isotope["nuclear_mass_me"]))
                jobs.add((z, charge, hi, isotope["a"],
                          isotope["nuclear_mass_me"]))
        before = db.execute("SELECT COUNT(*) FROM config_jobs").fetchone()[0]
        with db:
            db.executemany("""UPDATE line_jobs SET lower_configuration=?,
                upper_configuration=?,status=?,classification_source=?
                WHERE z=? AND charge=? AND row_index=?
                  AND status='missing_classification'""", updates)
            db.executemany("""INSERT OR IGNORE INTO config_jobs
                (z,charge,configuration,status,isotope_a,nuclear_mass_me)
                VALUES (?,?,?,'pending',?,?)""", sorted(jobs))
        after = db.execute("SELECT COUNT(*) FROM config_jobs").fetchone()[0]
        affected.update(db.execute("""SELECT DISTINCT z,charge FROM line_jobs
            WHERE classification_source='nist_level_id'"""))
        for z, charge in sorted(affected):
            for attempt in range(5):
                try:
                    # Claim the write lock before update_lines reads a
                    # snapshot. Otherwise concurrent workers can make a
                    # read-to-write upgrade fail with SQLITE_BUSY_SNAPSHOT.
                    db.execute("BEGIN IMMEDIATE")
                    update_lines(db, z, charge)
                    db.commit()
                    break
                except sqlite3.OperationalError as exc:
                    db.rollback()
                    if "locked" not in str(exc).lower() or attempt == 4:
                        raise
                    time.sleep(0.2 * (attempt + 1))
        return {"recovered_lines": len(updates),
                "new_configurations": after - before,
                "statuses": dict(statuses),
                "affected_ions": len(affected)}


def update_lines(db: sqlite3.Connection, z: int, charge: int) -> None:
    energies = {config: (status, energy, radius or 50.0)
                for config, status, energy, radius in db.execute(
        """SELECT configuration,status,energy_ha,r_max_bohr FROM config_jobs
           WHERE z=? AND charge=?""",
        (z, charge))}
    for index, lo, hi, observed, medium in db.execute(
            """SELECT row_index,lower_configuration,upper_configuration,
                      observed_nm,medium FROM line_jobs
               WHERE z=? AND charge=? AND status IN
               ('waiting_for_levels','calculation_failed','basis_validation_failed',
                'unresolved_hydrogenic_n_manifold',
                'inverted_configuration_order','compared',
                'configuration_average_only','radial_box_unresolved',
                'basis_domain_limited')""",
            (z, charge)):
        if (z - charge == 1 and one_electron_n(lo) is not None
                and one_electron_n(lo) == one_electron_n(hi)):
            db.execute("""UPDATE line_jobs SET status='unresolved_hydrogenic_n_manifold',
                calculated_nm=NULL,calculated_vacuum_nm=NULL,energy_error_percent=NULL
                WHERE z=? AND charge=? AND row_index=?""", (z, charge, index))
            continue
        lower = energies.get(lo)
        upper = energies.get(hi)
        if lower is None or upper is None:
            continue
        if lower[0] == "untrusted" or upper[0] == "untrusted":
            values = ("basis_validation_failed", None, None, None)
        elif lower[0] == "basis_limited" or upper[0] == "basis_limited":
            values = ("basis_domain_limited", None, None, None)
        elif lower[0] == "failed" or upper[0] == "failed":
            values = ("calculation_failed", None, None, None)
        elif lower[0] != "done" or upper[0] != "done":
            values = ("waiting_for_levels", None, None, None)
        else:
            gap = (upper[1] - lower[1]) * HARTREE_CM1
            if gap <= 0 or not math.isfinite(gap):
                values = ("inverted_configuration_order", None, None, None)
            else:
                vacuum = 1e7 / gap
                predicted = (vacuum / refractive_index_air(vacuum)
                             if medium == "air" else vacuum)
                observed_vacuum = observed_vacuum_nm(observed, medium)
                error = (100 * (observed_vacuum / vacuum - 1)
                         if observed_vacuum is not None else None)
                counts = {config: n for config, n in db.execute(
                    """SELECT configuration,n_states FROM config_state_counts
                       WHERE z=? AND charge=? AND configuration IN (?,?)""",
                    (z, charge, lo, hi))}
                status = ("configuration_average_only"
                          if (counts.get(lo, 0) != 1 or counts.get(hi, 0) != 1
                              or not configuration_has_single_level(lo)
                              or not configuration_has_single_level(hi))
                          else "radial_box_unresolved"
                          if (radial_box_unresolved(charge, lo, lower[2])
                              or radial_box_unresolved(charge, hi, upper[2]))
                          else "compared")
                values = (status, predicted, vacuum, error)
        db.execute("""UPDATE line_jobs SET status=?,calculated_nm=?,
            calculated_vacuum_nm=?,energy_error_percent=?
            WHERE z=? AND charge=? AND row_index=?""", (*values, z, charge, index))


def run(output: Path, max_calculations: int | None = None,
        refresh_site_every: int = 50, max_z: int = 118,
        recover_running: bool = True, retry_only: bool = False,
        min_z: int = 1, queue_status: str | None = None,
        min_charge: int = 0, max_charge: int = 117) -> None:
    if not 1 <= min_z <= max_z <= 118:
        raise ValueError("require 1 <= min_z <= max_z <= 118")
    if not 0 <= min_charge <= max_charge <= 117:
        raise ValueError("require 0 <= min_charge <= max_charge <= 117")
    if queue_status is not None and queue_status not in (
            "pending", "retry_pending", "repair_pending",
            "candidate_pending"):
        raise ValueError("unsupported calculation queue status")
    db = sqlite3.connect(output, timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("BEGIN IMMEDIATE")
    existing_columns = {row[1] for row in db.execute(
        "PRAGMA table_info(config_jobs)")}
    for column, definition in (("starting_potential", "TEXT"),
                               ("worker_pid", "INTEGER"),
                               ("worker_start_ticks", "INTEGER"),
                               ("claimed_from_status", "TEXT")):
        if column not in existing_columns:
            db.execute(f"ALTER TABLE config_jobs ADD COLUMN {column} {definition}")
    previous_scope = db.execute(
        "SELECT value FROM metadata WHERE key='active_max_z'").fetchone()
    recorded_max_z = max(max_z, int(previous_scope[0])) if previous_scope else max_z
    db.execute("INSERT OR REPLACE INTO metadata VALUES ('active_max_z',?)",
               (str(recorded_max_z),))
    if recover_running:
        db.execute("""UPDATE config_jobs SET status='pending',worker_pid=NULL,
                   worker_start_ticks=NULL,claimed_from_status=NULL
                   WHERE status='running' AND z BETWEEN ? AND ?
                   AND charge BETWEEN ? AND ?""",
                   (min_z, max_z, min_charge, max_charge))
        # Reassess levels generated by older runs before scoring or resuming.
        for z, charge, configuration, energy, mass in db.execute(
                """SELECT z,charge,configuration,energy_ha,nuclear_mass_me
                   FROM config_jobs WHERE status='done' AND z-charge=1
                   AND z BETWEEN ? AND MIN(20, ?)
                   AND charge BETWEEN ? AND ?""",
                (min_z, max_z, min_charge, max_charge)):
            if not one_electron_basis_valid(z, configuration, energy, mass):
                db.execute("""UPDATE config_jobs SET status='untrusted',
                    error='one-electron level fails analytic Dirac basis check'
                    WHERE z=? AND charge=? AND configuration=?""",
                    (z, charge, configuration))
        # Reconcile every line after a batch.  This reapplies current
        # validity gates to results computed by long-running older workers.
        for z, charge in db.execute("""SELECT DISTINCT z,charge FROM line_jobs
                                    WHERE z BETWEEN ? AND ?
                                    AND charge BETWEEN ? AND ?""",
                                    (min_z, max_z, min_charge,
                                     max_charge)).fetchall():
            update_lines(db, z, charge)
    db.commit()
    queue_status = queue_status or ("retry_pending" if retry_only else "pending")
    query = """SELECT z,charge,configuration,nuclear_mass_me,is_ground FROM config_jobs
               WHERE status=? AND z>=? AND z<=?
               AND charge>=? AND charge<=?
               ORDER BY is_ground DESC,z,charge,LENGTH(configuration),configuration
               LIMIT 512"""
    completed = 0
    worker_pid = os.getpid()
    worker_start_ticks = process_start_ticks(worker_pid)
    while max_calculations is None or completed < max_calculations:
        # SQLite's reserved write lock makes the claim atomic across workers.
        db.execute("BEGIN IMMEDIATE")
        candidates = db.execute(query, (queue_status, min_z, max_z,
                                        min_charge, max_charge)).fetchall()
        if not candidates:
            db.commit()
            break
        z, charge, configuration, mass, _ = min(
            candidates, key=pending_job_priority)
        db.execute("""UPDATE config_jobs SET status='running',worker_pid=?,
            worker_start_ticks=?,claimed_from_status=?
            WHERE z=? AND charge=? AND configuration=?""",
            (worker_pid, worker_start_ticks, queue_status,
             z, charge, configuration))
        db.commit()
        print(f"claimed Z={z} q={charge} {configuration} "
              f"worker={worker_pid}", flush=True)
        start = time.monotonic()
        radius = hmax = None
        try:
            basis, radius, hmax = basis_for_job(z, charge, configuration)
            result, starting = solve_configuration(
                z, configuration, mass, basis, retry_only=retry_only)
            if charge == z - 1 and not one_electron_basis_valid(
                    z, configuration, result.total_energy, mass):
                raise RuntimeError("one-electron level fails analytic Dirac basis check")
            db.execute("""UPDATE config_jobs SET status='done',energy_ha=?,
                nms_ha=?,converged=1,elapsed_seconds=?,error=NULL,
                r_max_bohr=?,grid_hmax_bohr=?,starting_potential=?,
                worker_pid=NULL,worker_start_ticks=NULL,claimed_from_status=NULL
                WHERE z=? AND charge=? AND configuration=?""",
                (float(result.total_energy),
                 float(result.energy_components.get("nms", 0.0)),
                 time.monotonic() - start, radius, hmax, starting,
                 z, charge, configuration))
            print(f"done Z={z} q={charge} {configuration} "
                  f"E={result.total_energy:.9f} Ha", flush=True)
        except Exception as exc:
            status = ("basis_limited" if isinstance(exc, RadialBasisLimitError)
                      else "failed")
            db.execute("""UPDATE config_jobs SET status=?,elapsed_seconds=?,
                error=?,converged=0,r_max_bohr=?,grid_hmax_bohr=?,
                worker_pid=NULL,worker_start_ticks=NULL,claimed_from_status=NULL
                WHERE z=? AND charge=? AND configuration=?""",
                (status, time.monotonic() - start,
                 f"{type(exc).__name__}: {exc}"[:1000], radius, hmax,
                 z, charge, configuration))
            print(f"failed Z={z} q={charge} {configuration}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        update_lines(db, z, charge)
        db.commit()
        completed += 1
        if refresh_site_every and completed % refresh_site_every == 0:
            rebuilt = subprocess.run(
                [sys.executable, str(ROOT / "scripts/compare_theory_nist.py")],
                cwd=ROOT, check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True)
            if rebuilt.returncode:
                print(f"site refresh failed: {rebuilt.stderr[-1000:]}",
                      file=sys.stderr, flush=True)
            else:
                print(f"site refreshed after {completed} calculations", flush=True)
    print(dict(db.execute("""SELECT status,COUNT(*) FROM config_jobs
                             WHERE z BETWEEN ? AND ? AND charge BETWEEN ? AND ?
                             GROUP BY status""",
                          (min_z, max_z, min_charge, max_charge))), flush=True)
    print(dict(db.execute("""SELECT status,COUNT(*) FROM line_jobs
                             WHERE z BETWEEN ? AND ? AND charge BETWEEN ? AND ?
                             GROUP BY status""",
                          (min_z, max_z, min_charge, max_charge))), flush=True)
    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--isotopes", type=Path, default=ISOTOPES)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--backfill-level-ids", action="store_true",
                        help="recover missing line endpoints using NIST level IDs")
    parser.add_argument("--max-calculations", type=int)
    parser.add_argument("--max-z", type=int, default=118,
                        help="calculate only elements through this atomic number")
    parser.add_argument("--min-z", type=int, default=1,
                        help="calculate only elements starting at this atomic number")
    parser.add_argument("--min-charge", type=int, default=0,
                        help="calculate only ion stages starting at this charge")
    parser.add_argument("--max-charge", type=int, default=117,
                        help="calculate only ion stages through this charge")
    parser.add_argument("--worker-only", action="store_true",
                        help="join an existing prepared ledger without resetting active jobs")
    parser.add_argument("--retry-only", action="store_true",
                        help="claim only jobs explicitly marked retry_pending")
    parser.add_argument("--queue-status", choices=("pending", "retry_pending",
                        "repair_pending", "candidate_pending"),
                        help="claim only this queue status")
    parser.add_argument("--refresh-site-every", type=int, default=50,
                        help="rebuild the interactive comparison after this many jobs; 0 disables")
    args = parser.parse_args()
    if args.backfill_level_ids:
        if not args.output.exists():
            parser.error("--backfill-level-ids requires an existing ledger")
        print(backfill_nist_level_ids(args.archive, args.output,
                                      args.isotopes), flush=True)
        return
    if args.worker_only:
        if not args.output.exists():
            parser.error("--worker-only requires an existing prepared ledger")
    else:
        summary = prepare(args.archive, args.output, args.isotopes,
                          ground_max_z=args.max_z)
        print(summary, flush=True)
    if not args.prepare_only:
        run(args.output, args.max_calculations, args.refresh_site_every,
            args.max_z, recover_running=not args.worker_only,
            retry_only=args.retry_only, min_z=args.min_z,
            min_charge=args.min_charge, max_charge=args.max_charge,
            queue_status=args.queue_status)


if __name__ == "__main__":
    main()
