#!/usr/bin/env python
"""Verify atomx excitation energies against the committed NIST ASD panel.

For every system the script climbs a fixed ladder of theory rungs with
the *public* atomx API and compares the result with the observed NIST
levels fetched by :mod:`benchmarks.fetch_nist_asd`:

===============  =====================================================
rung             how it is computed
===============  =====================================================
``DHF``          one jj-average Dirac-Fock SCF per configuration
                 (``Atom.solve(DiracFock(breit=False, qed=False))``);
                 every level of a configuration shares its energy
``DHF+B+QED``    the same with ``DiracFock(breit=True, qed=True)``
``MCDHF``        one CI over the model space
                 (``Atom(configs=[...], breit=False, qed=False)``),
                 so the (2J, parity) blocks are split
``MCDHF+B+QED``  the same with Breit + QED
``DHF+MBPT2``    the ``DHF`` rung with the *closed-shell ground
                 configuration* replaced by ``E_DHF + E2``
                 (``atomx.methods.mbpt2_energy``); the excited states are
                 unchanged, so this rung is explicitly composite
===============  =====================================================

Only **excitation energies relative to the computed ground state** are
compared: NIST ASD levels are relative to the ion's ground state, so an
atomx *total* energy is never comparable to a NIST number.

Matching computed states to NIST levels
---------------------------------------
The model space is a list of configurations; the (2J, parity) block
dimensions it implies are counted from
:func:`atomx.methods.mcdhf.config.enumerate_jj_csfs_many` (summed over
the configurations), and the NIST levels of the *same* configurations,
with a single printed J and a printed energy, are grouped the same way.
Within a block the two energy-ordered lists are paired positionally.
A block whose counts disagree is **excluded and reported**, never
silently re-paired.  ``mode="block-min"`` (used only where NIST's list
is a strict subset of the configuration's LSJ space, e.g. Fe I) pairs
the lowest ``k = min(n_calc, n_nist)`` states and additionally asserts
that the first unpaired computed state lies above the last paired NIST
level.

The model space itself is chosen from the declared candidate list by an
exhaustive search for the largest sub-list whose blocks balance exactly,
so the configuration space is limited by *what NIST lists*, not by
hand-tuning.

Outputs (all deterministic):

* ``benchmarks/results_nist_levels.csv`` — one row per (system, rung)
* ``benchmarks/NIST_VERIFICATION.md``    — the human-readable report
* ``benchmarks/_nist_work/cache.json``   — SCF/CI cache (git-ignored),
  so repeated runs are cheap; delete it or pass ``--refresh`` to force
  a full recomputation.

Usage::

    python benchmarks/verify_nist_levels.py                 # all systems
    python benchmarks/verify_nist_levels.py --system "Fe XVII"
    python benchmarks/verify_nist_levels.py --quick         # cheap systems only
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import benchmarks.nist_asd as nist  # noqa: E402
from atomx import Atom, Corrections  # noqa: E402
from atomx.methods import DiracFock, mbpt2_energy  # noqa: E402
from atomx.methods.mcdhf.config import enumerate_jj_csfs  # noqa: E402
from atomx.physics.dirac import hydrogenic_dirac_energy  # noqa: E402

WORK_DIR = REPO_ROOT / "benchmarks" / "_nist_work"
CACHE_PATH = WORK_DIR / "cache.json"
CSV_PATH = REPO_ROOT / "benchmarks" / "results_nist_levels.csv"
REPORT_PATH = REPO_ROOT / "benchmarks" / "NIST_VERIFICATION.md"

CM1 = nist.HA_CM1
EV = nist.HARTREE_EV


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """One system: NIST panel label + the candidate configuration space."""

    label: str
    z: int
    candidates: tuple[str, ...]
    ground: str
    mode: str = "exact"          # "exact" | "block-min"
    mbpt2: bool = False
    slow: bool = False
    limit: str = ""
    note: str = ""


def _case(label, z, candidates, ground, **kw) -> Case:
    return Case(label=label, z=z, candidates=tuple(candidates),
                ground=ground, **kw)


#: Full electron configurations (atomx spelling, explicit occupations —
#: ``solve(DiracFock(...))`` rejects the implicit-occupation form).
CASES: tuple[Case, ...] = (
    _case("H I", 1,
          ("1s1", "2s1", "2p1", "3s1", "3p1", "3d1"), "1s1",
          limit="one electron: no correlation; the residual is the "
                "finite-nuclear-size + QED treatment and the B-spline "
                "radial basis"),
    _case("He I", 2,
          ("1s2", "1s1 2s1", "1s1 2p1", "1s1 3s1", "1s1 3p1", "1s1 3d1"),
          "1s2", mbpt2=True,
          limit="two electrons: the model space is small, so the residual "
                "is dominated by missing correlation (and by the jj "
                "one-configuration average for the DHF rungs)"),
    _case("Li I", 3,
          ("1s2 2s1", "1s2 2p1", "1s2 3s1", "1s2 3p1", "1s2 3d1"), "1s2 2s1",
          limit="one valence electron over a frozen He-like core: core "
                "correlation and core polarisation are absent"),
    _case("Be I", 4,
          ("1s2 2s2", "1s2 2s1 2p1", "1s2 2p2", "1s2 2s1 3s1", "1s2 2s1 3p1"),
          "1s2 2s2", mbpt2=True,
          limit="four electrons, 2s-2p near-degeneracy: the 5-configuration "
                "model space is the limiting factor (see the rung walk)"),
    _case("C I", 6,
          ("1s2 2s2 2p2", "1s2 2s2 2p1 3s1", "1s2 2s2 2p1 3p1",
           "1s2 2s1 2p3"), "1s2 2s2 2p2",
          limit="open 2p shell: valence correlation and the 2s2p3 "
                "configuration interaction are only partly spanned"),
    _case("N I", 7,
          ("1s2 2s2 2p3", "1s2 2s2 2p2 3s1", "1s2 2s2 2p2 3p1",
           "1s2 2s1 2p4"), "1s2 2s2 2p3", mode="block-min",
          limit="half-filled 2p shell: NIST lists some levels of the "
                "excited configurations without an energy, so the model "
                "space cannot be balanced exactly"),
    _case("O I", 8,
          ("1s2 2s2 2p4", "1s2 2s2 2p3 3s1", "1s2 2s2 2p3 3p1",
           "1s2 2s1 2p5"), "1s2 2s2 2p4", mode="block-min",
          limit="open 2p shell; 30 NIST rows of these configurations carry "
                "no energy value at all"),
    _case("Ne I", 10,
          ("1s2 2s2 2p6", "1s2 2s2 2p5 3s1", "1s2 2s2 2p5 3p1",
           "1s2 2s2 2p5 3d1"), "1s2 2s2 2p6", mbpt2=True,
          limit="closed shell: the ground state is exact at the DHF level "
                "but the excited states need correlation"),
    _case("Na I", 11,
          ("1s2 2s2 2p6 3s1", "1s2 2s2 2p6 3p1", "1s2 2s2 2p6 4s1",
           "1s2 2s2 2p6 3d1", "1s2 2s2 2p6 4p1"), "1s2 2s2 2p6 3s1",
          limit="one valence electron over a Ne-like core: core "
                "polarisation is the leading missing term"),
    _case("Ar I", 18,
          ("1s2 2s2 2p6 3s2 3p6", "1s2 2s2 2p6 3s2 3p5 4s1",
           "1s2 2s2 2p6 3s2 3p5 4p1"), "1s2 2s2 2p6 3s2 3p6", mbpt2=True,
          limit="closed 3p shell; the 3p5 4s / 3p5 4p model space is small "
                "and the 3d correlating shell is absent"),
    _case("Fe I", 26,
          ("1s2 2s2 2p6 3s2 3p6 3d6 4s2",
           "1s2 2s2 2p6 3s2 3p6 3d6 4s1 4p1",
           "1s2 2s2 2p6 3s2 3p6 3d7 4s1"), "1s2 2s2 2p6 3s2 3p6 3d6 4s2",
          mode="block-min", slow=True,
          limit="open 3d shell with strong 3d-4s-4p mixing; NIST lists only "
                "the observed subset of each configuration's LSJ space "
                "(23 of the 34 3d6 4s2 CI states), so blocks cannot be "
                "balanced and the pairing is positional from the bottom"),
    _case("C VI", 6, ("1s1", "2s1", "2p1", "3s1", "3p1", "3d1"), "1s1",
          limit="H-like: no correlation; the residual is QED + nuclear size",
          note="H-like (the task brief calls this sequence 'He-like'); "
               "1 electron"),
    _case("Ne X", 10, ("1s1", "2s1", "2p1", "3s1", "3p1", "3d1"), "1s1",
          limit="H-like at Z=10: the self-energy is the largest missing "
                "term after the Dirac eigenvalue"),
    _case("Ar XVIII", 18, ("1s1", "2s1", "2p1", "3s1", "3p1", "3d1"), "1s1",
          limit="H-like at Z=18: QED grows as Z^4"),
    _case("Fe XXVI", 26, ("1s1", "2s1", "2p1", "3s1", "3p1", "3d1"), "1s1",
          limit="H-like at Z=26: the 1s self-energy is ~100 eV, so the QED "
                "rung dominates the error"),
    _case("C V", 6,
          ("1s2", "1s1 2s1", "1s1 2p1", "1s1 3s1", "1s1 3p1", "1s1 3d1"),
          "1s2", mbpt2=True,
          limit="He-like: the two-electron model space is complete for "
                "n<=3, so the residual is mostly QED + missing higher n"),
    _case("Ne IX", 10,
          ("1s2", "1s1 2s1", "1s1 2p1", "1s1 3s1", "1s1 3p1", "1s1 3d1"),
          "1s2", mbpt2=True,
          limit="He-like at Z=10: same structure as C V, QED ~Z^4 larger"),
    _case("Ar XVII", 18,
          ("1s2", "1s1 2s1", "1s1 2p1", "1s1 3s1", "1s1 3p1", "1s1 3d1"),
          "1s2", mbpt2=True,
          limit="He-like at Z=18: the MCDHF rung cannot carry the "
                "spin-dependent Breit part (see the MCDHF docstring)"),
    _case("Fe XXV", 26,
          ("1s2", "1s1 2s1", "1s1 2p1", "1s1 3s1", "1s1 3p1", "1s1 3d1"),
          "1s2", mbpt2=True, slow=True,
          limit="He-like at Z=26: fine structure and QED are both large and "
                "the CI model space holds only n<=3"),
    _case("Ne VIII", 10,
          ("1s2 2s1", "1s2 2p1", "1s2 3s1", "1s2 3p1", "1s2 3d1"), "1s2 2s1",
          limit="Li-like: one valence electron, so correlation is small and "
                "the residual is core polarisation + QED"),
    _case("Ar VIII", 18,
          ("1s2 2s2 2p6 3s1", "1s2 2s2 2p6 3p1", "1s2 2s2 2p6 3d1",
           "1s2 2s2 2p6 4s1", "1s2 2s2 2p6 4p1"), "1s2 2s2 2p6 3s1",
          limit="Na-like: one valence electron over a closed Ne-like core"),
    _case("Fe XVII", 26,
          ("1s2 2s2 2p6", "1s2 2s2 2p5 3s1", "1s2 2s2 2p5 3p1",
           "1s2 2s2 2p5 3d1"), "1s2 2s2 2p6", slow=True,
          limit="Ne-like closed shell: strong 2p-3s/3d correlation and a "
                "large spin-orbit splitting of the 2p hole"),
)

#: The task brief cites ``asd.026017`` as "Fe XVII"; that file is Fe XVIII
#: (F-like).  The fetcher validates the label embedded in each file, so
#: the mistake cannot propagate — recorded here for the report.
BRIEF_INDEX_ERRATUM = ("asd.026017 is Fe XVIII (F-like); Ne-like Fe XVII "
                       "is asd.026016 — see fetch_nist_asd.PANEL")


RUNGS = ("DHF", "DHF+B+QED", "MCDHF", "MCDHF+B+QED", "DHF+MBPT2")


# ---------------------------------------------------------------------------
# calculation cache
# ---------------------------------------------------------------------------

class Cache:
    """JSON-backed cache of SCF total energies and CI state tables."""

    def __init__(self, path: Path, refresh: bool = False):
        self.path = path
        self.data: dict = {} if refresh else self._load()
        self.hits = self.misses = 0

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))

    def scf(self, z: int, config: str, breit: bool, qed: bool) -> float:
        key = f"scf|{z}|{config}|{int(breit)}|{int(qed)}"
        if key in self.data:
            self.hits += 1
            return float(self.data[key])
        self.misses += 1
        result = Atom(Z=z, config=config).solve(DiracFock(breit=breit, qed=qed))
        self.data[key] = float(result.total_energy)
        return float(self.data[key])

    def scf_result(self, z: int, config: str):
        """The SCFResult object (needed by MBPT(2)); never cached."""
        return Atom(Z=z, config=config).solve(DiracFock(breit=False, qed=False))

    def mbpt2(self, z: int, config: str, result) -> float:
        key = f"mbpt2|{z}|{config}"
        if key in self.data:
            self.hits += 1
            return float(self.data[key])
        self.misses += 1
        self.data[key] = float(mbpt2_energy(result).correlation_energy)
        return float(self.data[key])

    def ci(self, z: int, model: tuple[str, ...], breit: bool, qed: bool):
        key = f"ci|{z}|{','.join(model)}|{int(breit)}|{int(qed)}"
        if key in self.data and self.data[key].get("converged") is True:
            self.hits += 1
            return self.data[key]
        self.misses += 1
        atom = Atom(
                    Z=z,
                    config=list(model),
                    corrections=Corrections(breit=breit, qed=qed),
               )
        atom.solve()
        result = atom.result
        mixing = np.asarray(result.mixing_coefficients)
        dominant = []
        for state in range(mixing.shape[1] if mixing.size else 0):
            index = int(np.argmax(np.abs(mixing[:, state])))
            dominant.append(_csf_configuration(result.csfs[index]))
        payload = {
            "energies": [float(e) for e in np.asarray(result.state_energies)],
            "two_j": [int(j) for j in result.state_J2],
            "parity": [int(p) for p in result.state_parity],
            "dominant": dominant,
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "orbital_residual": (float(result.orbital_optimization["orbital_residual"])
                                 if result.orbital_optimization.get("orbital_residual") is not None
                                 else None),
        }
        self.data[key] = payload
        return payload


def _csf_configuration(csf) -> str:
    """l-resolved configuration of a CSF, in the atomx spelling.

    A CSF carries jj subshells ``(n, kappa, q)``; summing the occupations
    over the two j of each ``(n, l)`` recovers the non-relativistic
    configuration, which is the spelling NIST uses, so the label can be
    compared against the model space and the NIST rows.
    """
    occupations: dict[tuple[int, int], float] = collections.defaultdict(float)
    for shell in csf.subshells:
        kappa = int(shell.kappa)
        l = kappa if kappa > 0 else -kappa - 1
        occupations[(int(shell.n), l)] += float(shell.occupation)
    return " ".join(f"{n}{'spdfghik'[l]}{round(q)}"
                    for (n, l), q in sorted(occupations.items()) if q > 1e-6)


# ---------------------------------------------------------------------------
# model-space selection
# ---------------------------------------------------------------------------

def candidate_blocks(config: str) -> collections.Counter:
    """(2J, parity) → number of jj-CSFs, for one configuration."""
    return collections.Counter((csf.total_J2, csf.parity)
                               for csf in enumerate_jj_csfs(config))


def attributable_levels(ion: nist.Ion, candidates) -> list[tuple]:
    """NIST levels usable in a comparison: energy and a single J printed."""
    out = []
    for level in ion.levels:
        if level.energy_cm1 is None or len(level.j_values) != 1:
            continue
        owners = [c for c in candidates
                  if nist.nist_config_matches(level.configuration, c)]
        if len(owners) != 1:
            continue
        out.append((owners[0], (int(round(2 * level.j_values[0])),
                                level.parity), level))
    return out


def choose_model(case: Case, ion: nist.Ion) -> tuple[tuple[str, ...], int, int]:
    """Largest candidate sub-list whose (2J, parity) blocks balance exactly.

    Returns ``(model, n_covered, n_attributable)``.  Falls back to the
    full candidate list when nothing balances (``mode="block-min"``).
    """
    blocks = {c: candidate_blocks(c) for c in case.candidates}
    levels = attributable_levels(ion, case.candidates)
    best = None
    for size in range(1, len(case.candidates) + 1):
        for subset in itertools.combinations(case.candidates, size):
            calc = collections.Counter()
            for config in subset:
                calc.update(blocks[config])
            observed = collections.Counter(
                block for config, block, _ in levels if config in subset)
            if set(calc) != set(observed):
                continue
            if any(calc[b] != observed[b] for b in calc):
                continue
            cover = sum(observed.values())
            if best is None or cover > best[0]:
                best = (cover, subset)
    if best is None:
        return case.candidates, len(levels), len(levels)
    return best[1], best[0], len(levels)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

@dataclass
class BlockResult:
    two_j: int
    parity: int
    n_paired: int
    n_calc: int
    n_nist: int
    excluded: bool
    reason: str = ""
    residuals_cm1: list[float] = field(default_factory=list)


@dataclass
class RungResult:
    system: str
    rung: str
    model: tuple[str, ...]
    n_levels: int
    n_excluded_blocks: int
    n_nist_available: int
    residuals_cm1: list[float]
    blocks: list[BlockResult] = field(default_factory=list)
    centroid_residuals_cm1: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    error: str = ""

    @property
    def stats(self) -> dict:
        if not self.residuals_cm1:
            return {}
        arr = np.asarray(self.residuals_cm1)
        return {
            "rms_cm1": float(np.sqrt(np.mean(arr ** 2))),
            "max_cm1": float(np.max(np.abs(arr))),
            "mean_cm1": float(np.mean(arr)),
            "rms_ev": float(np.sqrt(np.mean(arr ** 2)) / CM1 * EV),
            "max_ev": float(np.max(np.abs(arr)) / CM1 * EV),
        }

    @property
    def centroid_stats(self) -> dict:
        if not self.centroid_residuals_cm1:
            return {}
        arr = np.asarray(self.centroid_residuals_cm1)
        return {
            "rms_cm1": float(np.sqrt(np.mean(arr ** 2))),
            "max_cm1": float(np.max(np.abs(arr))),
            "n": int(arr.size),
        }


def _nist_block(ion: nist.Ion, model, two_j: int, parity: int):
    """NIST levels of the model configurations in one (2J, parity) block.

    Energy-ordered: the NIST level pages are *not* printed in energy
    order (He I lists 1s, 2p, 2s, 3p, 3s, ...), and the computed states
    are, so the block must be sorted before it is paired positionally.
    """
    rows = [lv for lv in ion.levels
            if lv.energy_cm1 is not None and len(lv.j_values) == 1
            and int(round(2 * lv.j_values[0])) == two_j
            and lv.parity == parity
            and any(nist.nist_config_matches(lv.configuration, c)
                    for c in model)]
    return sorted(rows, key=lambda lv: lv.energy_cm1)


def compare_config_rung(case: Case, ion: nist.Ion, rung: str, model,
                        energies: dict[str, float], wall: float) -> RungResult:
    """Level-by-level comparison for a per-configuration rung."""
    ground_e = energies[case.ground]
    residuals: list[float] = []
    n_available = 0
    for level in ion.levels:
        if level.energy_cm1 is None or len(level.j_values) != 1:
            continue
        owners = [c for c in model
                  if nist.nist_config_matches(level.configuration, c)]
        if len(owners) != 1:
            continue
        n_available += 1
        calc = (energies[owners[0]] - ground_e) * CM1
        residuals.append(calc - level.energy_cm1)

    # the same rung against the (2J+1)-weighted NIST configuration
    # centroid: isolates the fine-structure part of the error
    centroid_residuals: list[float] = []
    centroid = {}
    for config in model:
        rows = [lv for lv in ion.levels
                if lv.energy_cm1 is not None and len(lv.j_values) == 1
                and nist.nist_config_matches(lv.configuration, config)]
        if not rows:
            continue
        weights = np.array([2 * lv.j_values[0] + 1 for lv in rows])
        values = np.array([lv.energy_cm1 for lv in rows])
        centroid[config] = float(np.sum(weights * values) / np.sum(weights))
    if case.ground in centroid:
        for config, value in centroid.items():
            centroid_residuals.append(
                (energies[config] - energies[case.ground]) * CM1
                - (value - centroid[case.ground]))

    return RungResult(
        system=case.label, rung=rung, model=tuple(model),
        n_levels=len(residuals), n_excluded_blocks=0,
        n_nist_available=n_available, residuals_cm1=residuals,
        centroid_residuals_cm1=centroid_residuals, wall_seconds=wall)


def compare_ci_rung(case: Case, ion: nist.Ion, rung: str, model, payload,
                    wall: float) -> RungResult:
    """Block-by-block comparison for a CI rung."""
    energies = np.asarray(payload["energies"])
    physical_error = ci_model_physical_error(case, energies)
    if physical_error:
        return RungResult(system=case.label, rung=rung, model=tuple(model),
                          n_levels=0, n_excluded_blocks=1,
                          n_nist_available=0, residuals_cm1=[],
                          wall_seconds=wall, error=physical_error)
    two_j = np.asarray(payload["two_j"])
    parity = np.asarray(payload["parity"])
    if energies.size:
        ground = float(np.min(energies))
    else:
        return RungResult(system=case.label, rung=rung, model=tuple(model),
                          n_levels=0, n_excluded_blocks=0,
                          n_nist_available=0, residuals_cm1=[],
                          wall_seconds=wall, error="no CI states")

    dominant = payload.get("dominant", [])
    calc_blocks: dict[tuple[int, int], list[tuple[float, str]]] = \
        collections.defaultdict(list)
    for index, (e, j, p) in enumerate(zip(energies, two_j, parity, strict=False)):
        label = dominant[index] if index < len(dominant) else ""
        calc_blocks[(int(j), int(p))].append((float(e), _owner(label, model)))
    for key in calc_blocks:
        calc_blocks[key].sort(key=lambda item: item[0])

    residuals: list[float] = []
    blocks: list[BlockResult] = []
    excluded = 0
    n_available = 0
    for key in sorted(set(calc_blocks) | {
            (int(round(2 * lv.j_values[0])), lv.parity)
            for lv in ion.levels
            if lv.energy_cm1 is not None and len(lv.j_values) == 1
            and any(nist.nist_config_matches(lv.configuration, c)
                    for c in model)}):
        observed = _nist_block(ion, model, *key)
        n_available += len(observed)
        calculated = calc_blocks.get(key, [])
        if not observed and not calculated:
            continue
        nist_energies = [lv.energy_cm1 for lv in observed]
        nist_configs = [_owner(lv.configuration, model, nist_style=True)
                        for lv in observed]
        if len(calculated) == len(observed):
            pairs = list(zip(calculated, nist_energies, strict=False))
            reason = ""
        elif case.mode == "block-min" and calculated and observed:
            k = min(len(calculated), len(observed))
            pairs = list(zip(calculated[:k], nist_energies[:k], strict=False))
            # the unpaired computed states must all lie above the highest
            # paired NIST level, otherwise the positional pairing is unsafe
            tail_ok = (len(calculated) <= k
                       or (calculated[k][0] - ground) * CM1
                       > nist_energies[k - 1])
            reason = ("block-min pairing of the lowest "
                      f"{k} of {len(calculated)} computed / "
                      f"{len(observed)} NIST states"
                      + ("" if tail_ok else "; TAIL CHECK FAILED"))
            if not tail_ok:
                excluded += 1
                blocks.append(BlockResult(*key, 0, len(calculated),
                                          len(observed), True, reason))
                continue
        else:
            excluded += 1
            blocks.append(BlockResult(
                *key, 0, len(calculated), len(observed), True,
                f"dimension mismatch: {len(calculated)} computed vs "
                f"{len(observed)} NIST levels with an energy"))
            continue
        # Self-validation of the pairing: the energy-ordered sequence of
        # dominant configurations must reproduce the energy-ordered
        # sequence of NIST configurations.  If it does not, the CI space
        # and the observed level list are not describing the same states
        # in the same order, and the block is dropped rather than
        # silently mis-paired.
        calc_configs = [config for (_, config), _ in pairs]
        expected = nist_configs[:len(pairs)]
        if calc_configs != expected:
            excluded += 1
            blocks.append(BlockResult(
                *key, 0, len(calculated), len(observed), True,
                f"dominant-configuration sequence mismatch: "
                f"{calc_configs} vs {expected}"))
            continue
        for (e_calc, _), e_nist in pairs:
            residuals.append((e_calc - ground) * CM1 - e_nist)
        blocks.append(BlockResult(*key, len(pairs), len(calculated),
                                  len(observed), False, reason))

    return RungResult(system=case.label, rung=rung, model=tuple(model),
                      n_levels=len(residuals), n_excluded_blocks=excluded,
                      n_nist_available=n_available, residuals_cm1=residuals,
                      blocks=blocks, wall_seconds=wall)


def ci_model_physical_error(case: Case, energies) -> str:
    """Reject a He-like bound-state panel with a gross continuum pseudostate.

    Its one-electron ion has an analytic Dirac 1s threshold. A predicted
    1s-nl level well above that threshold cannot be paired with the bound
    NIST 1s-nl series, even when its dominant CSF has the requested name.
    The 0.1 Ha allowance exceeds the small nuclear/recoil threshold shift
    in the Z<=26 He-like panels while catching the observed multi-Ha roots.
    """
    if case.ground != "1s2" or not 2 <= case.z <= 26:
        return ""
    values = np.asarray(energies, dtype=float)
    if not values.size:
        return ""
    threshold = hydrogenic_dirac_energy(1, -1, case.z)
    excess = float(np.max(values) - threshold)
    if excess > 0.1:
        return (f"CI bound-state model contains a continuum pseudostate "
                f"{excess:.3f} Ha above the one-electron threshold")
    return ""


def _owner(configuration: str, model, *, nist_style: bool = False) -> str:
    """Which model configuration a label belongs to ("" if none)."""
    for candidate in model:
        if nist_style:
            if nist.nist_config_matches(configuration, candidate):
                return candidate
        elif (nist.normalise_configuration(configuration)
              == nist.normalise_configuration(candidate)):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_case(case: Case, cache: Cache, verbose: bool = True) -> list[RungResult]:
    """Run every rung of one system; a failing rung is recorded, not raised."""
    ion = nist.load(case.label, with_lines=False)
    model, covered, attributable = choose_model(case, ion)
    if verbose:
        print(f"[{case.label}] model={list(model)} "
              f"({covered}/{attributable} attributable NIST levels, "
              f"mode={case.mode})", flush=True)

    results: list[RungResult] = []
    energies: dict[tuple[bool, bool], dict[str, float]] = {}
    for breit, qed in ((False, False), (True, True)):
        t0 = time.time()
        rung = "DHF+B+QED" if breit else "DHF"
        try:
            table = {config: cache.scf(case.z, config, breit, qed)
                     for config in model}
            energies[(breit, qed)] = table
            outcome = compare_config_rung(case, ion, rung, model, table,
                                          time.time() - t0)
        except Exception as exc:                       # noqa: BLE001
            outcome = RungResult(system=case.label, rung=rung,
                                 model=tuple(model), n_levels=0,
                                 n_excluded_blocks=0, n_nist_available=0,
                                 residuals_cm1=[],
                                 wall_seconds=time.time() - t0,
                                 error=f"{type(exc).__name__}: {exc}")
        results.append(outcome)
        if verbose:
            stats = outcome.stats
            print(f"    {rung:12s} n={outcome.n_levels:3d} "
                  f"rms={stats.get('rms_cm1', float('nan')):12.3f} cm-1"
                  + (f"  ERROR {outcome.error}" if outcome.error else ""),
                  flush=True)

    if case.mbpt2 and (False, False) in energies:
        t0 = time.time()
        try:
            scf = cache.scf_result(case.z, case.ground)
            e2 = cache.mbpt2(case.z, case.ground, scf)
            table = dict(energies[(False, False)])
            table[case.ground] = table[case.ground] + e2
            outcome = compare_config_rung(case, ion, "DHF+MBPT2", model,
                                          table, time.time() - t0)
        except Exception as exc:                       # noqa: BLE001
            e2 = float("nan")
            outcome = RungResult(system=case.label, rung="DHF+MBPT2",
                                 model=tuple(model), n_levels=0,
                                 n_excluded_blocks=0, n_nist_available=0,
                                 residuals_cm1=[],
                                 wall_seconds=time.time() - t0,
                                 error=f"{type(exc).__name__}: {exc}")
        results.append(outcome)
        if verbose:
            stats = outcome.stats
            print(f"    {'DHF+MBPT2':12s} n={outcome.n_levels:3d} "
                  f"rms={stats.get('rms_cm1', float('nan')):12.3f} cm-1  "
                  f"(E2={e2:+.6f} Ha)"
                  + (f"  ERROR {outcome.error}" if outcome.error else ""),
                  flush=True)

    for breit, qed in ((False, False), (True, True)):
        t0 = time.time()
        rung = "MCDHF+B+QED" if breit else "MCDHF"
        try:
            payload = cache.ci(case.z, model, breit, qed)
            outcome = compare_ci_rung(case, ion, rung, model, payload,
                                      time.time() - t0)
        except Exception as exc:                       # noqa: BLE001
            outcome = RungResult(system=case.label, rung=rung,
                                 model=tuple(model), n_levels=0,
                                 n_excluded_blocks=0, n_nist_available=0,
                                 residuals_cm1=[],
                                 wall_seconds=time.time() - t0,
                                 error=f"{type(exc).__name__}: {exc}")
        results.append(outcome)
        if verbose:
            stats = outcome.stats
            print(f"    {rung:12s} n={outcome.n_levels:3d} "
                  f"excl={outcome.n_excluded_blocks} "
                  f"rms={stats.get('rms_cm1', float('nan')):12.3f} cm-1"
                  + (f"  ERROR {outcome.error}" if outcome.error else ""),
                  flush=True)
    return results


def write_csv(all_results: list[RungResult]) -> None:
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "system", "rung", "n_levels_compared", "n_blocks_excluded",
            "n_nist_available", "rms_cm1", "max_cm1", "mean_cm1",
            "rms_eV", "max_eV", "model", "wall_s"])
        for res in all_results:
            stats = res.stats
            writer.writerow([
                res.system, res.rung, res.n_levels,
                res.n_excluded_blocks, res.n_nist_available,
                f"{stats.get('rms_cm1', float('nan')):.4f}",
                f"{stats.get('max_cm1', float('nan')):.4f}",
                f"{stats.get('mean_cm1', float('nan')):.4f}",
                f"{stats.get('rms_ev', float('nan')):.6f}",
                f"{stats.get('max_ev', float('nan')):.6f}",
                " ".join(res.model), f"{res.wall_seconds:.2f}"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", action="append", default=None,
                        help="restrict to a NIST label (repeatable)")
    parser.add_argument("--quick", action="store_true",
                        help="skip cases flagged slow")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore and overwrite the calculation cache")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    cases = [c for c in CASES
             if (args.system is None or c.label in args.system)
             and not (args.quick and c.slow)]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    cache = Cache(CACHE_PATH, refresh=args.refresh)
    all_results: list[RungResult] = []
    for case in cases:
        all_results.extend(run_case(case, cache))
    cache.save()

    write_csv(all_results)
    print(f"wrote {CSV_PATH.relative_to(REPO_ROOT)} "
          f"({len(all_results)} rows; cache hits={cache.hits} "
          f"misses={cache.misses})")

    if not args.no_report and len(cases) == len(CASES):
        import benchmarks.nist_report as report
        report.write_report(all_results, CASES, REPORT_PATH)
        print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
