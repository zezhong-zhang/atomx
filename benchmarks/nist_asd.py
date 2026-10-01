#!/usr/bin/env python
"""Local, dependency-free reader for the committed NIST ASD v5.12 panel.

The data are the plain-text (XHTML-in-``<pre>``) NIST ASD v5.12 pages
fetched by :mod:`benchmarks.fetch_nist_asd` into
``benchmarks/nist_asd/{levels,lines}/asd.ZZZSSS``.  Nothing here touches
the network: the tests run offline against the committed bytes.

What the parser handles
-----------------------
* **Header-driven column layout.**  NIST emits between 6 and 8 level
  columns (``Leading percentages`` and ``Landé`` come and go) and three
  different line-column layouts; the field indices are read from the
  file's own header, never hard-coded.
* **Continuation rows.**  A blank ``Configuration`` and/or ``Term`` means
  "same as the row above" — the common NIST style for fine-structure
  multiplets.  All-blank separator rows are dropped.
* **Level flags.**  ``a`` (adjusted), ``?`` (uncertain), ``+x``/``+y``/
  ``+z`` (value is a lower bound), ``*`` (odd parity, carried by the
  term), square brackets, and combinations (``?a``, ``[131000.00]+x``).
* **Rows with no numeric level** (42 of them in the panel) are retained
  with ``energy_cm1 is None`` so callers can count them rather than
  silently lose them.
* **Ambiguous J.**  NIST writes ``0,1,2`` or ``1/2,3/2`` for an
  unresolved group and ``-----`` for "unknown"; ``Level.j`` is then
  ``None`` while ``Level.j_values`` keeps what was printed.
* **Line medium.**  Each lines page states whether the wavelengths are
  in vacuum or in air (``λ in: vacuum``); the value is exposed as
  ``SpectralLine.medium`` instead of being assumed.

Reference columns contain HTML anchors and are stripped; the NIST
bibliographic code (``L9172``, ``L3620c107``) is kept.

Units
-----
``HA_CM1 = 219474.6313632`` cm⁻¹ per hartree, ``HA_EV`` from
:mod:`atomx.constants`.  **NIST level energies are relative to the
ground state of the ion** (the lowest level of the file is 0), so only
*excitation* energies may be compared with a calculation; a computed
total energy is never comparable to a NIST level value.

CLI::

    python benchmarks/nist_asd.py --coverage   # parser coverage per file
    python benchmarks/nist_asd.py --ion "Fe XVII"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "benchmarks" / "nist_asd"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

sys.path.insert(0, str(REPO_ROOT / "src"))


# Compatibility exports; the installable package owns record parsing.
from atomx.reference_data.nist import (
    Level,
    SpectralLine,
    nist_config_matches,
    parse_levels_text,
    parse_lines_text,
)

# ---------------------------------------------------------------------------
# ion container + lookups
# ---------------------------------------------------------------------------

@dataclass
class Ion:
    """Everything the panel holds for one spectroscopic ion."""

    label: str
    z: int
    charge: int
    levels: list[Level] = field(default_factory=list)
    lines: list[SpectralLine] = field(default_factory=list)
    level_problems: list[str] = field(default_factory=list)
    line_problems: list[str] = field(default_factory=list)

    @property
    def n_electrons(self) -> int:
        return self.z - self.charge

    @property
    def ground(self) -> Level | None:
        """Lowest level with a numeric energy (NIST anchors it at 0)."""
        with_energy = [lv for lv in self.levels if lv.energy_cm1 is not None]
        return min(with_energy, key=lambda lv: lv.energy_cm1) \
            if with_energy else None

    # -- lookups ---------------------------------------------------------
    def by_configuration(self, configuration: str) -> list[Level]:
        """Levels of a (full) configuration, energy-ordered.

        ``configuration`` is given in the atomx spelling, e.g.
        ``"1s2 2s2 2p6 3s"``; NIST labels that elide the frozen core
        (``"2p6.3s"``) still match — see :func:`nist_config_matches`.
        """
        hits = [lv for lv in self.levels
                if nist_config_matches(lv.configuration, configuration)]
        return sorted(hits, key=lambda lv: (lv.energy_cm1 is None,
                                            lv.energy_cm1 or 0.0))

    def by_term(self, term: str) -> list[Level]:
        """Levels whose term label matches (whitespace-insensitive)."""
        key = term.replace(" ", "")
        hits = [lv for lv in self.levels if lv.term.replace(" ", "") == key]
        return sorted(hits, key=lambda lv: (lv.energy_cm1 is None,
                                            lv.energy_cm1 or 0.0))

    def find(self, *, configuration: str | None = None, term: str | None = None,
             j: float | None = None, odd_parity: bool | None = None,
             max_energy_cm1: float | None = None,
             require_energy: bool = True) -> list[Level]:
        """Levels matching every supplied selector, energy-ordered.

        Configuration matching is on the *full* subshell map, so
        ``find(configuration="1s2 2s2")`` will not match ``1s2 2s2 2p``.
        """
        wanted = configuration if configuration else None
        hits = []
        for level in self.levels:
            if require_energy and level.energy_cm1 is None:
                continue
            if wanted is not None and not nist_config_matches(
                    level.configuration, wanted):
                continue
            if term is not None and level.term.replace(
                    " ", "") != term.replace(" ", ""):
                continue
            if j is not None:
                values = level.j_values
                if len(values) != 1 or abs(values[0] - j) > 1e-9:
                    continue
            if odd_parity is not None and level.odd_parity != odd_parity:
                continue
            if (max_energy_cm1 is not None
                    and (level.energy_cm1 or 0.0) > max_energy_cm1):
                continue
            hits.append(level)
        return sorted(hits, key=lambda lv: lv.energy_cm1 or 0.0)

    def block(self, two_j: int, parity: int) -> list[Level]:
        """Energy-ordered levels with well-defined J in the (2J, parity) block."""
        hits = [lv for lv in self.levels
                if lv.energy_cm1 is not None
                and lv.two_j == two_j
                and lv.parity == parity
                and len(lv.j_values) == 1]
        return sorted(hits, key=lambda lv: lv.energy_cm1)

    def excitation_cm1(self, level: Level) -> float:
        ground = self.ground
        if ground is None or level.energy_cm1 is None:
            raise ValueError(f"{self.label}: no ground level / no energy")
        return level.energy_cm1 - ground.energy_cm1


@lru_cache(maxsize=None)
def load(label: str, *, with_lines: bool = True) -> Ion:
    """Load one panel ion from the committed data (cached)."""
    manifest = _manifest()
    record = next((r for r in manifest["files"]
                   if r["label"].replace(" ", "").upper()
                   == label.replace(" ", "").upper() and r["kind"] == "levels"),
                  None)
    if record is None:
        raise KeyError(f"{label!r} is not in the committed NIST panel; "
                       f"see benchmarks/nist_asd/manifest.json")
    ion = Ion(label=record["label"], z=record["Z"], charge=record["charge"])

    levels_path = DATA_ROOT / "levels" / f"asd.{record['Z']:03d}" \
                                         f"{record['charge']:03d}"
    if levels_path.exists():
        text = levels_path.read_text(errors="replace")
        ion.levels, ion.level_problems = parse_levels_text(
            text, ion=record["label"], z=record["Z"],
            charge=record["charge"], source=levels_path.name)
    if with_lines:
        lines_path = DATA_ROOT / "lines" / f"asd.{record['Z']:03d}" \
                                          f"{record['charge']:03d}"
        if lines_path.exists():
            text = lines_path.read_text(errors="replace")
            ion.lines, ion.line_problems = parse_lines_text(
                text, ion=record["label"], z=record["Z"],
                charge=record["charge"], source=lines_path.name)
    return ion


@lru_cache(maxsize=1)
def _manifest() -> dict:
    import json
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} is missing — run "
            f"`python benchmarks/fetch_nist_asd.py` to fetch the NIST ASD "
            f"panel (network required once; the files are then committed)")
    return json.loads(MANIFEST_PATH.read_text())


def panel_labels() -> tuple[str, ...]:
    """Spectroscopic labels of every ion in the committed panel."""
    return tuple(r["label"] for r in _manifest()["files"] if r["kind"] == "levels")


def levels_path(label: str) -> Path:
    record = next(r for r in _manifest()["files"]
                  if r["label"] == label and r["kind"] == "levels")
    return DATA_ROOT / record["path"]


def lines_path(label: str) -> Path:
    record = next(r for r in _manifest()["files"]
                  if r["label"] == label and r["kind"] == "lines")
    return DATA_ROOT / record["path"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _coverage() -> int:
    total_levels = total_lines = 0
    bad_levels = bad_lines = 0
    print(f"{'ion':>10s} {'Z':>3s} {'q':>3s} {'N':>3s} "
          f"{'levels':>7s} {'noE':>4s} {'badJ':>5s} {'lines':>7s} "
          f"{'noAki':>6s} {'bad':>4s}")
    for label in panel_labels():
        ion = load(label)
        no_energy = sum(1 for lv in ion.levels if lv.energy_cm1 is None)
        bad_j = sum(1 for lv in ion.levels
                    if lv.energy_cm1 is not None and lv.two_j is None)
        no_aki = sum(1 for ln in ion.lines if ln.aki is None)
        bad = len(ion.level_problems) + len(ion.line_problems)
        print(f"{label:>10s} {ion.z:3d} {ion.charge:3d} {ion.n_electrons:3d} "
              f"{len(ion.levels):7d} {no_energy:4d} {bad_j:5d} "
              f"{len(ion.lines):7d} {no_aki:6d} {bad:4d}")
        total_levels += len(ion.levels)
        total_lines += len(ion.lines)
        bad_levels += len(ion.level_problems)
        bad_lines += len(ion.line_problems)
    print(f"\ntotal: {total_levels} levels, {total_lines} lines; "
          f"unparsed rows: {bad_levels} level, {bad_lines} line")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coverage", action="store_true",
                        help="parse every committed file and print coverage")
    parser.add_argument("--ion", help="dump one ion's level table")
    args = parser.parse_args(argv)
    if args.ion:
        ion = load(args.ion)
        print(f"{ion.label}: Z={ion.z} charge={ion.charge} "
              f"N={ion.n_electrons}, {len(ion.levels)} levels, "
              f"{len(ion.lines)} lines")
        for level in ion.levels:
            print("  ", level)
        return 0
    return _coverage()


if __name__ == "__main__":
    raise SystemExit(main())
