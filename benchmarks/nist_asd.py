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
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "benchmarks" / "nist_asd"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from atomx.constants import HARTREE_EV  # noqa: E402  (needs the sys.path above)

#: CODATA hartree → cm⁻¹ (the value quoted in the task brief; identical to
#: :data:`atomx.constants.HARTREE_CM1` to all printed digits).
HA_CM1 = 219474.6313632

_TAG_RE = re.compile(r"<[^>]*>")
_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&eacute;": "é", "&times;": "x", "&#8722;": "-",
}
#: ``<sup></sup>Fe  XVII &nbsp;`` on a levels page.
_TITLE_LEVELS_RE = re.compile(
    r"<sup>\s*</sup>\s*(?P<label>[A-Za-z]{1,2}\s+[IVX]+)\s*&nbsp;")
#: ``<u>Fe XVII</u>: 67 Levels of Data Found`` on a lines page.
_TITLE_LINES_RE = re.compile(
    r"<u>(?P<label>[A-Za-z]{1,2}\s+[IVX]+)</u>\s*:")
_SEQ_RE = re.compile(
    r"<i>Z</i>\s*=\s*(?P<z>\d+),\s*(?P<seq>[A-Za-z]{1,2})\s+"
    r"isoelectronic sequence")
_MEDIUM_RE = re.compile(r"<i>&lambda;</i>\s*in</u>:\s*"
                        r"<span[^>]*>(?P<medium>\w+)</span>")
#: Tokens of a NIST configuration string: ``2s2``, ``3d``, ``5g``, ``10p``.
_SUBSHELL_RE = re.compile(r"(?P<n>\d{1,2})(?P<l>[spdfghik])(?P<occ>\d*)")
_L_VALUE = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _clean(cell: str) -> str:
    """Strip HTML markup and entities from one table cell."""
    text = _TAG_RE.sub("", cell)
    for entity, replacement in _ENTITIES.items():
        text = text.replace(entity, replacement)
    return text.strip()


def _to_float(text: str) -> float | None:
    match = _NUMBER_RE.fullmatch(text.strip())
    return float(match.group()) if match else None


def _parse_j(text: str) -> tuple[float, ...]:
    """``"3/2"`` → ``(1.5,)``; ``"0,1,2"`` → ``(0.0, 1.0, 2.0)``."""
    values: list[float] = []
    for chunk in text.replace(" ", "").split(","):
        if not chunk or set(chunk) <= {"-"}:
            continue
        if "/" in chunk:
            num, _, den = chunk.partition("/")
            num = num.strip("[]()*?")
            try:
                values.append(float(num) / float(den))
            except ValueError:
                continue
        else:
            value = _to_float(chunk.strip("[]()*?"))
            if value is not None:
                values.append(value)
    return tuple(values)


def normalise_configuration(text: str) -> str:
    """NIST configuration → canonical atomx spelling.

    ``'2s2.2p5.(2P*<3/2>).3s'`` → ``'2s2 2p5 3s'``;
    ``'1s.(2S).2s.2p2.(4P)'`` → ``'1s 2s 2p2'``.  Parenthesised
    intermediate-coupling parentage, ``?`` marks and stray markup are
    dropped; the subshells are sorted by ``(n, l)``.  Both the NIST
    dotted spelling and the atomx space-separated spelling are accepted,
    so a NIST configuration can be compared with an atomx one directly.
    """
    text = _clean(text)
    while "(" in text and ")" in text:          # drop (...(2P*<3/2>)...) groups
        start = text.index("(")
        end = text.index(")", start)
        text = text[:start] + text[end + 1:]
    subshells: list[tuple[int, int, int]] = []
    for token in re.split(r"[.\s]+", text):
        match = _SUBSHELL_RE.fullmatch(token.strip().replace("*", "")
                                       .replace("?", ""))
        if match is None:
            continue
        occupation = int(match.group("occ")) if match.group("occ") else 1
        subshells.append((int(match.group("n")),
                          _L_VALUE[match.group("l")], occupation))
    subshells.sort()
    return " ".join(f"{n}{'spdfghik'[l]}{occ}" for n, l, occ in subshells)


def configuration_map(text: str) -> dict[tuple[int, int], int]:
    """``'2s2.2p5.3s'`` → ``{(2, 0): 2, (2, 1): 5, (3, 0): 1}``."""
    out: dict[tuple[int, int], int] = {}
    for token in normalise_configuration(text).split():
        match = _SUBSHELL_RE.fullmatch(token)
        if match is not None:
            out[(int(match.group("n")), _L_VALUE[match.group("l")])] = int(
                match.group("occ"))
    return out


def _filled(n: int, l: int) -> int:
    return 2 * (2 * l + 1)


def nist_config_matches(nist_config: str, model_config: str) -> bool:
    """Does a NIST configuration label denote ``model_config``?

    The NIST pages do not print a frozen closed core consistently: the
    Be-like and Li-like files write ``1s2.2s2``, while the Na-like,
    Ne-like, C-like and Fe-like files write ``2p6.3s``, ``2s2.2p6``,
    ``2s2.2p2``, ``3d6.4s2`` — the 1s (or 1s2s2p) core is simply left
    out.  A match therefore requires

    * every subshell NIST prints to be present in the model with the
      same occupation, and
    * every model subshell NIST does *not* print to be completely
      filled (i.e. a core the page elided).

    An *open* subshell missing from the NIST label is not a match.
    """
    model = configuration_map(model_config)
    nist = configuration_map(nist_config)
    if any(key not in model or model[key] != occ for key, occ in nist.items()):
        return False
    return all(key in nist or occ == _filled(*key)
               for key, occ in model.items())


# ---------------------------------------------------------------------------
# unit helpers
# ---------------------------------------------------------------------------

def cm1_to_hartree(value_cm1: float) -> float:
    return value_cm1 / HA_CM1


def hartree_to_cm1(value_ha: float) -> float:
    return value_ha * HA_CM1


def cm1_to_ev(value_cm1: float) -> float:
    return value_cm1 / HA_CM1 * HARTREE_EV


def ev_to_cm1(value_ev: float) -> float:
    return value_ev / HARTREE_EV * HA_CM1


def hartree_to_ev(value_ha: float) -> float:
    return value_ha * HARTREE_EV


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    """One row of a NIST ASD levels table."""

    ion: str
    z: int
    charge: int
    index: int
    row: int
    configuration: str            # normalised atomx spelling
    configuration_nist: str       # exactly as printed by NIST
    term: str
    j_text: str
    energy_cm1: float | None
    uncertainty_cm1: float | None
    reference: str
    flags: tuple[str, ...]
    bracketed: bool

    @property
    def n_electrons(self) -> int:
        return self.z - self.charge

    @property
    def j_values(self) -> tuple[float, ...]:
        return _parse_j(self.j_text)

    @property
    def j(self) -> float | None:
        """The J quantum number, or ``None`` if NIST prints a group/``---``."""
        values = self.j_values
        return values[0] if len(values) == 1 else None

    @property
    def two_j(self) -> int | None:
        j = self.j
        return None if j is None else int(round(2 * j))

    @property
    def odd_parity(self) -> bool:
        """NIST marks odd parity with ``*`` on the term."""
        return "*" in self.term

    @property
    def parity(self) -> int:
        return -1 if self.odd_parity else 1

    @property
    def energy_hartree(self) -> float | None:
        return None if self.energy_cm1 is None else cm1_to_hartree(
            self.energy_cm1)

    @property
    def in_leading_percentages(self) -> bool:
        return "leading" in self.flags

    def __str__(self) -> str:
        energy = ("        " if self.energy_cm1 is None
                  else f"{self.energy_cm1:9.3f}")
        return (f"{self.ion:>10s} {self.configuration:<28s} "
                f"{self.term:<10s} J={self.j_text or '?':<8s} {energy} cm-1")


@dataclass(frozen=True)
class SpectralLine:
    """One row of a NIST ASD lines table."""

    ion: str
    z: int
    charge: int
    index: int
    row: int
    medium: str                        # "vacuum" | "air" | "" (unknown)
    observed_nm: float | None
    observed_unc_nm: float | None
    ritz_nm: float | None
    ritz_unc_nm: float | None
    aki: float | None                  # s^-1
    acc: str
    ei_cm1: float | None
    ek_cm1: float | None
    lower_configuration: str
    lower_term: str
    lower_j_text: str
    upper_configuration: str
    upper_term: str
    upper_j_text: str
    transition_type: str
    tp_reference: str
    line_reference: str

    @property
    def n_electrons(self) -> int:
        return self.z - self.charge

    @property
    def wavenumber_cm1(self) -> float | None:
        """Observed wavenumber, from ``Ek - Ei`` when NIST prints both."""
        if self.ei_cm1 is None or self.ek_cm1 is None:
            return None
        return self.ek_cm1 - self.ei_cm1


# ---------------------------------------------------------------------------
# table readers
# ---------------------------------------------------------------------------

def _read_table(text: str, header_predicate, *, max_header_rows: int = 3):
    """Yield ``(header_cells, data_cells, line_number)`` for one ``<pre>`` table.

    ``header_predicate`` selects the first header line (the one that names
    the columns).  The *last* header line of the block is the one whose
    field count matches the data rows — that is the row the field indices
    are taken from.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if header_predicate(line)),
                 None)
    if start is None:
        raise ValueError("no NIST table header found")
    header_rows: list[list[str]] = []
    cursor = start
    while (cursor < len(lines) and "|" in lines[cursor]
           and not set(lines[cursor].strip()) <= set("-| ")
           and len(header_rows) < max_header_rows):
        header_rows.append(lines[cursor].split("|"))
        cursor += 1
    ncols = len(header_rows[-1])
    for number in range(cursor, len(lines)):
        line = lines[number]
        if "</pre>" in line:
            break
        if "|" not in line:
            continue
        cells = line.split("|")
        if len(cells) != ncols:
            # NIST never emits this; a mismatch would silently shift every
            # column, so surface it instead.
            yield None, cells, number + 1
            continue
        if not any(cell.strip() for cell in cells):
            continue
        # NIST also rules off groups of levels with dash-only rows; those
        # carry no data either.
        if all(set(_clean(cell)) <= set("- ") for cell in cells):
            continue
        yield header_rows, cells, number + 1


def _reference_code(cell: str) -> str:
    """``"...>L9172</a>"`` → ``"L9172"`` (joined if several)."""
    text = _clean(cell)
    return ",".join(text.split())


def parse_levels_text(text: str, *, ion: str, z: int, charge: int,
                      source: str = "<text>") -> tuple[list[Level], list[str]]:
    """Parse a NIST ASD levels page into :class:`Level` records.

    Returns ``(levels, problems)``; ``problems`` lists rows that could not
    be interpreted (they are *not* silently dropped).
    """
    levels: list[Level] = []
    problems: list[str] = []
    last_config = ""
    last_term = ""
    for header_rows, cells, row in _read_table(
            text, lambda line: ("Configuration" in line
                                and "Level (cm-1)" in line)):
        header = header_rows[0] if header_rows else None
        if header is None:
            problems.append(f"{source}:{row}: {len(cells)} fields, expected "
                            f"the header's count")
            continue
        names = [_clean(cell) for cell in header]
        try:
            i_conf = names.index("Configuration")
            i_term = names.index("Term")
            i_j = names.index("J")
            i_level = next(i for i, n in enumerate(names)
                           if n.startswith("Level"))
        except (ValueError, StopIteration):
            problems.append(f"{source}:{row}: unrecognised header {names}")
            continue
        i_unc = next((i for i, n in enumerate(names)
                      if n.startswith("Uncertainty")), None)

        config_raw = _clean(cells[i_conf])
        term_raw = _clean(cells[i_term])
        if config_raw:
            last_config = config_raw
        if term_raw:
            last_term = term_raw
        config = config_raw or last_config
        term = term_raw or last_term

        level_raw = _clean(cells[i_level])
        flags: list[str] = []
        bracketed = level_raw.startswith("[") and "]" in level_raw
        core = level_raw
        if bracketed:
            core = core[core.index("[") + 1:core.rindex("]")]
        core = core.strip().strip("()")
        energy: float | None = None
        match = re.match(r"^(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<tail>.*)$",
                         core)
        if match is not None:
            energy = float(match.group("value"))
            tail = match.group("tail").strip()
            if tail:
                flags.append(tail)
        elif core:
            problems.append(f"{source}:{row}: unreadable level {level_raw!r}")
        elif not (config_raw or term_raw or _clean(cells[i_j])):
            continue

        uncertainty = None
        if i_unc is not None:
            uncertainty = _to_float(_clean(cells[i_unc]))

        if bracketed:
            flags.append("bracketed")
        levels.append(Level(
            ion=ion, z=z, charge=charge, index=len(levels), row=row,
            configuration=normalise_configuration(config),
            configuration_nist=config,
            term=term,
            j_text=_clean(cells[i_j]),
            energy_cm1=energy,
            uncertainty_cm1=uncertainty,
            reference=_reference_code(cells[names.index("Reference")])
            if "Reference" in names else "",
            flags=tuple(flags),
            bracketed=bracketed,
        ))
    return levels, problems


def _flat_header(header: list[str]) -> list[str]:
    """Semantic names of the *data* columns to the left of the level block.

    ``['Observed', 'Unc.', 'Ritz', 'Unc.', 'Aki', 'Acc.', 'Ei ...']`` →
    ``['observed', 'observed_unc', 'ritz', 'ritz_unc', 'aki', 'acc']``.
    The ``Unc.`` cell names the uncertainty of the column before it.
    """
    flat: list[str] = []
    for cell in header:
        name = _clean(cell)
        if name == "Unc.":
            if flat:
                flat[-1] = flat[-1] + "_unc"
            continue
        if name.startswith("Ei"):
            break
        flat.append({"Observed": "observed", "Ritz": "ritz",
                     "Aki": "aki", "Acc.": "acc"}.get(name, name.lower()))
    return flat


def _cell_float(cells: list[str], index: int | None) -> float | None:
    return None if index is None else _to_float(_clean(cells[index]))


def _split_energy_pair(text: str) -> tuple[float | None, float | None]:
    """``'0    -  9878000'`` → ``(0.0, 9878000.0)``."""
    values = [float(v) for v in _NUMBER_RE.findall(text)]
    if len(values) >= 2:
        return values[0], values[1]
    if len(values) == 1:
        return values[0], None
    return None, None


def parse_lines_text(text: str, *, ion: str, z: int, charge: int,
                     source: str = "<text>") -> tuple[list[SpectralLine],
                                                      list[str]]:
    """Parse a NIST ASD lines page into :class:`SpectralLine` records."""
    medium_match = _MEDIUM_RE.search(text)
    medium = medium_match.group("medium").lower() if medium_match else ""
    rows: list[SpectralLine] = []
    problems: list[str] = []
    for header_rows, cells, row in _read_table(
            text, lambda line: ("Observed" in line and "Aki" in line)):
        if not header_rows or not any('Conf.' in _clean(c)
                                      for c in header_rows[-1]):
            problems.append(f"{source}:{row}: no level-subcolumn "
                            f"header")
            continue
        header = header_rows[-1]
        names = [_clean(cell) for cell in header]
        try:
            i_lower = names.index("Conf.")
        except ValueError:
            problems.append(f"{source}:{row}: unrecognised header {names}")
            continue
        if names[i_lower:i_lower + 6] != ["Conf.", "Term", "J",
                                          "Conf.", "Term", "J"]:
            problems.append(
                f"{source}:{row}: unexpected level columns "
                f"{names[i_lower:i_lower + 6]}")
            continue
        # The number of *data* columns preceding the level block varies
        # between ions (N I has no observed-wavelength uncertainty and no
        # Ritz uncertainty, Li I has no observed uncertainty, He I has
        # both).  The first header row names them, and the columns to the
        # left of "Ei Ek" map one-to-one onto data fields; everything from
        # "Ei Ek" rightwards is expanded by the nested level table.
        flat = _flat_header(header_rows[0])
        if not any(_clean(cell).startswith("Ei") for cell in header_rows[0]):
            if not problems or "no 'Ei Ek'" not in problems[-1]:
                problems.append(f"{source}:{row}: no 'Ei Ek' column in "
                                f"{[_clean(c) for c in header_rows[0]]}")
            continue
        # ``flat`` names data fields 0..len(flat)-1 one-to-one; the next
        # data field is the combined "Ei  Ek" cell.
        i_eiek = len(flat)
        i_obs = flat.index("observed") if "observed" in flat else None
        i_obs_unc = (flat.index("observed_unc")
                     if "observed_unc" in flat else None)
        i_ritz = flat.index("ritz") if "ritz" in flat else None
        i_ritz_unc = (flat.index("ritz_unc") if "ritz_unc" in flat else None)
        i_aki = flat.index("aki") if "aki" in flat else None
        i_acc = flat.index("acc") if "acc" in flat else None

        ei, ek = _split_energy_pair(_clean(cells[i_eiek]))
        rows.append(SpectralLine(
            ion=ion, z=z, charge=charge, index=len(rows), row=row,
            medium=medium,
            observed_nm=_cell_float(cells, i_obs),
            observed_unc_nm=_cell_float(cells, i_obs_unc),
            ritz_nm=_cell_float(cells, i_ritz),
            ritz_unc_nm=_cell_float(cells, i_ritz_unc),
            aki=_cell_float(cells, i_aki),
            acc=_clean(cells[i_acc]) if i_acc is not None else "",
            ei_cm1=ei, ek_cm1=ek,
            lower_configuration=normalise_configuration(
                _clean(cells[i_lower])),
            lower_term=_clean(cells[i_lower + 1]),
            lower_j_text=_clean(cells[i_lower + 2]),
            upper_configuration=normalise_configuration(
                _clean(cells[i_lower + 3])),
            upper_term=_clean(cells[i_lower + 4]),
            upper_j_text=_clean(cells[i_lower + 5]),
            transition_type=_clean(cells[i_lower + 6]),
            tp_reference=_clean(cells[i_lower + 7]),
            line_reference=_clean(cells[i_lower + 8]),
        ))
    return rows, problems


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
