"""Constants hygiene (code review 2026-10-05, finding P4).

Unit-conversion factors must come from :mod:`atomx.constants` — the
single source of truth — not from literals re-typed at module edges,
where precision silently diverges.  The review found ``219474.63`` (a
*truncated* ``HARTREE_CM1``) next to ``219474.6313632`` (a CODATA-2018
copy) and ``hc`` in eV·nm at four precisions across four modules.

Two gates, in the style of ``test_review_m9_m13_gates.py`` (M12):

* **Source scan** — tokenize every ``src/atomx/**/*.py`` and flag any
  *numeric literal* within 1e-4 relative of a canonical conversion
  factor.  Tokenizing (not regex) means comments and docstrings —
  derivation notes, doctest output, FAC-provenance quotes such as
  ``numerics/shooting/constants.py``'s ``HREV=27.21138602D0`` — are
  never flagged; only executable literals are.  ``constants.py`` itself
  is the canonical home and is exempt.  The banned values are *derived
  from* the canonical module, so the gate cannot drift from the truth
  it enforces; a genuine non-constant collision (e.g. an input photon
  energy that happens to land in the window) is whitelisted explicitly
  below, with its reason.
* **Identity** — the module edges that previously carried literals now
  hold exactly the canonical value, so substituting a *different*
  number (not just a banned-precision copy) also fails.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

from atomx.constants import (
    BOLTZMANN_EV,
    HARTREE_CM1,
    HARTREE_EV,
    HC_EV_NM,
    RYDBERG_EV,
)

SRC = Path(__file__).resolve().parent.parent / "src" / "atomx"

#: Canonical conversion factors a literal copy of which is banned outside
#: ``constants.py`` (name, value).  Derived from the module itself.
_BANNED = {
    "HARTREE_EV": HARTREE_EV,
    "RYDBERG_EV": RYDBERG_EV,
    "HARTREE_CM1": HARTREE_CM1,
    "eV -> cm^-1 (HARTREE_CM1 / HARTREE_EV)": HARTREE_CM1 / HARTREE_EV,
    "HC_EV_NM": HC_EV_NM,
    "eV -> K (1 / BOLTZMANN_EV)": 1.0 / BOLTZMANN_EV,
    "hc in eV*angstrom (10 * HC_EV_NM)": 10.0 * HC_EV_NM,
}

#: Match window.  1e-4 relative catches every precision class the review
#: found (27.2114 … 27.211386245988, 219474.6 … 219474.6313632,
#: 1239.84 … 1239.8419843320026) while staying far from generic inputs
#: (a 13.6 eV photon energy is 4e-4 off RYDBERG_EV and passes).
_REL_TOL = 1e-4

#: (path relative to src/atomx, line) pairs whose literal is a reviewed
#: genuine non-constant use — empty today; add only with a comment
#: justifying why the value is not the conversion factor it resembles.
_WHITELIST: set[tuple[str, int]] = set()


def _numeric_literals(path: Path):
    """(lineno, source, value) of every NUMBER token in ``path``."""
    tokens = tokenize.generate_tokens(
        io.StringIO(path.read_text(encoding="utf-8")).readline)
    for tok in tokens:
        if tok.type != tokenize.NUMBER:
            continue
        try:
            value = float(tok.string)
        except ValueError:
            continue  # 0x/0o ints, complex "1j", ...
        yield tok.start[0], tok.string, value


def test_no_conversion_factor_literals_outside_constants():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel == Path("constants.py"):
            continue  # the canonical home
        for lineno, source, value in _numeric_literals(path):
            magnitude = abs(value)
            if magnitude == 0.0:
                continue
            for name, target in _BANNED.items():
                if abs(magnitude - target) / target < _REL_TOL:
                    site = (str(rel), lineno)
                    if site in _WHITELIST:
                        continue
                    offenders.append(
                        f"{rel}:{lineno}: {source} ~ {name} "
                        f"— use atomx.constants.{name.split(' ')[0]}")
    assert not offenders, (
        "conversion-factor literals outside atomx.constants (P4):\n"
        + "\n".join(offenders))


def test_review_literals_are_detectably_banned():
    """The gate has teeth: every literal the review caught lands inside
    the ban window, so re-introducing one fails this suite."""
    review_literals = {
        "HARTREE_EV": [27.211386245988, 27.2114],
        "RYDBERG_EV": [13.6057],
        "HARTREE_CM1": [219474.63, 219474.6313632, 219474.631],
        "eV -> cm^-1 (HARTREE_CM1 / HARTREE_EV)": [8065.54, 8065.544],
        "eV -> K (1 / BOLTZMANN_EV)": [11604.5, 11604.5250061657],
        "HC_EV_NM": [1239.84, 1239.841984, 1239.8419843320026],
        "hc in eV*angstrom (10 * HC_EV_NM)": [12398.4, 12398.42],
    }
    for name, literals in review_literals.items():
        for lit in literals:
            assert abs(abs(lit) - _BANNED[name]) / _BANNED[name] < _REL_TOL, \
                f"{lit} would escape the {name} ban window"


def test_module_edges_hold_canonical_values():
    """The sites fixed for P4 reference the canonical constants, not a
    look-alike number under a new name."""
    import inspect

    from atomx.methods.csf import crystal_field, slater_scaling
    from atomx.reference_data import nist

    # crystal_field.py / slater_scaling.py: the truncated 219474.63.
    assert crystal_field.CM_PER_HARTREE == HARTREE_CM1
    default_cm = inspect.signature(
        slater_scaling.free_ion_levels).parameters["cm"].default
    assert default_cm == HARTREE_CM1

    # reference_data/nist.py: the duplicated CODATA-2018 HA_CM1.
    assert nist.HA_CM1 == HARTREE_CM1
    assert nist.cm1_to_hartree(1.0) == pytest.approx(1.0 / HARTREE_CM1,
                                                     rel=1e-15)

    # The eV <-> nm sites: exact float identity through the public paths
    # (uta/moments.py and spectrum/uta.py; crm_synth.py and viz/mpl.py
    # are covered by the source scan above).
    from atomx.spectrum.uta import _uta_sigma_nm
    from atomx.uta import uta_moments

    moments = uta_moments([(100.0, 1.0)])
    assert moments.centroid_nm == HC_EV_NM / 100.0
    assert _uta_sigma_nm(100.0, 2.0) == HC_EV_NM * 2.0 / 100.0 ** 2
