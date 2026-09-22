"""A/B coverage for the public comparison helpers.

Annex A of ``docs/test_audit/S7_api_units_harness.md`` found ``theory_ladder``,
``what_changed``, ``Method`` and seven ``atomx.types`` enums exported from
``src/atomx/__init__.py`` with ZERO references anywhere in the suite.  This
file covers the two that compute physics, plus the enums a live public call
actually consumes (``TheoryLevel``, ``CouplingScheme``, ``EnergyUnit``,
``NuclearModel``, ``StartingPotential``; ``RelativisticCorrection`` is consumed
by the ladder's Breit-Pauli rungs).

Every expected value is closed-form or external — nothing here is a number
that only atomx could have produced:

* the two hydrogenic rungs are ``-Z^2/(2 n^2)`` and the Sommerfeld
  ``C^2 (gamma - 1)`` with ``gamma = sqrt(1 - (Z alpha)^2)``, both written out;
* the Dirac-Fock rung is compared with the live DBSR_HF Coulomb energies
  quoted in ``benchmarks/EXTERNAL_VERIFICATION.md`` (He -2.86181332 Ha,
  Ne -128.69192580 Ha);
* ``what_changed``'s energy line is checked against the Ha -> eV and
  Ha -> cm-1 identities and against the antisymmetry of its arguments;
* ``compare(..., unit=...)`` must print ``E * factor`` in that unit.

Still zero-coverage after this file (see ``docs/test_audit/S7_remediation.md``
for why each is plumbing rather than physics): ``Method`` is covered only
structurally below; ``Gauge`` and ``Multipole`` have no live consumer in
``src/atomx``; ``NuclearModel``/``StartingPotential`` members are *not* coerced
by ``Atom(nuclear=...)``/``Basis(starting_potential=...)`` — only their value
strings are, which is what the last test pins.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import (  # noqa: E402
    Atom,
    Basis,
    StartingPotential,
    theory_ladder,
    what_changed,
)
from atomx.compare import compare  # noqa: E402
from atomx.constants import ALPHA, HARTREE_CM1, HARTREE_EV, C  # noqa: E402
from atomx.types import (  # noqa: E402
    CouplingScheme,
    EnergyUnit,
    NuclearModel,
    TheoryLevel,
)


@pytest.fixture(scope="module")
def h_ladder():
    return theory_ladder(1)


@pytest.fixture(scope="module")
def he_ladder():
    return theory_ladder(2)


@pytest.fixture(scope="module")
def ne_ladder():
    return theory_ladder(10)


# ---------------------------------------------------------------------------
# theory_ladder: the closed-form hydrogenic rungs
# ---------------------------------------------------------------------------

def test_hydrogenic_ladder_is_the_two_closed_form_solutions(h_ladder):
    """Z=1 is the only case with an exactly known answer at every rung."""
    assert [r.method_name for r in h_ladder] == [
        "HydrogenicSchrodinger", "HydrogenicDirac"]
    assert [r.theory_level for r in h_ladder] == [TheoryLevel.HYDROGENIC] * 2
    assert [r.coupling_scheme for r in h_ladder] == [CouplingScheme.JJ] * 2

    e_schrodinger = float(h_ladder[0].total_energy)
    e_dirac = float(h_ladder[1].total_energy)

    # Schrodinger: E_1 = -Z^2 / (2 n^2) = -1/2 Ha exactly for Z = n = 1.
    assert e_schrodinger == -0.5

    # Dirac: E = C^2 (gamma - 1), gamma = sqrt(kappa^2 - (Z alpha)^2) with
    # |kappa| = 1, n_r = 0 for the 1s1/2 ground state.
    gamma = np.sqrt(1.0 - ALPHA ** 2)
    assert e_dirac == pytest.approx(C ** 2 * (gamma - 1.0), rel=1e-12)

    # Relativity *lowers* the 1s energy: the Dirac electron is more bound.
    assert e_dirac < e_schrodinger


# ---------------------------------------------------------------------------
# theory_ladder: the Dirac-Fock rung against DBSR_HF
# ---------------------------------------------------------------------------

def test_ladder_dirac_fock_rung_matches_dbsr_hf(he_ladder, ne_ladder):
    """The ladder's last rung is DHF, and DHF is what DBSR_HF computes.

    Reference numbers (``benchmarks/EXTERNAL_VERIFICATION.md`` section 1,
    live ``dbsr_hf an=Z term=AV mbreit=0``): He -2.86181332 Ha,
    Ne -128.69192580 Ha.  Measured here: -2.861813320655851 and
    -128.6919258404333 (4e-8 Ha), well inside the 1e-6 gate.
    """
    panel = {"He": (he_ladder, -2.86181332),
             "Ne": (ne_ladder, -128.69192580)}
    for name, (rungs, dbsr_hf) in panel.items():
        dhf = rungs[-1]
        assert dhf.theory_level is TheoryLevel.DIRAC_HARTREE_FOCK, name
        assert dhf.coupling_scheme is CouplingScheme.JJ, name
        assert float(dhf.total_energy) == pytest.approx(dbsr_hf, abs=1e-6), name

        # The non-relativistic rung below it is LS-coupled and *less* bound:
        # the relativistic contraction of the s shells lowers the total
        # energy.  This is what makes the ladder a ladder.
        non_relativistic = rungs[0]
        assert non_relativistic.coupling_scheme is CouplingScheme.LS, name
        assert float(non_relativistic.total_energy) > float(
            dhf.total_energy), name


def test_ladder_rungs_are_the_documented_method_objects(h_ladder):
    """``theory_ladder`` is the ``Method`` API's public entry point."""
    from atomx.methods import HydrogenicDirac, HydrogenicSchrodinger, Method

    for cls, rung in zip(
        (HydrogenicSchrodinger, HydrogenicDirac), h_ladder, strict=True
    ):
        assert issubclass(cls, Method)
        assert rung.method_name == cls.__name__


# ---------------------------------------------------------------------------
# what_changed: the energy line is a unit identity
# ---------------------------------------------------------------------------

_ENERGY_LINE = re.compile(
    r"Energy change: (-?\d+\.\d+) Ha = (-?\d+\.\d+) eV = (-?\d+\.\d+) cm-1")


def test_what_changed_reports_the_energy_change_in_three_units(ne_ladder):
    before, after = ne_ladder[0], ne_ladder[-1]
    text = what_changed(before, after)
    assert "Dirac" in text     # the keyed explanation, not the fallback

    match = _ENERGY_LINE.search(text)
    assert match, text
    d_ha, d_ev, d_cm1 = (float(group) for group in match.groups())
    delta = float(after.total_energy) - float(before.total_energy)
    assert d_ha == pytest.approx(delta, rel=1e-5)
    # 1 Ha = HARTREE_EV eV = HARTREE_CM1 cm-1; the printed line must use the
    # same constants the rest of the package does (no stray CODATA year).
    assert d_ev == pytest.approx(d_ha * HARTREE_EV, rel=1e-4)
    assert d_cm1 == pytest.approx(d_ha * HARTREE_CM1, rel=1e-4)


def test_what_changed_is_antisymmetric_in_its_arguments(ne_ladder):
    before, after = ne_ladder[0], ne_ladder[-1]
    forward = float(_ENERGY_LINE.search(what_changed(before, after)).group(1))
    backward = float(_ENERGY_LINE.search(what_changed(after, before)).group(1))
    assert backward == pytest.approx(-forward, rel=1e-5)


# ---------------------------------------------------------------------------
# compare: the unit conversion is the only thing `unit=` may change
# ---------------------------------------------------------------------------

def test_compare_prints_each_energy_in_the_requested_unit(ne_ladder, capsys):
    before, after = ne_ladder[0], ne_ladder[-1]
    compare(before, after, unit=EnergyUnit.CM1)
    out = capsys.readouterr().out
    assert "cm-1" in out
    assert f"{float(before.total_energy) * HARTREE_CM1:.6f}" in out
    assert f"{float(after.total_energy) * HARTREE_CM1:.6f}" in out


# ---------------------------------------------------------------------------
# The enums a public keyword actually consumes
# ---------------------------------------------------------------------------

def test_finite_nuclear_size_reduces_the_binding():
    """The Fermi nucleus is a *shallower* potential inside r < r_nucleus.

    Spreading the charge over the nuclear volume lowers the potential seen by
    the inner electrons, so the finite-nucleus total energy must lie *above*
    the point-nucleus one (less bound).  Measured (corrections off, so the
    comparison is not mixed with Breit/QED): Ne -128.6299783505843 (Fermi) vs
    -128.63002194708503 (point), Fe XVII -1146.731865120602 vs
    -1146.7363760402477.
    """
    shifts = {}
    for Z, charge in ((10, 0), (26, 16)):
        point = Atom(Z=Z, charge=charge, nuclear="point",
                     corrections="none").solve()
        finite = Atom(Z=Z, charge=charge, nuclear="fermi",
                      corrections="none").solve()
        assert point.total_energy < finite.total_energy, Z
        shifts[Z] = finite.total_energy - point.total_energy
    # A contact effect of the inner shells: it grows with Z.
    assert shifts[26] > shifts[10]


def test_nuclear_and_starting_potential_enum_values_are_accepted_spellings():
    """``atomx.types`` documents member/value equivalence with the strings.

    Neither surface coerces the *member* (``Atom(10, nuclear=NuclearModel.POINT)``
    raises ValueError — reported, not frozen here), so what has to stay in
    sync is the value spelling.
    """
    for member in NuclearModel:
        assert Atom(10, nuclear=member.value).nuclear == member.value
    for member in StartingPotential:
        assert Basis(starting_potential=member.value).starting_potential \
            == member.value
