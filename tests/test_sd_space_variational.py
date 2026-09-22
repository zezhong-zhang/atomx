"""Post-review finding: ``correlation="sd"`` produces a non-variational energy.

Discovered 2026-09-19 while fixing M8 (same-subshell doubles).  The missing
a²→b² doubles were added, the Be SD space grew from 19 to 39 CSFs, and the
resulting energy moved from already-unphysical to further unphysical:

| space | CSFs | E (Ha) | when |
|---|---:|---:|---|
| ``1s2 2s2`` alone | 1 | -14.575892 | current (reference solve) |
| ``correlation="s"`` | 9 | **-14.578968** | current (exact CI matrix) |
| ``correlation="s"`` | 9 | -14.659475 | with the removed legacy kernel |
| ``correlation="sd"`` | 39 | **-15.053612** | the 2026-09-19 defect |
| ``correlation="sd"`` | 399 | **-14.619159** | current (exact CI matrix) |

The exact non-relativistic ground state of Be is -14.66736 Ha (Drake), so the
pre-fix SD space sat 0.386 Ha *below* it — impossible for a variational CI in
a space that contains the singles space.  The exact CI matrix
(``docs/CORRELATION_SD_RESEARCH_2026-09-20.md`` §8 option A) removed that
signature: the space now lands 0.048 Ha above the exact bound.

The same fix also removed the *spurious* 84 mHa "correlation" the old kernel
gave the 9-CSF singles space: with the exact matrix the singles gain over the
reference determinant is 3.1 mHa, which is what Brillouin's theorem predicts
for a singles-only expansion on (near-)self-consistent orbitals.  The 0.084 Ha
in the row above is the legacy kernel's error, not physics.

Diagnosis of the *old* defect (2026-09-19, for the record).  It was in the
CI/CSF layer, not the orbital optimisation: switching to frozen orbitals made
it worse rather than better (``fac_frozen``: singles -15.400, SD -16.358),
and every value was below the exact ground state.

Observations that are still open:

* the 399-CSF SD space expands from 39 configurations with only 99 distinct
  occupation vectors (up to 16 CSFs sharing one), including core-excited
  configurations such as ``2s2 2p2`` (no 1s);
* the SD protocol's Guerra average is far from the best available: the SD
  space gains 43 mHa over the reference determinant where the exact
  correlation energy is 91 mHa.  That is not a variational error (the space
  is below the singles space, as it must be) but it is the reason
  ``test_sd_space_is_a_superset_of_the_singles_space`` exists: if a future
  change makes the enlarged space land *above* the smaller one, the orbital
  protocol has broken.

Gates in this file are two-sided wherever a bound exists: the exact
non-relativistic ground state is the floor, and the 1-CSF reference solve of
the same file is the ceiling for any space that contains it.  A one-sided
gate is what let a 0.18 Ha drift in the recorded SD number stay invisible
(the docstring recorded -14.4369 Ha while the code produced -14.619159 Ha,
and both satisfy ``E > -14.6674``).

See ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` §"Found during
remediation" and ``docs/test_audit/S2_ci_mcdhf_correlation.md`` F2/F8.
"""

from __future__ import annotations

import pytest
from _be_reference import BE_DC_LOWER_BOUND, BE_EXACT_NR, BE_HF_1CSF

from atomx import Atom

#: Dirac-Coulomb 2p orbital energy for a bare Z=4 nucleus — the value an
#: unscreened orbital sits at.  Closed form (hydrogenic Dirac), used as the
#: upper end of the screening bracket.
BE_HYDROGENIC_2P = -2.00047

#: The same orbital fully screened to Z_eff = 1, still hydrogenic and still
#: closed form: E(2p, Z=1) = -1/8 Ha.  A 2p with *any* partial screening has
#: to lie between this and ``BE_HYDROGENIC_2P``.
BE_FULLY_SCREENED_2P = -0.125


def _be(correlation):
    atom = Atom(
                Z=4,
                config="1s2 2s2",
                correlation=correlation,
                engine="bspline",
                corrections="none",
           )
    atom.solve()
    return atom


@pytest.fixture(scope="module")
def be_reference():
    """The 1-CSF ``1s2 2s2`` solve — the ceiling every larger space must beat."""
    return _be(None)


@pytest.fixture(scope="module")
def be_singles():
    return _be("s")


@pytest.fixture(scope="module")
def be_sd():
    return _be("sd")


class TestVariationalSanity:
    @pytest.mark.snapshot
    def test_single_configuration_is_the_reference(self, be_reference):
        """atomx's own 1-CSF fixed point (frozen self-regression).

        ``BE_HF_1CSF`` is a measured atomx value, not an external one; the
        textbook Hartree-Fock limit is 2.9 mHa lower (``BE_HF_LIMIT``).
        """
        assert be_reference.total_energy == pytest.approx(BE_HF_1CSF, abs=2e-3)

    def test_singles_are_variational(self, be_reference, be_singles):
        """A 9-CSF singles space must sit inside the variational window.

        Floor: the exact non-relativistic ground state (any variational
        energy is above it).  Ceiling: the 1-CSF reference of the same
        protocol, which the singles space contains — a minimum eigenvalue
        over a superset cannot be higher.  The gain is additionally bounded
        by the total correlation energy ``E_ref - BE_DC_LOWER_BOUND``,
        because the singles space cannot recover more correlation than
        exists between the reference determinant and the exact ground state.
        """
        e_singles = be_singles.total_energy
        e_ref = be_reference.total_energy
        assert e_singles > BE_EXACT_NR, (
            f"singles CI = {e_singles:.6f} Ha is below the exact Be "
            f"ground state {BE_EXACT_NR} Ha")
        assert e_singles < e_ref, (
            f"singles CI = {e_singles:.6f} Ha is above its own 1-CSF "
            f"reference {e_ref:.6f} Ha — the space contains that CSF")
        gain = e_ref - e_singles
        assert 0.0 < gain < e_ref - BE_DC_LOWER_BOUND, (
            f"singles recovered {gain:.6f} Ha, more than the total "
            f"correlation energy {e_ref - BE_DC_LOWER_BOUND:.6f} Ha")

    @pytest.mark.slow
    def test_sd_space_is_variational(self, be_reference, be_sd):
        """The SD space must sit inside the same window as the singles space.

        Was a strict xfail: the space returned -15.0536 Ha, 0.386 Ha BELOW
        the exact non-relativistic ground state -14.66736 Ha.  With the
        exact CI matrix (``docs/CORRELATION_SD_RESEARCH_2026-09-20.md`` §8,
        option A) it lands at -14.619159 Ha, inside the window.  The window
        is two-sided: the assertion below fails both for a non-variational
        collapse and for a space that fails to gain anything over the
        reference determinant it contains.
        """
        e_sd = be_sd.total_energy
        e_ref = be_reference.total_energy
        assert e_sd > BE_EXACT_NR, (
            f"correlation='sd' CI = {e_sd:.6f} Ha is below the "
            f"exact Be ground state {BE_EXACT_NR} Ha — non-variational")
        assert e_sd < e_ref - 1e-6, (
            f"correlation='sd' CI = {e_sd:.6f} Ha is not below the 1-CSF "
            f"reference {e_ref:.6f} Ha although the SD space contains it")

    @pytest.mark.slow
    def test_sd_space_is_a_superset_of_the_singles_space(
            self, be_singles, be_sd):
        """Superset monotonicity: E(SD) <= E(singles).

        The SD CSF list contains every singles CSF, so with one common
        orbital set the enlarged space cannot be higher; and since the
        singles-optimal orbitals are also available to the SD run, the
        variationally optimised SD energy must be at or below the
        singles-optimal one.  Measured: -14.619159 (SD) below -14.578968
        (singles).  The removed legacy kernel satisfied this too (it was
        *below* everything), which is why the floor assertion above, not
        this one, is what catches it.
        """
        assert be_sd.total_energy <= be_singles.total_energy, (
            f"SD {be_sd.total_energy:.6f} Ha > singles "
            f"{be_singles.total_energy:.6f} Ha although the SD space "
            f"contains the singles space")

    def test_orbitals_are_screened(self, be_singles):
        """Be ``correlation="s"`` 2p must not sit at the bare-nucleus value.

        Was a strict xfail until 2026-09-20: the 2p sat at the hydrogenic
        Z=4 value -2.0005 Ha, i.e. it felt no electron screening at all
        (``docs/CORRELATION_SD_RESEARCH_2026-09-20.md`` §3.7).  The
        parent-capped asymmetric mean field (R7) repairs it.

        Two independent statements: a *closed-form* bracket between the two
        hydrogenic endpoints (bare Z=4 and fully screened Z_eff=1), which is
        what makes this a physics gate rather than a pin, and the measured
        value of the R7 field, which is atomx's own number and is labelled
        as such.
        """
        eps_2p = be_singles.orbital_energies.get("2p1/2")
        assert eps_2p is not None
        assert BE_HYDROGENIC_2P < eps_2p < BE_FULLY_SCREENED_2P, (
            f"2p1/2 = {eps_2p:.6f} Ha is outside the hydrogenic bracket "
            f"({BE_HYDROGENIC_2P}, {BE_FULLY_SCREENED_2P}) — it is either "
            f"unscreened (bare Z=4) or bound by less than a bare proton")
        # atomx's own R7 value, measured on this machine; the audit found the
        # old gate (eps_2p > -1.50047) satisfied by a half-collapsed virtual
        # at -1.4 Ha, 1.3 Ha away from the produced -0.228585.
        assert eps_2p == pytest.approx(-0.2286, abs=0.02), (
            f"2p1/2 = {eps_2p:.6f} Ha is not the parent-capped mean-field "
            f"value -0.2286 Ha")
