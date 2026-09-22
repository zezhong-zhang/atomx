"""External, literature-anchored gate: Be I levels against NIST ASD.

The CI matrix is validated against an independent m-scheme oracle on
identical orbitals in ``test_ci_matrix_vs_mscheme_reference.py``.  This
file anchors the *same* levels to an external measurement instead, so a
shared misconception between the two implementations cannot pass: the
residual is the orbital basis, not the CI matrix.

Reference (NIST ASD, ``asd.004000``, ground ``1s2 2s2 1S0`` = 0 cm-1;
1 Ha = 219474.6313632 cm-1; the exact non-relativistic Be ground state is
-14.66736 Ha, Drake):

======================  ================
level                   cm-1
======================  ================
1s2 2s2p 3P*0           21978.31
1s2 2s2p 3P*1           21978.925
1s2 2s2p 3P*2           21981.26
1s2 2s2p 1P*1           42565.4502
======================  ================

Measured here (``breit=False, qed=False``, ``ns=113, ks=9``), excitation
energies above the 1s2 2s2 ground state:

===========================  ============  ============  ===========
protocol                    3P*0          3P*1          1P*1
===========================  ============  ============  ===========
``fac_frozen``              13822.5       13858.9       43594.4
``mcdhf`` (default)         18404.6       18283.9       51880.9
``mcdhf`` before R7         118643.6      118592.8      139877.5
===========================  ============  ============  ===========

The 2p of this space is a spectator virtual: its CSFs are the odd-parity
``1s2 2s1 2p1`` ones, which carry exactly zero amplitude in the even
ground state, so its Guerra population is zero and the pre-R7 solver gave
it a *bare-nucleus* 2p (ε = −2.0 Ha, the hydrogenic Z=4 value) — hence the
118000 cm-1 row.  The R7 asymmetric mean field (GRASP's layer protocol,
``MCDHFSolver._mean_field_matrices``) solves it in the reference
configuration's potential instead, which brings the manifold down to
18.0–18.4 kcm-1 for the triplets.

What is left is the *averaged-level* residual, not a CI error: the
orbitals are optimised for the ground configuration, while this space's
2p only ever appears in ``1s2 2s1 2p1``, and the parent-capped anchor
contributes 1s^2 2s^1 (the physical field of that configuration).  The
``3P``-``1P`` separation is ``2 G^1(2s,2p)``, which is set by the 2p
radial extent this protocol gives (-0.59 Ha, i.e. more compact than the
state-optimised MCHF 2p): the computed separation is 33.6 kcm-1 against
the NIST 20.6 kcm-1.  The fine-structure ordering is also still inverted,
because the two jj components are solved as independent radial functions
in the frozen field; the ordering test below is therefore written on the
``fac_frozen`` fixture, where the jj pair is solved together with the tiny
virtual occupations and the ordering does come out physical.

Both residuals are *orbital* (the same levels agree with the m-scheme
oracle to 1e-14 on the very orbitals used here), and both gates are
recorded as strict xfails carrying the measured numbers rather than being
widened into meaningless tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom, Correlation
from atomx.constants import HARTREE_CM1

#: NIST ASD excitation energies (cm-1) for the 1s2 2s2p levels.
NIST_CM1 = {
    "3P0": 21978.31,
    "3P1": 21978.925,
    "3P2": 21981.26,
    "1P1": 42565.4502,
}

CONFIGS = ["1s2 2s2", "1s2 2s1 2p1"]


def _solve(protocol):
    atom = Atom(
                Z=4,
                config=CONFIGS,
                corrections="none",
                correlation=Correlation(radial_optimization=protocol),
           )
    atom.solve()
    return atom


@pytest.fixture(scope="module")
def fac_frozen_levels():
    """``(cm-1 excitation energies by 2J, ground Ha)`` for the FAC protocol."""
    atom = _solve("fac_frozen")
    result = atom._result
    energies = np.asarray(result.state_energies)
    two_j = np.asarray([
        int(getattr(csf, "total_J2", 0))
        for csf in result.csfs
    ])[np.argmax(np.abs(result.mixing_coefficients), axis=0)]
    ground = float(energies.min())
    excit = {}
    for j2, e in zip(two_j, energies, strict=True):
        if e - ground < 1e-6:
            continue
        excit.setdefault(int(j2), []).append((e - ground) * HARTREE_CM1)
    return {j2: sorted(v) for j2, v in excit.items()}, ground


def test_ground_state_is_the_reference_configuration(fac_frozen_levels):
    _excit, ground = fac_frozen_levels
    assert ground == pytest.approx(-14.5759, abs=2e-3)


def test_three_p_ordering_is_physical(fac_frozen_levels):
    """3P0 < 3P1 < 3P2 < 1P1 must come out in that order.

    The old kernel produced five scrambled levels with a "ground state"
    0.042 Ha below the exact Be ground state; the ordering gate is the
    cheapest statement that the J-block structure is now physical.
    """
    excit, _ground = fac_frozen_levels
    assert sorted(excit) == [0, 2, 4]
    p0 = excit[0][0]
    j1 = sorted(excit[2])          # 3P1 (low) and 1P1 (high)
    p2 = excit[4][0]
    assert len(j1) == 2
    assert p0 < j1[0] < p2 < j1[1], (p0, j1, p2)


def test_singlet_p_excitation_matches_nist(fac_frozen_levels):
    """The 1P1 level, the least orbital-sensitive of the four."""
    excit, _ground = fac_frozen_levels
    singlet = max(excit[2])
    assert singlet == pytest.approx(NIST_CM1["1P1"], abs=3000.0)


@pytest.mark.xfail(strict=True, reason=(
    "OPEN, orbital protocol (docs/CORRELATION_SD_RESEARCH_2026-09-20.md "
    "section 3.7): ``fac_frozen`` optimises the first CSF and keeps every "
    "other orbital as a 0.001-occupation virtual in that frozen potential, "
    "which for this space anchors the 2p on 1s2 2s2 rather than on its own "
    "parents 1s2 2s1 2p1.  Over-screening costs the 3P term 8150 cm-1 "
    "(measured 13822.5 vs NIST 21978.31 cm-1); the 1P1 level is within "
    "1100 cm-1.  The R7 asymmetric mean field fixes the Guerra branch, not "
    "this protocol: ``fac_frozen`` is unchanged by design, and the same "
    "parent-capping that fixed the mcdhf branch is not applied here "
    "because the protocol *is* 'frozen reference potential'.  The matrix is "
    "not the cause: the same five levels agree with the independent "
    "m-scheme oracle to 1e-14 on these orbitals."))
def test_triplet_p_excitation_matches_nist(fac_frozen_levels):
    excit, _ground = fac_frozen_levels
    assert excit[0][0] == pytest.approx(NIST_CM1["3P0"], abs=3000.0)


@pytest.mark.xfail(strict=True, reason=(
    "OPEN, orbital protocol: R7 (the asymmetric mean field) removed the "
    "bare-nucleus 2p here — the manifold moved from 118000 cm-1 to 18.0-18.4 "
    "kcm-1 — and the 3P1 level is now 3695 cm-1 below NIST (measured "
    "18283.9 vs 21978.925, gate tolerance 3000), so the gate is still "
    "strictly red rather than passing on a widened tolerance.  The residual "
    "is the averaged-level protocol itself: the Guerra orbitals are "
    "optimised for the 1s2 2s2 ground configuration while this space's 2p "
    "only appears in 1s2 2s1 2p1, and the parent-capped anchor gives that "
    "configuration's field (1s^2 2s^1).  The 2p that comes out is more "
    "compact than a state-optimised MCHF 2p (ε = −0.59 Ha), so the "
    "3P-1P separation is 33.6 kcm-1 against NIST 20.6 kcm-1 and the fine "
    "structure is inverted (the two jj components are independent radial "
    "functions in the frozen field).  The CI matrix is exact on those "
    "orbitals (oracle agreement 2e-15); the orbitals are."))
def test_levels_match_nist_in_default_protocol():
    atom = _solve("mcdhf")
    result = atom._result
    energies = np.asarray(result.state_energies)
    ground = float(energies.min())
    two_j = np.asarray([
        int(getattr(csf, "total_J2", 0)) for csf in result.csfs
    ])[np.argmax(np.abs(result.mixing_coefficients), axis=0)]
    excit = {}
    for j2, e in zip(two_j, energies, strict=True):
        if e - ground < 1e-6:
            continue
        excit.setdefault(int(j2), []).append((e - ground) * HARTREE_CM1)
    assert sorted(excit[2])[0] == pytest.approx(NIST_CM1["3P1"], abs=3000.0)
