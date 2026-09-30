"""Analytic (orbital-independent) gates on the jj-CI Hamiltonian.

These two identities hold for the EXACT matrix elements
``<CSF_r|H|CSF_s>`` and for nothing else, so they cannot be satisfied by a
configuration-average kernel:

1. **He-like ``1s2p`` singlet-triplet splitting.**  In the
   non-relativistic limit the two ``J=1`` levels of an ``ns n'p``
   configuration split by exactly ``2 G^1(ns,n'p)/3``.  The ratio is
   *independent of the radial functions* — a wrong angular coefficient
   shows up as a constant multiplicative factor across the whole
   isoelectronic sequence.  Measured on the old configuration-average
   kernel: 3.533 at Z=2, 3.533 at Z=6, 3.530 at Z=10.  After the exact
   ``<CSF|H|CSF>`` construction the ratio is 1.0000 at Z=2, 1.0019 at
   Z=6 and 1.0119 at Z=10 — the residual growth is the relativistic
   correction to the non-relativistic LS identity (at Z=18 it reaches
   1.10), which is exactly the expected behaviour and is why the gate is
   restricted to Z <= 10 with a 5% tolerance.

   Note the configuration string: ``"1s1 2p1"`` (two open subshells, 5
   CSFs).  ``"1s2 2p1"`` is 1s^2 2p — one electron outside a closed
   shell, 2 CSFs, no singlet/triplet structure at all.

2. **Be 2s^2/2p^2 near degeneracy.**  The two-configuration MCHF value
   is -14.6166 Ha (literature, state-optimised).  atomx optimises the
   Guerra average over all six CSFs of the two configurations, which is a
   *different protocol* and sits ~5 mHa above it (measured -14.6115), so
   the literature comparison keeps a 6 mHa band; the physics gates are the
   variational sandwich (below the exact bound, below atomx's own live
   1-CSF reference) and a >= 0.03 Ha gain over that reference.  The 1-CSF
   reference is *measured in-test* (``_be_reference.be_1csf_energy``), not
   typed: the old hardcoded -14.5734 was 2.5 mHa away from the produced
   -14.575892, and the textbook Hartree-Fock *limit* (-14.573023) is a
   further 2.9 mHa below atomx's finite basis.
"""

from __future__ import annotations

import numpy as np
import pytest
from _be_reference import BE_DC_SANITY_FLOOR, be_1csf_energy

import atomx.physics.csf.matrix as MAT
import atomx.methods.csf.solver as SOLVER
from atomx import Atom

BE_2CONFIG_MCHF = -14.6166


def _solve_capturing_orbitals(Z, configs, **kwargs):
    """Solve and return (result, orbitals_dict, bsp_data)."""
    captured = {}
    original = SOLVER.ci_solve

    def spy(csfs, orbitals_dict, orbital_basis, bsp_data, Z_, ns, **kw):
        out = original(csfs, orbitals_dict, orbital_basis, bsp_data, Z_, ns, **kw)
        captured["orbitals_dict"] = dict(orbitals_dict)
        captured["bsp_data"] = bsp_data
        return out

    SOLVER.ci_solve = spy
    try:
        atom = Atom(Z=Z, config=configs, corrections="none", **kwargs)
        atom.solve(tol=1e-9, max_iter=80)
        result = atom._result
    finally:
        SOLVER.ci_solve = original
    return result, captured["orbitals_dict"], captured["bsp_data"]


@pytest.mark.parametrize("Z", [2, 6, 10])
def test_helike_1s2p_j1_splitting_is_two_thirds_g1(Z):
    """``E(1P1) - E(3P1) == 2 G^1(1s,2p)/3`` to within relativistic corrections."""
    result, orbitals, bsp = _solve_capturing_orbitals(Z, ["1s2", "1s1 2p1"])
    energies = np.asarray(result.state_energies)
    j2 = np.asarray(result.state_J2)
    j1 = np.sort(energies[j2 == 2])
    assert j1.size == 2, f"expected two J=1 levels, got {j1.size}"

    P1s, Q1s = orbitals[(1, -1)]
    P2p, Q2p = orbitals[(2, -2)]
    g1 = MAT._gk(P1s, Q1s, P2p, Q2p, bsp, 1)
    predicted = 2.0 * g1 / 3.0
    assert g1 > 0.0, "G^1(1s,2p) must be positive"

    ratio = (j1[-1] - j1[0]) / predicted
    # Sub-percent relativistic correction at Z=2; a few percent at Z=10.
    assert ratio == pytest.approx(1.0, abs=0.05), (
        f"Z={Z}: J=1 splitting {j1[-1] - j1[0]:.6f} Ha vs 2G^1/3 = "
        f"{predicted:.6f} Ha (ratio {ratio:.3f}). The old configuration-average "
        f"kernel gave 3.533 / 3.533 / 3.530 at Z = 2 / 6 / 10.")


def test_be_two_configuration_near_degeneracy():
    """Be ``1s2 2s2 + 1s2 2p2``: variational, and the near degeneracy is captured."""
    result, _orbitals, _bsp = _solve_capturing_orbitals(
        4, ["1s2 2s2", "1s2 2p2"])
    energy = result.total_energy
    assert energy >= BE_DC_SANITY_FLOOR, (
        f"{energy} is below the diagnostic Be energy floor "
        f"{BE_DC_SANITY_FLOOR}")
    e_ref = be_1csf_energy()   # atomx's own 1s2 2s2 solve, measured live
    assert energy < e_ref, (
        f"{energy}: the 2s^2/2p^2 space contains the 1-CSF reference "
        f"({e_ref}) and cannot sit above it")
    assert energy <= e_ref - 0.03, (
        f"{energy}: the 2s^2/2p^2 near degeneracy must lower the energy by "
        f">= 0.03 Ha from the live 1-CSF reference {e_ref}")
    # The literature state-optimised 2-configuration MCHF value; atomx's
    # Guerra-average protocol sits ~5 mHa above it (measured -14.6115).
    assert energy == pytest.approx(BE_2CONFIG_MCHF, abs=6e-3)
