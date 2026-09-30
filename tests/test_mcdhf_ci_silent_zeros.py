"""Guards against silently-uncorrelated CI builds and dropped SD doubles.

Two Medium findings in ``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md``:

**M7 — missing orbital silently yields the uncorrelated energy.**
``physics/csf/matrix.py`` zero-filled the orbital a CI element asked
for when it was absent from the radial workspace, so a multi-CSF run
whose correlation orbital had never been solved came back as the
single-configuration energy with no diagnostic at all (the review
measured Be ``configs=["1s2", "1s2 2p2"]`` as ``E(2 CSF) == E(1 CSF)``).
:func:`build_ci_matrix` / :func:`ci_solve` now raise
:class:`MissingOrbitalError` naming every missing ``(n, kappa)``.

**M8 — same-subshell doubles dropped.**  ``generate_sd_excitations`` took
*three* electrons out of a doubly-occupied source subshell in its
``i == j`` branch (``src1`` was decremented once more on top of the
unconditional ``src1``/``src2`` decrements), so the resulting negative
occupation was rejected by the validation below and every same-subshell
double (1s² → 2s², 1s² → 2p², …) was silently dropped.  For a
single-active-shell reference such as the review's ``"1s2"`` that is
*every* double, so ``correlation="sd"`` generated singles only; for
multi-shell references the ``i != j`` doubles still came through, but
none of the a² → b² / a² → b¹c¹ ones did.
"""

from __future__ import annotations

import numpy as np
import pytest
from _be_reference import BE_DC_SANITY_FLOOR

from atomx import Atom
from atomx.methods.ci import ci_solve
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.config.csf import get_shared_orbital_basis, make_csf_list
from atomx.methods.mcdhf.excitations import (
    generate_sd_excitations,
    suggest_virtual_orbitals,
)
from atomx.physics.csf.matrix import MissingOrbitalError, build_ci_matrix


def _sparse_workspace(ns=40, ks=7):
    """A valid B-spline workspace for Z=4 — DFSolver construction only.

    ``_check_orbitals_present`` fires before any integral is evaluated, so
    no SCF (and no real orbital arrays) are needed for the guard tests.
    """
    subs = [Subshell(n=1, kappa=-1, occupation=2.0),
            Subshell(n=2, kappa=-1, occupation=2.0)]
    dhf = DFSolver(Z=4, subshells=subs, ns=ns, ks=ks)
    return dhf.bsp_data, dhf.ns


def _orbitals_without_2p(ns):
    """Only the two s orbitals "solved" — 2p is intentionally absent."""
    return {(1, -1): (np.zeros(ns), np.zeros(ns)),
            (2, -1): (np.zeros(ns), np.zeros(ns))}


class TestSameSubshellDoubles:
    """M8 — ``1s² → 2s²`` and friends must actually be generated."""

    VIRTUALS = ["2s", "2p", "3s"]

    def test_same_subshell_doubles_are_generated(self):
        singles = generate_sd_excitations(
            "1s2", self.VIRTUALS, max_excitations=1)
        sd = generate_sd_excitations("1s2", self.VIRTUALS, max_excitations=2)
        doubles = set(sd) - set(singles)
        assert doubles == {
            "2s2", "2p2", "3s2",                    # a² → b²
            "2s1 2p1", "2s1 3s1", "2p1 3s1",        # a² → b¹c¹
        }
        # C(3 virtuals + 1, 2) = 6 unordered virtual pairs → six doubles.
        assert len(doubles) == 6
        assert len(sd) == len(singles) + 6

    def test_doubles_from_different_subshells_still_generated(self):
        """The ``i != j`` branch (a¹b¹ → c²) must survive the M8 fix."""
        singles = generate_sd_excitations(
            "1s2 2s1", ["3s", "3p"], max_excitations=1)
        sd = generate_sd_excitations(
            "1s2 2s1", ["3s", "3p"], max_excitations=2)
        doubles = set(sd) - set(singles)
        assert {"1s1 3s2", "2s1 3s2"} <= doubles

    def test_correlation_sd_plumbing_requests_doubles(self):
        """``correlation="sd"`` must ask for the doubles, not just singles."""
        virtuals = suggest_virtual_orbitals(4, n_virtual=1)
        singles = generate_sd_excitations(
            "1s2 2s2", virtuals, core_orbitals=None, max_excitations=1)
        atom = Atom(Z=4, config="1s2 2s2", correlation="sd")
        assert set(singles) <= set(atom._config_strings)
        assert len(atom._config_strings) > len(singles)


class TestMissingOrbitalGuard:
    """M7 — a CSF that needs an unsolved orbital must raise, not zero-fill."""

    def test_build_ci_matrix_rejects_unsolved_orbital(self):
        bsp_data, ns = _sparse_workspace()
        csfs = make_csf_list(Z=4, config_strings=["1s2 2s2", "1s2 2p2"])
        orbitals = _orbitals_without_2p(ns)
        basis = get_shared_orbital_basis(csfs)
        with pytest.raises(MissingOrbitalError) as excinfo:
            build_ci_matrix(csfs, orbitals, basis, bsp_data, Z=4, ns=ns)
        message = str(excinfo.value)
        # Every missing (n, kappa) required by the CSF list is named, so the
        # user knows which orbital to add to the radial set.
        needed = {(sh.n, sh.kappa) for csf in csfs for sh in csf.subshells}
        missing = sorted(needed - set(orbitals))
        assert missing, "the fixture must actually be missing an orbital"
        for key in missing:
            assert str(key) in message
        assert "never" in message

    def test_empty_orbital_map_is_rejected_in_production(self):
        """``build_ci_matrix`` must not accept a CI with no radial data."""
        bsp_data, ns = _sparse_workspace()
        csfs = make_csf_list(Z=4, config_strings=["1s2 2s2"])
        basis = get_shared_orbital_basis(csfs)
        with pytest.raises(MissingOrbitalError):
            build_ci_matrix(csfs, {}, basis, bsp_data, Z=4, ns=ns)

    def test_ci_solve_rejects_unsolved_orbital(self):
        bsp_data, ns = _sparse_workspace()
        csfs = make_csf_list(Z=4, config_strings=["1s2 2s2", "1s2 2p2"])
        basis = get_shared_orbital_basis(csfs)
        with pytest.raises(MissingOrbitalError):
            ci_solve(csfs, _orbitals_without_2p(ns), basis, bsp_data,
                     Z=4, ns=ns)

    def test_complete_orbital_set_still_builds(self):
        """The guard must not fire for a workspace that has every orbital."""
        subs = [Subshell(n=1, kappa=-1, occupation=2.0),
                Subshell(n=2, kappa=-1, occupation=-2.0 + 4.0)]
        dhf = DFSolver(Z=4, subshells=subs, ns=40, ks=7)
        result = dhf.solve(max_iter=40, tol=1e-5)
        orbitals = {key: (p, q) for key, (p, q, _) in result.orbitals_by_subshell().items()}
        csfs = make_csf_list(Z=4, config_strings=["1s2 2s2"])
        basis = get_shared_orbital_basis(csfs)
        ci = ci_solve(csfs, orbitals, basis, dhf.bsp_data, Z=4, ns=dhf.ns)
        assert np.isfinite(ci.ground_energy)


class TestCorrelationIsNotSilentlyDropped:
    """End-to-end M7 gate: the correlation CSF must lower the energy."""

    def test_two_csf_be_lies_below_single_csf(self):
        single = Atom(Z=4, config=["1s2 2s2"], engine="bspline")
        single.solve()
        two = Atom(Z=4, config=["1s2 2s2", "1s2 2p2"], engine="bspline")
        two.solve()
        # 2p² is a 4-electron configuration of the SAME ion, so mixing it in
        # must recover correlation energy.  Before M7 this could silently
        # come back as exactly the 1-CSF energy when 2p was never solved.
        assert two.total_energy < single.total_energy - 1.0e-3
        # Two-sided: the old one-sided form was satisfied by the -15.05 Ha
        # non-variational regression it was written to catch, because any
        # collapse below the reference is "lower" too.
        assert two.total_energy > BE_DC_SANITY_FLOOR, (
            f"two-CSF CI = {two.total_energy:.6f} Ha is below the "
            f"diagnostic Be energy floor "
            f"{BE_DC_SANITY_FLOOR} Ha")
