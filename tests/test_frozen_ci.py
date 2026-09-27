"""Guards for the frozen-CI variational surrogate (frozen line search).

The surrogate (``CorrelatedOrbitalFunctional.freeze``) fixes the exact-CI
amplitudes and evaluates energy/Hellmann–Feynman gradient from the radial
integral pass alone — no CI Hamiltonian assembly, no diagonalization.  It
backs the ``frozen_line_search`` Armijo trials in ``optimize_orbitals``.

Physics pinned here:

* at the freeze point the surrogate equals the exact ground energy and its
  gradient (Hellmann–Feynman at the exact eigenstate);
* away from it the surrogate stays a variational upper bound, because
  orbital rotations are unitary and the fixed determinant amplitudes keep
  their norm — this is what makes Armijo acceptance on it sound;
* an optimization driven by frozen trials converges to the same energy as
  the exact-trial path (the accepted point is always re-evaluated exactly,
  and the convergence gate is untouched).
"""

from __future__ import annotations

import numpy as np
import pytest

import tests.test_ci_matrix_vs_mscheme_reference as _ref
from atomx.methods.mcdhf.variational import (
    CorrelatedOrbitalFunctional,
    optimize_orbitals,
)


def _mix(orbitals, key_a, key_b, angle):
    """Metric-orthonormal two-orbital rotation (keeps the variational bound)."""
    mixed = dict(orbitals)
    pa, qa = orbitals[key_a]
    pb, qb = orbitals[key_b]
    c, s = np.cos(angle), np.sin(angle)
    mixed[key_a] = (c * pa - s * pb, c * qa - s * qb)
    mixed[key_b] = (s * pa + c * pb, s * qa + c * qb)
    return mixed


@pytest.fixture(scope="module")
def solved():
    """Be ground + singles: a real CI with off-diagonal two-body keys."""
    result, csfs, captured = _ref._run_capturing_ci(
        ["1s2 2s2", "1s2 2s1 2p1"], radial_optimization="fac_frozen")
    assert result.converged
    functional = CorrelatedOrbitalFunctional(
        captured["csfs"], captured["bsp_data"], captured["Z"],
        captured.get("nuclear_potential"), captured.get("convol_cache"))
    return functional, captured


def test_frozen_energy_matches_exact_at_freeze_point(solved):
    functional, captured = solved
    energy, ci = functional.evaluate(captured["orbitals_dict"], gradient=False)
    frozen_energy = functional.freeze(ci).energy(captured["orbitals_dict"])
    assert frozen_energy == pytest.approx(energy, abs=1e-9)


def test_frozen_gradient_matches_exact_at_freeze_point(solved):
    functional, captured = solved
    _e, exact_gradient, ci = functional.evaluate(captured["orbitals_dict"])
    frozen_energy, frozen_gradient = functional.freeze(ci).evaluate(
        captured["orbitals_dict"])
    for key in functional.keys:
        np.testing.assert_allclose(
            frozen_gradient[key], exact_gradient[key], atol=1e-8)
    assert frozen_energy == pytest.approx(_e, abs=1e-9)


def test_frozen_energy_tracks_gauge_rotation(solved):
    """A unitary rotation among fully-occupied orbitals is pure gauge: the
    closed-shell ground CSF is invariant, so both the exact CI and the
    frozen surrogate must return the freeze-point energy unchanged.  This
    pins the contraction structure (one ordering per key, no double count)
    under a non-trivial orbital change, complementing the fixed-point tests
    which cannot see gauge directions.
    """
    functional, captured = solved
    rotated = _mix(captured["orbitals_dict"], (1, -1), (2, -1), 0.3)
    energy, ci = functional.evaluate(captured["orbitals_dict"], gradient=False)
    frozen = functional.freeze(ci)
    frozen_energy = frozen.energy(rotated)
    exact_energy, _ = functional.evaluate(rotated, gradient=False)
    assert frozen_energy == pytest.approx(energy, abs=1e-9)
    assert exact_energy == pytest.approx(energy, abs=1e-9)
    # But the rotated basis is a genuinely different coordinate system: the
    # excited J=0 level must move, i.e. the probe is not a no-op.
    _e_rot, ci_rot = functional.evaluate(rotated, gradient=False)
    assert ci_rot.state_energies[0] != pytest.approx(
        ci_rot.state_energies[1], abs=1e-3)


def test_screening_drops_small_weights(solved):
    """The screen is relative to the largest weight: keys below
    ``screen_tol * max|w|`` are dropped.  Handcrafted weights keep this
    independent of the fixture's weight distribution (a tiny closed-shell
    CI has only O(1) occupation weights, where any screen is a big change).
    """
    functional, captured = solved
    weights = {
        ("h", 1, -1, 1): 1.0,
        ("h", 2, -1, 2): 1.0,
        ("r", 0, (1, -1), (1, -1), (2, -1), (2, -1)): 4.0,
        ("r", 1, (1, -1), (1, -1), (1, -1), (2, -1)): 1e-9,
    }
    from atomx.methods.mcdhf.variational import FrozenCI
    fr = FrozenCI(functional, weights, screen_tol=1e-6)
    assert ("r", 1, (1, -1), (1, -1), (1, -1), (2, -1)) not in fr.weights
    assert len(fr.weights) == 3
    # The dropped key carried ~1e-9 weight, so the energy is untouched at
    # the freeze point to well below any convergence tolerance.
    e_full = FrozenCI(functional, weights, screen_tol=0.0).energy(
        captured["orbitals_dict"])
    e_screened = fr.energy(captured["orbitals_dict"])
    assert abs(e_full - e_screened) < 1e-8


def test_optimize_converges_same_energy_with_frozen_trials(solved):
    functional, captured = solved
    start = _mix(captured["orbitals_dict"], (1, -1), (2, -1), 0.1)
    orbitals_true, _ci_true, diag_true = optimize_orbitals(
        functional, start, max_iter=40, tolerance=1e-6,
        frozen_line_search=False)
    orbitals_frozen, _ci_frozen, diag_frozen = optimize_orbitals(
        functional, start, max_iter=40, tolerance=1e-6,
        frozen_line_search=True)
    assert diag_true["converged"] and diag_frozen["converged"]
    e_true, _ = functional.evaluate(orbitals_true, gradient=False)
    e_frozen, _ = functional.evaluate(orbitals_frozen, gradient=False)
    assert e_true == pytest.approx(e_frozen, abs=1e-6)
