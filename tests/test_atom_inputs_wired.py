"""The grouped inputs are wired to real physics, not recorded and ignored.

Four constructor inputs used to be accepted and then dropped on the floor:

* ``plasma_ne``/``plasma_Te`` (now ``environment=Environment(...)``) were
  stored on the atom and never read — every "plasma" run returned exactly
  the vacuum number;
* ``Sampling.r_max`` / ``Sampling.n_grid`` (now reachable through
  ``basis=Basis(sampling=...)``) never reached the solvers;
* ``qed_backend`` was ignored on multi-configuration runs, whose MCDHF path
  resolved ``DEFAULT_SE_METHOD`` regardless;
* ``slater_method`` was accepted on the B-spline engine, which evaluates its
  Slater integrals with its own quadrature (that one now raises — see
  ``test_atom_input_surface.py::test_slater_method_only_exists_where_it_is_read``).

Each test below measures the effect rather than asserting the attribute
round-trips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Basis, Corrections, Environment, Sampling  # noqa: E402
from atomx.plasma.screening import (  # noqa: E402
    debye_length,
    debye_potential,
    ion_sphere_radius,
    screening_shift,
    stewart_pyatt_potential,
    stewart_pyatt_screening_length,
)

NE_PLASMA = dict(plasma_ne=1e22, plasma_Te=100.0)


#: Ne^2+ (8 electrons) is the panel ion: its net charge keeps the screened
#: potential attractive at long range, so both engines solve the same
#: physical problem.  A *neutral* atom at this density is a different
#: story — see test_nuclear_screening_ionizes_a_neutral_atom_at_high_density.
ION = dict(Z=10, charge=2)


def _unscreened(**kw):
    return Atom(corrections="none", **{**ION, **kw}).solve()


def _screened(**kw):
    return Atom(corrections="none",
                environment=Environment(**NE_PLASMA),
                **{**ION, **kw}).solve()


# ----------------------------------------------------------------------
# Plasma screening (Debye / Stewart-Pyatt) inside the SCF
# ----------------------------------------------------------------------

def test_screening_shift_is_the_model_potential_minus_the_bare_nucleus():
    """The wired shift is the module's own potential, not a lookalike."""
    r = np.linspace(0.01, 25.0, 64)
    Z = 10
    debye = screening_shift(Z, r, 1e22, 100.0, "debye")
    assert np.allclose(debye, debye_potential(Z, 1e22, 100.0)(r) + Z / r)
    stewart = screening_shift(Z, r, 1e22, 100.0, "stewart_pyatt", z_ion=2)
    assert np.allclose(
        stewart, stewart_pyatt_potential(Z, 2, 1e22, 100.0)(r) + Z / r)
    # The shift is finite at the origin — unlike V_model, which diverges
    # as -Z/r — and starts at the continuum shift Z/lambda_D, decaying as
    # Z/r beyond the screening length.  Boundary electrons therefore move
    # up by ~Z/lambda_D: that is the ionisation potential depression.
    lam = debye_length(1e22, 100.0)
    assert screening_shift(Z, np.array([0.0]), 1e22, 100.0)[0] == \
        pytest.approx(Z / lam, rel=1e-3)
    assert np.all(np.diff(debye) < 0.0)          # monotone, no barrier
    assert debye[0] == pytest.approx(Z / lam, rel=0.1)
    assert debye[-1] < debye[0]                  # decays as Z/r beyond lambda_D


def test_plasma_raises_every_orbital_by_the_predicted_amount():
    """Screening lifts the orbitals; the size is the model's Z/lambda_D."""
    bare = _unscreened()
    dense = _screened()
    lam = debye_length(1e22, 100.0)
    continuum = 10 / lam
    for label, e_bare in bare.orbital_energies.items():
        shift = dense.orbital_energies[label] - e_bare
        assert shift > 0.0, label
        # Debye-Huckel: dV(0) = Z/lambda_D and dV -> Z/lambda_D at large r,
        # so the whole spectrum moves up by that scale, relaxed by the
        # self-consistent readjustment of the orbitals (measured 0.685-0.691
        # Ha for Ne^2+ at ne=1e22 cm^-3, Te=100 eV vs Z/lambda_D = 0.712 Ha).
        assert 0.5 * continuum < shift < 1.05 * continuum, (label, shift)
    # Eight electrons, each lifted by ~0.69 Ha: the ionisation energy drops
    # by the first-order sum over the occupied orbitals,
    #     sum_i w_i <dV>_i     (dV from screening_shift, density and weights
    #                           from the *unscreened* orbitals),
    # which is the independent side of the comparison.  Measured
    # sum_i w_i <dV>_i = 5.566352 Ha vs the SCF 5.566119 Ha — 4.2e-5
    # relative, so the old "5.5 ± 0.5 Ha" band (±9 %) was ~2000x wider than
    # the physics it was bounding.
    assert pytest.approx(_first_order_total(bare, "debye"), rel=1e-3, abs=0.0) == (
        dense.total_energy - bare.total_energy)


def _first_order_total(bare, model):
    """sum_i w_i <dV>_i over the occupied unscreened orbitals (Hartree)."""
    r = bare.r
    dV = screening_shift(10, r, 1e22, 100.0, model, z_ion=2)
    total = 0.0
    for orb in bare.orbitals.values():
        density = np.asarray(orb.P) ** 2 + np.asarray(orb.Q) ** 2
        total += float(orb.occupation) * float(np.trapezoid(dV * density, r))
    return total


def test_plasma_shift_matches_first_order_theory_from_the_module():
    """Compare the SCF shift with <dV> on the *unscreened* orbitals.

    ``dV`` comes straight from :func:`atomx.plasma.screening.screening_shift`
    and the density from the unscreened run, so this is a genuine
    model-vs-answer comparison rather than a self-consistency check.
    """
    bare = _unscreened()
    dense = _screened()
    r = bare.r
    for label in ("1s1/2", "2p3/2"):
        orb = bare.orbitals[label]
        density = np.asarray(orb.P) ** 2 + np.asarray(orb.Q) ** 2
        dV = screening_shift(10, r, 1e22, 100.0, "debye")
        first_order = float(np.trapezoid(dV * density, r))
        scf_shift = dense.orbital_energies[label] - bare.orbital_energies[label]
        # measured: 1s 0.7079 -> 0.6910, 2p3/2 0.7078 -> 0.6858 (the SCF
        # relaxes the orbitals, so it lands 2-3 % below first order).
        assert scf_shift == pytest.approx(first_order, rel=0.10), label
        assert scf_shift < first_order


def test_plasma_is_applied_on_both_engines():
    """Both engines must solve the same screened Hamiltonian."""
    out = {}
    for engine in ("shooting", "bspline"):
        bare = _unscreened(engine=engine)
        dense = _screened(engine=engine)
        shift = dense.orbital_energies["1s1/2"] - bare.orbital_energies["1s1/2"]
        assert shift > 0.5, engine
        assert dense.total_energy - bare.total_energy > 5.0, engine
        out[engine] = (shift, dense.total_energy - bare.total_energy)
    assert out["shooting"][0] == pytest.approx(out["bspline"][0], abs=1e-5)
    assert out["shooting"][1] == pytest.approx(out["bspline"][1], abs=1e-5)


def test_nuclear_screening_ionizes_a_neutral_atom_at_high_density():
    """A neutral atom has no net charge to screen — and the model says so.

    Screening only the nucleus leaves ``V_eff -> (N - Z e^{-r/lambda})/r``,
    which is repulsive for a neutral atom: at ne = 1e22 cm^-3, Te = 100 eV
    the model predicts complete pressure ionisation of Ne, and the B-spline
    solver reports exactly that instead of returning the core as if it were
    the atom.
    """
    from atomx.methods.dirac_fock import UnboundSubshellError
    with pytest.raises(UnboundSubshellError, match="not bound"):
        Atom(10, corrections="none", engine="bspline",
             environment=Environment(plasma_ne=1e22, plasma_Te=100.0)).solve()


def test_stewart_pyatt_and_debye_are_different_models():
    """The SP shift is the Debye one times the analytic length ratio.

    At n_e = 1e22 cm^-3, T_e = 100 eV the coupling parameter is
    u = R_IS/lambda_D = 0.488, where the Stewart-Pyatt interpolation length
    ``stewart_pyatt_screening_length(lambda_D, R_IS) = 14.312 a_0`` is
    *longer* than ``lambda_D = 14.048 a_0``: SP is the less attractive
    model (measured first-order sums 4.9762 vs 5.5664 Ha — the audit's
    expectation that SP exceeds Debye at fixed (n_e, T_e) is wrong here and
    the sign is asserted from the measured ratio, not assumed).  Both SCF
    shifts are then checked against first-order theory on the unscreened
    orbitals, and their ratio against the first-order ratio (measured
    0.89400 vs 0.89391, 1e-4 relative).
    """
    bare = _unscreened()
    debye = Atom(10, charge=2, corrections="none",
                 environment=Environment(screening="debye", **NE_PLASMA)
                 ).solve()
    stewart = Atom(10, charge=2, corrections="none",
                   environment=Environment(screening="stewart_pyatt",
                                           **NE_PLASMA)).solve()
    assert stewart.total_energy != debye.total_energy
    assert stewart.screening_model == "stewart_pyatt"

    lam_d = debye_length(1e22, 100.0)
    r_is = ion_sphere_radius(2, 1e22)
    lam_sp = stewart_pyatt_screening_length(lam_d, r_is)
    assert pytest.approx(0.48801, rel=1e-3, abs=0.0) == r_is / lam_d
    assert lam_sp > lam_d  # SP is the weaker screening at this coupling

    shift_debye = debye.total_energy - bare.total_energy
    shift_stewart = stewart.total_energy - bare.total_energy
    first_debye = _first_order_total(bare, "debye")
    first_stewart = _first_order_total(bare, "stewart_pyatt")
    assert pytest.approx(first_debye, rel=1e-3, abs=0.0) == shift_debye
    assert pytest.approx(first_stewart, rel=1e-3, abs=0.0) == shift_stewart
    # the models differ by the first-order ratio: the relaxation cancels
    assert pytest.approx(first_stewart / first_debye, rel=1e-3, abs=0.0) == (
        shift_stewart / shift_debye)
    assert shift_stewart < shift_debye


def test_plasma_is_recorded_in_provenance():
    bare = _unscreened()
    assert "plasma" not in bare.provenance
    dense = _screened()
    assert dense.provenance["plasma"] == {"plasma_ne": 1e22,
                                          "plasma_Te": 100.0,
                                          "screening": "debye"}


def test_no_plasma_means_no_screening_shift():
    """The default is exactly the old (unscreened) number."""
    bare = Atom(10, corrections="none")
    assert bare._screening_shift() is None
    assert Atom(10, environment=Environment())._screening_shift() is None


# ----------------------------------------------------------------------
# The radial box and the shooting grid
# ----------------------------------------------------------------------

def test_explicit_r_max_reaches_both_solvers():
    shoot = Atom(10, corrections="none", basis=Basis(r_max=120.0)).solve()
    assert shoot.engine == "shooting"
    assert shoot.r[-1] == pytest.approx(120.0)
    spline = Atom(10, corrections="none", engine="bspline",
                  basis=Basis(r_max=120.0)).solve()
    assert spline.r[-1] == pytest.approx(120.0, abs=1.0)


def test_sampling_tier_r_max_reaches_the_box():
    """``Sampling(r_max=...)`` is read — it used to be dropped."""
    atom = Atom(10, corrections="none",
                basis=Basis(sampling=Sampling(r_max=120.0)))
    assert atom.r_max == 120.0
    atom.solve()
    assert atom.r[-1] == pytest.approx(120.0)
    # Sampling.quick/publication set r_max explicitly too; standard's
    # 75.0 is the schema default and deliberately leaves the engine's own
    # auto box (max(75, 5 n^2/Z)) in charge.
    quick = Atom(10, corrections="none",
                 basis=Basis(sampling=Sampling.quick)).solve()
    assert quick.r[-1] == pytest.approx(60.0)
    public = Atom(10, corrections="none",
                  basis=Basis(sampling=Sampling.publication)).solve()
    assert public.r[-1] == pytest.approx(100.0)
    assert Atom(10, basis=Basis(sampling=Sampling.standard)).r_max is None


def test_radial_box_reaches_the_hamiltonian():
    """A diffuse orbital is only bound if the box is wide enough.

    H 6s needs ~180 Bohr; the default B-spline knot grid stops at ~51, so
    the auto-box run misses the 6s eigenvalue by 0.29 eV.  Widening the box
    reproduces the box-independent (shooting) eigenvalue — proof that
    ``Basis(r_max=...)`` reaches the Hamiltonian, not just a report field.
    """
    narrow = Atom(1, config="6s1", engine="bspline", corrections="none").solve()
    wide = Atom(1, config="6s1", engine="bspline", corrections="none",
                basis=Basis(r_max=120.0)).solve()
    assert narrow.r[-1] < 60.0
    assert wide.r[-1] == pytest.approx(120.0, abs=1.0)
    assert narrow.total_energy - wide.total_energy > 0.005
    shooting = Atom(1, config="6s1", engine="shooting",
                    corrections="none").solve()
    assert wide.total_energy == pytest.approx(shooting.total_energy, abs=1e-6)


def test_too_small_a_box_fails_loudly():
    with pytest.raises(Exception, match="box too small"):
        Atom(1, config="6s1", engine="shooting",
             basis=Basis(r_max=60.0)).solve()


def test_sampling_tier_n_grid_reaches_the_shooting_grid():
    coarse = Atom(2, corrections="none", engine="shooting",
                  basis=Basis(sampling=Sampling(n_grid=500))).solve()
    fine = Atom(2, corrections="none", engine="shooting").solve()
    assert coarse._n_grid == 500
    assert fine._n_grid is None
    assert "n_grid=500" in coarse.resolved_inputs
    # The SGRID really changed: the quadrature moves the total energy at
    # the 1e-8 Ha level (measured 1.39e-8 Ha for He at n_grid 500 vs 1000).
    assert abs(coarse.total_energy - fine.total_energy) > 1e-9


def test_solve_time_sampling_still_only_carries_tol_and_max_iter():
    """The basis fields of a solve-time tier are a documented non-goal.

    ``solve(sampling=...)`` drives convergence control only; the radial box
    and the shooting grid belong to construction
    (``basis=Basis(sampling=...)``), where the solver is built.
    """
    atom = Atom(10, corrections="none")
    atom.solve(sampling=Sampling.quick)
    assert atom.r_max is None            # not taken from the quick tier
    assert atom._n_grid is None
    assert "n_grid=1000" in atom.resolved_inputs


# ----------------------------------------------------------------------
# The multi-configuration self-energy backend
# ----------------------------------------------------------------------

CONFIGS = ["1s2 2s2 2p6", "1s2 2s2 2p5 3s1"]


def test_qed_backend_reaches_the_multiconfig_solver():
    """``Corrections(qed_backend=...)`` used to be dropped by MCDHFSolver."""
    auto = Atom(10, config=CONFIGS).solve()
    welton = Atom(10, config=CONFIGS,
                  corrections=Corrections(qed_backend="welton")).solve()
    assert welton.qed_backend == "welton"
    se_auto = auto.energy_components["E_SE"]
    se_welton = welton.energy_components["E_SE"]
    # Same vacuum polarisation (same physics, same orbitals) but a
    # different self energy: measured 0.012119 -> 0.017502 Ha at Z=10.
    assert se_welton != se_auto
    assert abs(se_welton - se_auto) > 5e-3
    assert (welton.energy_components["E_VP"]
            == pytest.approx(auto.energy_components["E_VP"]))
    assert welton.total_energy != auto.total_energy


def test_multiconfig_default_backend_is_still_auto():
    """The wiring must not move a run that did not ask for a backend."""
    auto = Atom(10, config=CONFIGS).solve()
    mss = Atom(10, config=CONFIGS,
               corrections=Corrections(qed_backend="mss")).solve()
    # auto resolves to MSS/Mohr for Z <= 36, so the numbers agree exactly.
    assert mss.total_energy == auto.total_energy


def test_multiconfig_backend_reaches_the_shooting_engine_too():
    """The Guerra-CI route can run either engine; both take the backend."""
    cfgs = ["1s2 2s2", "1s2 2s1 2p1"]
    auto = Atom(6, config=cfgs).solve()
    welton = Atom(6, config=cfgs,
                  corrections=Corrections(qed_backend="welton")).solve()
    assert welton.energy_components["E_SE"] \
        != auto.energy_components["E_SE"]
