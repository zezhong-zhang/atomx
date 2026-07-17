"""Unified Atom-facade API across every result flavor.

Exercises the same set of accessors on:

- :class:`SCFResult` (B-spline DHF — ``Atom(Z).solve()``).
- :class:`MCDHFResult` (multi-config — ``Atom(Z, configs=[...]).solve()``).
- :class:`ShootingResult` (ODE shooting — ``Atom(Z, numerics='shooting', exchange='slater').solve()``).

The intent is to lock in the Phase-3 promise: every ``Atom`` accessor
delegates uniformly through ``self._result`` regardless of which method
produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.corrections import CorrectionResult, CorrectionStack
from atomx.methods.dirac_fock import SCFResult
from atomx.methods.mcdhf.solver import MCDHFResult
from atomx.numerics.shooting.solver import ShootingResult
from atomx.results import AtomicResult, Waves


# ---------------------------------------------------------------------------
# Fixtures: one Atom of each flavor.  Module-scoped so the SCF only runs once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def he_dhf():
    return Atom(Z=2).solve()


@pytest.fixture(scope="module")
def he_shooting():
    return Atom(Z=2, numerics="shooting", exchange="slater").solve()


@pytest.fixture(scope="module")
def be_mcdhf():
    return Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"]).solve()


# ---------------------------------------------------------------------------
# Result-flavor introspection
# ---------------------------------------------------------------------------


def test_dhf_result_is_scfresult(he_dhf):
    assert isinstance(he_dhf._result, SCFResult)
    assert he_dhf._mcdhf_result is None


def test_mcdhf_result_is_mcdhfresult(be_mcdhf):
    assert isinstance(be_mcdhf._result, MCDHFResult)
    # Back-compat alias: _mcdhf_result returns the same object.
    assert be_mcdhf._mcdhf_result is be_mcdhf._result


def test_shooting_result_is_shootingresult(he_shooting):
    assert isinstance(he_shooting._result, ShootingResult)
    assert he_shooting._mcdhf_result is None


def test_every_result_satisfies_atomicresult_protocol(he_dhf, be_mcdhf, he_shooting):
    for atom in (he_dhf, be_mcdhf, he_shooting):
        assert isinstance(atom._result, AtomicResult), (
            f"{type(atom._result).__name__} fails the AtomicResult protocol."
        )


# ---------------------------------------------------------------------------
# Uniform protocol delegators (must work on every flavor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atom_fixture", ["he_dhf", "be_mcdhf", "he_shooting"])
def test_basic_protocol_accessors(atom_fixture, request):
    atom = request.getfixturevalue(atom_fixture)
    assert atom.converged
    assert atom.iterations > 0
    assert atom.total_energy < 0.0
    assert atom.total_energy_eV == pytest.approx(
        atom.total_energy * 27.211386245988, rel=1e-6
    )
    assert isinstance(atom.orbital_energies, dict) and atom.orbital_energies
    assert isinstance(atom.energy_breakdown, dict)
    assert atom.theory_level is not None
    assert isinstance(atom.method_name, str) and atom.method_name
    assert atom.coupling_scheme is not None


@pytest.mark.parametrize("atom_fixture", ["he_dhf", "be_mcdhf", "he_shooting"])
def test_to_table_summary(atom_fixture, request):
    atom = request.getfixturevalue(atom_fixture)
    t = atom.to_table()
    assert t["converged"] is True
    assert t["Z"] == atom.Z
    assert t["theory_level"] is not None
    assert t["coupling"] is not None
    assert t["total_energy_Ha"] == pytest.approx(atom.total_energy, rel=1e-12)


# ---------------------------------------------------------------------------
# Multi-state behaviour
# ---------------------------------------------------------------------------


def test_dhf_state_energies_is_singleton(he_dhf):
    se = he_dhf.state_energies
    assert se.shape == (1,)
    assert se[0] == pytest.approx(he_dhf.total_energy, rel=1e-12)
    assert he_dhf.excitation_energies("cm-1").tolist() == [0.0]


def test_mcdhf_state_energies_has_multiple(be_mcdhf):
    se = be_mcdhf.state_energies
    assert se.shape[0] >= 2
    # Ground state coincides with total_energy.
    assert se[0] == pytest.approx(be_mcdhf.total_energy, rel=1e-12)


def test_shooting_state_energies_is_singleton(he_shooting):
    se = he_shooting.state_energies
    assert se.shape == (1,)


# ---------------------------------------------------------------------------
# waves + density
# ---------------------------------------------------------------------------


def test_dhf_waves_have_orbitals(he_dhf):
    w = he_dhf.waves
    assert isinstance(w, Waves)
    assert "1s1/2" in w.orbitals


def test_mcdhf_waves_have_meaningful_occupations(be_mcdhf):
    w = be_mcdhf.waves
    # Be 1s2 2s2 has the dominant CSF, so 1s and 2s are ~doubly occupied.
    occs = {lbl: orb.occupation for lbl, orb in w.orbitals.items()}
    assert occs["1s1/2"] == pytest.approx(2.0, abs=0.05)
    assert occs["2s1/2"] == pytest.approx(2.0, abs=0.05)


def test_dhf_density_is_positive_finite(he_dhf):
    grid = np.array([0.5, 1.0, 2.0])
    rho = he_dhf.density(grid)
    assert rho.shape == grid.shape
    assert np.all(np.isfinite(rho))
    assert np.all(rho > 0)


def test_mcdhf_density_is_positive_finite(be_mcdhf):
    grid = np.array([0.3, 0.5, 1.0, 2.0])
    rho = be_mcdhf.density(grid)
    assert rho is not None
    assert rho.shape == grid.shape
    assert np.all(np.isfinite(rho))
    assert np.all(rho >= 0)
    # The 1s peak (~0.3 Bohr for Be) sits well above the long-tail
    # density at 2 Bohr.
    assert rho[0] > rho[-1]


def test_shooting_density_known_limitation(he_shooting):
    # Current shooting backend doesn't surface per-orbital wavefunctions,
    # so density() returns None.  Pin this so the limitation is explicit.
    assert he_shooting.density() is None


# ---------------------------------------------------------------------------
# corrections() unified accessor
# ---------------------------------------------------------------------------


def test_dhf_corrections_runs_and_returns_dict(he_dhf):
    c = he_dhf.corrections(CorrectionStack(se_backend="mohr",
                                              include_vp=True,
                                              include_breit=True))
    expected = {"self_energy[mohr]", "uehling_vp", "breit"}
    assert expected <= set(c)
    for name in expected:
        assert isinstance(c[name], CorrectionResult)


def test_shooting_corrections_run_grid_natively(he_shooting):
    """Dirac shooting results are no longer skipped: the adapters
    evaluate Breit/SE/VP directly on the shooting grid orbitals
    (see tests/test_shooting_corrections.py for cross-engine gates)."""
    c = he_shooting.corrections(CorrectionStack(se_backend="mohr",
                                                   include_vp=True,
                                                   include_breit=True))
    expected = {"self_energy[mohr]", "uehling_vp", "breit"}
    assert expected <= set(c)
    assert "_skipped" not in c
    for name in expected:
        assert isinstance(c[name], CorrectionResult)
        assert np.isfinite(c[name].energy_shift)


# ---------------------------------------------------------------------------
# Unit-aware accessors
# ---------------------------------------------------------------------------


def test_total_energy_as_in_eV(he_dhf):
    pytest.importorskip("pint")
    q = he_dhf.total_energy_as("eV")
    # Just check that pint produced a Quantity with eV-compatible
    # dimensionality and the magnitude matches a hand conversion.
    assert "electron_volt" in str(q.units) or "eV" in str(q.units)
    assert float(q.magnitude) == pytest.approx(
        he_dhf.total_energy * 27.211386245988, rel=1e-6
    )


def test_orbital_energy_as_raises_for_unknown_label(he_dhf):
    with pytest.raises(KeyError):
        he_dhf.orbital_energy_as("9z9/2", "eV")


# ---------------------------------------------------------------------------
# plot3d shortcut
# ---------------------------------------------------------------------------


def test_plot3d_shortcut(he_dhf):
    pytest.importorskip("plotly.graph_objects")
    fig = he_dhf.plot3d("1s1/2", n_grid=16)
    assert fig is not None


# ---------------------------------------------------------------------------
# Per-orbital access: Atom.orbital(label), Atom.orbitals, Atom.orbital_table()
# ---------------------------------------------------------------------------


def test_orbital_lookup_returns_orbitalwavefunction(he_dhf):
    from atomx.results import OrbitalWavefunction

    # Single match: returns an OrbitalWavefunction directly.
    orb = he_dhf.orbital("1s")
    assert isinstance(orb, OrbitalWavefunction)
    assert orb.label == "1s1/2"
    assert orb.n == 1
    assert orb.l == 0
    assert orb.kappa == -1
    assert orb.j == 0.5
    assert orb.s == 0.5
    assert orb.l_symbol == "s"
    assert orb.occupation == pytest.approx(2.0, abs=0.01)
    assert orb.P.shape == he_dhf.waves.grid.shape
    assert orb.Q is not None and orb.Q.shape == orb.P.shape
    assert orb.is_relativistic
    assert orb.expectation_r(he_dhf.waves.grid) == pytest.approx(0.75, abs=0.2)


def test_orbital_lookup_multiple_match_returns_dict(he_dhf):
    """When label matches multiple j components, orbital() returns a dict."""
    # He has only 1s1/2, so let's use Ne for a real multi-match.
    ne = Atom(Z=10).solve()
    result = ne.orbital("2p")
    assert isinstance(result, dict)
    assert "2p1/2" in result and "2p3/2" in result


def test_orbital_lookup_raises_for_unknown_label(he_dhf):
    with pytest.raises(KeyError, match="not found"):
        he_dhf.orbital("9z9/2")


def test_orbitals_dict_contains_every_subshell(he_dhf):
    orbs = he_dhf.orbitals
    assert isinstance(orbs, dict)
    assert "1s1/2" in orbs


def test_orbital_table_summary_shape(he_dhf):
    rows = he_dhf.orbital_table()
    assert isinstance(rows, list) and rows
    row = rows[0]
    expected_keys = {
        "label", "n", "l", "l_symbol", "kappa", "j", "s",
        "occupation", "energy_Ha", "is_relativistic", "n_grid_points",
    }
    assert expected_keys <= set(row)


def test_mcdhf_orbital_carries_guerra_occupation(be_mcdhf):
    orb = be_mcdhf.orbital("2s1/2")
    # Be ground state has 2s1/2 ~ doubly occupied (with a tiny 2p mix
    # bleeding off some weight).
    assert orb.occupation == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# Hyperfine + Zeeman: gracefully NotImplemented for unsupported flavors
# ---------------------------------------------------------------------------


def test_hyperfine_constants_works_on_dhf(he_dhf):
    hf = he_dhf.hyperfine_constants(isotope=(2, 3))  # 3-He
    # 3-He has only s-electrons in DHF ground state — hyperfine A
    # should be a finite real for 1s1/2.
    assert "1s1/2" in hf.A_MHz
    assert np.isfinite(hf.A_MHz["1s1/2"])


def test_hyperfine_constants_raises_on_shooting(he_shooting):
    with pytest.raises(NotImplementedError, match="hyperfine"):
        he_shooting.hyperfine_constants(isotope=(2, 3))
