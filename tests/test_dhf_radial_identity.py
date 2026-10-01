"""Recorded radial identities of specific DHF branches, independent of campaign gates.

Nonlocal HF has no general local-potential node theorem. Each signature below
is scoped to a reproduced configuration and constraint mode. The stationary
Sc X energy is an independent DBSR_HF reference; other cases compare branches.
"""

import numpy as np
import pytest

from atomx import Atom, Basis
from atomx.methods.dirac_fock import DiracFock
from atomx.types import Exchange


def has_node_signature(result, expected):
    found = set()
    for shell in result.subshells:
        key = (shell.n, shell.l)
        if key not in expected:
            continue
        found.add(key)
        values = np.asarray(result.waves.orbitals[shell.label].P).ravel()
        significant = values[np.abs(values) > 0.01 * np.max(np.abs(values))]
        nodes = np.count_nonzero(
            np.signbit(significant[:-1]) != np.signbit(significant[1:])
        )
        if nodes != expected[key]:
            return False
    return found == set(expected)


def energies_ordered(result):
    for kappa in {s.kappa for s in result.subshells}:
        shells = sorted(
            (s for s in result.subshells if s.kappa == kappa), key=lambda s: s.n
        )
        if np.any(np.diff([result.orbital_energies[s.label] for s in shells]) <= 0):
            return False
    return True


@pytest.mark.parametrize("z", [10, 11])
def test_neon_and_sodium_core_p_branches_preserve_distinct_node_patterns(z):
    config = "1s2 2s2 2p5 3p1"
    results = {
        start: Atom(Z=z, config=config, engine="bspline", basis=Basis()).solve(
            DiracFock(
                core_hole_stationary=False,
                breit=True,
                qed=True,
                nuclear_mass_me=40000.0,
                starting_potential=start,
            ),
            max_iter=200,
        )
        for start in ("gsz", "tfda")
    }
    assert all((r.converged for r in results.values()))
    assert not has_node_signature(
        results["gsz"], {(3, 1): 1} if z == 10 else {(2, 1): 0, (3, 1): 1}
    )
    assert has_node_signature(
        results["tfda"], {(3, 1): 1} if z == 10 else {(2, 1): 0, (3, 1): 1}
    )
    assert results["tfda"].total_energy < results["gsz"].total_energy - 0.05


def test_sc_x_stationary_branch_matches_independent_dbsr_reference():
    config = "1s2 2s2 2p6 3p1 4p1"
    result = Atom(Z=21, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(nuclear_mass_number=45, starting_potential="tfda"),
        tol=1e-09,
        max_iter=250,
    )
    assert result.converged
    dc = result.energy_components["E_1e"] + result.energy_components["E_2e"]
    assert abs(dc - -729.2225473824016) < 3e-07
    assert has_node_signature(result, {(3, 1): 2, (4, 1): 2})
    assert energies_ordered(result)


def test_mn_ii_core_d_branches_have_distinct_node_patterns():
    config = "1s2 2s2 2p6 3s2 3p6 3d5 4d1"
    mass = 100120.9277024357
    bad = Atom(Z=25, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="gsz",
            scf_damping=0.35,
        ),
        max_iter=100,
    )
    good = Atom(Z=25, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="tfda",
            scf_damping=0.35,
        ),
        max_iter=100,
    )
    assert bad.converged and good.converged
    assert energies_ordered(bad) and energies_ordered(good)
    assert not has_node_signature(bad, {(3, 2): 0, (4, 2): 1})
    assert has_node_signature(good, {(3, 2): 0, (4, 2): 1})
    assert good.total_energy < bad.total_energy - 0.1


def test_co_ii_slater_seed_recovers_core_d_node_pattern():
    config = "1s2 2s2 2p6 3s2 3p6 3d7 4d1"
    mass = 107401.64133110592
    bad = Atom(Z=27, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="tfda",
            scf_damping=0.35,
        ),
        max_iter=100,
    )
    seed = Atom(Z=27, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            exchange=Exchange.SLATER,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="tfda",
            scf_damping=0.35,
        ),
        max_iter=120,
        tol=0.001,
    )
    assert seed.converged
    good = Atom(Z=27, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="tfda",
            scf_damping=0.35,
            seed_orbitals=seed.orbitals_raw,
        ),
        max_iter=100,
    )
    assert bad.converged and good.converged
    assert energies_ordered(bad) and energies_ordered(good)
    assert not has_node_signature(bad, {(3, 2): 0, (4, 2): 1})
    assert has_node_signature(good, {(3, 2): 0, (4, 2): 1})


def test_rb_x_ni_like_core_d_branches_have_distinct_node_patterns():
    config = "1s2 2s2 2p6 3s2 3p6 3d9 4d1"
    mass = 154747.72386288052
    bad = Atom(Z=37, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="gsz",
        ),
        max_iter=100,
    )
    good = Atom(Z=37, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="tfda",
            scf_damping=0.35,
        ),
        max_iter=100,
    )
    assert bad.converged and good.converged
    assert energies_ordered(bad) and energies_ordered(good)
    assert not has_node_signature(bad, {(3, 2): 0, (4, 2): 1})
    assert has_node_signature(good, {(3, 2): 0, (4, 2): 1})
    assert good.total_energy < bad.total_energy - 11


def test_sr_iii_starting_potential_selects_distinct_4p_5p_branches():
    config = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p5 5p1"
    mass = 160204.12890588387
    bad = Atom(Z=38, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="gsz",
        ),
        max_iter=100,
    )
    good = Atom(Z=38, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(
            core_hole_stationary=False,
            breit=True,
            qed=True,
            nuclear_mass_me=mass,
            starting_potential="tfda",
            scf_damping=0.35,
        ),
        max_iter=100,
    )
    assert bad.converged and good.converged
    assert energies_ordered(bad) and energies_ordered(good)
    assert not has_node_signature(bad, {(4, 1): 2, (5, 1): 3})
    assert has_node_signature(good, {(4, 1): 2, (5, 1): 3})
    assert good.total_energy < bad.total_energy - 0.5
