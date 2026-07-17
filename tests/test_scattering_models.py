"""Composable scattering model tests."""

from __future__ import annotations

import numpy as np

from atomx import Atom
from atomx.scattering import ContinuumModel, PotentialModel
from atomx.scattering.distorted_wave.solver import _extract_scf_potential
from atomx.scattering.electron_ionization import eii_cross_section
from atomx.scattering.radiative_recombination import rr_cross_section


def test_potential_model_bare_callable():
    model = PotentialModel.bare()
    V = model.build_callable(None, Z=2)
    np.testing.assert_allclose(V(np.array([1.0, 2.0])), np.array([-2.0, -1.0]))


def test_continuum_model_fac_residual_core_shape():
    model = ContinuumModel.fac_residual_core(n_grid=800, r_max=40.0)
    assert model.method == "shooting"
    assert model.shooting_n_grid == 800
    assert model.shooting_r_max == 40.0
    assert model.potential is not None
    assert model.potential.include_exchange
    assert model.potential.exchange == "fac"
    assert model.potential.fac_self_interaction_correction
    assert model.potential.latter_correction


def test_fac_residual_core_potential_uses_coulomb_tail():
    atom = Atom(Z=2, config="1s2", numerics="bspline", breit=False, qed=False)
    atom.solve()
    model = PotentialModel.static_hartree_exchange(exchange="fac", n_radial=200, r_max=8.0)
    V = model.build_callable(atom._result)
    r = np.array([16.0, 32.0])
    values = V(r)
    assert values[0] < 0.0
    assert values[1] < 0.0
    np.testing.assert_allclose(values[0] / values[1], 2.0, rtol=1.0e-12)


def test_dw_local_exchange_potential_matches_bspline_shape():
    atom = Atom(Z=2, config="1s2", numerics="bspline", breit=False, qed=False)
    atom.solve()
    V_PP, V_QQ = _extract_scf_potential(atom, include_local_exchange=True)
    ns = atom._bsp_data["ns"]
    assert V_PP.shape == (ns, ns)
    assert V_QQ.shape == (ns, ns)


def test_eii_accepts_continuum_model_for_dw_exchange():
    atom = Atom(Z=2, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()
    model = ContinuumModel(
        method="bspline",
        potential=PotentialModel.static_hartree_exchange(),
    )
    out = eii_cross_section(
        atom._result,
        np.array([100.0]),
        method="DW",
        kappa_max=2,
        n_outgoing_energies=3,
        continuum_model=model,
    )
    assert out.provenance["include_local_exchange"] is True
    assert out.sigma_total_cm2[0] > 0.0


def test_eii_dw_uses_shooting_mesh_from_continuum_model():
    atom = Atom(Z=2, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()
    model = ContinuumModel.shooting(n_grid=700, r_max=35.0)
    out = eii_cross_section(
        atom._result,
        np.array([100.0]),
        method="DW",
        kappa_max=2,
        n_outgoing_energies=3,
        continuum_model=model,
    )
    orb_info = next(iter(out.provenance["per_orbital"].values()))
    assert orb_info["shooting_n_grid"] == 700
    assert orb_info["shooting_r_max_a0"] == 35.0
    assert out.sigma_total_cm2[0] > 0.0


def test_rr_accepts_continuum_model_without_changing_default_bspline():
    atom = Atom(Z=2, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()
    model = ContinuumModel.bspline()
    out = rr_cross_section(
        atom._result,
        np.array([100.0]),
        capture_orbitals=["1s1/2"],
        continuum_model=model,
    )
    assert out.provenance["continuum_model"] == model
    assert out.sigma_total_a0_sq[0] > 0.0
