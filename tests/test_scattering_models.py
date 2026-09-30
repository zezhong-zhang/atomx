"""Composable scattering model tests."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.numerics.quadrature import GridResourceError, make_fine_grid
from atomx.scattering import ContinuumModel, PotentialModel
from atomx.scattering.distorted_wave.solver import (
    _extract_scf_potential,
    _shooting_points,
)
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
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    model = PotentialModel.static_hartree_exchange(exchange="fac", n_radial=200, r_max=8.0)
    V = model.build_callable(atom._result)
    r = np.array([16.0, 32.0])
    values = V(r)
    assert values[0] < 0.0
    assert values[1] < 0.0
    np.testing.assert_allclose(values[0] / values[1], 2.0, rtol=1.0e-12)


def test_dw_local_exchange_potential_matches_bspline_shape():
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_PP, V_QQ = _extract_scf_potential(atom, include_local_exchange=True)
    ns = atom._bsp_data["ns"]
    assert V_PP.shape == (ns, ns)
    assert V_QQ.shape == (ns, ns)


def test_eii_accepts_continuum_model_for_dw_exchange():
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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


def test_eii_rejects_nan_inf_inputs():
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
    atom.solve()
    with pytest.raises(ValueError, match="finite"):
        eii_cross_section(atom._result, np.array([100.0, np.nan]), method="DW")
    with pytest.raises(ValueError, match="finite integer"):
        eii_cross_section(atom._result, np.array([100.0]), method="DW", kappa_max=np.inf)


def test_eii_dw_batch_wave_failure_fails_closed(monkeypatch):
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
    atom.solve()

    def fail_many(requests, *args, **kwargs):
        return kwargs["interp_grid"], {req: None for req in requests}

    monkeypatch.setattr(
        "atomx.numerics.shooting.dirac_continuum.solve_dirac_continuum_many",
        fail_many,
    )
    with pytest.raises(RuntimeError, match="fail-closed.*kappa=.*n_grid"):
        eii_cross_section(
            atom._result,
            np.array([100.0]),
            method="DW",
            kappa_max=1,
            n_outgoing_energies=3,
        )


def test_eii_dw_diagnostic_mode_reports_skipped_waves(monkeypatch):
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
    atom.solve()

    def fail_many(requests, *args, **kwargs):
        return kwargs["interp_grid"], {req: None for req in requests}

    monkeypatch.setattr(
        "atomx.numerics.shooting.dirac_continuum.solve_dirac_continuum_many",
        fail_many,
    )
    out = eii_cross_section(
        atom._result,
        np.array([100.0]),
        method="DW",
        kappa_max=1,
        n_outgoing_energies=3,
        continuum_failure_mode="diagnostic",
    )
    assert out.provenance["continuum_failure_mode"] == "diagnostic"
    failures = out.provenance["continuum_failures"]
    assert failures
    assert {"kappa", "energy_eV", "r_max_a0", "shooting_n_grid"} <= failures[0].keys()


@pytest.mark.parametrize("radius", [35., 100.])
def test_eii_dw_uses_shooting_mesh_from_continuum_model(radius):
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
    atom.solve()
    knots = atom._result._bsp_data["t"].copy()
    orbital_energy = dict(atom.orbital_energies)
    model = ContinuumModel.shooting(n_grid=700, r_max=radius)
    out = eii_cross_section(
        atom._result,
        np.array([100.0]),
        method="DW",
        kappa_max=2,
        n_outgoing_energies=3,
        continuum_model=model,
        radial_refinement=2,
    )
    orb_info = next(iter(out.provenance["per_orbital"].values()))
    assert orb_info["shooting_n_grid"] == 700
    assert orb_info["shooting_r_max_a0"] == radius
    assert orb_info["bound_r_max_a0"] == knots[-1]
    np.testing.assert_array_equal(atom._result._bsp_data["t"], knots)
    assert atom.orbital_energies == orbital_energy
    assert orb_info["radial_refinement"] == 2
    assert out.provenance["collision_operator"] == "instantaneous_coulomb"
    assert out.provenance["transverse_exchange"] is False
    assert out.sigma_total_cm2[0] > 0.0


def test_fine_grid_cap_hit_raises_resource_error():
    with pytest.raises(GridResourceError, match=r"required_points.*cap=10.*p=.*r_max"):
        make_fine_grid(50.0, 2.0, z_inner=2.0, max_points=10)


def test_shooting_points_cap_hit_raises_resource_error():
    with pytest.raises(GridResourceError, match=r"required_points=.*cap=10.*p=.*r_max"):
        _shooting_points(50.0, 2.0, max_points=10)


def test_rr_accepts_continuum_model_without_changing_default_bspline():
    atom = Atom(Z=2, config="1s1", engine="bspline", corrections="none")
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
