"""Tests for parity non-conservation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.properties._pseudoscalar import pseudoscalar_angular
from atomx.properties.pnc import (
    PNCResult,
    _default_finite_nuclear_density,
    _pnc_unit_diagnostics,
    pnc_amplitude_dirac_only,
    weak_charge_qw,
    weak_hamiltonian_radial_me,
)


def test_weak_charge_is_neutron_dominated():
    assert weak_charge_qw(55, 78) < -70.0


def test_pnc_amplitude_runs_on_hydrogen():
    """Low-level PNC amplitude runs on H 1s1/2 -> 2s1/2 (no p intermediate)."""
    atom = Atom(Z=1, config="1s1", breit=False, qed=False)
    atom.solve()
    # H has no p_{1/2} virtuals in a minimal basis, so this returns a
    # PNCResult with zero amplitude (no intermediate states to sum over).
    result = pnc_amplitude_dirac_only(atom._result, "1s1/2", "1s1/2", N=0)
    assert result.E_pnc_au == 0.0
    assert result.N == 0
    assert result.k_pnc == 0.0
    assert result.provenance["n_intermediates"] == 0


def test_compute_pnc_high_level_api():
    """High-level compute_pnc accepts a solved Atom and returns PNCResult."""
    from atomx.methods.pnc import compute_pnc

    atom = Atom(Z=3, config="1s2 2s1", breit=False, qed=False)
    atom.solve()
    result = compute_pnc(atom, "2s1/2", "1s1/2", N=4)
    assert hasattr(result, "E_pnc_au")
    assert hasattr(result, "k_pnc")
    assert hasattr(result, "E_pnc_per_qw")
    assert result.N == 4
    assert np.isfinite(result.E_pnc_au)
    assert np.isfinite(result.E_pnc_per_qw)


def test_pnc_pseudoscalar_excludes_p_three_halves():
    """Weak pseudoscalar mixing allows s1/2-p1/2, not s1/2-p3/2."""
    assert pseudoscalar_angular(-1, +1) != 0.0
    assert pseudoscalar_angular(-1, -2) == 0.0


def test_weak_hamiltonian_radial_me_is_antisymmetric():
    bsp_data = {
        "gr": np.array([0.1, 0.2, 0.3]),
        "grw": np.array([0.1, 0.1, 0.1]),
        "nv": 3,
        "ks": 1,
        "ns": 3,
        "bsp_p": np.eye(3)[:, :, None],
        "bsp_q": np.eye(3)[:, :, None],
        "ksp": 1,
        "ksq": 1,
        "nsp": 3,
        "nsq": 3,
    }
    P_a = np.array([1.0, 0.0, 0.0])
    Q_a = np.array([0.0, 1.0, 0.0])
    P_b = np.array([0.0, 1.0, 0.0])
    Q_b = np.array([1.0, 0.0, 0.0])
    ab = weak_hamiltonian_radial_me(
        P_a, Q_a, P_b, Q_b, bsp_data,
        rho_nuc_func=lambda _r: 1.0,
        Q_W=-1.0,
    )
    ba = weak_hamiltonian_radial_me(
        P_b, Q_b, P_a, Q_a, bsp_data,
        rho_nuc_func=lambda _r: 1.0,
        Q_W=-1.0,
    )
    assert ab == pytest.approx(-ba)


def test_weak_hamiltonian_radial_me_runs_on_arrays():
    bsp_data = {
        "gr": np.array([0.1, 0.2, 0.3]),
        "grw": np.array([0.1, 0.1, 0.1]),
        "nv": 3,
        "ks": 1,
        "ns": 3,
        "bsp_p": np.eye(3)[:, :, None],
        "bsp_q": np.eye(3)[:, :, None],
        "ksp": 1,
        "ksq": 1,
        "nsp": 3,
        "nsq": 3,
    }
    P_a = np.array([1.0, 0.0, 0.0])
    Q_a = np.array([0.0, 1.0, 0.0])
    P_b = np.array([0.0, 1.0, 0.0])
    Q_b = np.array([1.0, 0.0, 0.0])
    me = weak_hamiltonian_radial_me(
        P_a, Q_a, P_b, Q_b, bsp_data,
        rho_nuc_func=lambda _r: 1.0,
        Q_W=-1.0,
    )
    assert np.isfinite(me)


def test_pnc_unit_conversion_conventional_coefficient():
    E_pnc_au = -5.5e-11
    Q_W = -73.2
    N = 78
    k_pnc, per_qw, status = _pnc_unit_diagnostics(E_pnc_au, Q_W, N)

    assert status == "ok"
    assert k_pnc == pytest.approx(-(N / Q_W) * E_pnc_au / 1.0e-11)
    assert E_pnc_au == pytest.approx(k_pnc * 1.0e-11 * (-Q_W / N))
    assert per_qw == pytest.approx(E_pnc_au / (Q_W * 1.0e-11))
    assert per_qw == pytest.approx(-k_pnc / N)
    assert abs(k_pnc) != pytest.approx(abs(per_qw))


def test_pnc_units_without_neutron_number_do_not_substitute():
    E_pnc_au = -5.5e-11
    Q_W = -73.2
    k_pnc, per_qw, status = _pnc_unit_diagnostics(E_pnc_au, Q_W, None)
    result = PNCResult(
        E_pnc_au=E_pnc_au,
        k_pnc=k_pnc,
        E_pnc_per_qw=per_qw,
        E_pnc_iea=per_qw,
        Q_W=Q_W,
        N=None,
        provenance={"k_pnc_status": status},
    )

    assert result.k_pnc is None
    assert result.N is None
    assert result.provenance["k_pnc_status"] == "k_pnc_unavailable_without_N"
    assert result.E_pnc_per_qw == pytest.approx(E_pnc_au / (Q_W * 1.0e-11))


def test_isotope_dependence_records_N_QW_density_and_k():
    bsp_data = _resolved_nuclear_grid()
    rho_133, prov_133 = _default_finite_nuclear_density(55, 78, bsp_data)
    rho_135, prov_135 = _default_finite_nuclear_density(55, 80, bsp_data)
    Q_133 = weak_charge_qw(55, 78)
    Q_135 = weak_charge_qw(55, 80)
    E_pnc_au = -5.5e-11
    k_133, _, _ = _pnc_unit_diagnostics(E_pnc_au, Q_133, 78)
    k_135, _, _ = _pnc_unit_diagnostics(E_pnc_au, Q_135, 80)
    result = PNCResult(E_pnc_au=E_pnc_au, k_pnc=k_133, Q_W=Q_133, N=78)

    assert Q_133 != Q_135
    assert prov_133["density_A"] == 133
    assert prov_135["density_A"] == 135
    assert prov_133["density_fermi_c_bohr"] != pytest.approx(
        prov_135["density_fermi_c_bohr"]
    )
    assert rho_133(1.0e-4) != pytest.approx(rho_135(1.0e-4))
    assert k_133 != pytest.approx(k_135)
    assert result.N == 78


def test_default_density_is_3d_normalized_on_real_grid():
    atom = Atom(
        Z=3, config="1s2 2s1", numerics="bspline", ns=60,
        breit=False, qed=False,
    )
    atom.solve()
    bsp_data = atom._result._bsp_data
    rho, provenance = _default_finite_nuclear_density(3, 4, bsp_data)
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    rho_grid = np.vectorize(rho)(gr.ravel()).reshape(gr.shape)
    norm = 4.0 * np.pi * np.sum(grw * gr * gr * rho_grid)

    assert norm == pytest.approx(1.0, abs=1e-6)
    assert provenance["density_model"] == "fermi"
    assert provenance["density_A"] == 7
    assert provenance["density_N"] == 4
    assert provenance["density_norm_residual"] <= 1e-6


def test_default_density_requires_N_when_QW_is_supplied():
    atom = Atom(Z=3, config="1s2 2s1", breit=False, qed=False)
    atom.solve()
    with pytest.raises(ValueError, match="requires N"):
        pnc_amplitude_dirac_only(atom._result, "2s1/2", "1s1/2", Q_W=-3.0)


def test_default_density_rejects_unresolved_grid():
    bsp_data = {"gr": np.array([10.0, 20.0, 30.0]), "grw": np.ones(3)}
    with pytest.raises(ValueError, match="normalize finite nuclear density"):
        _default_finite_nuclear_density(55, 78, bsp_data)


def test_pnc_provenance_keeps_both_sos_terms_observable():
    atom = Atom(Z=3, config="1s2 2s1", breit=False, qed=False)
    atom.solve()
    result = pnc_amplitude_dirac_only(atom._result, "2s1/2", "1s1/2", N=4)

    assert "term_initial" in result.provenance
    assert "term_final" in result.provenance
    assert np.isfinite(result.provenance["term_initial"])
    assert np.isfinite(result.provenance["term_final"])


def _resolved_nuclear_grid():
    gr = np.geomspace(1.0e-8, 2.0e-3, 800)
    edges = np.empty(gr.size + 1)
    edges[1:-1] = np.sqrt(gr[:-1] * gr[1:])
    edges[0] = gr[0] ** 2 / edges[1]
    edges[-1] = gr[-1] ** 2 / edges[-2]
    return {"gr": gr, "grw": np.diff(edges)}
