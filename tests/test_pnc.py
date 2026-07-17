"""Tests for parity non-conservation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.properties._pseudoscalar import pseudoscalar_angular
from atomx.properties.pnc import (
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
    assert result.provenance["n_intermediates"] == 0


def test_compute_pnc_high_level_api():
    """High-level compute_pnc accepts a solved Atom and returns PNCResult."""
    from atomx.methods.pnc import compute_pnc

    atom = Atom(Z=3, config="1s2 2s1", breit=False, qed=False)
    atom.solve()
    result = compute_pnc(atom, "2s1/2", "1s1/2", N=4)
    assert hasattr(result, "E_pnc_au")
    assert hasattr(result, "E_pnc_iea")
    assert np.isfinite(result.E_pnc_au)
    assert np.isfinite(result.E_pnc_iea)


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
