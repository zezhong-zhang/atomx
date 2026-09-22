"""Tests for parity non-conservation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.properties.pnc import (
    pnc_amplitude_dirac_only,
    weak_charge_qw,
    weak_hamiltonian_radial_me,
)


def test_weak_charge_is_neutron_dominated():
    assert weak_charge_qw(55, 78) < -70.0


def test_pnc_amplitude_scaffold_raises_on_hydrogen():
    """Low-level PNC amplitude stays loud until the SOS implementation lands."""
    atom = Atom(Z=1, config="1s1", corrections="none")
    atom.solve()
    with pytest.raises(NotImplementedError, match="opposite-parity"):
        pnc_amplitude_dirac_only(atom._result, "1s1/2", "1s1/2", N=0)


def test_compute_pnc_high_level_api_scaffold_raises_not_typeerror():
    """High-level compute_pnc preserves the scaffold raise contract."""
    from atomx.methods.pnc import compute_pnc

    atom = Atom(Z=3, config="1s2 2s1", corrections="none")
    atom.solve()
    with pytest.raises(NotImplementedError, match="opposite-parity"):
        compute_pnc(atom, "2s1/2", "1s1/2", N=4)


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
