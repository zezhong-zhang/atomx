"""Solver-agnostic protocol tests."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pytest

from atomx.scattering.protocols import (
    AtomicResultLike,
    SubshellLike,
    get_orbital,
    grid_data,
    list_occupied_orbitals,
)

# ── Mock backend with no SCF ──────────────────────────────


@dataclass
class MockSubshell:
    label: str
    kappa: int
    occupation: float


@dataclass
class MockResult:
    Z: int = 10
    subshells: List[MockSubshell] = field(default_factory=list)
    orbital_energies: Dict[str, float] = field(default_factory=dict)
    orbitals_raw: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = field(
        default_factory=dict
    )


def _make_mock():
    P = np.linspace(0, 1, 10)
    Q = np.zeros(10)
    res = MockResult(Z=10)
    res.subshells = [
        MockSubshell("1s1/2", -1, 2.0),
        MockSubshell("2s1/2", -1, 2.0),
        MockSubshell("2p1/2", 1, 2.0),
        MockSubshell("2p3/2", -2, 4.0),
    ]
    res.orbital_energies = {
        "1s1/2": -10.0,
        "2s1/2": -1.5,
        "2p1/2": -1.0,
        "2p3/2": -0.95,
    }
    res.orbitals_raw = {
        -1: [(P, Q, 2.0), (P * 0.5, Q, 2.0)],
        1: [(P * 0.3, Q, 2.0)],
        -2: [(P * 0.4, Q, 4.0)],
    }
    return res


def test_mock_satisfies_protocol():
    """Duck-typed mock satisfies AtomicResultLike."""
    res = _make_mock()
    assert isinstance(res, AtomicResultLike)
    for sh in res.subshells:
        assert isinstance(sh, SubshellLike)


def test_get_orbital_returns_tuple():
    res = _make_mock()
    P, Q, kappa, eps, occ = get_orbital(res, "1s1/2")
    assert kappa == -1
    assert eps == -10.0
    assert occ == 2.0
    assert P.shape == Q.shape


def test_get_orbital_handles_multiple_subshells_same_kappa():
    """Multiple n with same κ map to distinct orbitals_raw entries."""
    res = _make_mock()
    P_1s, _, _, _, _ = get_orbital(res, "1s1/2")
    P_2s, _, _, _, _ = get_orbital(res, "2s1/2")
    assert not np.allclose(P_1s, P_2s)


def test_get_orbital_unknown_label():
    res = _make_mock()
    with pytest.raises(KeyError):
        get_orbital(res, "9z9/2")


def test_list_occupied_orbitals():
    res = _make_mock()
    occ = list_occupied_orbitals(res)
    labels = {label for (label, _, _, _) in occ}
    assert labels == {"1s1/2", "2s1/2", "2p1/2", "2p3/2"}


def test_grid_data_returns_none_if_absent():
    res = _make_mock()
    assert grid_data(res) is None


def test_real_scfresult_satisfies_protocol():
    """The actual SCFResult is also AtomicResultLike."""
    from atomx import Atom
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    assert isinstance(atom._result, AtomicResultLike)
    P, Q, kappa, eps, occ = get_orbital(atom._result, "1s1/2")
    assert kappa == -1
    assert eps < 0
