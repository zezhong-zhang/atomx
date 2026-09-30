"""Native reference-provider tests (HYP-904).

The native atomx solvers already satisfy ``AtomicResultLike``; wrapping one
must yield a valid payload exposing the radial-orbital surface with an
orthonormal (absent) overlap.  Uses a duck-typed mock result so the test
does not depend on running a full SCF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from atomx.providers.contract import Capability
from atomx.providers.native_adapter import NativeProvider, from_atomic_result
from atomx.state_protocols import AtomicResultLike


@dataclass
class _Subshell:
    label: str
    kappa: int
    occupation: float


@dataclass
class _MockResult:
    Z: int = 10
    method_name: str = "DHF"
    subshells: List[_Subshell] = field(default_factory=list)
    orbital_energies: Dict[str, float] = field(default_factory=dict)
    orbitals_raw: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = field(
        default_factory=dict
    )


def _make(relativistic=True):
    P = np.linspace(0, 1, 8)
    Q = 1e-2 * P if relativistic else None
    res = _MockResult(Z=10)
    res.subshells = [
        _Subshell("1s1/2", -1, 2.0),
        _Subshell("2s1/2", -1, 2.0),
        _Subshell("2p1/2", 1, 2.0),
        _Subshell("2p3/2", -2, 4.0),
    ]
    res.orbital_energies = {
        "1s1/2": -30.0,
        "2s1/2": -1.5,
        "2p1/2": -1.0,
        "2p3/2": -0.95,
    }
    res.orbitals_raw = {
        -1: [(P, Q, 2.0), (0.5 * P, Q, 2.0)],
        1: [(0.3 * P, Q, 2.0)],
        -2: [(0.4 * P, Q, 4.0)],
    }
    return res


def test_mock_result_is_atomic_result_like():
    assert isinstance(_make(), AtomicResultLike)


def test_native_provider_builds_valid_payload():
    payload = NativeProvider(_make()).build_payload()
    assert payload.Z == 10
    caps = payload.capabilities()
    assert Capability.RADIAL_ORBITALS in caps
    assert Capability.ORBITAL_ENERGIES in caps
    assert Capability.OCCUPATIONS in caps
    # native orbitals are orthonormal → overlap absent, S treated as I.
    assert Capability.OVERLAP_S not in caps
    assert payload.overlap is None
    assert payload.provenance.provider == "atomx-native"
    assert payload.provenance.method == "DHF"


def test_native_payload_is_atomic_result_like():
    payload = from_atomic_result(_make())
    assert isinstance(payload, AtomicResultLike)
    # orbital energies preserved (already Hartree).
    assert payload.orbital_energies["1s1/2"] == -30.0


def test_relativistic_flag_sets_soc_spinblock():
    rel = NativeProvider(_make(relativistic=True)).build_payload()
    assert rel.spin is not None and rel.spin.soc
    nonrel = NativeProvider(_make(relativistic=False)).build_payload()
    assert nonrel.spin is None


def test_non_atomic_result_rejected():
    import pytest

    with pytest.raises(TypeError):
        NativeProvider(object())


def test_zero_q_is_not_a_relativistic_declaration():
    result = _make()
    result.orbitals_raw = {k: [(p, np.zeros_like(p), occ) for p, q, occ in rows]
                           for k, rows in result.orbitals_raw.items()}
    result._equation = 'schrodinger'
    payload = NativeProvider(result).build_payload()
    assert payload.spin is None
    assert payload.metadata['equation_source'] == 'declared'
    del result._equation
    payload = NativeProvider(result).build_payload()
    assert payload.spin is None
    assert payload.metadata['equation_source'] == 'legacy_Q_heuristic'


def test_provider_preserves_declared_physics_and_basis():
    from atomx.calculation_spec import CalculationSpec
    result = _make(relativistic=False)
    result.calculation_spec = CalculationSpec(10, 'DHF', 'bspline',
        {'equation': 'dirac', 'exchange': 'hf'}, {}, {}, {'include_breit': True})
    result._bsp_data = {'ks': 9}
    payload = NativeProvider(result).build_payload()
    assert payload.spin.soc
    assert payload.provenance.basis == 'atomx-bspline'
    assert payload.metadata['calculation_spec'] == result.calculation_spec.to_dict()
