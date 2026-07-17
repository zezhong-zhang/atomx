"""Per-orbital QED shifts for scattering energetics (review M7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.corrections.orbital_shifts import (  # noqa: E402
    qed_orbital_shifts, shifted_orbital_energies)


@pytest.fixture(scope="module")
def fe17():
    a = Atom(Z=26, config="1s2 2s2 2p6", charge=16, numerics="bspline", breit=False, qed=False)
    a.solve()
    return a


def test_fe_k_shift_matches_fac_scale(fe17):
    """Fe 1s QED shift ~ +9 eV/electron (SE - Uehling); this is the
    documented -8 eV DR resonance-energy offset vs FAC-with-QED."""
    sh = qed_orbital_shifts(fe17._result)
    ev = sh["1s1/2"] * HARTREE_EV
    assert 7.0 <= ev <= 11.0
    # SE dominance ordering: s-states positive and decreasing with n;
    # p-states near zero.
    assert sh["1s1/2"] > sh["2s1/2"] > abs(sh["2p1/2"])


def test_shifted_energies_consistency(fe17):
    res = fe17._result
    sh = qed_orbital_shifts(res)
    eps = shifted_orbital_energies(res)
    for lab, e in res.orbital_energies.items():
        assert eps[lab] == pytest.approx(e + sh.get(lab, 0.0), abs=1e-14)
    # binding is reduced (1s less bound with QED)
    assert eps["1s1/2"] > res.orbital_energies["1s1/2"]
