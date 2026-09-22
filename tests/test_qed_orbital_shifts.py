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
    qed_orbital_shifts,
    shifted_orbital_energies,
)


@pytest.fixture(scope="module")
def fe17():
    a = Atom(
             Z=26,
             config="1s2 2s2 2p6",
             charge=16,
             engine="bspline",
             corrections="none",
        )
    a.solve()
    return a


def test_fe_k_shift_matches_fac_scale(fe17):
    """Fe^16+ 1s QED shift = +3.93 eV/electron (SE 4.28 - Uehling 0.35).

    Re-pinned 2026-09-19.  The previous 7-11 eV window was satisfied by
    two cancelling errors: the self-energy table was 4.7x too large at
    Z=26 (20.2 eV instead of 4.28 eV) and ``qed_orbital_shifts`` divided
    the already per-electron SE by the occupancy (2 for the 1s), halving
    it.  The self-energy now comes from FAC's F(Zα) and the occupancy
    division is gone, so this is a real external-scale check.
    """
    sh = qed_orbital_shifts(fe17._result)
    ev = sh["1s1/2"] * HARTREE_EV
    assert 3.5 <= ev <= 4.4
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
