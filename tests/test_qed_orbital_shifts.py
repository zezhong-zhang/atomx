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


@pytest.mark.snapshot
def test_fe_k_shift_matches_fac_scale(fe17):
    """Fe^16+ 1s QED shift = +3.93 eV/electron (SE 4.28 - Uehling 0.35).

    Frozen self-regression, NOT an external gate: the FAC numbers in the
    line above are prose — the assertion is a 3.5–4.4 eV window on a
    value (3.9274 eV) that commit 200d657 re-pinned in the same change
    that fixed the kernel, so it is atomx's own output wearing a band.
    It does still discriminate the two defect classes it was written for
    (the 3.9x Uehling kernel error moves it to 2.91 eV, the 4.7x SE
    table error to ~14 eV), which is why it is kept rather than deleted.
    To make it a B-class gate, assert the FAC Fe XVII 1s self-energy
    literal with rel=0.02 — that number is not in the repo yet.
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
