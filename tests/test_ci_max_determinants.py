"""Correlation(max_determinants=...) lifts the exact-CI block cap.

Uses the measured lanthanide case from the Eu3+ probe series (2026-09-30):
reference [Xe]4f6 + singles into {5d, 6s} needs 2256 determinants for the
2J=0 block — above the default cap of 2000, below it when overridden.
Marked slow: the solve itself takes minutes.
"""
import pytest

from atomx import Atom, Correlation
from atomx.methods.mcdhf.excitations import generate_sd_excitations

CORE = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p"]
REF = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f6"


def _eu3_configs():
    cfgs = generate_sd_excitations(REF, ["5d", "6s"], core_orbitals=CORE,
                                   max_excitations=1)
    assert len(cfgs) == 3
    return cfgs


@pytest.mark.slow
def test_default_cap_rejects_oversized_block():
    cfgs = _eu3_configs()
    atom = Atom(
        Z=63, charge=3, config=cfgs,
        correlation=Correlation(j_resolved_csfs=True,
                                radial_optimization="fac_frozen"),
        corrections="none", engine="bspline",
    )
    with pytest.raises(Exception) as ei:
        atom.solve()
    assert "determinant" in str(ei.value).lower()


@pytest.mark.slow
def test_max_determinants_override_runs_and_keeps_ground_identity():
    cfgs = _eu3_configs()
    atom = Atom(
        Z=63, charge=3, config=cfgs,
        correlation=Correlation(j_resolved_csfs=True,
                                radial_optimization="fac_frozen",
                                max_determinants=4000),
        corrections="none", engine="bspline",
    )
    atom.solve()
    res = atom.result
    # Same ground-state symmetry as the single-configuration run:
    # 7F0 at 2J = 0, and the 7F ladder starts below 400 cm^-1.
    assert res.state_J2[0] == 0
    de = res.excitation_energies(unit="cm-1")
    assert 200.0 < de[1] < 450.0
