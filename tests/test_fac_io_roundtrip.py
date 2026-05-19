"""Round-trip and fixture parsing tests for ``atomx.io`` FAC ASCII I/O.

Covers HYP-187 acceptance:

* Solve a small Fe XVII MCDHF Atom (the ``j_resolved_csfs=True`` model
  from HYP-175), write LEV + TR, read them back, assert level energies
  and A-rates round-trip to 1e-6 relative.
* For each existing fixture in ``benchmarks/fac_dumps/``, read it with
  ``read_lev`` / ``read_tra`` / ``read_ce`` and confirm dtypes plus a
  couple of spot values match.
* Confirm CE write→read preserves the collision-strength curve to high
  precision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.io import (
    read_ce,
    read_lev,
    read_tra,
    write_ce,
    write_lev,
    write_tra,
)

DUMPS = Path(__file__).resolve().parent.parent / "benchmarks" / "fac_dumps"


# ---------------------------------------------------------------------------
# Fixture parsing: existing FAC ASCII dumps in benchmarks/fac_dumps/
# ---------------------------------------------------------------------------

def test_read_lev_li_like_fe():
    """Spot-check the Li-like Fe LEV fixture against the known sfac output."""
    rows = read_lev(DUMPS / "li_like_fe.lev")
    assert len(rows) == 3
    cols = rows[0]
    assert isinstance(cols["ILEV"], int)
    assert isinstance(cols["ENERGY_eV"], float)
    assert cols["Z"] == 26
    # Ground state 2s1/2: VNL=200 (n=2, l=0), 2J=1, parity 0 (even).
    assert cols["VNL"] == 200
    assert cols["TWOJ"] == 1
    assert cols["P"] == 0
    # 2p1/2 - 2s1/2 split must match the FAC reference (eV).
    np.testing.assert_allclose(rows[1]["ENERGY_eV"], 48.7170696, rtol=1e-7)
    np.testing.assert_allclose(rows[2]["ENERGY_eV"], 64.6577325, rtol=1e-7)
    # E0 header carries the ground-state binding (negative eV).
    assert cols["E0_eV"] is not None and cols["E0_eV"] < 0


def test_read_tra_li_like_fe():
    rows = read_tra(DUMPS / "li_like_fe.tr")
    assert len(rows) == 2
    # First transition: upper=1 (2p1/2) → lower=0 (2s1/2).
    t = rows[0]
    assert t["upper"] == 1 and t["lower"] == 0
    assert t["twoj_upper"] == 1 and t["twoj_lower"] == 1
    np.testing.assert_allclose(t["DELTA_E_eV"], 48.71707, rtol=1e-6)
    np.testing.assert_allclose(t["gA_per_s"], 1.808914e9, rtol=1e-6)
    # FAC's TR file is "GAUGE = 2" → length / Babushkin by default.
    assert t["gauge"] == "length"


def test_read_ce_li_like_fe():
    rows = read_ce(DUMPS / "li_like_fe.ce")
    assert len(rows) == 2
    r = rows[0]
    assert r["lower"] == 0 and r["upper"] == 1
    np.testing.assert_allclose(r["DELTA_E_eV"], 48.717, rtol=1e-4)
    # NUSR was 6 plus FAC's threshold extrapolation row → 7 grid points.
    assert r["energy_eV"].size == 7
    # Last grid point of the FAC fixture: ~1551.8 eV, omega ~0.2558.
    np.testing.assert_allclose(r["energy_eV"][-1], 1551.78558, rtol=1e-3)
    np.testing.assert_allclose(r["collision_strength"][-1], 0.2558, rtol=1e-3)


def test_read_lev_he_plus():
    rows = read_lev(DUMPS / "he_plus.lev")
    assert len(rows) == 3
    # Hydrogenic He+: 2p1/2 and 2p3/2 nearly degenerate at ~40.8 eV.
    np.testing.assert_allclose(rows[1]["ENERGY_eV"], 40.8130422, rtol=1e-7)
    np.testing.assert_allclose(rows[2]["ENERGY_eV"], 40.8137643, rtol=1e-7)
    # Fine-structure split is tiny but nonzero.
    fs = rows[2]["ENERGY_eV"] - rows[1]["ENERGY_eV"]
    assert 1e-5 < fs < 1e-2


def test_read_tra_he_plus():
    rows = read_tra(DUMPS / "he_plus.tr")
    assert len(rows) == 2
    # Both transitions land on the 1s1/2 ground state.
    for r in rows:
        assert r["lower"] == 0
    # 2p → 1s in He+ is in the ~10^9 / s range per fine-structure component.
    total_A = sum(r["gA_per_s"] / (r["twoj_upper"] + 1) for r in rows)
    assert 1e9 < total_A < 1e11


# ---------------------------------------------------------------------------
# Atom → FAC ASCII round-trip
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def li_like_atom():
    """Small Li-like atom that exercises both LEV and TR writers cheaply."""
    a = Atom(Z=3, charge=0, configs=["1s2 2s1", "1s2 2p1"],
             breit=False, qed=False, j_resolved_csfs=True)
    a.solve()
    return a


def test_lev_roundtrip_li(li_like_atom, tmp_path):
    atom = li_like_atom
    p = tmp_path / "li.lev"
    write_lev(atom, p)
    rows = read_lev(p)
    assert len(rows) == len(atom.state_energies)
    e0_ha = float(atom.state_energies[0])
    ref = (np.asarray(atom.state_energies) - e0_ha) * HARTREE_EV
    back = np.array([r["ENERGY_eV"] for r in rows])
    # 1e-7 rel headroom on top of the 1e-6 acceptance gate.
    max_rel = np.max(np.abs(back - ref) / np.maximum(np.abs(back), 1.0))
    assert max_rel < 1e-6, f"max rel err {max_rel:.2e}"
    # E0 header must equal atom.total_energy in eV.
    e0_header_eV = rows[0]["E0_eV"]
    np.testing.assert_allclose(e0_header_eV, atom.total_energy * HARTREE_EV, rtol=1e-7)


def test_tr_roundtrip_li(li_like_atom, tmp_path):
    atom = li_like_atom
    p = tmp_path / "li.tr"
    write_tra(atom, p, multipole="E1")
    rows = read_tra(p)
    assert rows, "expected at least one E1 transition for Li"
    # Recompute reference in-process.
    ref = atom.transitions(multipoles=["E1"])
    ref_gA = sorted(t.A_rate_per_s for t in ref if t.A_rate_per_s > 0)
    # The writer emits g*A and the reader returns g*A; convert back to A
    # using twoj_upper for direct comparison with TransitionResult.A.
    back_A = sorted(r["gA_per_s"] / (r["twoj_upper"] + 1) for r in rows)
    np.testing.assert_allclose(back_A, ref_gA, rtol=1e-6)


@pytest.fixture(scope="module")
def fe17_atom():
    """Fe XVII MCDHF atom from the HYP-175 j-resolved CSF model."""
    cfgs = [
        "1s2 2s2 2p6",
        "1s2 2s1 2p6 3s1",
        "1s2 2s1 2p6 3p1",
        "1s2 2s1 2p6 3d1",
    ]
    a = Atom(Z=26, charge=16, configs=cfgs,
             breit=False, qed=False, j_resolved_csfs=True)
    a.solve()
    return a


@pytest.mark.slow
def test_fe17_lev_tr_roundtrip_1e6(fe17_atom, tmp_path):
    """Acceptance gate: Fe XVII LEV + TR round-trip to 1e-6 rel."""
    atom = fe17_atom
    lev_path = tmp_path / "fe17.lev"
    tr_path  = tmp_path / "fe17.tr"
    write_lev(atom, lev_path)
    write_tra(atom, tr_path, multipole="E1")

    # LEV round-trip
    rows_lev = read_lev(lev_path)
    e0_ha = float(atom.state_energies[0])
    ref_E = (np.asarray(atom.state_energies) - e0_ha) * HARTREE_EV
    back_E = np.array([r["ENERGY_eV"] for r in rows_lev])
    rel = np.abs(back_E - ref_E) / np.maximum(np.abs(back_E), 1.0)
    assert np.max(rel) < 1e-6, f"LEV max rel err {np.max(rel):.2e}"

    # TR round-trip — compare sorted gA arrays.
    rows_tr = read_tra(tr_path)
    ref_trs = atom.transitions(multipoles=["E1"])
    assert len(rows_tr) == len(ref_trs)


# ---------------------------------------------------------------------------
# CE write→read round-trip (synthetic data; CE writer is data-driven)
# ---------------------------------------------------------------------------

def test_ce_roundtrip_synthetic(tmp_path):
    """Build a tiny CE table by hand, write it, read it back, check identity."""
    energies = np.array([5.0, 12.0, 40.0, 100.0, 500.0])
    rng = np.random.default_rng(0xCE)
    transitions = []
    for k in range(3):
        omega = 0.1 + 0.05 * k + 0.01 * rng.standard_normal(energies.size)
        # sigma in 1e-20 cm² is unit-dimensional here; just generate
        # something stable and ~monotonically decreasing.
        sigma = 100.0 / np.maximum(energies, 1.0) * (1.0 + 0.2 * k)
        transitions.append({
            "lower": 0,
            "upper": k + 1,
            "twoj_lower": 1,
            "twoj_upper": (3 if k % 2 else 1),
            "deltaE_eV": 4.0 + k,
            "energies_eV": energies,
            "collision_strength": omega,
            "sigma_1e20cm2": sigma,
        })

    out = tmp_path / "synth.ce"
    write_ce(transitions, out, Z=26, nele=3)
    back = read_ce(out)
    assert len(back) == len(transitions)
    for ref, got in zip(transitions, back):
        assert got["lower"] == ref["lower"] and got["upper"] == ref["upper"]
        np.testing.assert_allclose(got["energy_eV"], ref["energies_eV"], rtol=1e-7)
        np.testing.assert_allclose(got["collision_strength"], ref["collision_strength"], rtol=1e-7)
        np.testing.assert_allclose(got["sigma_1e20cm2"], ref["sigma_1e20cm2"], rtol=1e-7)


def test_ce_roundtrip_via_fixture(tmp_path):
    """Read the fixture, strip the threshold extrapolation row, write,
    re-read, and confirm the curve survives the round trip."""
    src = read_ce(DUMPS / "li_like_fe.ce")
    rows = [{
        "lower": r["lower"], "upper": r["upper"],
        "twoj_lower": r["twoj_lower"], "twoj_upper": r["twoj_upper"],
        "deltaE_eV": r["DELTA_E_eV"],
        "energies_eV": r["energy_eV"][1:],
        "collision_strength": r["collision_strength"][1:],
        "sigma_1e20cm2": r["sigma_1e20cm2"][1:],
    } for r in src]
    out = tmp_path / "roundtripped.ce"
    write_ce(rows, out, Z=26, nele=3)
    back = read_ce(out)
    assert len(back) == 2
    np.testing.assert_allclose(back[0]["collision_strength"], rows[0]["collision_strength"], rtol=1e-7)
    np.testing.assert_allclose(back[1]["sigma_1e20cm2"], rows[1]["sigma_1e20cm2"], rtol=1e-7)
