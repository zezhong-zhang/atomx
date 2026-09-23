"""End-to-end: atomx shooting with ``exchange='gu'`` vs FAC reference.

Reference total-binding energies come from ``sfac`` runs captured during
Task 0 (``docs/plans/2026-04-20-fac-calibration/calibration.csv``).  The
panel is a 13-row *sample* spanning Z = 2..92 (Z = 2, 3, 4, 6, 10, 18,
26 x2, 36, 54, 74, 82, 92) — not every Z in that range.

Flags must match the reference (2026-09-22 physics-audit remediation)
--------------------------------------------------------------------
``extract_AHX.py:56-59`` generated the ``E0_eV`` column with
``SetSE(0)``, ``SetVP(0)``, ``SetMS(0,0)`` and ``SetBreit(0)`` — i.e. **no
self-energy, no vacuum polarisation, no Breit**.  The test used to run
``Atom(..., corrections="qed")`` against that QED-free panel, so part of
every residual was a flag mismatch rather than an atomx accuracy error:
the atomx QED shift alone is 370.0 eV at Z=74 and 960.7 eV at Z=92, the
same size as the gate itself (1e-3 x 494345 eV = 494 eV at Z=92).  The run
is now ``corrections="none"``, i.e. the same operators as the reference.

The two panels are different physics — ``results_fac_structure.csv`` (used
by ``test_fac_structure_snapshot.py``) *is* a QED-on/Breit-off FAC run, this
calibration panel is not.

Gate sizing
-----------
``MEASURED_REL`` below records the matched-flag residual |ΔE|/|E| of every
panel row, measured 2026-09-22.  The gate is **1.5 x the measured value for
that row**: tight enough that any growth is caught, and strictly tighter
than the tiered 1e-3 (light) / 2e-3 (panel) gates it replaces on *every*
row — including Z=92, whose residual is dominated by a non-QED DFS
local-exchange/screening model difference of 466.8 eV (9.4e-4 relative)
that matched flags only reduce from 9.99e-4.

The residual is what is left after the flags match: the Gu local-exchange
model itself, the shooting-vs-FAC numerics, and at Z >= 74 the Latter /
screening-tail treatment.  It is *not* a self-pin: the reference column is
FAC 1.1.5's own output.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from atomx import Atom
from atomx.constants import HARTREE_EV

CALIBRATION = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "plans"
    / "2026-04-20-fac-calibration"
    / "calibration.csv"
)

#: Matched-flag residual |ΔE|/|E| per panel row, measured 2026-09-22 with
#: ``corrections="none"`` — the operators ``extract_AHX.py`` used for the
#: reference (SetSE/SetVP/SetMS/SetBreit all 0).  Keys are the CSV ``tag``
#: column, so a row added to the panel fails loudly here instead of silently
#: re-using another row's gate.
MEASURED_REL = {
    "Z002_N02_1s2": 1.3444e-04,
    "Z003_N03_1s2_2s1": 4.1562e-05,
    "Z004_N04_1s2_2s2": 5.3180e-05,
    "Z006_N06_1s2_2s2_2p2": 6.4561e-04,
    "Z010_N10_1s2_2s2_2p6": 1.2123e-05,
    "Z018_N18_1s2_2s2_2p6_3s2_3p6": 7.2315e-07,
    "Z026_N10_1s2_2s2_2p6": 3.2731e-06,
    "Z026_N12_1s2_2s2_2p6_3s2": 2.9810e-06,
    "Z036_N18_1s2_2s2_2p6_3s2_3p6": 9.2595e-06,
    "Z054_N10_1s2_2s2_2p6": 4.6682e-05,
    "Z074_N10_1s2_2s2_2p6": 2.2248e-04,
    "Z082_N18_1s2_2s2_2p6_3s2_3p6": 3.6010e-04,
    "Z092_N10_1s2_2s2_2p6": 9.4426e-04,
}
#: Gate = this multiple of the measured residual.  1.5x leaves room for FP
#: drift while still failing on any real accuracy loss.
GATE_MARGIN = 1.5
#: Fallback for a panel row with no recorded measurement (re-measure!).
UNMEASURED_GATE = 1.0e-3


def _load_reference(max_Z: int | None = None) -> list[dict]:
    """Load panel rows from the Task 0 calibration CSV.

    Heavy atoms (Z >= 74) previously hung the shooting SCF for
    minutes (``_slater_integral``'s power-series ODE had no
    step-halving cap).  After that fix every panel row runs in
    ~0.1-0.2 s warm, so we exercise the full Z=2..92 panel.
    """
    if not CALIBRATION.exists():
        return []
    with CALIBRATION.open() as f:
        rows = list(csv.DictReader(f))
    if max_Z is not None:
        rows = [r for r in rows if int(r["Z"]) <= max_Z]
    return rows


_REFERENCE = _load_reference()


def _nel_from_config(cfg: str) -> int:
    total = 0
    for tok in cfg.split():
        try:
            total += int(tok[2:])
        except ValueError:
            continue
    return total


# Parametrised over the Task 0 calibration panel.
_IDS = [f"Z{row['Z']}_N{int(float(row['N_fac']))}" for row in _REFERENCE]


def test_calibration_reference_present_and_spans_z2_to_z92():
    """The reference file must exist and cover the claimed Z range.

    This replaces the silent self-skip: a wrong/missing path now fails
    loudly instead of reporting "skipped".
    """
    assert CALIBRATION.exists(), f"reference file missing: {CALIBRATION}"
    Zs = [int(r["Z"]) for r in _REFERENCE]
    assert min(Zs) == 2 and max(Zs) == 92
    assert len(_REFERENCE) >= 13
    assert all(float(r["E0_eV"]) < 0.0 for r in _REFERENCE)


@pytest.mark.external_fac
@pytest.mark.parametrize("row", _REFERENCE, ids=_IDS or None)
def test_atomx_fac_total_energy_matches_sfac(row):
    Z = int(row["Z"])
    cfg = row["config"]
    nel = _nel_from_config(cfg)
    e0_ref_eV = float(row["E0_eV"])
    charge = max(Z - nel, 0)

    # The reference column was produced with SetSE(0)/SetVP(0)/SetMS(0,0)/
    # SetBreit(0) (extract_AHX.py:56-59), so the atomx run must use the same
    # operators: corrections="none" (no QED, no Breit).  Running with
    # corrections="qed" here compared two different physics questions.
    atom = Atom(
                Z=Z,
                charge=charge,
                config=cfg,
                engine="shooting",
                exchange="gu",
                corrections="none",
           )
    atom.solve()
    e_atomx_eV = atom.total_energy * HARTREE_EV

    rel = abs(e_atomx_eV - e0_ref_eV) / abs(e0_ref_eV)
    measured = MEASURED_REL.get(row["tag"])
    gate = (GATE_MARGIN * measured) if measured is not None else UNMEASURED_GATE
    assert rel < gate, (
        f"Z={Z}, cfg={cfg}: atomx={e_atomx_eV:.3f} eV, "
        f"fac={e0_ref_eV:.3f} eV, rel={rel:.3e} (gate={gate:.3e}"
        + (f" = 1.5 x measured {measured:.3e}" if measured is not None
           else ", unmeasured row — re-measure and record it")
        + ")"
    )


def test_gu_vs_slater_alpha_values_differ_for_z26():
    """Sanity: shooting exchange='gu' and exchange='slater' produce
    different ground-state energies for Fe (Z=26), proving they take
    different SCF paths. If we accidentally wired them to the same
    back-end this assertion flags it."""
    gu = Atom(Z=26, charge=16, config="1s2 2s2 2p6", engine="shooting", exchange="gu")
    gu.solve()
    slater = Atom(
                  Z=26,
                  charge=16,
                  config="1s2 2s2 2p6",
                  engine="shooting",
                  exchange="slater",
             )
    slater.solve()
    assert gu.total_energy != pytest.approx(slater.total_energy, rel=1e-6)
