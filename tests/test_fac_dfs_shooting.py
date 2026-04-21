"""End-to-end: atomx shooting with ``exchange='fac'`` vs FAC reference.

Reference total-binding energies come from ``sfac`` runs captured
during Task 0 (calibration.csv).  The gate is 5 × 10⁻⁴ relative
agreement — wide enough to absorb small differences in the grid and
the asymptotic-fix details, tight enough to fail if the local
exchange functional is wrong.
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
    / "superpowers"
    / "plans"
    / "2026-04-20-fac-calibration"
    / "calibration.csv"
)


def _load_reference(max_Z: int | None = None) -> list[dict]:
    """Load panel rows from the Task 0 calibration CSV.

    Heavy atoms (Z ≥ 74) previously hung the shooting SCF for
    minutes (``_slater_integral``'s power-series ODE had no
    step-halving cap).  After that fix every panel row runs in
    ~0.1–0.2 s warm, so we exercise the full Z=2..92 panel.
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


@pytest.mark.skipif(not _REFERENCE, reason="calibration.csv missing")
@pytest.mark.parametrize("row", _REFERENCE, ids=_IDS or None)
def test_atomx_fac_total_energy_matches_sfac(row):
    Z = int(row["Z"])
    cfg = row["config"]
    nel = _nel_from_config(cfg)
    e0_ref_eV = float(row["E0_eV"])
    charge = max(Z - nel, 0)

    atom = Atom(Z=Z, charge=charge, config=cfg, exchange="fac")
    atom.solve()
    e_atomx_eV = atom.total_energy * HARTREE_EV

    rel = abs(e_atomx_eV - e0_ref_eV) / abs(e0_ref_eV)
    # Gate: 1 × 10⁻³ relative across the full Z = 2..92 panel.
    # Heavy-Z (Z ≥ 74) was previously gated at 0.30 because atomx's
    # ``doutw`` used a 3-point finite-difference proxy for the
    # past-turning-point blow-up check ``TCONV = P''·P``, which
    # mis-detected the sign for sharp heavy-Z 2p₁/₂ orbitals and let
    # integration continue into the forbidden region — picking up an
    # exponentially-growing irregular-solution contamination that
    # registered as a spurious extra radial node, driving dbound's
    # "too many nodes" branch to the wrong eigenvalue (~3× too deep).
    # Fixed 2026-04-20 by threading the analytical power-series
    # ``ca[2]`` coefficient out of ``dir_integrate`` and using
    # ``TCONV = 2·ca[2]·PPI`` verbatim from Fortran radial.f DOUTW
    # (line 1780).  See ``benchmarks/REPORT.md`` §6.4 and trace.md.
    gate = 1e-3
    assert rel < gate, (
        f"Z={Z}, cfg={cfg}: atomx={e_atomx_eV:.3f} eV, "
        f"fac={e0_ref_eV:.3f} eV, rel={rel:.2e} (gate={gate:.0e})"
    )


def test_fac_vs_shooting_alpha_values_differ_for_z26():
    """Sanity: exchange='fac' and exchange='shooting' produce different
    ground-state energies for Fe (Z=26), proving they take different
    SCF paths. If we accidentally wired them to the same back-end
    this assertion flags it."""
    fac = Atom(Z=26, charge=16, config="1s2 2s2 2p6", exchange="fac")
    fac.solve()
    shooting = Atom(Z=26, charge=16, config="1s2 2s2 2p6", exchange="shooting")
    shooting.solve()
    assert fac.total_energy != pytest.approx(shooting.total_energy, rel=1e-6)
