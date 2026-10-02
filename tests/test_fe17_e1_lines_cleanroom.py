"""Fe XVII E1 line strengths vs the cleanroom FAC TR reference (2026-10-02).

The 2026-10-02 radiative audit isolated the mscheme-manifold A_r
over-estimate (8-15x on the doubly-excited LMM resonances) to the
manifold's own one-body contraction.  This gate certifies the OTHER
E1 engine — ``ci_line_strengths`` — against FAC's own line RMEs
(tests/data/fe17_tr_fac_cleanroom.json, FAC 1.1.5 single-threaded,
2026-07-17): the 2p5 3s/3d -> 2p6 lines including the classic 3C
(826.7 eV) and 3D (812.3 eV) x-ray lines agree to ~2 % on the
shooting+gu comparison contract.  The element, the angular factors
and the CI line contraction are thereby NOT the source of the
manifold discrepancy — the remaining gap is the mscheme
``radiative_channels`` contraction (two code paths, one certified).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.properties.transitions.ci_lines import ci_line_strengths  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "fe17_tr_fac_cleanroom.json"
BAND = 0.05  # measured 2026-10-02: worst |ratio-1| = 0.021


@pytest.fixture(scope="module")
def fe17_ground():
    """shooting+gu: the FAC-comparison orbital contract."""
    atom = Atom(Z=26, charge=16, config="1s2 2s2 2p6",
                engine="shooting", exchange="gu", corrections="none")
    atom.solve()
    return atom._result


def _matched_lines(result, config):
    lineset = ci_line_strengths(result, config, "2p6", multipole="E1")
    fixture = json.loads(FIXTURE.read_text())["lines"]
    fac = [L for L in fixture if L["config"] == config]
    ground_eV = None
    pairs = []
    for ln in lineset.lines:
        if ground_eV is None:
            ground_eV = ln.lower.energy_au * 27.211386245988
        e_exc = ln.upper.energy_au * 27.211386245988 - ground_eV
        best = min(fac, key=lambda L: abs(L["E_upper_eV"] - e_exc))
        if abs(best["E_upper_eV"] - e_exc) > 8.0:
            continue  # no counterpart (levels split differently)
        pairs.append((best, ln, e_exc))
    return pairs


def test_3s_and_3d_lines_vs_fac_rmes(fe17_ground):
    rows = []
    for config in ("2p5 3s1", "2p5 3d1"):
        for best, ln, e_exc in _matched_lines(fe17_ground, config):
            s_fac = best["rme"] ** 2 / 2.0
            rows.append((best.get("note") or best["fac_name"],
                         e_exc, best["E_upper_eV"],
                         ln.S / s_fac))
    assert len(rows) >= 4, f"expected the 3s/3D/3C line set, got {rows}"
    report = "\n".join(
        f"  {name}: E {e:.1f} (fac {ef:.1f}) S/S_fac {r:.3f}"
        for name, e, ef, r in rows)
    for name, _e, _ef, r in rows:
        assert abs(r - 1.0) <= BAND, (
            f"{name}: S/S_fac = {r:.3f} outside 1+-{BAND} "
            f"(E1 chain regression); lines:\n{report}"
        )
