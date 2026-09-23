"""Fe XVII KLL Auger validation panel vs FAC AITable (HYP-530).

Loads the 5 dominant KLL Auger resonances from
``tests/data/fe17_kll_auger_fac.json`` and compares the atomx
``auger_rate_csf_average`` per-resonance Auger width to the FAC value.

Background
----------
HYP-394 landed the ab-initio Auger width plumbing but
``auger_rate_csf_average`` was producing ~1e5 s⁻¹ for KLL channels —
eight orders of magnitude below FAC.  Root cause (HYP-530):

1. ``_channel_population_factor`` was reading solved-atom orbital
   occupations, which collapse to ~1e-9 for the essentially-unoccupied
   captured 3l orbitals of a doubly-excited DR resonance.  The CSF-average
   entry point should return the per-resonance rate (one source pair,
   one hole), so the prefactor is now hard-coded to 1.0 when
   ``state_index is None``.
2. The B-spline ``solve_continuum`` pseudo-state (discrete box eigenvalue
   scaled by ``sqrt(2ρ/π)``) mis-estimates the inner-profile amplitude
   for KLL continuum energies by ~25x.  Default to the
   δ(E)-normalised Dirac shooting solver (``shooting_n_grid=400``) for
   the Slater integral.

Together these bring the per-pair KLL Auger width into the 1e14-1e16 s⁻¹
range, matching the FAC order of magnitude.

De-duplication (S5 finding 17): the live panel, its magnitude band
(``[1e13, 1e17]`` s^-1) and the per-resonance ratio band now live in
``tests/test_auger_fac_panel.py``, which computes the same five widths from
the same ``Atom(Z=26, charge=16, config=[...])`` solve and additionally
cross-checks the panel against the independent AITable record.  The
duplicated magnitude test was deleted here; the 10 % gate is kept (it is the
second, independent statement of the same aspirational claim) and is now a
*loud* strict xfail.

The strict 10 % per-resonance gate stays ``xfail`` because the
average-of-configuration angular factor ``exchange_lambda`` over-counts
magnetic substates that do not contribute to a specific J-resolved
resonance (notably the k=0 monopole term for 3p3/2 -> 2p3/2 and the
high-R^k 3d-3d couplings).  A proper Cowan-Grant J-resolved angular
treatment is the remaining gap (follow-up).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.scattering.autoionization import auger_rate_csf_average  # noqa: E402

FIXTURE_PATH = Path(__file__).parent / "data" / "fe17_kll_auger_fac.json"

# Aspirational per-resonance tolerance.  Currently xfailed because the
# average-of-configuration angular factor over-counts magnetic substates
# vs the J-resolved FAC rate.  RE-MEASURED 2026-09-22 on the five panel
# resonances (the R7 asymmetric-mean-field fix moved them): atomx/FAC =
# 0.50 / 1.20 / 5.13 / 1.38 / 0.81, so the worst |atomx/FAC - 1| is 4.13
# (the 1.13-14.82 quoted here on 2026-09-17 is stale).
STRICT_TOLERANCE = 0.10
MEASURED_WORST_RATIO_DEVIATION = 4.13


@pytest.fixture(scope="module")
def fac_panel() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"FAC KLL fixture missing: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def fe17_result(fac_panel):
    """Solve Fe XVII once with ground + 2p^5 3l CSFs for the radial functions.

    Only the *radial* orbitals feed the Slater integrals in
    ``auger_rate_csf_average``, so a non-j-resolved MCDHF solve with the
    doubly-excited configs added is sufficient — the 3l virtual orbitals
    are well-defined even though their ground-state occupation is ~0.
    """
    atom = Atom(
                Z=26,
                charge=16,
                config=[
            "1s2 2s2 2p6",
            "1s2 2s2 2p5 3s1",
            "1s2 2s2 2p5 3p1",
            "1s2 2s2 2p5 3d1",
        ],
                correlation=None,
                corrections="none",
           )
    atom.solve()
    return atom._result


def _per_resonance_rates(fac_panel, fe17_result) -> list[dict]:
    out = []
    for row in fac_panel["resonances"]:
        result = auger_rate_csf_average(
            fe17_result,
            hole_label=row["atomx_hole_label"],
            source_label=row["atomx_source_label"],
            source2_label=row["atomx_source2_label"],
            resonance_energy_eV=row["E_resonance_eV"],
        )
        atomx = float(result["Gamma_per_s"])
        fac = float(row["gamma_auger_per_s"])
        out.append({
            "ilev": row["ilev"],
            "label": row["fac_label"],
            "E_eV": row["E_resonance_eV"],
            "J": row["J"],
            "channel": (
                f"{row['atomx_source_label']}+{row['atomx_source2_label']}"
                f" -> {row['atomx_hole_label']}+eps"
            ),
            "atomx_per_s": atomx,
            "fac_per_s": fac,
            "ratio": atomx / fac if fac > 0 else float("nan"),
        })
    return out


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Strict 10 % per-resonance KLL Auger gate.  MEASURED live "
        "atomx/FAC = 0.50 / 1.20 / 5.13 / 1.38 / 0.81 on the five panel "
        "resonances (2026-09-22, post-R7; the 1.13-14.82 measured 2026-09-17 "
        "is stale), i.e. the worst |atomx/FAC - 1| = 4.13 against the 0.10 "
        "gate.  "
        "2026-07-14 progress: "
        "atomx.scattering.autoionization.jj_level.auger_rate_jj_level now "
        "computes the exact jj-coupled level amplitude (explicit "
        "m-summation with hole conjugation, direct-exchange antisymmetry, "
        "delta(E) shooting continuum, Gamma = 2*pi*|A|^2 == FAC's "
        "4*sum p^2/(2J+1) given FAC's 1/sqrt(k) waves).  The amplitude is "
        "verified M-independent and improves the panel from 1.2-15x "
        "(config-average proxy) to 0.5-4x.  The remaining deviation has "
        "two identified sources: (i) orbital-basis difference — FAC "
        "optimizes radials on the ground config while atomx's MCDHF "
        "average-level solve relaxes them over the excited configs (the "
        "3d radials differ most, matching the largest residual at ILEV "
        "190), and (ii) CI mixing of the FAC levels within each "
        "(J, parity) manifold vs atomx's pure basis states.  Closing "
        "this requires wiring atomx's j-resolved CSF eigenvectors into "
        "the jj_level amplitudes."
    ),
)
def test_kll_auger_per_resonance_10pct(fac_panel, fe17_result):
    """Per-resonance |ΔΓ|/Γ ≤ 10 % gate — aspirational, currently xfailed."""
    rates = _per_resonance_rates(fac_panel, fe17_result)
    worst = max(abs(r["ratio"] - 1.0) for r in rates)
    report = "\n".join(
        f"  ILEV {r['ilev']} J={r['J']} {r['channel']}: "
        f"atomx={r['atomx_per_s']:.3e}  FAC={r['fac_per_s']:.3e}  "
        f"|ΔΓ|/Γ={abs(r['ratio']-1.0):.2f}"
        for r in rates
    )
    assert worst <= STRICT_TOLERANCE, (
        f"Worst |ΔΓ|/Γ = {worst:.2f} exceeds {STRICT_TOLERANCE:.0%}; "
        f"per-resonance panel:\n{report}"
    )
