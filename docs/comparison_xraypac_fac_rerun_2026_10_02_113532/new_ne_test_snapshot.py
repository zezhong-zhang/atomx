"""Ne KLL Auger channel gate vs XATOM (Son-Young-Santra PRA 83, 033402).

Pins the LS-coupled channel chain in ``auger_rate_csf_average`` — the
exact transcription of XATOM's ``Auger_group_rate_extended``
(decay.f90): coherent K-sum per (l4, L), exchange interference, the
(2L+1)/(2l3+1) statistical weights, ``N12`` pair normalisation per LS
subshell and ``tau^2 = 1/2`` for same-(n,l) pairs.

Reference values were produced 2026-10-02 with the vendored XATOM
binary (xraypac, non-relativistic HFS, ``-conf '1s1 2s2 2p6' -auger``),
per channel in a.u.: 2s2s 8.70451e-4, 2s2p 2.65841e-3, 2p2p 6.15067e-3
(total 9.67952e-3).  The angular chain alone reproduces these to 1.0000
given XATOM's own radial integrals, so the residual band below measures
the ORBITAL-MODEL difference (atomx Dirac fac_frozen radials are ~10 %
more compact than HFS), not formalism error.

Convergence history of the coarse path vs this reference: 12x (2026-09)
-> 7.25x (N12+tau^2) -> 3.22x (fine grid + final-state field) -> 3.485x
(FAC jj sd/se, 4th attempt) -> 1.025x total (LS chain + full
final-state continuum field, 2026-10-02).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Correlation  # noqa: E402
from atomx.scattering.autoionization import auger_rate_csf_average  # noqa: E402

AU_RATE_PER_S = 4.1341373336561e16

# channel: (rate a.u., Auger KE eV) from the vendored XATOM run
XATOM_NE_KLL = {
    "2s2s": (8.70451e-4, 815.26),
    "2s2p": (2.65841e-3, 837.50),
    "2p2p": (6.15067e-3, 859.74),
}
XATOM_TOTAL_AU = 9.67952e-3

# Bands: wide enough for orbital-model drift, tight enough to catch any
# factor-2 class formalism regression (the pre-LS-chain state sat at
# 4.4x/3.0x on 2s2p/2p2p; the pre-rho_full-fix state at 0.44x on 2p2p).
CHANNEL_BAND = (0.6, 1.8)
TOTAL_BAND = (0.7, 1.4)

CHANNEL_SOURCES = {
    "2s2s": ("2s1/2", "2s1/2"),
    "2s2p": ("2s1/2", "2p3/2"),
    "2p2p": ("2p3/2", "2p3/2"),
}


@pytest.fixture(scope="module")
def ne_khole_result():
    """Solve the Ne+ K-hole resonance once for the radial functions."""
    atom = Atom(
        Z=10,
        charge=1,
        config=["1s1 2s2 2p6"],
        correlation=Correlation(radial_optimization="fac_frozen"),
        corrections="none",
    )
    atom.solve()
    return atom._result


def _channel_rates(ne_khole_result) -> dict[str, float]:
    out = {}
    for name, (src, src2) in CHANNEL_SOURCES.items():
        _, ke = XATOM_NE_KLL[name]
        result = auger_rate_csf_average(
            ne_khole_result,
            hole_label="1s1/2",
            source_label=src,
            source2_label=src2,
            resonance_energy_eV=ke,
        )
        out[name] = float(result["Gamma_per_s"])
    return out


def test_ne_kll_channels_vs_xatom(ne_khole_result):
    """Live channel ratios stay in the orbital-model band."""
    rates = _channel_rates(ne_khole_result)
    lines = []
    for name, mine in rates.items():
        ref = XATOM_NE_KLL[name][0] * AU_RATE_PER_S
        ratio = mine / ref
        lines.append(f"{name}: atomx={mine:.4e} xatom={ref:.4e} ratio={ratio:.3f}")
        assert CHANNEL_BAND[0] <= ratio <= CHANNEL_BAND[1], (
            f"Ne KLL {name}: atomx/XATOM = {ratio:.3f} outside "
            f"{CHANNEL_BAND}; panel:\n" + "\n".join(lines)
        )


def test_ne_kll_total_vs_xatom(ne_khole_result):
    """Total KLL width within +-40 % of XATOM (measured 1.025x)."""
    rates = _channel_rates(ne_khole_result)
    total = sum(rates.values())
    ref = XATOM_TOTAL_AU * AU_RATE_PER_S
    ratio = total / ref
    assert TOTAL_BAND[0] <= ratio <= TOTAL_BAND[1], (
        f"Ne KLL total: atomx/XATOM = {ratio:.3f} outside {TOTAL_BAND} "
        f"(atomx={total:.4e}, xatom={ref:.4e})"
    )
