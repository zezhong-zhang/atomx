"""Diagnostic only: FAC resolved E1 strengths at one native photon energy."""

import json
import sys
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path.insert(0, str(ROOT / "benchmarks"))
from fac_runner import parse_levels, parse_transitions  # noqa: E402

from atomx.constants import HARTREE_EV  # noqa: E402

HERE = Path(__file__).resolve().parent
raw = HERE.parent / "systematic_xray_2026_10_02/raw/C/fac-2s-hole"
levels = parse_levels(raw / "levels.txt")
lines = [t for t in parse_transitions(raw / "tr.txt") if t["upper"] < 8]
r = json.loads((HERE / "C-gu.json").read_text())
a = r["configurations"]
energy = (a["1s2 2s1 2p2"]["total_Ha"] - a["1s2 2s2 2p1"]["total_Ha"]) * HARTREE_EV
g = sum(l["TWOJ"] + 1 for l in levels[:8])
width = sum((levels[t["upper"]]["TWOJ"] + 1) * t["A_per_s"] for t in lines) / g
strength = (
    sum(
        (levels[t["upper"]]["TWOJ"] + 1) * t["A_per_s"] / t["DELTA_E_eV"] ** 3
        for t in lines
    )
    / g
)
collapsed = strength * energy**3
native = next(
    x["atomx_per_s"]
    for x in r["fluorescence"]
    if x["hole"] == "2s" and x["energy_mode"] == "relaxed"
)
result = {
    "native_delta_scf_photon_eV": energy,
    "fac_line_energy_range_eV": [
        min(t["DELTA_E_eV"] for t in lines),
        max(t["DELTA_E_eV"] for t in lines),
    ],
    "fac_width_per_s": width,
    "fac_same_line_strengths_at_native_centroid_per_s": collapsed,
    "native_gu_per_s": native,
    "native_gu_over_fac_collapsed": native / collapsed,
    "native_gu_over_fac_resolved": native / width,
    "control_only": True,
    "interpretation": "Replace FAC line frequencies by the native configuration centroid while retaining weighted strengths. Tests sensitivity to resolved energy/strength correlations; no control is used in the native benchmark.",
}
(HERE / "C_2s_resolved_control.json").write_text(json.dumps(result, indent=2) + "\n")
print(result)
