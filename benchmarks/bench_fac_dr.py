#!/usr/bin/env python3
"""Benchmark atomx ``dr_calculate`` against FAC AI + TR for Fe XVII -> Fe XVI.

FAC 0.9.3+ removed the ``DRTable`` convenience wrapper, so the canonical
parity protocol is the same one ``DRTable`` would have run internally:
``Structure`` + ``AITable`` + ``TRTable``, fed to the Burgess detailed-balance
kernel (Gu 2003, ApJ 590, 1131).  We compute the FAC reference α_DR(T) by
feeding FAC's level energies, AI rates, and stabilising TR rates into the same
:func:`atomx.scattering.autoionization.dr_rate_coefficient` used for the
atomx self-driven number.  The two sides therefore differ only in the AI and
radiative width *physics* — the kinematic kernel is identical, which is
exactly the property the ≤ 20 % parity gate is supposed to test.

The script writes ``benchmarks/results_fac_dr.csv`` and the JSON fixture at
``tests/data/fe17_dr_fac.json`` consumed by ``tests/test_dr_vs_fac.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from atomx.scattering.autoionization import (  # noqa: E402
    AutoionizationState,
    dr_calculate,
    dr_rate_coefficient,
    dr_strengths,
)
from fac_runner import parse_ai, parse_levels, parse_transitions, run_sfac  # noqa: E402


K_B_EV_PER_K = 8.617333262e-5
TEMPERATURES_MK = [1.0, 5.0, 10.0, 50.0]
TEMPERATURES_EV = [T * 1.0e6 * K_B_EV_PER_K for T in TEMPERATURES_MK]


# ---------------------------------------------------------------------------
# atomx side
# ---------------------------------------------------------------------------

class FeXVIIModel:
    """Lightweight Atom-like input for ``dr_calculate``.

    The orbital energies are FAC-like screened one-electron eigenvalues in
    Hartree (Fe XVII Ne-like ground).  They keep the benchmark end-to-end
    without a separate structure solve and are only used to *rank* resonance
    energies; the parity check compares α_DR(T) summed over resonances.
    """

    charge = 16
    converged = True
    _config_strings = ["1s2 2s2 2p6"]
    result = None
    orbital_energies = {
        "1s1/2": -262.0,
        "2s1/2": -34.8,
        "2p1/2": -31.4,
        "2p3/2": -30.6,
        "3s1/2": -13.7,
        "3p1/2": -12.9,
        "3p3/2": -12.7,
        "3d3/2": -12.1,
        "3d5/2": -12.0,
        "4s1/2": -7.6,
        "4p1/2": -7.3,
        "4p3/2": -7.2,
        "4d3/2": -6.9,
        "4d5/2": -6.9,
    }


def run_atomx():
    rc = dr_calculate(
        FeXVIIModel(),
        "1s2 2s2 2p6 3s1",
        n_max=10,
        l_max=5,
        T_eV=TEMPERATURES_EV,
    )
    rows = dr_strengths(rc)
    dominant_idx = int(np.argmin([abs(t - 10.0e6 * K_B_EV_PER_K) for t in TEMPERATURES_EV]))
    dominant = sorted(
        rows,
        key=lambda r: float(r["alpha_cm3_s"][dominant_idx]),
        reverse=True,
    )[:12]
    return rc, dominant


# ---------------------------------------------------------------------------
# FAC side: AITable + TRTable -> Burgess
# ---------------------------------------------------------------------------

FAC_SCRIPT = """# Fe XVII (Ne-like) + e -> Fe XVI doubly-excited [AI+TR] -> Fe XVI bound + photon
# AITable + TRTable in lieu of DRTable, which was removed in FAC 0.9.3.
# Channels: Δn=0 KLL (2{s,p} hole + two captured n=3 electrons)
#           Δn=1 KLM (2{s,p} hole + one captured n=3 + one captured n=4)
# Final stabilising states: Fe XVI 2s²2p⁶ nl (n<=4) + 2p hole + nl.
SetAtom('Fe')
Closed('1s')
Config('rec', '2s2 2p6')
Config('na3', '2s2 2p6 3*1')
Config('na4', '2s2 2p6 4*1')
Config('dr33', '2s2 2p5 3*2', '2s1 2p6 3*2')
Config('dr34', '2s2 2p5 3*1 4*1', '2s1 2p6 3*1 4*1')

SetAICut(1e-4)
SetTransitionCut(1e-4)
ConfigEnergy(0)
OptimizeRadial(['rec'])
ConfigEnergy(1)

Structure('fe.lev.b', ['rec'])
Structure('fe.lev.b', ['na3', 'na4'])
Structure('fe.lev.b', ['dr33', 'dr34'])

MemENTable('fe.lev.b')
PrintTable('fe.lev.b', 'fe.lev', 1)

AITable('fe.ai.b', ['dr33', 'dr34'], ['rec'])
PrintTable('fe.ai.b', 'fe.ai', 1)

TRTable('fe.tr.b', ['dr33', 'dr34'], ['na3', 'na4'], -1)
PrintTable('fe.tr.b', 'fe.tr', 1)
"""


def build_fac_resonances(
    levels: list[dict],
    ai_rows: list[dict],
    tr_rows: list[dict],
) -> tuple[list[AutoionizationState], int]:
    """Stitch FAC ASCII tables into ``AutoionizationState`` objects.

    Returns the resonance list and the FAC ILEV identifying the recombining
    ion ground state (lowest-energy NELE=10 level), so the caller can pick the
    correct g_i and resonance-energy reference.
    """

    levels_by_idx = {lv["ILEV"]: lv for lv in levels}
    # The recombining ion ground state is the ILEV that appears most often as
    # the autoionising "lower" target.  Avoids fragile config-string parsing.
    from collections import Counter
    if not ai_rows:
        return [], -1
    rec_idx = Counter(r["lower"] for r in ai_rows).most_common(1)[0][0]

    auger_by_upper: dict[int, float] = {}
    e_resonance_by_upper: dict[int, float] = {}
    twoj_by_upper: dict[int, int] = {}
    for row in ai_rows:
        if row["lower"] != rec_idx:
            continue
        u = row["upper"]
        auger_by_upper[u] = auger_by_upper.get(u, 0.0) + float(row["rate_per_s"])
        e_resonance_by_upper[u] = float(row["DELTA_E_eV"])
        twoj_by_upper[u] = int(row["TWOJ_upper"])

    radiative_by_upper: dict[int, float] = {}
    channels_by_upper: dict[int, dict[str, float]] = {}
    for row in tr_rows:
        u = row["upper"]
        if u not in auger_by_upper:
            continue
        twoj = twoj_by_upper.get(u)
        if twoj is None:
            continue
        g = float(twoj + 1)
        a = float(row["gA_per_s"]) / g
        if a <= 0.0:
            continue
        radiative_by_upper[u] = radiative_by_upper.get(u, 0.0) + a
        channels_by_upper.setdefault(u, {})[f"->{row['lower']}"] = a

    states: list[AutoionizationState] = []
    for u, gamma_a in auger_by_upper.items():
        twoj = twoj_by_upper[u]
        e_res = e_resonance_by_upper[u]
        a_r = radiative_by_upper.get(u, 0.0)
        if gamma_a <= 0.0 or a_r <= 0.0 or e_res <= 0.0:
            continue
        lv = levels_by_idx.get(u, {})
        label = f"FAC ILEV {u}: {lv.get('config', '')}"[:96]
        states.append(AutoionizationState(
            label=label,
            E_resonance_eV=e_res,
            J=0.5 * twoj,
            gamma_auger_per_s=gamma_a,
            a_radiative_per_s=a_r,
            channels=channels_by_upper.get(u, {}),
        ))
    return states, rec_idx


def run_fac(timeout: float, keep_dir: Path | None = None) -> tuple[list[AutoionizationState], dict[float, float], str]:
    """Drive sfac on FAC_SCRIPT and return parsed FAC-side resonances."""

    workdir_ctx = None
    if keep_dir is not None:
        wd = keep_dir
        wd.mkdir(parents=True, exist_ok=True)
    else:
        workdir_ctx = tempfile.TemporaryDirectory()
        wd = Path(workdir_ctx.name)
    try:
        script = wd / "fe_dr.sf"
        script.write_text(FAC_SCRIPT)
        try:
            _stdout, _elapsed, _ = run_sfac(script, workdir=wd, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            msg = " ".join(str(exc).split())
            return [], {}, "FAC failed: " + msg
        try:
            levels = parse_levels(wd / "fe.lev")
            ai_rows = parse_ai(wd / "fe.ai")
            tr_rows = parse_transitions(wd / "fe.tr")
        except FileNotFoundError as exc:
            return [], {}, "FAC produced no output: " + str(exc)
        states, rec_idx = build_fac_resonances(levels, ai_rows, tr_rows)
        rc = dr_rate_coefficient(
            states,
            TEMPERATURES_EV,
            initial_state_g=1.0,
            initial_state_label=f"FAC ILEV {rec_idx} (Fe XVII 2p6)",
        )
        return states, {T: float(a) for T, a in zip(TEMPERATURES_EV, rc.alpha_cm3_s)}, ""
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()


def rms_relative(rows: list[dict]) -> float:
    vals = [
        (r["alpha_atomx_cm3_s"] - r["alpha_fac_cm3_s"]) / r["alpha_fac_cm3_s"]
        for r in rows
        if r["alpha_fac_cm3_s"] and r["alpha_fac_cm3_s"] > 0.0
    ]
    if not vals:
        return math.nan
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results_fac_dr.csv"))
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "tests" / "data" / "fe17_dr_fac.json"),
        help="Write a JSON FAC reference fixture for tests/test_dr_vs_fac.py",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-fac", action="store_true")
    parser.add_argument("--keep-fac-dir", default=None,
                        help="Persist sfac working files in this directory for debugging")
    args = parser.parse_args()

    rc_atomx, dominant = run_atomx()
    if args.skip_fac:
        fac_alpha: dict[float, float] = {}
        note = "FAC skipped (--skip-fac)"
        fac_states: list[AutoionizationState] = []
    else:
        keep = Path(args.keep_fac_dir) if args.keep_fac_dir else None
        fac_states, fac_alpha, note = run_fac(args.timeout, keep_dir=keep)

    rows = []
    for i, (T_mk, T_eV) in enumerate(zip(TEMPERATURES_MK, TEMPERATURES_EV)):
        fac = fac_alpha.get(T_eV, 0.0)
        atomx_val = float(rc_atomx.alpha_cm3_s[i])
        rel = ((atomx_val - fac) / fac) if fac > 0 else ""
        rows.append({
            "T_MK": T_mk,
            "T_eV": T_eV,
            "alpha_atomx_cm3_s": atomx_val,
            "alpha_fac_cm3_s": fac,
            "rel_diff": rel,
            "note": note,
        })
    rms = rms_relative(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T_MK",
                "T_eV",
                "alpha_atomx_cm3_s",
                "alpha_fac_cm3_s",
                "rel_diff",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "T_MK": "rms",
            "T_eV": "",
            "alpha_atomx_cm3_s": "",
            "alpha_fac_cm3_s": "",
            "rel_diff": "" if math.isnan(rms) else f"{rms:.4f}",
            "note": "target |rel| rms <= 0.20",
        })

    fixture = Path(args.fixture)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({
        "source": "FAC 1.1.5 AITable+TRTable, combined via Burgess kernel",
        "script": FAC_SCRIPT,
        "T_MK": TEMPERATURES_MK,
        "T_eV": TEMPERATURES_EV,
        "alpha_fac_cm3_s": [fac_alpha.get(T, 0.0) for T in TEMPERATURES_EV],
        "n_fac_resonances": len(fac_states),
        "initial_state_g": 1.0,
        "resonances": [
            {
                "label": d.label,
                "E_resonance_eV": d.E_resonance_eV,
                "J": d.J,
                "gamma_auger_per_s": d.gamma_auger_per_s,
                "a_radiative_per_s": d.a_radiative_per_s,
            }
            for d in fac_states
        ],
        "note": note,
    }, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"Wrote {fixture}")
    print(f"FAC reference resonances: {len(fac_states)}")
    print("Dominant atomx resonances (at ~10 MK):")
    for r in dominant:
        print(f"  {r['label']}: {float(r['alpha_cm3_s'][2]):.3e} cm^3/s")
    if note:
        print(note)


if __name__ == "__main__":
    main()
