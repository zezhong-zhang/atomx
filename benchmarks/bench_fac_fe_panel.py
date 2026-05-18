#!/usr/bin/env python3
"""Real sfac runs for the Fe XVII..Fe XXVI L-shell panel.

For each Fe charge state {16..25}, run sfac end-to-end (no fixtures) to
obtain the ground-state to (2*-1 + 3*1) E1 line list. Outputs:

  benchmarks/results_fac_fe_panel.csv
    columns: ion, charge, n_levels, n_e1, top_DE_eV, top_wl_A, top_gf,
             top_f_lower, top_cfg_up

  benchmarks/results_fac_fe17_lines.csv
    columns: name, upper_ILEV, DE_eV, wl_A, gf, f_lower, cfg_up

Used by ``benchmarks/REPORT.md`` § 10.1 to cross-validate against CHIANTI
v10 reference values (Del Zanna et al. 2021, ApJ 909, 38).

Run with: ``uv run python benchmarks/bench_fac_fe_panel.py``
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_runner import parse_levels, parse_transitions, run_sfac  # noqa: E402

HC_EV_A = 12398.42  # h*c in eV·Å (CODATA)


# ---------------------------------------------------------------------------
# sfac script templates
# ---------------------------------------------------------------------------

# Closed-1s L-shell ions (Fe XVII..Fe XXIV): n2 = 2*<nval>, excited = 2*<nval-1> 3*1
SCRIPT_LSHELL = """\
SetAtom('Fe')
Closed('1s')
Config('2*{nval}', group='n2')
Config('n3', '2*{nm1} 3*1')
ConfigEnergy(0)
OptimizeRadial(['n2'])
ConfigEnergy(1)
Structure('out.lev.b', ['n2', 'n3'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['n2'], ['n3'])
PrintTable('out.tr.b', 'out.tr', 1)
"""

# He-like Fe XXV (q=24): ground 1s², excited 1s¹ n=2/3
SCRIPT_HELIKE = """\
SetAtom('Fe')
Config('gnd', '1s2')
Config('exc', '1s1 2*1')
Config('n3', '1s1 3*1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('out.lev.b', ['gnd', 'exc', 'n3'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['gnd'], ['n3'])
PrintTable('out.tr.b', 'out.tr', 1)
"""

# H-like Fe XXVI (q=25): ground 1s¹, excited 2*1, 3*1
SCRIPT_HLIKE = """\
SetAtom('Fe')
Config('gnd', '1s1')
Config('exc', '2*1')
Config('n3', '3*1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('out.lev.b', ['gnd', 'exc', 'n3'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['gnd'], ['n3'])
PrintTable('out.tr.b', 'out.tr', 1)
"""

ROMAN = (
    "XVII", "XVIII", "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI",
)
CHARGES = tuple(range(16, 26))


def script_for(charge: int) -> str:
    n_electrons = 26 - charge
    if charge == 25:
        return SCRIPT_HLIKE
    if charge == 24:
        return SCRIPT_HELIKE
    # Fe XVII..Fe XXIV: closed 1s, n_valence = n_electrons - 2
    nval = n_electrons - 2
    return SCRIPT_LSHELL.format(nval=nval, nm1=nval - 1)


def run_one(charge: int) -> dict:
    """Run sfac for one Fe charge state. Returns parsed levels + transitions."""
    with tempfile.TemporaryDirectory(prefix=f"fac_fe{charge}_") as tmp:
        tmp_path = Path(tmp)
        sf = tmp_path / "run.sf"
        sf.write_text(script_for(charge))
        _out, elapsed, wd = run_sfac(sf, workdir=tmp_path, timeout=300.0)
        lev = parse_levels(wd / "out.lev")
        trs = parse_transitions(wd / "out.tr")
        return {"charge": charge, "levels": lev, "transitions": trs, "elapsed": elapsed}


def summarise(stage: dict) -> dict:
    """Pull the strongest ground-state E1 line and panel-level diagnostics."""
    lev = stage["levels"]
    trs = stage["transitions"]
    lev_by_idx = {l["ILEV"]: l for l in lev}
    # Ground = lowest-ILEV (FAC convention puts ground at ILEV=0 with E=0).
    ground = min(lev, key=lambda l: l["ENERGY_eV"])["ILEV"]
    gs_trs = [t for t in trs if t["lower"] == ground]
    if not gs_trs:
        return {
            "ion": f"Fe {ROMAN[stage['charge'] - 16]}", "charge": stage["charge"],
            "n_levels": len(lev), "n_e1": len(trs),
            "top_DE_eV": "", "top_wl_A": "", "top_gf": "", "top_f_lower": "",
            "top_cfg_up": "",
        }
    top = max(gs_trs, key=lambda t: t["gf"])
    up_lev = lev_by_idx.get(top["upper"], {})
    twoj_low = lev_by_idx.get(ground, {}).get("TWOJ", 0)
    g_low = (twoj_low + 1) if isinstance(twoj_low, int) else 1
    f_low = top["gf"] / max(g_low, 1)
    return {
        "ion": f"Fe {ROMAN[stage['charge'] - 16]}",
        "charge": stage["charge"],
        "n_levels": len(lev),
        "n_e1": len(trs),
        "top_DE_eV": top["DELTA_E_eV"],
        "top_wl_A": HC_EV_A / top["DELTA_E_eV"] if top["DELTA_E_eV"] else "",
        "top_gf": top["gf"],
        "top_f_lower": f_low,
        "top_cfg_up": (up_lev.get("config", "") or "")[:48],
    }


def fe17_line_list(stage: dict) -> list[dict]:
    """All ground-state E1 transitions of Fe XVII, sorted by gf desc."""
    if stage["charge"] != 16:
        raise ValueError("fe17_line_list expects the Fe XVII stage")
    lev = stage["levels"]
    trs = stage["transitions"]
    lev_by_idx = {l["ILEV"]: l for l in lev}
    ground = 0  # Fe XVII ground = 2p⁶ ¹S₀ at ILEV=0 (J=0 ⇒ g_lower=1)
    rows = []
    for t in sorted([t for t in trs if t["lower"] == ground], key=lambda x: -x["gf"]):
        de = t["DELTA_E_eV"]
        wl = HC_EV_A / de if de else 0.0
        up_lev = lev_by_idx.get(t["upper"], {})
        twoj_up = up_lev.get("TWOJ", -1)
        rows.append({
            "upper_ILEV": t["upper"],
            "two_J_up": twoj_up,
            "DE_eV": de,
            "wl_A": wl,
            "gf": t["gf"],
            "f_lower": t["gf"],  # g_lower=1 for ground J=0
            "cfg_up": (up_lev.get("config", "") or "")[:60],
        })
    return rows


def name_3c_3d(line_rows: list[dict]) -> list[dict]:
    """Tag the two highest-f lines in [14.5, 15.5] Å as 3C / 3D."""
    in_band = [r for r in line_rows if 14.5 <= r["wl_A"] <= 15.5]
    in_band_sorted = sorted(in_band, key=lambda r: -r["gf"])
    name_map = {}
    if len(in_band_sorted) >= 1:
        name_map[in_band_sorted[0]["upper_ILEV"]] = "3C"
    if len(in_band_sorted) >= 2:
        name_map[in_band_sorted[1]["upper_ILEV"]] = "3D"
    out = []
    for r in line_rows:
        rr = dict(r)
        rr["name"] = name_map.get(r["upper_ILEV"], "")
        out.append(rr)
    # Re-order: 3C/3D first, then by gf desc
    out.sort(key=lambda r: (0 if r["name"] in ("3C", "3D") else 1, -r["gf"]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-summary",
        default=str(ROOT / "benchmarks" / "results_fac_fe_panel.csv"),
    )
    parser.add_argument(
        "--out-fe17",
        default=str(ROOT / "benchmarks" / "results_fac_fe17_lines.csv"),
    )
    parser.add_argument("--only-fe17", action="store_true",
                        help="Skip Fe XVIII..Fe XXVI (faster CI)")
    args = parser.parse_args(argv)

    summary_rows: list[dict] = []
    stages: dict[int, dict] = {}
    targets = [16] if args.only_fe17 else list(CHARGES)
    for q in targets:
        ion = f"Fe {ROMAN[q - 16]}"
        try:
            print(f"Running sfac for {ion} (charge={q}) …", flush=True)
            stage = run_one(q)
            stages[q] = stage
            summary_rows.append(summarise(stage))
            print(
                f"  {ion}: {len(stage['levels'])} levels, "
                f"{len(stage['transitions'])} E1, {stage['elapsed']:.2f} s",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover — surface FAC failure
            print(f"  {ion} failed: {exc}", flush=True)
            summary_rows.append({
                "ion": ion, "charge": q, "n_levels": 0, "n_e1": 0,
                "top_DE_eV": "", "top_wl_A": "", "top_gf": "",
                "top_f_lower": "", "top_cfg_up": f"FAILED: {exc!s:.60}",
            })

    out_sum = Path(args.out_summary)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    with out_sum.open("w", newline="") as f:
        fields = [
            "ion", "charge", "n_levels", "n_e1",
            "top_DE_eV", "top_wl_A", "top_gf", "top_f_lower", "top_cfg_up",
        ]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {out_sum}")

    if 16 in stages:
        fe17_rows = name_3c_3d(fe17_line_list(stages[16]))
        out_fe17 = Path(args.out_fe17)
        with out_fe17.open("w", newline="") as f:
            fields = [
                "name", "upper_ILEV", "two_J_up",
                "DE_eV", "wl_A", "gf", "f_lower", "cfg_up",
            ]
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(fe17_rows)
        print(f"Wrote {out_fe17}")
        # Echo the 3C/3D pair for the PR body.
        c = next((r for r in fe17_rows if r["name"] == "3C"), None)
        d = next((r for r in fe17_rows if r["name"] == "3D"), None)
        if c and d:
            ratio = c["f_lower"] / d["f_lower"] if d["f_lower"] else float("nan")
            print(
                f"Fe XVII real-FAC f(3C)={c['f_lower']:.3f} @ {c['wl_A']:.3f} Å, "
                f"f(3D)={d['f_lower']:.3f} @ {d['wl_A']:.3f} Å, "
                f"ratio={ratio:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
