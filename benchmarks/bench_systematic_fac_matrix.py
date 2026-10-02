"""Evidence inventory for the nine-cell atomic-process comparison matrix.

Presence of a test file is not a physics accuracy claim. New cells report
measured results, reference type and sample count. Historical RR anchors
are kept distinct from native neutral-atom PI. Fe's resonance panel is LMM,
not KLL. Photon scattering uses nonresonant f0 and angle-integrated WH;
FAC scattering is independent spinor postprocessing, not a native FAC table.

Run bench_systematic_xray.py first; then this file (--report selects results).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def span(data, key):
    values = [float(r[key]) for r in data if r.get(key)]
    return f"{min(values):.3f}–{max(values):.3f}×" if values else "no nonzero ratios"


def matrix(report, consistency=None, integration=None):
    historical = rows(ROOT / "benchmarks/results_fac_pi_rr.csv")
    he = [r for r in historical if r["case"].startswith("Fe25+")]
    li = [r for r in historical if r["case"].startswith("Fe24+")]
    pi = rows(report / "pi.csv")
    fl = rows(report / "fluorescence_totals.csv")
    y = rows(report / "yield.csv")
    sc = rows(report / "scattering.csv")
    ne_path = report / "ne_ls.json"
    ne = json.loads(ne_path.read_text()) if ne_path.exists() else {}
    fe_path = report / "fe_jj.json"
    fe = json.loads(fe_path.read_text())["rows"] if fe_path.exists() else []

    # Fresh new-panel statistics are derived from output, never copied from
    # an aspirational tolerance or inferred merely from file presence.
    def rr_status(data):
        if not data:
            return "MISSING"
        return f"archived FAC; {len(data)} points; |error| max {max(abs(float(r['rel_diff'])) for r in data):.1%}"

    def panel_status(data, text):
        return text if data else "MISSING: run bench_systematic_xray.py"

    ray = [
        dict(
            r,
            ratio=str(float(r["atomx_rayleigh_barn"]) / float(r["fac_rayleigh_barn"])),
        )
        for r in sc
    ]
    comp = [
        dict(
            r,
            ratio=str(
                float(r["atomx_compton_KN_barn"]) / float(r["fac_compton_KN_barn"])
            ),
        )
        for r in sc
    ]
    cells = [
        dict(
            family="PI/RR",
            observable="RR capture into He-like Fe24+ 1s²",
            evidence="benchmarks/results_fac_pi_rr.csv",
            status=rr_status(he),
        ),
        dict(
            family="PI/RR",
            observable="RR capture into Li-like Fe23+ 2s/2p",
            evidence="benchmarks/results_fac_pi_rr.csv",
            status=rr_status(li),
        ),
        dict(
            family="PI",
            observable="neutral K/L PI, C/Ne/Ar/Fe/Kr/Xe",
            evidence=str(report / "pi.csv"),
            status=panel_status(
                pi,
                f"measured {len(pi)} points; native/FAC {span(pi, 'atomx_over_fac')}",
            ),
        ),
        dict(
            family="AUGER",
            observable="Ne KLL configuration average (LS)",
            evidence=str(ne_path),
            status=panel_status(
                ne,
                f"separate public Atom anchor: {ne.get('total_ratio', float('nan')):.3f}× XATOM; radial settings differ from Z sweep",
            ),
        ),
        dict(
            family="AUGER",
            observable="Fe XVI LMM → Fe XVII, five resonances (jj)",
            evidence=str(fe_path),
            status=panel_status(
                fe,
                f"archived FAC: {len(fe)} live points; {span(fe, 'ratio')}; 10% accuracy unmet",
            ),
        ),
        dict(
            family="FLUOR",
            observable="K/L per-vacancy radiative widths",
            evidence=str(report / "fluorescence_totals.csv"),
            status=panel_status(
                fl,
                f"measured {len(fl)} hole widths; native/FAC {span(fl, 'atomx_over_fac')}; E1 only",
            ),
        ),
        dict(
            family="YIELD",
            observable="omega_K=A_r/(A_r+Gamma_A), Z panel",
            evidence=str(report / "yield.csv"),
            status=panel_status(
                y,
                f"measured {len(y)} Z points; native/FAC {span(y, 'atomx_over_fac')}; LS Auger + E1",
            ),
        ),
        dict(
            family="RAYLEIGH",
            observable="nonresonant f0, Ne/Ar/Fe",
            evidence=str(report / "scattering.csv"),
            status=panel_status(
                sc,
                f"{len(sc)} energy/element points; native/FAC-spinor integral {span(ray, 'ratio')}",
            ),
        ),
        dict(
            family="COMPTON",
            observable="WH S(q), Thomson/KN weighted, Ne/Ar/Fe",
            evidence=str(report / "scattering.csv"),
            status=panel_status(
                sc,
                f"{len(sc)} energy/element points; native/FAC-spinor KN integral {span(comp, 'ratio')}; no DDCS",
            ),
        ),
    ]
    if consistency is not None:
        native_pi = rows(consistency / "pi.csv")
        native_fl = rows(consistency / "fluorescence.csv")
        if len(native_pi) != 432 or len(native_fl) != 108:
            raise ValueError(
                "the consistency panel must contain 432 PI and 108 fluorescence rows"
            )
        for family, data, filename in [
            ("PI", native_pi, "pi.csv"),
            ("FLUOR", native_fl, "fluorescence.csv"),
        ]:
            selected = [
                r
                for r in data
                if r["energy_mode"] == "relaxed"
                and (family != "PI" or r["field"] == "frozen")
            ]
            hf = [r for r in selected if r["exchange"] == "hf"]
            gu = [r for r in selected if r["exchange"] == "gu"]
            cell = next(c for c in cells if c["family"] == family)
            cell["evidence"] = str(consistency / filename)
            cell["status"] = (
                f"statistical moments + native delta-SCF; {len(gu)} samples per model; "
                f"HF/FAC {span(hf, 'ratio')}; Gu/FAC {span(gu, 'ratio')}; "
                + (
                    "frozen Gu continuum"
                    if family == "PI"
                    else "E1 only; C 2s discrepancy retained"
                )
            )
        next(c for c in cells if c["family"] == "YIELD")["status"] += (
            "; archived model, not recomputed with consistency-panel radiative widths"
        )
    if integration is not None:
        panel = json.loads((integration / "fe_lmm.json").read_text())
        rows_fe = panel["rows"]
        if len(rows_fe) != 5 or not panel["band_15pct_passed"]:
            raise ValueError("the integrated Fe panel must pass its five-anchor gate")
        cell = next(
            c for c in cells if c["family"] == "AUGER" and "Fe XVI" in c["observable"]
        )
        cell["evidence"] = str(integration / "fe_lmm.json")
        cell["status"] = (
            f"full nl=3 determinant CI: five FAC anchors, ratio {span(rows_fe, 'ratio')}; "
            f"worst error {panel['max_abs_fractional_error']:.1%}; reference-frozen orbitals; "
            "strongest-width candidate within +/-2.5 eV in the same J block; "
            "not Hamiltonian-relaxed; 10% accuracy unmet"
        )
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", type=Path, default=ROOT / "docs/systematic_xray_2026_10_02"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--consistency", type=Path, help="completed HF/Gu native-energy panel"
    )
    p.add_argument(
        "--integration", type=Path, help="completed main integration Fe CI panel"
    )
    args = p.parse_args()
    cells = matrix(
        args.report.resolve(),
        args.consistency.resolve() if args.consistency else None,
        args.integration.resolve() if args.integration else None,
    )
    missing = sum(c["status"].startswith("MISSING") for c in cells)
    if args.json:
        print(
            json.dumps(
                dict(
                    cells=cells,
                    missing=missing,
                    scope="measured coverage, not uniform physical accuracy",
                ),
                indent=2,
            )
        )
    else:
        for i, c in enumerate(cells, 1):
            print(
                f"{i}. {c['family']:8} {c['observable']}\n   {c['status']}\n   {c['evidence']}"
            )
        print(
            f"{missing} missing evidence cells of {len(cells)}; accuracy/model restrictions remain explicit."
        )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
