#!/usr/bin/env python3
"""Benchmark atomx DR driver against FAC ``SetDR`` for Fe XVII -> Fe XVI.

The script always writes ``benchmarks/results_fac_dr.csv`` with the atomx
driver output.  When the local sfac binary supports the DR-table calls, FAC
columns are populated as well.  This repository's bundled sfac currently has
no ``SetDR`` command and removed ``DRTable`` in FAC 0.9.3, so the script also
records the exact legacy script and failure text for review.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from atomx.scattering.autoionization import dr_calculate, dr_strengths  # noqa: E402
from fac_runner import run_sfac  # noqa: E402


TEMPERATURES_EV = [100.0, 300.0, 1000.0, 3000.0, 10000.0]

# Placeholder for the acceptance fallback requested in HYP-146.  The Savin
# 2001 ApJ 562, 1259 measured DR curve needs to be digitized into these five
# temperatures before the fallback gate can be treated as verified.
SAVIN_2001_FE17_TO_FE16_CM3_S: dict[float, float] = {}


class FeXVIIModel:
    """Lightweight Atom-like object for the benchmark driver.

    These orbital energies are FAC-like screened one-electron values in
    Hartree, used so the benchmark exercises the end-to-end DR API without
    requiring a long structure solve before every comparison run.
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
    }


FAC_SCRIPT = """SetAtom('Fe')
Closed('1s')
Config('fe17', '2*8')
Config('fe16g', '2*8 3s1')
Config('kll', '2*7 2*1 3s1')
OptimizeRadial(['fe17'])
Structure('fe17.lev.b', ['fe17'])
Structure('fe16.lev.b', ['fe16g', 'kll'])
SetDR('fe_dr', ['fe17'], ['kll'])
DRTable('fe_dr.tbl', 'fe_dr', {temps})
PrintTable('fe_dr.tbl', 'fe_dr.txt', 1)
"""


def run_atomx():
    rc = dr_calculate(
        FeXVIIModel(),
        "1s2 2s2 2p6 3s1",
        n_max=10,
        l_max=5,
        T_eV=TEMPERATURES_EV,
    )
    rows = dr_strengths(rc)
    dominant = sorted(
        rows,
        key=lambda r: float(r["alpha_cm3_s"][2]),
        reverse=True,
    )[:12]
    return rc, dominant


def parse_fac_dr_table(path: Path) -> dict[float, float]:
    rates: dict[float, float] = {}
    if not path.exists():
        return rates
    for line in path.read_text().splitlines():
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", line)]
        if len(nums) < 2:
            continue
        T, alpha = nums[0], nums[-1]
        if any(abs(T - want) / want < 1e-6 for want in TEMPERATURES_EV):
            rates[T] = alpha
    return rates


def run_fac(timeout: float) -> tuple[dict[float, float], str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "fe_dr.sf"
        script.write_text(FAC_SCRIPT.format(temps=TEMPERATURES_EV))
        try:
            _stdout, _elapsed, wd = run_sfac(script, workdir=tmp_path, timeout=timeout)
            rates = parse_fac_dr_table(wd / "fe_dr.txt")
            return rates, "" if rates else "FAC DR table parser found no rows"
        except Exception as exc:  # noqa: BLE001 - benchmark should still emit atomx rows
            msg = " ".join(str(exc).split())
            if "Syntax Error" in msg and ("SetDR" in msg or "DRTable" in FAC_SCRIPT):
                msg += " | local FAC has no SetDR symbol and ChangeLog notes DRTable removal"
            return {}, "FAC failed: " + msg


def rms_relative(rows: list[dict]) -> float:
    vals = [
        (r["alpha_atomx_cm3_s"] - r["alpha_reference_cm3_s"]) / r["alpha_reference_cm3_s"]
        for r in rows
        if r["alpha_reference_cm3_s"] and r["alpha_reference_cm3_s"] > 0.0
    ]
    if not vals:
        return math.nan
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results_fac_dr.csv"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-fac", action="store_true")
    args = parser.parse_args()

    rc, dominant = run_atomx()
    fac_rates, note = ({}, "FAC skipped") if args.skip_fac else run_fac(args.timeout)

    rows = []
    for i, T in enumerate(TEMPERATURES_EV):
        fac = fac_rates.get(T, 0.0)
        savin = SAVIN_2001_FE17_TO_FE16_CM3_S.get(T, 0.0)
        ref = fac or savin
        source = "FAC SetDR" if fac else ("Savin 2001" if savin else "")
        rows.append({
            "T_eV": T,
            "alpha_atomx_cm3_s": float(rc.alpha_cm3_s[i]),
            "alpha_fac_cm3_s": fac,
            "alpha_savin2001_cm3_s": savin,
            "alpha_reference_cm3_s": ref,
            "reference_source": source,
            "rel_diff": ((float(rc.alpha_cm3_s[i]) - ref) / ref) if ref else "",
            "note": note,
        })
    rms = rms_relative(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T_eV",
                "alpha_atomx_cm3_s",
                "alpha_fac_cm3_s",
                "alpha_savin2001_cm3_s",
                "alpha_reference_cm3_s",
                "reference_source",
                "rel_diff",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "T_eV": "rms",
            "alpha_atomx_cm3_s": "",
            "alpha_fac_cm3_s": "",
            "alpha_savin2001_cm3_s": "",
            "alpha_reference_cm3_s": "",
            "reference_source": "",
            "rel_diff": "" if math.isnan(rms) else rms,
            "note": "target <= 0.20",
        })
        writer.writerow({
            "T_eV": "legacy_sfac_script",
            "alpha_atomx_cm3_s": "",
            "alpha_fac_cm3_s": "",
            "alpha_savin2001_cm3_s": "",
            "alpha_reference_cm3_s": "",
            "reference_source": "",
            "rel_diff": "",
            "note": FAC_SCRIPT.format(temps=TEMPERATURES_EV).replace("\n", "\\n"),
        })

    print(f"Wrote {out}")
    print("Dominant resonance contributions at 1000 eV:")
    for r in dominant:
        print(f"  {r['label']}: {float(r['alpha_cm3_s'][2]):.3e} cm^3/s")
    if note:
        print(note)


if __name__ == "__main__":
    main()
