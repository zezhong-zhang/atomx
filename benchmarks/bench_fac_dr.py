#!/usr/bin/env python3
"""Public native DR calculation against the matched FAC Fe dr33 reference.

The reference retains 237 levels in four separately mixed hole/parity
spaces, all ground-channel Auger rates, and finite-frequency E1 stabilization
into the five n=3 final levels. Both calculations use independently optimized
local ground potentials, point nuclei, infinite mass and no corrections.
FAC also disables configuration shifts and small-coefficient pruning.

Thermal rates combine each code's energies and widths through the same
Burgess kernel; this does not independently check the kernel prefactor.
The 2% aggregate gate does not imply 2% agreement of individual widths.
Raw external output is retained under
benchmarks/runs/2026-09-30-dr-radiative/complete-fe-fixed/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Basis, Sampling  # noqa: E402
from atomx.scattering.autoionization import (  # noqa: E402
    AutoionizationState,
    dr_calculate,
    dr_rate_coefficient,
)

K_B_EV_PER_K = 8.617333262e-5
TEMPERATURES_MK = [1.0, 5.0, 10.0, 50.0]
TEMPERATURES_EV = [T * 1.0e6 * K_B_EV_PER_K for T in TEMPERATURES_MK]

FIXTURE_MATCHED = ROOT / "tests" / "data" / "fac_fe_dr33_matched.json"

RMS_TARGET = 0.02

def run_atomx():
    """Solve Fe XVII and drive the public manifold-backed DR API."""
    atom = Atom(26, charge=16, config='1s2 2s2 2p6', engine='shooting',
                exchange='gu', corrections='none', nuclear='point',
                basis=Basis(sampling=Sampling(n_grid=4000)))
    atom.solve(tol=1e-10, max_iter=150)
    if not atom.converged:
        raise RuntimeError('ground SCF did not converge')
    return dr_calculate(atom, "1s2 2s2 2p6 3s1", T_eV=TEMPERATURES_EV)


def load_fac_reference():
    """Matched FAC dr33 energies/widths -> alpha_FAC(T) via the same kernel."""
    data = json.loads(FIXTURE_MATCHED.read_text())
    states = [
        AutoionizationState(
            label=str(r["ilev"]),
            E_resonance_eV=float(r["energy_eV"]),
            J=float(r["two_J"])/2,
            gamma_auger_per_s=float(r["auger_per_s"]),
            a_radiative_per_s=float(r["radiative_per_s"]),
        )
        for r in data["resonances"]
    ]
    rc = dr_rate_coefficient(states, TEMPERATURES_EV, initial_state_g=1.0)
    return rc, data["source"], len(states)


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
    args = parser.parse_args()

    rc_fac, provenance, n_res = load_fac_reference()
    note = "ref: matched FAC fac_fe_dr33_matched.json; native local Gu, point nucleus, dr33"
    rc_atomx = run_atomx()

    rows = []
    for i, (T_mk, T_eV) in enumerate(zip(TEMPERATURES_MK, TEMPERATURES_EV, strict=False)):
        fac = float(rc_fac.alpha_cm3_s[i])
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
            lineterminator="\n",
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
            "note": (
                f"target |rel| rms <= {RMS_TARGET:.2f} "
                "(matched frozen-CI spaces; independently optimized local orbitals)"
            ),
        })

    print(f"Wrote {out}")
    print(f"FAC reference: {provenance} ({n_res} resonances)")
    print(f"atomx manifold resonances: {len(rc_atomx.resonances)}")
    for r in rows:
        ratio = r["alpha_atomx_cm3_s"] / r["alpha_fac_cm3_s"]
        print(
            f"  T={r['T_MK']:>4} MK  atomx={r['alpha_atomx_cm3_s']:.3e}  "
            f"FAC={r['alpha_fac_cm3_s']:.3e}  ratio={ratio:.3f}"
        )
    print(f"rms |rel_diff| = {rms:.4f} (target <= {RMS_TARGET})")


if __name__ == "__main__":
    main()
