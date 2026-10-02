"""Live public APIs against the existing five-level Fe DR FAC fixture.

FAC is archived for this supplementary panel, unlike the fresh Ne/Ar runs.
The resonances have 2p holes and two n=3 electrons (11-electron Fe), and
decay to the 10-electron Fe XVII target. They are not single-K-hole states.
"""

import json
from pathlib import Path

import compare as panel
from public_auger import load_previous_api

from atomx import Atom, Correlation
from atomx.scattering.autoionization import auger_rate_csf_average
from atomx.scattering.autoionization.jj_level import auger_rate_jj_level

HERE = Path(__file__).resolve().parent
INTERMEDIATE_J2 = {32: 0, 71: 0, 72: 0, 142: 0, 190: 2}


def main():
    reference = json.loads((HERE / "fe-reference.json").read_text())
    atom = Atom(
        Z=26,
        charge=16,
        config=["1s2 2s2 2p6", "1s2 2s2 2p5 3s1", "1s2 2s2 2p5 3p1", "1s2 2s2 2p5 3d1"],
        correlation=Correlation(radial_optimization="fac_frozen"),
        corrections="none",
    )
    atom.solve()
    previous = load_previous_api()
    rows = []
    for row in reference["resonances"]:
        args = dict(
            hole_label=row["atomx_hole_label"],
            source_label=row["atomx_source_label"],
            source2_label=row["atomx_source2_label"],
            resonance_energy_eV=row["E_resonance_eV"],
        )
        rates = {
            "old_coarse": previous.auger_rate_csf_average(atom._result, **args)[
                "Gamma_per_s"
            ],
            "new_coarse": auger_rate_csf_average(atom._result, **args)["Gamma_per_s"],
            "exact_jj_level": auger_rate_jj_level(
                atom._result,
                hole_label=args["hole_label"],
                e1_label=args["source_label"],
                e2_label=args["source2_label"],
                two_J_inter=INTERMEDIATE_J2[row["ilev"]],
                two_J_res=int(round(2 * row["J"])),
                resonance_energy_eV=args["resonance_energy_eV"],
                scheme="pair"
                if args["source_label"] == args["source2_label"]
                else "sequential",
            ),
        }
        out = {
            "ilev": row["ilev"],
            "J": row["J"],
            "hole": args["hole_label"],
            "sources": [args["source_label"], args["source2_label"]],
            "electron_eV": args["resonance_energy_eV"],
            "fac_per_s": row["gamma_auger_per_s"],
        }
        for name, value in rates.items():
            out[name + "_per_s"] = float(value)
            out[name + "_ratio_to_fac"] = float(value / row["gamma_auger_per_s"])
        rows.append(out)
    panel.dump(
        HERE / "fe-panel.json",
        {"reference_type": "archived FAC fixture; no fresh FAC Fe run", "rows": rows},
    )
    for row in rows:
        print(
            row["ilev"],
            row["old_coarse_ratio_to_fac"],
            row["new_coarse_ratio_to_fac"],
            row["exact_jj_level_ratio_to_fac"],
            flush=True,
        )


if __name__ == "__main__":
    main()
