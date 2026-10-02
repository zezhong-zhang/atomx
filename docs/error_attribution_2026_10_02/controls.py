"""Code-comparison diagnostics; external controls never replace native results."""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b
from fac_runner import parse_levels, parse_transitions

from atomx.constants import HARTREE_EV

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "atomic_consistency_2026_10_02"
REF = HERE.parent / "systematic_xray_2026_10_02"
refs = json.loads((REF / "references.json").read_text())


def save(name, rows):
    if not rows:
        raise ValueError("empty diagnostic")
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fluorescence():
    totals, channels = [], []
    for el, (_, shells) in b.CASES.items():
        for model in ("hf", "gu"):
            record = json.loads((OLD / f"{el}-{model}.json").read_text())
            configs = record["configurations"]
            for hole in ("1s", "2s", "2p"):
                ref = refs[el]["decay"][hole]
                raw = REF / "raw" / el / f"fac-{hole}-hole"
                levels = parse_levels(raw / "levels.txt")
                lines = parse_transitions(raw / "tr.txt")
                n = ref["fac"]["initial_levels"]
                weights = {r["ILEV"]: r["TWOJ"] + 1 for r in levels[:n]}
                g = sum(weights.values())
                initial = dict(shells)
                initial[hole] -= 1
                initial_cfg = b.config(initial)
                Ei = configs[initial_cfg]["total_Ha"]
                # Independent check of the FAC input-configuration centroid.
                fac_i = sum(r["ENERGY_eV"] * (r["TWOJ"] + 1) for r in levels[:n]) / g
                offset = n
                collapsed = 0.0
                x_same_energy = 0.0
                fac_resolved = 0.0
                for source in shells:
                    if source == hole:
                        continue
                    final = dict(shells)
                    final[source] -= 1
                    end = offset + b.csf_count(final)
                    group = levels[offset:end]
                    selected = [
                        t
                        for t in lines
                        if t["upper"] < n and offset <= t["lower"] < end
                    ]
                    offset = end
                    if not selected:
                        continue
                    final_cfg = b.config(final)
                    E = (Ei - configs[final_cfg]["total_Ha"]) * HARTREE_EV
                    strength = (
                        sum(
                            weights[t["upper"]] * t["A_per_s"] / t["DELTA_E_eV"] ** 3
                            for t in selected
                        )
                        / g
                    )
                    width = (
                        sum(weights[t["upper"]] * t["A_per_s"] for t in selected) / g
                    )
                    gf = sum(r["TWOJ"] + 1 for r in group)
                    fac_f = sum(r["ENERGY_eV"] * (r["TWOJ"] + 1) for r in group) / gf
                    centroid = fac_i - fac_f
                    control = strength * max(0.0, E) ** 3
                    collapsed += control
                    fac_resolved += width
                    xr = ref["xatom"]["fluorescence"].get(f"{source}->{hole}")
                    x_energy = xr["energy_eV"] if xr else None
                    x_width = xr["rate_per_s"] if xr else 0.0
                    # XATOM has one energy per n-l source. Keep its strength,
                    # replace only its printed photon energy by the native one.
                    x_control = x_width * (E / x_energy) ** 3 if xr and E > 0 else 0.0
                    x_same_energy += x_control
                    channels.append(
                        dict(
                            element=el,
                            exchange=model,
                            hole=hole,
                            source=source,
                            native_photon_eV=E,
                            fac_degeneracy_centroid_eV=centroid,
                            fac_line_min_eV=min(t["DELTA_E_eV"] for t in selected),
                            fac_line_max_eV=max(t["DELTA_E_eV"] for t in selected),
                            fac_strength_effective_photon_eV=(width / strength)
                            ** (1 / 3),
                            fac_resolved_per_s=width,
                            fac_same_native_photon_per_s=control,
                            fac_resolved_over_own_centroid=width
                            / (strength * centroid**3),
                            xatom_photon_eV=x_energy,
                            xatom_per_s=x_width,
                            xatom_same_native_photon_per_s=x_control,
                        )
                    )
                # K-hole runs append a further N-2 Auger block.
                assert offset <= len(levels)
                assert all(t["lower"] < offset for t in lines if t["upper"] < n)
                native = next(
                    r["atomx_per_s"]
                    for r in record["fluorescence"]
                    if r["hole"] == hole and r["energy_mode"] == "relaxed"
                )
                fac = ref["fac"]["radiative_per_s"]
                x = ref["xatom"]["radiative_per_s"]
                assert np.isclose(fac_resolved, fac, rtol=1e-12)
                totals.append(
                    dict(
                        element=el,
                        exchange=model,
                        hole=hole,
                        native_per_s=native,
                        fac_per_s=fac,
                        native_over_fac=native / fac if fac else None,
                        fac_same_native_photon_per_s=collapsed,
                        native_over_fac_same_photon=native / collapsed
                        if collapsed
                        else None,
                        fac_energy_factor=collapsed / fac if fac else None,
                        xatom_per_s=x,
                        native_over_xatom=native / x if x else None,
                        xatom_same_native_photon_per_s=x_same_energy,
                        native_over_xatom_same_photon=native / x_same_energy
                        if x_same_energy
                        else None,
                        xatom_energy_factor=x_same_energy / x if x else None,
                    )
                )
    save("fluorescence_totals.csv", totals)
    save("fluorescence_channels.csv", channels)


def pi_inventory():
    rows = []
    archived = list(csv.DictReader((OLD / "pi.csv").open()))
    for el in b.CASES:
        for shell in ("1s", "2s", "2p"):
            raw = REF / "raw" / el / f"fac-pi-{shell}"
            levels = parse_levels(raw / "levels.txt")
            ni = b.csf_count(b.CASES[el][1])
            gi = sum(r["TWOJ"] + 1 for r in levels[:ni])
            Ei = sum(r["ENERGY_eV"] * (r["TWOJ"] + 1) for r in levels[:ni]) / gi
            gf = sum(r["TWOJ"] + 1 for r in levels[ni:])
            Ef = sum(r["ENERGY_eV"] * (r["TWOJ"] + 1) for r in levels[ni:]) / gf
            mean = Ef - Ei
            for photon in refs[el]["photons_eV"]:
                row = dict(
                    element=el,
                    shell=shell,
                    photon_eV=photon,
                    fac_threshold_centroid_eV=mean,
                    fac_threshold_min_eV=min(r["ENERGY_eV"] for r in levels[ni:])
                    - max(r["ENERGY_eV"] for r in levels[:ni]),
                    fac_threshold_max_eV=max(r["ENERGY_eV"] for r in levels[ni:])
                    - min(r["ENERGY_eV"] for r in levels[:ni]),
                )
                for model in ("hf", "gu"):
                    r = json.loads((OLD / f"{el}-{model}.json").read_text())
                    label = next(
                        k for k in r["thresholds_eV"]["relaxed"] if k.startswith(shell)
                    )
                    row[f"{model}_threshold_eV"] = r["thresholds_eV"]["relaxed"][label]
                    row[f"{model}_threshold_minus_fac_eV"] = (
                        row[f"{model}_threshold_eV"] - mean
                    )
                    for field, mode in [
                        ("residual", "orbital"),
                        ("frozen", "orbital"),
                        ("frozen", "frozen"),
                        ("frozen", "relaxed"),
                    ]:
                        x = next(
                            x
                            for x in archived
                            if x["element"] == el
                            and x["shell"] == shell
                            and x["exchange"] == model
                            and float(x["photon_eV"]) == photon
                            and x["field"] == field
                            and x["energy_mode"] == mode
                        )
                        row[f"{model}_{field}_{mode}_over_fac"] = float(x["ratio"])
                        if mode == "relaxed":
                            row[f"{model}_over_xatom"] = float(x["ratio_xatom"])
                rows.append(row)
    save("pi_inventory.csv", rows)


def fac_pi_grid(variant="focused", elements=None):
    def job(case):
        el, shell = case
        old = REF / "raw" / el / f"fac-pi-{shell}"
        levels = parse_levels(old / "levels.txt")
        ni = b.csf_count(b.CASES[el][1])
        threshold_max = max(r["ENERGY_eV"] for r in levels[ni:]) - min(
            r["ENERGY_eV"] for r in levels[:ni]
        )
        photons = refs[el]["photons_eV"]
        emin = 0.8 * (min(photons) - threshold_max)
        assert emin > 0
        emax = max(photons) * 1.2
        script = (old / "run.sf").read_text()
        original = next(x for x in script.splitlines() if x.startswith("SetPEGrid("))
        script = script.replace(original, f"SetPEGrid(60,{emin:.15g},{emax:.15g})")
        if variant == "point":
            script = script.replace(f"SetAtom('{el}')", f"SetAtom('{el}',-1,-1,0)")
        elif variant in ("radial6400", "radial12800"):
            radial = next(
                x for x in script.splitlines() if x.startswith("SetRadialGrid(")
            )
            from fac_runner import radial_grid_command

            nrad = int(variant.removeprefix("radial"))
            script = script.replace(
                radial, radial_grid_command(nrad, density_scale=nrad / 800)
            )
        out = HERE / "raw" / el / f"fac-pi-{shell}-{variant}"
        b.fac_job(out, script.splitlines())
        weights = np.array([r["TWOJ"] + 1 for r in levels[:ni]])
        values = (
            sum(
                np.loadtxt(out / f"pi-{i}.txt", ndmin=2)[:, 1] * weights[i]
                for i in range(ni)
            )
            / weights.sum()
            * 0.01
        )
        rows = [
            dict(
                element=el,
                shell=shell,
                photon_eV=E,
                fac_original_Mb=refs[el]["fac_pi"][shell][i],
                fac_focused_Mb=float(values[i]),
                focused_over_original=float(values[i] / refs[el]["fac_pi"][shell][i]),
                electron_grid_min_eV=emin,
                electron_grid_max_eV=emax,
                grid_points=60,
            )
            for i, E in enumerate(photons)
        ]
        print(
            el,
            shell,
            "FAC focused grid:",
            [r["focused_over_original"] for r in rows],
            flush=True,
        )
        return rows

    cases = [
        (el, shell) for el in (elements or b.CASES) for shell in ("1s", "2s", "2p")
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = [row for result in executor.map(job, cases) for row in result]
    save(f"fac_pi_{variant}.csv" if variant != "focused" else "fac_pi_grid.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fac-grid", action="store_true")
    parser.add_argument(
        "--fac-variant",
        choices=["focused", "point", "radial6400", "radial12800"],
        default="focused",
    )
    parser.add_argument("--elements", nargs="+", choices=list(b.CASES))
    args = parser.parse_args()
    if not args.fac_grid:
        fluorescence()
        pi_inventory()
    if args.fac_grid:
        fac_pi_grid(args.fac_variant, args.elements)
