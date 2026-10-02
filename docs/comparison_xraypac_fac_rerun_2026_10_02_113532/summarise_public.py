"""Build the public-API comparison tables, diagnostics and standalone figure."""

from __future__ import annotations

import json
from pathlib import Path

import compare as panel
import numpy as np

HERE = Path(__file__).resolve().parent
VARIANTS = [
    "previous_API_default",
    "new_LS_default",
    "new_LS_16000",
    "new_LS_FAC_energy_16000",
    "new_LS_orbital_energy_16000",
    "new_JJ_16000",
]


def numerical(value, prefix=""):
    if isinstance(value, dict):
        return {
            key: v
            for k, item in value.items()
            if "seconds" not in k
            for key, v in numerical(item, prefix + "/" + k).items()
        }
    if isinstance(value, list):
        return {
            key: v
            for i, item in enumerate(value)
            for key, v in numerical(item, prefix + "/" + str(i)).items()
        }
    return {prefix: value} if isinstance(value, (int, float)) else {}


def main():
    public = json.loads((HERE / "public-auger.json").read_text())
    refs = json.loads((HERE / "xatom.json").read_text())
    baseline = Path(json.loads((HERE / "snapshot.json").read_text())["baseline_report"])
    rows, radial_rows, metrics = [], [], {}
    for element, record in public.items():
        for channel in [*record["channels"], "TOTAL"]:
            row = {
                "element": element,
                "source_pair": channel,
                "hole": "1s",
                "ls_radial_representative": "j=l+1/2",
            }
            if channel == "TOTAL":
                row["xatom_per_s"] = record["totals"]["xatom_per_s"]
                row["fac_per_s"] = record["totals"]["fac_per_s"]
                values = record["totals"]
            else:
                data = record["channels"][channel]
                row["xatom_per_s"], row["fac_per_s"] = (
                    data["xatom_per_s"],
                    data["fac_per_s"],
                )
                values = {
                    name: data[name]["rate_per_s"] for name in VARIANTS if name in data
                }
            for name in VARIANTS:
                v = values.get(name)
                row[name + "_per_s"] = v
                row[name + "_relative_to_xatom"] = (
                    v / row["xatom_per_s"] - 1 if v is not None else None
                )
                row[name + "_relative_to_fac"] = (
                    v / row["fac_per_s"] - 1 if v is not None else None
                )
            rows.append(row)
        for channel, data in record["channels"].items():
            a, b = channel.split("+")
            na, la = panel.shell_key(a)
            nb, lb = panel.shell_key(b)
            for trace in data["new_LS_16000"]["radial_trace"]:
                for ordering in ["R12", "R21"]:
                    first, second = (
                        ((na, la), (nb, lb))
                        if ordering == "R12"
                        else ((nb, lb), (na, la))
                    )
                    for rank, value in trace[ordering].items():
                        key = f"{rank},{first[0]}{first[1]},{second[0]}{second[1]},10,{trace['l4']}"
                        ref = refs[element]["auger"]["radial"].get(key)
                        radial_rows.append(
                            {
                                "element": element,
                                "source_pair": channel,
                                "ordering": ordering,
                                "rank": rank,
                                "continuum_l": trace["l4"],
                                "atomx_radial": value,
                                "xatom_radial": ref,
                                "ratio": value / ref if ref else None,
                            }
                        )
        metrics[element] = {
            "default_to_16000_total_relative_change": record["totals"]["new_LS_default"]
            / record["totals"]["new_LS_16000"]
            - 1,
            "default_to_16000_max_channel_relative_change": max(
                abs(
                    v["new_LS_default"]["rate_per_s"] / v["new_LS_16000"]["rate_per_s"]
                    - 1
                )
                for v in record["channels"].values()
            ),
            "production_LS_same_radial_max_relative_error": max(
                abs(v["ratio"] - 1)
                for v in record["same_radial_production_LS_audit"].values()
            ),
        }
    panel.write_csv("public-auger.csv", rows)
    panel.write_csv("radial-integrals.csv", radial_rows)
    comparisons = {}
    for name in ["atomx", "atomx-gu", "pi-refinement", "fac", "xatom"]:
        new = numerical(json.loads((HERE / (name + ".json")).read_text()))
        old = numerical(json.loads((baseline / (name + ".json")).read_text()))
        assert new.keys() == old.keys()
        changed = [k for k in new if new[k] != old[k]]
        comparisons[name] = {
            "numeric_fields_compared": len(new),
            "changed_fields": changed,
            "max_absolute_change": max(
                (abs(new[k] - old[k]) for k in new), default=0.0
            ),
        }
    panel.dump(
        HERE / "rerun-checks.json",
        {
            "public_api": metrics,
            "independent_driver_vs_previous": comparisons,
            "comparisons_exclude_timings": True,
        },
    )
    fe = json.loads((HERE / "fe-panel.json").read_text())["rows"]
    panel.write_csv(
        "fe-panel.csv",
        [
            {k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items()}
            for r in fe
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.9), gridspec_kw={"width_ratios": [1, 2.4]}
    )
    for ax, (element, record) in zip(axes, public.items(), strict=True):
        channels = list(record["channels"]) + ["TOTAL"]
        for key, title, color, marker in [
            (
                "new_LS_default",
                "New public LS, XATOM energy (400 points)",
                "#2563eb",
                "o",
            ),
            ("new_LS_16000", "Same, 16000 points", "#059669", "x"),
            (
                "new_LS_orbital_energy_16000",
                "New public LS, orbital-sum energy",
                "#7c3aed",
                "^",
            ),
            ("fac_per_s", "FAC (its own level energies)", "#d97706", "s"),
        ]:
            values = []
            for ch in channels:
                if ch == "TOTAL":
                    r = record["totals"]
                    value = r[key] / r["xatom_per_s"] - 1
                else:
                    r = record["channels"][ch]
                    value = (
                        r[key] if key == "fac_per_s" else r[key]["rate_per_s"]
                    ) / r["xatom_per_s"] - 1
                values.append(100 * value)
            ax.plot(
                np.arange(len(channels)),
                values,
                ls="none",
                color=color,
                marker=marker,
                label=title,
            )
        ax.axhline(0, color="#27272a", lw=0.9, label="XATOM")
        ax.set_xticks(np.arange(len(channels)), channels, rotation=55, ha="right")
        ax.set_title(f"{element}+ single 1s vacancy")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylim(-35, 55)
    axes[0].set_ylabel("Difference from XATOM (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"public-auger.{ext}", dpi=180)
    plt.close(fig)
    print(
        f"{len(rows)} public API rows; {len(radial_rows)} radial records; 5 Fe levels; rerun checks and plots written."
    )


if __name__ == "__main__":
    main()
