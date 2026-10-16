"""Plot only admitted, paired fixed-work CPU measurements from performance.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(owner=None):
    owner = Path(__file__).resolve().parent if owner is None else Path(owner)
    accepted = defaultdict(list)
    with (owner / "performance.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["controlled"] != "True" or not row["speedup"]:
                continue
            accepted[row["workload"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for name, group in accepted.items():
        group.sort(key=lambda row: int(row["workers"]))
        workers = np.array([int(row["workers"]) for row in group])
        for axis, value, low, high in (
            (axes[0], "speedup", "speedup_min", "speedup_max"),
            (axes[1], "efficiency", "efficiency_min", "efficiency_max"),
        ):
            middle = np.array([float(row[value]) for row in group])
            lower = np.array([float(row[low]) for row in group])
            upper = np.array([float(row[high]) for row in group])
            spread = np.vstack((middle - lower, upper - middle))
            scale = 100 if value == "efficiency" else 1
            axis.errorbar(
                workers,
                middle * scale,
                yerr=spread * scale,
                marker="o",
                capsize=3,
                label=name,
            )
    axes[0].plot([1, 2, 4, 8], [1, 2, 4, 8], "k--", lw=1, label="Ideal")
    axes[1].axhline(100, color="black", ls="--", lw=1)
    axes[0].set_ylabel("Paired speedup S = T1 / Tp")
    axes[1].set_ylabel("Logical team efficiency E = S / p (%)")
    for axis in axes:
        axis.set_xticks([1, 2, 4, 8])
        axis.set_xlabel("Requested workers p (fixed work)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Actual CPU strong scaling: distinct recorded workload scopes")
    fig.supxlabel(
        "Medians and full observed paired ranges; not confidence intervals. "
        "Only admitted CPU groups plotted; rejected CPU/GPU observations remain in reports.",
        fontsize=9,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(owner / f"parallel_efficiency.{suffix}", dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner", type=Path, help="Directory containing the admitted performance.csv"
    )
    main(parser.parse_args().owner)
