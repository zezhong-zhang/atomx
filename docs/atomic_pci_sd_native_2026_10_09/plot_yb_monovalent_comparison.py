"""Export the completed frozen nominal energy comparison, without refitting."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OWNER = Path(__file__).resolve().parent


def main():
    out = OWNER / "yb_monovalent_experiment003"
    source = out / "comparison.json"
    result = json.loads(source.read_text())
    rows = [r for r in result["rows"] if r["relative_error"] is not None]
    rows.extend(result["splittings"])
    names = [
        r"$6p_{1/2}$ excitation",
        r"$6p_{3/2}$ excitation",
        r"$5d_{3/2}$ excitation",
        r"$5d_{5/2}$ excitation",
        r"$6p$ fine splitting",
        r"$5d$ fine splitting",
    ]
    errors = [100 * r["relative_error"] for r in rows]
    gate = 100 * result["relative_error_gate"]
    with plt.rc_context({"font.size": 11, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(figsize=(10, 4.7))
        axes.axvspan(-gate, gate, color="#e3e8ed", label="Original 1% target")
        axes.axvline(0, color="#59636d", linewidth=0.8)
        axes.barh(
            range(6),
            errors,
            color=["#2b6e99" if r["gate_pass"] else "#b94d2b" for r in rows],
            height=0.6,
        )
        for position, value in enumerate(errors):
            axes.text(
                value + (0.13 if value >= 0 else -0.13),
                position,
                f"{value:+.3f}%",
                va="center",
                ha="left" if value >= 0 else "right",
            )
        axes.set_yticks(range(6), names)
        axes.invert_yaxis()
        axes.set_xlim(-11.2, 6.2)
        axes.set_xlabel("Signed relative energy error against frozen NIST Handbook (%)")
        axes.set_title("Yb II: complete declared linearized SD space")
        axes.legend(loc="lower right", frameon=False)
        axes.spines[["top", "right"]].set_visible(False)
        figure.text(
            0.02,
            0.015,
            "68 core electrons; 414 radial virtuals; l <= 3; Emax = 5 Ha; "
            "same computed 6s zero; no fitted shift. Energy comparison only.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        for suffix in ("svg", "png"):
            figure.savefig(out / f"energy_errors.{suffix}", dpi=180)
        plt.close(figure)
    (out / "plot_metadata.json").write_text(
        json.dumps(
            dict(
                comparison_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                matplotlib_version=matplotlib.__version__,
                scope="Derived nominal energy error plot; no new calculation, fit, uncertainty, line strength or convergence claim",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
