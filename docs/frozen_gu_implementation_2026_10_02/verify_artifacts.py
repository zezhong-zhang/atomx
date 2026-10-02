"""Check frozen-field panel controls, immutable archives, and source identity."""

import argparse
import contextlib
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
OLD = HERE.parent / "atomic_consistency_2026_10_02"
PRIOR = HERE.parent / "error_attribution_2026_10_02"
REF = HERE.parent / "systematic_xray_2026_10_02" / "references.json"
REFERENCE_HASH = "1b30f8cb8c140074a0f001852572fc5aee69dc8be97b686f88fcade2429dc59a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(live=False):
    old_identity = json.loads((OLD / "result_identity.json").read_text())
    for filename, expected in old_identity["files_sha256"].items():
        assert sha(OLD / filename) == expected, filename
    assert sha(REF) == REFERENCE_HASH, "FAC reference archive changed"
    identity = HERE / "result_identity.json"
    if identity.exists():
        for filename, expected in json.loads(identity.read_text())[
            "files_sha256"
        ].items():
            assert sha(ROOT / filename) == expected, filename

    with (HERE / "pi_validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 288
    keys = [
        (r["element"], r["ns"], r["variant"], r["shell"], r["photon_eV"]) for r in rows
    ]
    assert len(set(keys)) == len(keys)
    for row in rows:
        for name in ("sigma_Mb", "archived_fac_Mb", "ratio"):
            assert np.isfinite(float(row[name])) and float(row[name]) > 0
        np.testing.assert_allclose(
            float(row["ratio"]),
            float(row["sigma_Mb"]) / float(row["archived_fac_Mb"]),
            rtol=1e-14,
        )
    records = [json.loads(p.read_text()) for p in sorted(HERE.glob("*-ns*.json"))]
    assert len(records) == 8
    shooting_changes = []
    ratios = {}
    references = json.loads(REF.read_text())
    for record in records:
        el, ns = record["element"], record["ns"]
        values = record["variants_Mb"]
        for shell in ("1s", "2s", "2p"):
            np.testing.assert_array_equal(
                values["production"][shell], values["production_nrad128000"][shell]
            )
            for variant in values:
                panel_rows = [
                    r
                    for r in rows
                    if r["element"] == el
                    and int(r["ns"]) == ns
                    and r["variant"] == variant
                    and r["shell"] == shell
                ]
                assert len(panel_rows) == 3
                np.testing.assert_array_equal(
                    [float(r["sigma_Mb"]) for r in panel_rows], values[variant][shell]
                )
            if ns == 180:
                prior = json.loads((PRIOR / f"pi-{el}.json").read_text())[
                    "variants_Mb"
                ]["native_16000"][shell]
                np.testing.assert_allclose(
                    values["rebuilt_legacy"][shell], prior, rtol=1e-11, atol=0
                )
                shooting_changes.extend(
                    abs(
                        np.asarray(values["production_shooting32000"][shell])
                        / values["production"][shell]
                        - 1
                    )
                )
        if ns == 180:
            ratio = np.concatenate(
                [
                    np.asarray(values["production"][sh]) / references[el]["fac_pi"][sh]
                    for sh in ("1s", "2s", "2p")
                ]
            )
            ratios[el] = {"minimum": float(ratio.min()), "maximum": float(ratio.max())}
    assert "100 passed" in (HERE / "tests.log").read_text()
    if live:
        from validate_panel import run

        archived = next(r for r in records if r["element"] == "Ne" and r["ns"] == 180)
        # Check the final public API while leaving the panel archive immutable.
        import tempfile

        import validate_panel as panel

        original_here = panel.HERE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                panel.HERE = Path(tmp)
                with contextlib.redirect_stdout(sys.stderr):
                    current = run("Ne")
        finally:
            panel.HERE = original_here
        for variant in archived["variants_Mb"]:
            for shell in ("1s", "2s", "2p"):
                np.testing.assert_array_equal(
                    current["variants_Mb"][variant][shell],
                    archived["variants_Mb"][variant][shell],
                )
    summary = dict(
        pi_values=len(rows),
        bound_results=len(records),
        legacy_points=54,
        legacy_relative_tolerance=1e-11,
        n_radial_controls="bitwise identical",
        shooting_max_relative_change_ns180=float(max(shooting_changes)),
        production_fac_ratios_ns180=ratios,
        unchanged_original_archive_files=len(old_identity["files_sha256"]),
        unchanged_reference_sha256=REFERENCE_HASH,
        final_public_api_ne_panel="bitwise identical" if live else "not rerun",
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-check", action="store_true")
    args = parser.parse_args()
    verify(args.live_check)
