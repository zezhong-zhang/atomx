"""Package final native source and sufficient frozen inputs; retain raw poles locally."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / "Atomx_Yb_matrix_source_replay.zip"
TESTS = (
    "test_mbpt_matrix.py", "test_mbpt_transitions.py", "test_mbpt2.py",
    "test_ci_mbpt_sigma1.py", "test_mbpt_sigma2.py",
    "test_architecture_boundaries.py", "test_numerical_design_contracts.py",
    "test_qed_approximation_coverage.py", "test_qed_self_energy_table.py",
    "test_pair_ladder.py", "test_ladder_and_sampling.py",
    "test_review_regressions.py",
)


def main():
    files = set((REPO / "src/atomx").rglob("*.py"))
    files.update(REPO / name for name in ("pyproject.toml", "README.md", "LICENSE", "conftest.py"))
    files.update(REPO / "tests" / name for name in TESTS)
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(x in p.parts for x in ("__pycache__", "preview", "previews")):
            continue
        if p.suffix in (".zip", ".pyc") or p.name == "matrix_poles.npz":
            continue
        if p.name.startswith("delivery_receipt") or p.name in ("package_file_list.json",):
            continue
        files.add(p)
    for topic in ("yb_diagnosis_repair_2026_10_08", "yb_energy_dependent_sigma_2026_10_08",
                  "yb_correlation_tail_2026_10_08", "yb_literature_repair_strategy_2026_10_08",
                  "rare_earth_yb_spectrum_2026_10_08"):
        files.add(REPO / "docs" / topic / "README.md")
    files.add(REPO / "docs/yb_diagnosis_repair_2026_10_08/common.py")
    files.add(REPO / "docs/rare_earth_yb_spectrum_2026_10_08/render_report.py")
    files.add(REPO / "docs/yb_correlation_tail_2026_10_08/absolute_anchor/results.json")
    for reference in ("v68_default", "v68_refined_newton"):
        base = REPO / "docs/rare_earth_yb_spectrum_2026_10_08/results" / reference
        files.update(base / name for name in (
            "result.pkl", "status.json", "core.json", "structure.json",
            "orbitals_workspace.npz", "inputs.json", "source_identity.json",
            "experiment_contract.json",
        ))
    for case in ("coarse_l8_e200", "fine_l6_e200"):
        for label in ("6s_1_2", "5d_3_2", "5d_5_2", "6p_1_2", "6p_3_2"):
            files.add(REPO / "docs/yb_correlation_tail_2026_10_08/results" / case / label / "results.json")
    entries = {}
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(files):
            payload = p.read_bytes()
            name = str(p.relative_to(REPO))
            entries[name] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
            archive.writestr(name, payload)
        manifest = dict(files=entries, excluded={
            "full_matrix_poles": "Ten production and five independent-validation matrix_poles.npz files remain locally; regenerate with supplied source/inputs",
            "previous_archives": "Only previous frozen inputs, scalar anchors and report context included; full old archive audit requires original repo",
            "nested_source_zips": "Stopped-attempt source archives remain in original repo; final source is src/atomx in this bundle",
        })
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    (ROOT / "package_file_list.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(path=str(OUTPUT.relative_to(REPO)), included_files=len(files), bytes=OUTPUT.stat().st_size)))


if __name__ == "__main__":
    main()
