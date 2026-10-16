"""Seal measured identities and a self-contained source/input evidence bundle."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from common import ROOT, REPO, OLD, save, sha, source_identity


def main():
    result=json.loads((ROOT/"results/l6_e50_all/results.json").read_text())
    current=source_identity()
    measured=json.loads((ROOT/"results/l6_e50_all/identity.json").read_text())["source_sha256"]
    changes=[p for p in current if current[p]!=measured[p]]
    assert set(changes)=={"src/atomx/corrections/qed.py","src/atomx/corrections/qedmod/mohr.py",
                         "src/atomx/corrections/qedmod/indelicato.py","src/atomx/methods/ci_mbpt.py"}
    assert current["src/atomx/methods/mbpt_transitions.py"]==measured["src/atomx/methods/mbpt_transitions.py"]
    with zipfile.ZipFile(ROOT/"Atomx_MBPT_measurement_source.zip") as z:
        for path,digest in measured.items():assert hashlib.sha256(z.read(path)).hexdigest()==digest
    initial=json.loads((ROOT/"contract.json").read_text())
    save(ROOT/"source_chain.json",dict(initial_clean_HEAD="233fec1a9710e95cf42bf64ecb821d287d4336e2",
        previous_baseline_changes=initial["previous_baseline_changes"],
        measurement_source_archive_sha256=sha(ROOT/"Atomx_MBPT_measurement_source.zip"),
        final_source_sha256=current, changed_after_full_MBPT_measurement=changes,
        justification="Only analytic-QED warnings and documentation changed after complete sums. MBPT implementation and kernels are identical. Current-source Yb CI injection and unchanged QED values were actually replayed.",
        early_pilots="l2_n4 predates virtual-manifest/base-selection diagnostic additions. Its recorded identity and outputs are retained; the sealed primary complete-window drivers correspond to l4_e50_all and l6_e50_all.",
        measured_source_unchanged=result["source_unchanged"]))
    testlog=(ROOT/"final_physics_tests.log").read_text()
    assert "129 passed" in testlog and not re.search(r"\bFAILED\b",testlog)
    for path in ["baseline_audit_v3/results.json","gauge_audit_v3/results.json",
                 "results/l6_e50_all/ci_injection/results.json","qed_coverage_after.json"]:
        assert json.loads((ROOT/path).read_text())["passed"],path
    summary=json.loads((ROOT/"report_summary.json").read_text())
    pdf=(ROOT/"pdf_extracted.txt").read_text()
    for number in ["23798.59","25232.04","28568.40","32103.42","578.14","861"]:
        assert number in pdf,number
    for md in [ROOT/"README.md",REPO/"docs/README.md",REPO/"docs/AGENT_REFERENCE.md",
               REPO/"docs/AGENT_INSIGHTS.md",REPO/"docs/ARCHITECTURE_CAPABILITIES.md"]:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)",md.read_text()):
            if target.startswith(("https://","http://","#")):continue
            if target.startswith("../") or md==ROOT/"README.md" or "yb_diagnosis_repair" in target:
                path=(md.parent/target.split("#")[0])
                if path.name in ("verification.json","source_chain.json"):continue
                assert path.exists(),str(path)
    save(ROOT/"verification.json",dict(affected_physics_tests=129,failed_tests=0,
        coverage_warnings=27, test_command="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_mbpt_transitions.py tests/test_ci_mbpt_sigma1.py tests/test_mbpt2.py tests/test_mbpt_sigma2.py tests/test_qed_self_energy_table.py tests/test_qed_backend_auto.py tests/test_qed_indelicato_vs_fac.py tests/test_qed_orbital_shifts.py tests/test_qed_approximation_coverage.py tests/test_velocity_gauge_full_order.py tests/test_finite_frequency_electric.py -q",
        baseline_replay=True,independent_DF_and_dipole=True,Fock_commutator_closure=True,
        native_Yb_CI_injection=True, unchanged_QED_value_replay=True,
        measurement_source_archive_verified=True,local_links_checked=True,
        PDF_numbers_checked=True, PDF_pages_visually_checked=[1,3],
        source_lint="ruff check passed for all five changed production modules and two affected test files",
        preserved_failures=["baseline_audit adapter initial_state_index error",
            "baseline_audit_v2 omitted required kappas in finite-frequency operator",
            "gauge_audit_v2 raw signed publication comparison: different phases and small-denominator printed-energy mismatch",
            "qed_coverage_before: 3 failed, 1 passed"],
        physical_limits=summary))
    save(ROOT/"documentation_disposition.json",dict(
        updated=["docs/README.md","docs/AGENT_REFERENCE.md","docs/AGENT_INSIGHTS.md",
            "docs/ARCHITECTURE_CAPABILITIES.md","docs/TESTING.md","docs/VERIFICATION.md",
            "src/atomx/methods/ci_mbpt.py docstring","src/atomx/corrections/qed.py docstrings"],
        retained={"docs/ARCHITECTURE_DESIGN.md":"Lower-level method/numerics/operator ownership preserved; no new Atom.solve backend or result type",
            "original Yb report and bundles":"Immutable dated baseline evidence; newest continuation is indexed as current diagnosis",
            "AGENTS.md":"No new root principle; evidence merged into the existing convention-first insight"},
        registered="Nonlocal Fock gauge assumptions are a scoped supplement to convention-first, not a correlated-rate certification"))
    patch=subprocess.run(["git","diff","--","src/atomx/methods/mbpt_transitions.py","src/atomx/methods/ci_mbpt.py",
        "src/atomx/corrections/qed.py","src/atomx/corrections/qedmod/mohr.py","src/atomx/corrections/qedmod/indelicato.py",
        "tests/test_mbpt_transitions.py"],cwd=REPO,text=True,capture_output=True,check=True).stdout
    newtest=subprocess.run(["git","diff","--no-index","--","/dev/null","tests/test_qed_approximation_coverage.py"],cwd=REPO,text=True,capture_output=True)
    assert newtest.returncode==1
    (ROOT/"source_changes.patch").write_text(patch+newtest.stdout)
    with zipfile.ZipFile(ROOT/"Atomx_final_source.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted((REPO/"src/atomx").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:z.write(p,p.relative_to(REPO))
        z.write(REPO/"pyproject.toml","pyproject.toml")
        for name in ["test_mbpt_transitions.py","test_ci_mbpt_sigma1.py","test_review_regressions.py","test_qed_approximation_coverage.py"]:
            z.write(REPO/"tests"/name,"tests/"+name)
    ignored={"artifact_manifest.json","bundle_identity.json","delivery_receipts.json"}
    entries=[p for p in sorted(ROOT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
        and p.name not in ignored and not p.name.startswith("Yb谱线定位修复_复现数据")
        and not (p.parent.name=="references" and p.suffix in (".pdf",".png"))]
    save(ROOT/"artifact_manifest.json",{str(p.relative_to(ROOT)):sha(p) for p in entries})
    bundle=ROOT/"Yb谱线定位修复_复现数据.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for p in entries+[ROOT/"artifact_manifest.json"]:z.write(p,p.relative_to(REPO))
        for base in ["v68_default","v68_box90_newton"]:
            for name in ["result.pkl","core.json","structure.json","status.json","orbitals_workspace.npz","experiment_contract.json"]:
                p=OLD/"results"/base/name;z.write(p,p.relative_to(REPO))
        for name in ["README.md","render_report.py"]:
            p=OLD/name;z.write(p,p.relative_to(REPO))
        # Full original papers/screenshots are local reading aids, not redistributed.
        z.writestr("REPLAY.txt","Extract in a repository root. Atomx_final_source.zip contains final package source and pyproject; Atomx_MBPT_measurement_source.zip seals the actual full-window source. Use the existing .venv or install declared dependencies. Inspect source_changes.patch and README before replay; choose a new output label. The original Yb evidence is retained under docs/rare_earth_yb_spectrum_2026_10_08.\n")
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for p in entries:assert hashlib.sha256(z.read(str(p.relative_to(REPO)))).hexdigest()==sha(p)
    save(ROOT/"bundle_identity.json",dict(filename=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),
        manifest_entries=len(entries),ZIP_CRC_and_manifest_payloads_verified=True,
        external_papers_redistributed=False))
    print("SEALED",bundle.stat().st_size,"bytes",len(entries),"hashed artifacts",flush=True)


if __name__=="__main__":main()
