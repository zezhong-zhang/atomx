"""Seal full local evidence; deliver source, fixtures, virtual snapshots and summaries."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PRIOR=ROOT.parent/"yb_diagnosis_repair_2026_10_08"
DYSON=ROOT.parent/"yb_energy_dependent_sigma_2026_10_08"
OLD=ROOT.parent/"rare_earth_yb_spectrum_2026_10_08"


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    source=json.loads((ROOT/"capture_source_sha256.json").read_text())
    if any(sha(REPO/name)!=digest for name,digest in source.items()):
        raise ValueError("Production source changed before sealing")
    if not json.loads((ROOT/"capture_coverage/results.json").read_text())["passed"]:
        raise ValueError("Capture coverage failed")
    # The validated affected tests are carried as their unchanged prior archive.
    (ROOT/"REPLAY.txt").write_text('''Yb correlation-tail and radial controls, 2026-10-08.

The delivery includes complete production source, coarse/refined DF inputs, all measured JSON/CSV summaries, the 200-Ha virtual-orbital snapshots, and scientific drivers. Full large poles.npz coupling arrays are excluded from this delivery, retained locally and individually sealed in artifact_manifest.json. The bundle therefore supports native recapture, not immediate offline full-pole replay.

Extract Atomx_capture_source.zip into the same directory as this bundle, preserving src/ and pyproject.toml. Use the existing numerical environment or install pyproject.toml dependencies in an isolated environment. Rendering uses a separate Markdown/WeasyPrint environment; numerical dependencies must not change to render a PDF. For the test panel, extract the included unchanged Affected_tests_source.zip into the same checkout.

Drivers reject overwriting scientific evidence. Retain a complete unchanged copy of the extracted delivery first. To recapture in a second independent checkout, move this topic's existing results/, analysis/, coarse_preliminary/, capture_coverage/ and absolute_anchor/ aside. Keep the bundled preflight/, representation_audit_v4/, literature_check/ and earlier-stage reference inputs unchanged. Set PYTHONPATH=src and run:
python docs/yb_correlation_tail_2026_10_08/run_controls.py
python docs/yb_correlation_tail_2026_10_08/analyze_controls.py
python docs/yb_correlation_tail_2026_10_08/verify_capture_coverage.py
python docs/yb_correlation_tail_2026_10_08/absolute_energy_anchor.py
python docs/yb_correlation_tail_2026_10_08/build_report.py

The original full-workspace audit_representation.py checks the native Gram matrices and also complete manifests of the two original sealed evidence folders. Those earlier folders include large artifacts excluded here, so that archival-preservation audit requires the original workspace. For an independent bundle recapture, retain the included already sealed representation_audit_v4/ result; do not move it aside or describe it as a new preservation pass. verify_capture_coverage.py checks the newly constructed complete virtual manifests against the same bundled, metric-validated snapshots. The original audit script is retained as evidence of the actual measurement.

Pure Dirac-Coulomb, frozen-orbital scalar Sigma2 Dyson roots. No fitted shifts or imported higher-order corrections. Space increments, numerical residuals and experimental errors are distinct. The report states which gates failed.
''')
    exclusions={"artifact_manifest.json","bundle_identity.json","delivery_receipts.json","Yb相关尾部_复现包.zip"}
    files=[p for p in sorted(ROOT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name not in exclusions]
    manifest={str(p.relative_to(REPO)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    (ROOT/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    extras=[REPO/"README.md",REPO/"LICENSE",PRIOR/"common.py",DYSON/"replay_and_ci/subsets.json",DYSON/"final_source_sha256.json",DYSON/"Affected_tests_source.zip",OLD/"render_report.py"]
    for l in (4,6):extras.append(PRIOR/f"results/l{l}_e50_all/results.json")
    for base in ("v68_default","v68_refined_newton"):
        extras.extend(OLD/"results"/base/name for name in ("result.pkl","core.json","structure.json","status.json","orbitals_workspace.npz"))
    extra_identity={str(p.relative_to(REPO)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in extras}
    (ROOT/"bundle_identity.json").write_text(json.dumps(dict(
        HEAD=subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip(),
        source_archive_sha256=sha(ROOT/"Atomx_capture_source.zip"),
        source_unchanged_during_and_after_capture=True,
        manifest_sha256=sha(ROOT/"artifact_manifest.json"),extra_delivery_inputs=extra_identity,
        omitted_full_pole_arrays=[name for name in manifest if name.endswith("/poles.npz")],
        policy="Only large coupling-pole arrays are omitted; all small full virtual-orbital snapshots and both complete DF input fixtures are included."),indent=2)+"\n")
    deliver=[p for p in files if p.name!="poles.npz"]+extras+[ROOT/"artifact_manifest.json",ROOT/"bundle_identity.json"]
    bundle=ROOT/"Yb相关尾部_复现包.zip"
    seen=set()
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in deliver:
            name=str(p.relative_to(REPO))
            if name not in seen:archive.write(p,name);seen.add(name)
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:raise ValueError("Bundle CRC failure")
        for p in deliver:
            if hashlib.sha256(archive.read(str(p.relative_to(REPO)))).hexdigest()!=sha(p):
                raise ValueError("Bundle payload hash failure")
    print("SEALED",len(manifest),"local artifacts; delivery",len(seen),"files",bundle.stat().st_size,"bytes")


if __name__=="__main__":main()
