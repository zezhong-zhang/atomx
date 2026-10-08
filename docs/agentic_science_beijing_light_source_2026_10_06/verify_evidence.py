"""Verify documentary evidence, never import or run the archived physics code."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify():
    checks = []

    def check(name, condition, details=None):
        checks.append({"name": name, "pass": bool(condition), "details": details})

    index = read("evidence_index.json")
    for f in index["files"]:
        path = ROOT / f["path"]
        check("evidence hash: " + f["path"], path.exists() and digest(path) == f["sha256"])
    for f in read("sources/index.json"):
        path = ROOT / f["path"]
        # The distribution ZIP omits the complete third-party source documents.
        if path.exists():
            check("source hash: " + f["path"], digest(path) == f["sha256"])
    archive_manifest = Path(index["archive_root"]) / "manifest.json"
    if archive_manifest.exists():
        check("original sealed archive manifest unchanged", digest(archive_manifest) == index["archive_manifest_sha256"])
    analysis = read("evidence/e3200/analysis.json")
    text = (ROOT / "参赛项目书.md").read_text(encoding="utf-8")
    singles = read("evidence/e3200/singles_analysis.json")
    breit = read("evidence/breit_path_3200.json")
    tail = read("evidence/tail_model.json")
    freq = read("evidence/frequency_reference_projection.json")
    poles = read("evidence/wall_center_comparison.json")
    numeric = []
    for state in ["q", "r"]:
        t = analysis["transitions"][state]
        numeric.extend([
            (state+" photon eV", t["energy_eV"],9),
            (state+" photon bias meV", t["minus_experiment_meV"],6),
            (state+" complete tail meV", t["extension_increment_meV"],6),
            (state+" singles meV", singles["new_singles_photon_increment_meV"][state],6),
            (state+" Breit switch meV", breit["photons"][state]["exact_Breit_increment_meV"],6),
            (state+" Breit nonlinear meV", breit["photons"][state]["already_included_nonlinear_meV"],6),
            (state+" Breit paper difference meV", breit["photons"][state]["exact_Breit_minus_2017_printed_meV"],6),
            (state+" measured forecast meV", tail["rows"][state]["predeclared_p15_prediction_meV"],6),
            (state+" forecast error meV", tail["rows"][state]["p15_prediction_error_meV"],6),
            (state+" reference frequency meV", freq["photons"][state]["first_order_reference_frequency_response_meV"],6),
            (state+" reference bound meV", freq["photons"][state]["finite_ordered_eigenvalue_Weyl_photon_bound_meV"],6),
        ])
        check(state+" local increment remains failed", t["local_increment_gate_pass"] is False)
        check(state+" forecast remains failed", tail["rows"][state]["p15_prediction_gate_pass"] is False)
    numeric.extend([
        ("split meV", analysis["split_meV"],6),
        ("split minus joint experiment meV",analysis["split_minus_experiment_meV"],6),
        ("q rank2 pole transition meV", poles["rows"]["N3_K3"]["transition_real_increment_meV"],6),
        ("r rank2 pole transition meV", poles["rows"]["N3_K1"]["transition_real_increment_meV"],6),
    ])
    for name, value, digits in numeric:
        spelling = f"{value:.{digits}f}"
        check("document scalar: "+name, spelling in text.replace("−","-"), {"value":value,"document_spelling":spelling})
    check("full correlation remains uncompleted", analysis["full_correlation_converged"] is False)
    check("physical pole remains uncertified", analysis["physical_pole_certified"] is False and poles["physical_poles_certified"] is False)
    val=read("evidence/validation.json")
    check("original finite-model independent validation passed", val["pass_"] is True)
    check("source snapshot file count", val["source_files_verified"]==332)
    seal=read("evidence/seal_check.json")
    check("original archive count and seal", seal["pass_"] and seal["files_verified"]==50834 and seal["manifest_sha256"]==index["archive_manifest_sha256"])
    links=[];external_repo=[]
    for name in ["参赛项目书.md","评审摘要.md","README.md"]:
        content=(ROOT/name).read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            if urlsplit(target).scheme or target.startswith("#"):continue
            relative=unquote(target.split("#",1)[0]);path=(ROOT/relative).resolve()
            if path == ROOT / "verification.json":
                # This report is emitted below; its pre-existing presence is not an input gate.
                continue
            if not path.is_relative_to(ROOT):
                external_repo.append({"document":name,"target":relative,"exists_in_workspace":path.exists()})
                # The original archive is separately required for a full replay.
                continue
            links.append({"document":name,"target":relative,"exists":path.exists()})
            check("local link: "+name+" -> "+relative,path.exists())
    check("evidence subset count matches document",len(index["files"])==21)
    output={"pass":all(x["pass"] for x in checks),"scope":"Documentary hashes, selected reported scalars, preservation of failed scientific gates, and package-local links. No physics computation, no proof of remaining physical error, no platform submission.","checks":checks,"selected_scalar_count":len(numeric),"evidence_files":len(index["files"]),"package_local_links":len(links),"external_repository_links":external_repo,"physical_correlation_converged":False,"physical_poles_certified":False,"no_physics_imports":True}
    (ROOT/"verification.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pass":output["pass"],"checks":len(checks),"evidence_files":len(index["files"]),"selected_scalars":len(numeric),"failed_checks":[c for c in checks if not c["pass"]]},ensure_ascii=False))
    return output["pass"]


if __name__=="__main__":
    sys.exit(0 if verify() else 1)
