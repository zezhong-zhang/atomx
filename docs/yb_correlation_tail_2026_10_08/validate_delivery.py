"""Check actual source, numerical gates, current links and report numbers before delivery."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DOCS=ROOT.parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import save,sha,source_identity


def main():
    for p in ROOT.glob("*.py"):ast.parse(p.read_text(),filename=str(p))
    current=[DOCS/n for n in ("README.md","AGENT_REFERENCE.md","AGENT_INSIGHTS.md", "VERIFICATION.md","TESTING.md","ARCHITECTURE_CAPABILITIES.md")]+[ROOT/"README.md"]
    broken=[];checked=0
    for p in current:
        for target in re.findall(r"\]\(([^)]+)\)",p.read_text()):
            if target.startswith(("http:","https:","#","mailto:")):continue
            local=target.split("#",1)[0]
            if not local:continue
            checked+=1
            if not (p.parent/local).exists():broken.append(dict(file=str(p.relative_to(REPO)),target=target))
    if broken:raise ValueError(f"Broken current documentation links: {broken}")
    diff=subprocess.run(["git","diff","--check"],cwd=REPO,text=True,capture_output=True,check=True)
    if source_identity()!=json.loads((ROOT/"capture_source_sha256.json").read_text()):
        raise ValueError("Capture source changed")
    for path in ("analysis/status.json","capture_coverage/results.json","representation_audit_v4/results.json"):
        r=json.loads((ROOT/path).read_text())
        if not r.get("completed",r.get("passed",r.get("numerical_passed",False))):
            raise ValueError(f"Incomplete verification: {path}")
    analysis=json.loads((ROOT/"analysis/results.json").read_text())
    report=(ROOT/"README.md").read_text()
    for p in analysis["primaries"].values():
        for row in p["records"][1:]:
            if f"{row['excitation_dyson_cm1']:.2f}" not in report:
                raise ValueError("Published excitation missing/mismatched")
    summary=json.loads((ROOT/"report_summary.json").read_text())
    dispositions={"production API/tests":"Retained unchanged from preceding validated source; this increment changes physical setup and evidence only.",
        "current README/reference/verification":"Updated to completed common-space tail/radial measurements and separate numerical/spatial/experimental status.",
        "capabilities/testing":"Evidence links updated; supported method/approximation boundaries remain unchanged.",
        "insights":"Merged virtual capacity versus actual transition response into the existing transition-feedback rule; no root rule added.",
        "older scientific folders":"Preserved both sealed manifests (93 and 52 entries); no historical evidence edited."}
    save(ROOT/"documentation_review.json",dict(dispositions=dispositions,
        current_guide_sha256={str(p.relative_to(REPO)):sha(p) for p in current},
        local_links_checked=checked,broken_links=[],driver_syntax_passed=True,
        diff_whitespace_passed=True,source_unchanged=True,
        physical_accuracy_limits="Numerical passes do not turn failed 1 cm-1 correlation-space or 5% experimental targets into passes.",summary=summary))
    print("DELIVERY CHECK PASS",checked,"local links; driver syntax, source, scientific values and whitespace")


if __name__=="__main__":main()
