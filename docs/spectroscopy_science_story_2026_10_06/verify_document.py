"""Portable consistency checks; not a physics regression or accuracy test."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
MAIN=ROOT/"研究体系与案例设计.md"
checks=[]


def check(name,value):
    checks.append({"name":name,"passed":bool(value)})


def verify(write_report=False):
    text=MAIN.read_text(encoding="utf-8")
    registry=json.loads((ROOT/"case_registry.json").read_text(encoding="utf-8"))
    ids=[f"S{i:02}" for i in range(1,13)]
    check("12 unique document cases",re.findall(r"^### (S\d{2}) · ",text,re.M)==ids)
    check("12 matching registry cases",[c["case_id"] for c in registry["cases"]]==ids)
    for case in registry["cases"]:
        check(case["case_id"]+" is a plan",case["status"]=="planned_full_case_not_executed" and case["scientific_acceptance_passed"] is None)
        for field in ("motivation","system_and_starting_input","model_and_software","procedure","observables_and_validation","atomx_role_and_next_case"):
            check(case["case_id"]+" "+field,case[field] in text and len(case[field])>50)
    for name in ("研究体系与案例设计","故事线摘要"):
        check(name+" PDF",(ROOT/f"{name}.pdf").read_bytes().startswith(b"%PDF"))
        with zipfile.ZipFile(ROOT/f"{name}.docx") as z:
            xml=z.read("word/document.xml").decode()
            check(name+" DOCX content", "Atomx" in xml and "谱学" in xml)
            if name.startswith("研究"):
                check("all case IDs in editable Word",all(i in xml for i in ids))
        check(name+" HTML has source",'lang="zh-CN"' in (ROOT/f"{name}.html").read_text(encoding="utf-8"))
    for path in (MAIN,ROOT/"README.md",ROOT/"故事线摘要.md"):
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if target.startswith(("http://","https://","#")):continue
            local=path.parent/target.split("#")[0]
            # A portable document bundle intentionally omits source and sibling archives.
            if ".." not in Path(target).parts:check(f"local link {path.name}: {target}",local.exists())
            elif (REPO/"src/atomx/atom.py").exists():check(f"repository link {target}",local.exists())
    results=ROOT/"foundation_results"
    contract=json.loads((results/"contract.json").read_text())
    co=json.loads((results/"co_provider.json").read_text())
    mg=json.loads((results/"mgo_embedding.json").read_text())
    ne=json.loads((results/"ne_x_ground.json").read_text())
    check("software scope explicitly bounded",contract["purpose"]=="software_contract_demonstrations_only")
    check("CO actual electron count",abs(co["electron_count"]-14)<contract["acceptance"]["electron_count_absolute_error"])
    check("CO actual metric",co["MO_metric_residual"]<contract["acceptance"]["MO_metric_residual"])
    check("Mg core identity",mg["core_dominance"]>contract["acceptance"]["core_dominance_min"])
    check("Ne X charge definition",ne["Z"]-ne["charge"]==1 and ne["corrections"]=="none")
    identity=json.loads((results/"source_identity.json").read_text())
    source_present=(REPO/"src/atomx/atom.py").exists()
    if source_present:
        for name,expected in identity["audited_entrypoint_sha256"].items():
            check("audited source still matches "+name,hashlib.sha256((REPO/name).read_bytes()).hexdigest()==expected)
    report={"passed":all(c["passed"] for c in checks),"check_count":len(checks),"scope":"document, registry, artifact and saved software-contract consistency only", "source_available":source_present,"full_scientific_cases_verified":False,"checks":checks}
    if write_report:
        (ROOT/"verification_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="checks"},ensure_ascii=False))
    failed=[c["name"] for c in checks if not c["passed"]]
    if failed:raise SystemExit("Failed: "+", ".join(failed))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report",action="store_true",help="persist the current report; default checks are read-only")
    verify(parser.parse_args().write_report)
