"""Close source, saved-result, documentation and retained-evidence identities."""

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
BASE = HERE.parent / "native_virtual_p003"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    checks = []
    def gate(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)
    source = read(BASE / "source.json")
    old = read(BASE / "increment_receipt.json")
    before = read(HERE / "before.json")
    contract = read(HERE / "contract.json")
    gate("unchanged362production_files", all(sha(ROOT/n)==h for n,h in source["files"].items()))
    gate("unchanged61sealed_predecessor_artifacts", all(sha(ROOT/n)==h for n,h in old["artifacts"].items()))
    gate("unchanged_bound_source_and_original_inputs", all(sha(ROOT/n)==h for n,h in contract["inputs"].items()))
    gate("test_source_unchanged", sha(ROOT / "tests/test_tensor_coupled_cluster.py")==before["test_source_sha256"])
    for folder,status in (("probe001","PASS_COMPLETE_5P_OFFSHELL_PROBE"),("actual001","PASS_COMPLETE_5P_NATIVE_CCSDT_R3")):
        r=read(HERE/folder/"result.json")
        gate(folder+":PASS_status_and_all_controls",r["status"]==status and all(c["passed"] for c in r["controls"]))
        gate(folder+":same_experiment_contract",r["contract_sha256"]==sha(HERE/"contract.json"))
    result=read(HERE/"actual001/result.json")
    gate("serialized_solve_inputs_unchanged",all(sha(ROOT/n)==h for n,h in result["solve_inputs"].items()))
    review=read(HERE/"saved_review.json")
    gate("fresh_saved_replay_PASS",review["status"]=="PASS_COMPLETE_5P_SAVED_REPLAY" and all(c["passed"] for c in review["controls"]))
    gate("same_replayed_result",review["original_result_sha256"]==sha(HERE/"actual001/result.json"))
    rc=read(HERE/"review_contract.json")
    gate("unchanged_review_consumer",rc["consumer_sha256"]==sha(HERE/"review_saved.py"))
    gate("same_review_experiment_contract",rc["experiment_contract_sha256"]==sha(HERE/"contract.json"))
    comparison=read(HERE/"alignment_comparison.json")
    gate("comparison_uses_actual_saved_inputs",all(sha(Path(n))==h for n,h in comparison["inputs"].items()))
    snapshot=read(HERE/"limiting_equation_snapshot.json")
    gate("retained_tail_snapshot",sha(HERE/snapshot["checkpoint"])==snapshot["checkpoint_sha256"])
    suites=list(ET.parse(HERE/"tensor_tests.xml").getroot().iter("testsuite"))
    gate("12current_tensor_tests_no_failures_or_skips",sum(int(s.get("tests","0")) for s in suites)==12 and all(int(s.get(k,"0"))==0 for s in suites for k in ("errors","failures","skipped")))
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text())
        gate(path.name+":syntax_and_no_trailing_whitespace",all(line==line.rstrip() for line in path.read_text().splitlines()))
    impact=read(HERE/"documentation_impact.json")
    for name in impact["current_docs_to_retain"]:
        gate(name+":retained_guide_unchanged",sha(ROOT/name)==before["current_guides_before"][name])
    changed=impact["current_docs_to_update"]
    missing=[]
    count=0
    for path in [HERE/"README.md"]+[ROOT/n for n in changed]:
        for value in re.findall(r"\[[^\]]*\]\(([^)]+)\)",path.read_text()):
            target=value.split("#",1)[0].strip("<>")
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:",target):
                continue
            count+=1
            if not (path.parent/target).resolve().exists():
                missing.append(dict(source=str(path.relative_to(ROOT)),target=target))
    save(HERE/"documentation_links.json",dict(scope="Present working-tree targets, including ignored/untracked evidence; not a clean-checkout publication check. External URLs/anchors not mechanically checked.",checked=count,missing=missing))
    gate("all_working_tree_local_documentation_targets_exist",not missing)
    gate("task_sessions_reaped",read(HERE/"execution_record.json")["all_task_sessions_reaped"])
    save(HERE/"validation_receipt.json",dict(status="PASS",checks=checks,source_map_sha256=source["source_map_sha256"],scope="Actual source/input/result bindings, saved-equation replay,12 tests, local links. No independent large FCI or full-paper/experimental certificate."))
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name!="increment_receipt.json"}
    save(HERE/"increment_receipt.json",dict(status="PASS",timestamp_UTC=datetime.now(timezone.utc).isoformat(),source_map_sha256=source["source_map_sha256"],changed_production_files=[],production_files_verified=len(source["files"]),sealed_predecessor_artifacts_verified=len(old["artifacts"]),artifacts=artifacts,current_guides={n:sha(ROOT/n) for n in before["current_guides_before"]},guide_before_sha256=before["current_guides_before"],owning_report=str((HERE/"README.md").relative_to(ROOT)),all_task_sessions_reaped=True,scope="Fixed-orbital22core/18virtual finite EA-CCSDT/R3 discriminator; experimental errors grow. No full-paper matching, f/g calculation, controlled speedup, physical exp(T) state or full-H residual claim."))
    print("PASS_INCREMENT",len(checks),"checks",len(artifacts),"artifacts",count,"local links",flush=True)


if __name__=="__main__":
    main()
