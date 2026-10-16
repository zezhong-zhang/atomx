"""Recompute ten matrices in a new directory; preserve packaged evidence."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("fine_l6_e200", "coarse_l6_e200", "all"), default="all")
    parser.add_argument("--output", default="yb_matrix_reproduction")
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        parser.error("output must be one new directory name under docs/")
    target = ROOT.parent/args.output
    target.mkdir(exist_ok=False)
    for name in ("run_matrix.py", "contract.json"):
        shutil.copy2(ROOT/name, target/name)
    shutil.copytree(ROOT/"preflight", target/"preflight")
    contract = json.loads((target/"contract.json").read_text())
    if args.case != "all":
        contract["captures"] = [c for c in contract["captures"] if c["name"] == args.case]
        (target/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    subprocess.run([sys.executable, str(target/"run_matrix.py")], check=True)
    status = json.loads((target/"capture_status.json").read_text())
    # run_matrix's completed flag adapts to the selected job count.
    assert status["completed"]
    errors = []
    for job in status["jobs"]:
        rel = Path("results")/job["case"]/job["label"].replace("/", "_")/"results.json"
        expected, actual = json.loads((ROOT/rel).read_text()), json.loads((target/rel).read_text())
        expected_variants = {v["external_dimension"]: v for v in expected["variants"]}
        actual_variants = {v["external_dimension"]: v for v in actual["variants"]}
        assert set(expected_variants) == set(actual_variants) == {1, 2, 3}
        for dimension in (1, 2, 3):
            old, new = expected_variants[dimension], actual_variants[dimension]
            delta = abs(old["root"]["energy_Ha"]-new["root"]["energy_Ha"])
            assert delta <= 1e-10
            errors.append(delta)
    result = dict(recomputed_states=len(status["jobs"]), maximum_root_error_Ha=max(errors), passed=True,
                  scope="Same supplied frozen reference and native source; large poles regenerated, not included in delivery")
    (target/"bundle_replay.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
