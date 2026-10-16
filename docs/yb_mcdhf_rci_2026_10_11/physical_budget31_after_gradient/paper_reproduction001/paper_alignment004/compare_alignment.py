"""Own the common-observable arithmetic; never add literature corrections."""

import hashlib
import json
from pathlib import Path

from atomx.constants import HARTREE_CM1

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "native_virtual_p003"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(targets, lines):
    experiment = {"D3": 22960.80, "D5": 24332.69}
    result = dict(
        EA_cm1={k: v["EA_cm1"] for k, v in targets.items()},
        transitions_cm1=lines,
        calculation_minus_NIST_cm1={k: lines[k]-experiment[k] for k in experiment},
        relative_error_percent={k: 100*(lines[k]-experiment[k])/experiment[k] for k in experiment},
        fine_splitting_cm1=lines["D5"]-lines["D3"],
    )
    result["fine_splitting_error_cm1"] = result["fine_splitting_cm1"]-1371.89
    return result


def main():
    old = json.loads((BASE / "actual001/result.json").read_text())
    assert old["status"] == "PASS_COMPLETE_6P_NATIVE_CCSDT_R3"
    reference_EA = dict(
        DHF={"S": -90790, "D3": -66517, "D5": -66038},
        RCCSD={"S": -97642, "D3": -73525, "D5": -72268},
        RCCSDT={"S": -97859, "D3": -74198, "D5": -72915},
        Final={"S": -97982, "D3": -74952, "D5": -73646},
    )
    literature = {
        name: row({k: {"EA_cm1": v} for k, v in values.items()},
                  {k: values[k]-values["S"] for k in ("D3", "D5")})
        for name, values in reference_EA.items()
    }
    result = dict(
        scope="Same electronic transitions, common EA arithmetic; different orbital/core/method inputs. All literature quantities are reference-only and never applied to native values.",
        references=dict(
            Sahoo="https://arxiv.org/html/2504.01438v1",
            Hur="https://arxiv.org/html/2201.03578v2",
            Kawasaki="https://arxiv.org/html/2402.13541v2",
            NIST="https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable6_a.htm",
        ),
        NIST_electronic_levels_cm1={"S": 0.0, "D3": 22960.80, "D5": 24332.69},
        Sahoo_Table1=literature,
        uncertainty="Sahoo individual final EA uncertainties are300/350/350cm^-1; covariance unavailable. Central transition deviations are not theoretical uncertainties.",
        old18core=row(old["targets"], old["lines_cm1"]),
        inputs={str(BASE / "actual001/result.json"): sha(BASE / "actual001/result.json")},
        unknown_paper_inputs=["nuclear radius and charge density parameters", "40GTO exponent lists and contraction details", "actual frozen/correlated core partition", "virtual energy and triples cuts", "actual numerical residual controls", "the numerical program and original Yb input deck"],
        strict_input_reproduction_complete=False,
        method_equivalence_claimed=False,
    )
    path = HERE / "actual001/result.json"
    if path.exists():
        new = json.loads(path.read_text())
        result["new_run_status"] = new["status"]
        result["inputs"][str(path)] = sha(path)
        if new["status"] == "PASS_COMPLETE_5P_NATIVE_CCSDT_R3":
            result["new22core"] = row(new["targets"], new["lines_cm1"])
            result["conditional_core_closure_shifts_cm1"] = new["core_closure_shifts_cm1"]
            result["state_EA_shifts_cm1"] = new["EA_changes_cm1"]
            result["core_energy_shift_cm1"] = (
                new["core"]["energy_relative_Ha"]-old["core"]["energy_relative_Ha"]
            )*HARTREE_CM1
            result["state_total69_relative_energy_shifts_cm1"] = {
                name: (new["targets"][name]["total69_relative_Ha"]
                       -old["targets"][name]["total69_relative_Ha"])*HARTREE_CM1
                for name in new["targets"]
            }
            for name, value in result["state_total69_relative_energy_shifts_cm1"].items():
                assert abs(value-result["core_energy_shift_cm1"]
                           -result["state_EA_shifts_cm1"][name]) < 1e-7
    (HERE / "alignment_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("old18core", "new_run_status", "new22core", "conditional_core_closure_shifts_cm1") if k in result}, indent=2))


if __name__ == "__main__":
    main()
