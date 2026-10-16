"""Fresh-process equation replay, identities and arithmetic, without rerunning SCF."""

import json
import resource
import signal
import time
import traceback

import numpy as np

from align_core import (
    HERE, BASE, ROOT, AngularCoordinates, TensorCoreResult, check_inputs,
    environment, load_operator, save, sha,
)
from atomx.constants import HARTREE_CM1


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    check_inputs(contract)
    result = json.loads((HERE / "actual001/result.json").read_text())
    assert result["status"] == "PASS_COMPLETE_5P_NATIVE_CCSDT_R3"
    report = dict(status="RUNNING", controls=[],
                  original_result_sha256=sha(HERE / "actual001/result.json"),
                  scope="Complete equations/partners freshly reapplied with the same production action; replay is not an independent full22e FCI oracle.")
    started = time.monotonic()
    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(dict(name=name, measured=measured, cap=cap, passed=bool(passed)))
        if not passed:
            raise AssertionError((name, measured, cap))
    try:
        report["environment"] = environment(contract, "probe")
        op, ledger = load_operator(HERE, 22)
        gate("same_operator", op.operator_sha256 == json.loads((HERE / "probe001/result.json").read_text())["operator_sha256"])
        with np.load(HERE / "actual001/core.npz") as z:
            amplitudes = z["amplitudes"]
        residual = op.core_residual(amplitudes)
        gate("all1292379_core_equations_Ha", np.max(abs(residual)), 1e-10)
        energy = op.core_energy(amplitudes)
        gate("same_core_energy_Ha", abs(energy-result["core"]["energy_relative_Ha"]), 1e-12)
        angular_core = AngularCoordinates(op, False)
        gate("core_Jplus", np.linalg.norm(angular_core.ladder(amplitudes, 1)), 1e-7)
        del angular_core
        core = TensorCoreResult(amplitudes,energy,energy,float(np.max(abs(residual))),0,op.operator_sha256)
        angular = AngularCoordinates(op, True)
        with np.load(HERE / "actual001/all_target_partners.npz") as z:
            for name,j in (("S",0.5),("D3",1.5),("D5",2.5)):
                ea = result["targets"][name]["EA_Ha"]
                gate(name+":EA_units",abs(ea*HARTREE_CM1-result["targets"][name]["EA_cm1"]),1e-8)
                gate(name+":absolute69e_ledger",abs(ledger["absolute_energy_origin_Ha"]+energy+ea-result["targets"][name]["total69_absolute_Ha"]),1e-10)
                for m2 in range(-int(2*j),int(2*j)+1,2):
                    vector=z[name+f"_M{m2}"]
                    gate(name+f":M{m2}:norm",abs(np.linalg.norm(vector)-1),1e-8)
                    gate(name+f":M{m2}:J2",np.linalg.norm(angular.j_squared(vector,m2/2)-j*(j+1)*vector),1e-7)
                    gate(name+f":M{m2}:full191880_R3_Ha",np.max(abs(op.attachment_action(core,vector)-ea*vector)),1e-10)
                    print("REPLAY",name,m2,flush=True)
        baseline=json.loads((BASE / "actual001/result.json").read_text())
        for name in ("D3","D5"):
            line=(result["targets"][name]["EA_Ha"]-result["targets"]["S"]["EA_Ha"])*HARTREE_CM1
            gate(name+":same_core_transition",abs(line-result["lines_cm1"][name]),1e-8)
            gate(name+":conditional_closure_shift",abs(line-baseline["lines_cm1"][name]-result["core_closure_shifts_cm1"][name]),1e-8)
        check_inputs(contract)
        gate("all_production_and_original_inputs_unchanged",True)
        gate("live_controls_all_pass",all(c["passed"] for c in result["controls"]))
        report["status"]="PASS_COMPLETE_5P_SAVED_REPLAY"
    except Exception:
        report.update(status="FAILED",exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"]=time.monotonic()-started
        report["peak_RSS_MiB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save(HERE / "saved_review.json",report)
        print("END_REPLAY",report["status"],flush=True)


if __name__ == "__main__":
    main()
