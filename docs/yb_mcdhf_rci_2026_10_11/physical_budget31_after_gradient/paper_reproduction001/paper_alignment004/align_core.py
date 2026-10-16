"""Fixed-orbital Yb alignment discriminator: unfreeze the complete 5p3/2 shell.

Local experiment only. Reuses the qualified native integral, SDT, angular,
DIIS and highest-weight Davidson implementations; no production API change.
"""

import argparse
import hashlib
import itertools
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
BASE = HERE.parent / "native_virtual_p003"
DHF = HERE.parent / "yb_dhf_table001"
sys.path.insert(0, str(BASE))
import prepare_virtuals as previous  # noqa: E402
import solve as previous_solve  # noqa: E402
from angular import AngularCoordinates, act  # noqa: E402
from diis_core import solve as solve_diis  # noqa: E402

from atomx.constants import HARTREE_CM1  # noqa: E402
from atomx.methods.dirac_fock import SCFResult  # noqa: E402
from atomx.methods.tensor_coupled_cluster import (  # noqa: E402
    CCSDTTensorOperator,
    TensorCoreResult,
    resource_plan,
)
from atomx.physics.csf.spinor_integrals import (  # noqa: E402
    SpinorIntegrals,
    build_spinor_integrals,
)
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare():
    inputs = [
        Path(__file__), BASE / "source.json", BASE / "operator.json",
        BASE / "integrals.npz", BASE / "actual001/core.npz",
        BASE / "actual001/result.json", BASE / "probe001/virtuals.npz",
        BASE / "increment_receipt.json", BASE / "prepare_virtuals.py",
        BASE / "solve.py", BASE / "diis_core.py",
        HERE.parent / "native_tensor_4f002/angular.py",
        DHF / "actual004/blocks_ns180.npz", DHF / "actual004/core_ns180.h5",
    ] + [BASE / f"actual001/{name}_root.npz" for name in ("S", "D3", "D5")]
    source = json.loads((BASE / "source.json").read_text())
    assert all(sha(ROOT / n) == h for n, h in source["files"].items())
    contract = dict(
        hypothesis="At fixed owned orbitals/DC Hamiltonian/18 virtual spinors, opening the complete frozen 5p3/2 quartet changes differential core/valence correlation. The embedded predecessor must reproduce all its equations before solving any new amplitudes.",
        fixed=dict(
            nucleus="empirical A171 Fermi, identical to predecessor; exact paper nuclear parameters unknown",
            radial="same ns180/order9/R500/RKB; same complete6s,5d3/2,5d5/2,6p1/2,6p3/2 virtuals",
            Hamiltonian="Dirac-Coulomb, common original 68e HF energy zero, no empirical shift/Breit/QED/recoil",
            equations="complete SDT core and R3 attachment; exact M/parity, highest-weight J, all12 magnetic partners; canonical R norm only",
            old_core="18 correlated:5s2,5p1/2^2,4f14;50 frozen",
            new_core="22 correlated:old18 plus complete5p3/2^4;46 frozen",
            method_scope="finite EA-CCSDT/R3; not a claim of valence-universal FS-RCCSDT or full-paper reproduction",
        ),
        gates=dict(
            identity_Ha=1e-10, core_residual_Ha=1e-10,
            EA_residual_Ha=1e-10, J2=1e-7, core_Jplus=1e-7,
            canonical_R_norm=1e-8,
        ),
        resources=dict(
            cpu=27, threads=1, AS_bytes=24 * 1024**3,
            workspace_MiB=16384, previous_workspace_MiB=8192,
            reason_for_workspace_change="22x18 scalar conservative payload is12.583GB, above prior8GiB; existing process AS24GiB unchanged",
            probe_wall_seconds=360, solve_wall_seconds=2400,
            core_calls_max=60, Davidson_basis_per_target_max=48,
            load1_max=8, available_min_bytes=32 * 1024**3,
            plan=resource_plan(22, 18),
            rejected_next_full_f_plan=resource_plan(22, 32),
        ),
        distinguishing_outcomes="First establish exact old-coordinate/energy/Fock/ERI recovery and nonzero newly opened residuals. A solved new spectrum is accepted only after complete equations and partners pass; improvement against experiment is measured, never a gate.",
        stopping="First identity/source/resource/equation failure stops and is retained. No automatic rank, orbital, residual, root-weight or model changes. Independent call and wall limits apply. No full-f allocation under this contract.",
        source_map_sha256=source["source_map_sha256"],
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
    )
    save(HERE / "contract.json", contract)
    print("PREPARED", contract["resources"]["plan"], flush=True)


def environment(contract, phase):
    assert os.getloadavg()[0] <= contract["resources"]["load1_max"]
    available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    assert available >= contract["resources"]["available_min_bytes"]
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        assert os.environ.get(name) == "1", name
    os.sched_setaffinity(0, {contract["resources"]["cpu"]})
    cap = contract["resources"]["AS_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    def alarm(*_):
        raise TimeoutError("predeclared " + phase + " wall cap")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(contract["resources"][phase + "_wall_seconds"])
    return dict(load=os.getloadavg(), available_bytes=available, cpu=27, threads=1)


def check_inputs(contract):
    source = json.loads((BASE / "source.json").read_text())
    assert all(sha(ROOT / n) == h for n, h in source["files"].items())
    assert all(sha(ROOT / n) == h for n, h in contract["inputs"].items())


def load_operator(folder, occupied):
    ledger = json.loads((folder / "operator.json").read_text())
    with np.load(folder / "integrals.npz") as z:
        integrals = SpinorIntegrals(tuple(map(tuple, ledger["labels"])), occupied,
            z["h"], z["fock"], z["g"], ledger["shifted_frozen_energy_Ha"],
            0.0, ledger["identity"])
    return CCSDTTensorOperator(integrals, memory_MiB=16384,
                              projection_storage="canonical"), ledger


def embed(vector, old, new):
    assert old.virtual == new.virtual == 18
    result = np.zeros(new.size, dtype=np.asarray(vector).dtype)
    selected = []
    for (old_slice, op, oh, _), (new_slice, np_, nh, _) in zip(old.blocks, new.blocks, strict=True):
        rank = op.shape[1]
        count = len(list(itertools.combinations(range(old.virtual), rank)))
        old_holes = list(itertools.combinations(range(old.occupied), oh.shape[1]))
        new_holes = {h: i for i, h in enumerate(itertools.combinations(range(new.occupied), nh.shape[1]))}
        rows = np.array([new_holes[h] for h in old_holes])
        indices = new_slice.start + rows[:, None]*count + np.arange(count)
        indices = indices.ravel()
        assert np.array_equal(np_[indices-new_slice.start], op)
        assert np.array_equal(nh[indices-new_slice.start], oh)
        result[indices] = vector[old_slice]
        selected.extend(indices)
    return result, np.asarray(selected, dtype=int)


def run(phase):
    contract = json.loads((HERE / "contract.json").read_text())
    check_inputs(contract)
    out = HERE / ("probe001" if phase == "probe" else "actual001")
    out.mkdir(exist_ok=False)
    report = dict(status="RUNNING", controls=[], contract_sha256=sha(HERE / "contract.json"), source_map_sha256=contract["source_map_sha256"])
    started = time.monotonic()
    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(dict(name=name, measured=measured, cap=cap, passed=bool(passed)))
        if not passed:
            raise AssertionError((name, measured, cap))
    try:
        report["environment"] = environment(contract, phase)
        if phase == "probe":
            probe(report, gate, out)
        else:
            solve(report, gate, out)
        check_inputs(contract)
        gate("source_and_bound_inputs_unchanged", True)
    except Exception:
        report.update(status="FAILED", exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_RSS_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save(out / "result.json", report)
        print("END", phase, report["status"], flush=True)


def probe(report, gate, out):
    core = SCFResult.load(DHF / "actual004/core_ns180.h5")
    core._nuclear_potential = make_nuclear_potential(70, "fermi", 171)
    with np.load(DHF / "actual004/blocks_ns180.npz") as z:
        orbitals = {tuple(map(int, k)):(p,q) for k,p,q in zip(z["all_orbital_keys"],z["all_P"],z["all_Q"],strict=True)}
    with np.load(BASE / "probe001/virtuals.npz") as z:
        for k in (1,-2):
            x=z[f"selected_k{k}"]
            orbitals[6,k] = x[:180],x[180:]
    occupied = ((5,-1),(5,1),(4,3),(4,-4),(5,-2))
    virtual = ((6,-1),(5,2),(5,-3),(6,1),(6,-2))
    frozen = tuple(k for k in core.orbitals_by_subshell() if k not in occupied)
    gate("22correlated_46frozen", sum(2*abs(k) for n,k in frozen)==46)
    print("BUILD_NATIVE_INTEGRALS", flush=True)
    original = build_spinor_integrals(orbitals,core._bsp_data,70,180,
        occupied_keys=occupied,virtual_keys=virtual,frozen_keys=frozen,
        nuclear_potential=core._nuclear_potential,max_integral_bytes=256*1024**2,
        metric_tolerance=1e-10)
    origin = original.reference_energy_Ha
    identity = dict(original.identity)
    identity.update(experiment="complete5p closure;22correlated/46frozen/18virtual",energy_origin_Ha=origin)
    save(HERE / "operator.json",dict(noccupied=22,labels=original.labels,identity=identity,
        absolute_energy_origin_Ha=origin,shifted_frozen_energy_Ha=original.frozen_energy_Ha-origin))
    np.savez(HERE / "integrals.npz",h=original.one_body_Ha,fock=original.fock_Ha,g=original.packed_eri_Ha)
    op,ledger=load_operator(HERE,22)
    old,oldledger=load_operator(BASE,18)
    physical_map=np.array([op.labels.index(x) for x in old.labels])
    gate("all_old_spinor_identity", np.array_equal(physical_map,np.r_[np.arange(18),np.arange(22,40)]))
    gate("same68e_reference_origin_Ha",abs(origin-oldledger["absolute_energy_origin_Ha"]),1e-10)
    gate("same_all_old_Fock_Ha",np.max(abs(op.f[np.ix_(physical_map,physical_map)]-old.f)),1e-10)
    gate("same_all_old_ERI_Ha",np.max(abs(op.g[np.ix_(physical_map,physical_map,physical_map,physical_map)]-old.g)),1e-10)
    for coordinates in (old.core,old.attachment):
        for _,particles,holes,_ in coordinates.blocks:
            for hole in set(map(tuple, holes)):
                p=tuple(range(particles.shape[1]))
                phases=[]
                for occ in (18,22):
                    _,phase=act((1<<occ)-1,tuple((occ+i,True) for i in p)+tuple((int(i),False) for i in reversed(hole)))
                    phases.append(phase)
                gate("old_reference_phase:"+str(p)+str(hole),phases[0]==phases[1])
    with np.load(BASE / "actual001/core.npz") as z:
        initial,indices=embed(z["amplitudes"],old.core,op.core)
    np.savez(HERE / "fixed_inputs.npz",core=initial,old_indices=indices,physical_map=physical_map)
    print("PROBE_FULL_CORE",op.core.size,flush=True)
    stamp=time.perf_counter()
    residual=op.core_residual(initial)
    report["core_residual_seconds"]=time.perf_counter()-stamp
    energy=op.core_energy(initial)
    old_result=json.loads((BASE / "actual001/result.json").read_text())
    gate("embedded_core_energy_Ha",abs(energy-old_result["core"]["energy_relative_Ha"]),1e-10)
    gate("all689589_old_SDT_equations_Ha",np.max(abs(residual[indices])),1e-10)
    new_indices=np.setdiff1d(np.arange(op.core.size),indices)
    gate("opened_5p3_core_residual_nonzero",np.max(abs(residual[new_indices]))>1e-8)
    trial=TensorCoreResult(initial,energy,energy,float(np.max(abs(residual))),0,op.operator_sha256)
    products={"core_residual":residual}
    report["EA_action_seconds"]={}
    for name in ("S","D3","D5"):
        with np.load(BASE / f"actual001/{name}_root.npz") as z:
            right,ea_indices=embed(z["vector"],old.attachment,op.attachment)
            ea=float(z["energy"])
        stamp=time.perf_counter()
        product=op.attachment_action(trial,right)
        report["EA_action_seconds"][name]=time.perf_counter()-stamp
        error=product-ea*right
        gate(name+":all127620_old_R3_equations_Ha",np.max(abs(error[ea_indices])),1e-10)
        products[name+"_embedded"]=right
        products[name+"_EA_action"]=product
        products[name+"_old_indices"]=ea_indices
        print("PROBE_EA",name,report["EA_action_seconds"][name],flush=True)
    np.savez(out / "values.npz",**products)
    report.update(status="PASS_COMPLETE_5P_OFFSHELL_PROBE",operator_sha256=op.operator_sha256,
        core_amplitudes=op.core.size,attachment_coordinates=op.attachment.size,
        embedded_core_energy_Ha=energy,new_core_residual_Ha=float(np.max(abs(residual[new_indices]))))


def solve(report,gate,out):
    probe_result=json.loads((HERE / "probe001/result.json").read_text())
    gate("previous_probe_pass",probe_result["status"]=="PASS_COMPLETE_5P_OFFSHELL_PROBE")
    solve_inputs={str(p.relative_to(ROOT)):sha(p) for p in (HERE/"operator.json",HERE/"integrals.npz",HERE/"fixed_inputs.npz",HERE/"probe001/result.json")}
    report["solve_inputs"]=solve_inputs
    op,ledger=load_operator(HERE,22)
    gate("same_probe_operator",op.operator_sha256==probe_result["operator_sha256"])
    with np.load(HERE/"fixed_inputs.npz") as z:
        initial=z["core"]
    residual_action=op.core_residual
    calls=[]
    def counted(vector,**kwargs):
        if len(calls)>=60:
            raise RuntimeError("predeclared60 core call cap")
        stamp=time.perf_counter()
        result=residual_action(vector,**kwargs)
        calls.append(dict(seconds=time.perf_counter()-stamp,maximum_Ha=float(np.max(abs(result)))))
        save(out/"core_progress.json",dict(calls=calls))
        np.savez(out/"last_core_evaluation.npz",amplitudes=vector,residual=result)
        print("CORE_RESIDUAL",len(calls),calls[-1],flush=True)
        return result
    op.core_residual=counted
    core,history=solve_diis(op,initial=initial)
    op.core_residual=residual_action
    save(out/"DIIS_history.json",history)
    gate("all1292379_SDT_equations_Ha",np.max(abs(residual_action(core.amplitudes))),1e-10)
    angular_core=AngularCoordinates(op,False)
    gate("core_scalar_Jplus",np.linalg.norm(angular_core.ladder(core.amplitudes,1)),1e-7)
    del angular_core
    np.savez(out/"core.npz",amplitudes=core.amplitudes)
    report["core"]=dict(energy_relative_Ha=core.total_energy_Ha,energy_absolute_Ha=ledger["absolute_energy_origin_Ha"]+core.total_energy_Ha,
        residual_Ha=core.amplitude_residual_Ha,correlation_Ha=core.correlation_energy_Ha,
        calls=calls,rank_norms={str(k+1):float(np.linalg.norm(core.amplitudes[s])) for k,(s,*_) in enumerate(op.core.blocks)})
    save(out/"partial.json",report)
    print("CORE_CONVERGED",core.total_energy_Ha,flush=True)
    angular=AngularCoordinates(op,True)
    previous_solve.HERE=HERE
    progress,targets,partners={},{},{}
    for name,key,j in previous_solve.TARGETS:
        energy,vector,indices,history=previous_solve.solve_target(op,core,angular,name,key,j,gate,progress)
        states={int(2*j):vector}
        state,m=vector,j
        while m>-j:
            state=angular.ladder(state,-1)/np.sqrt((j+m)*(j-m+1))
            m-=1
            states[int(2*m)]=state
        gate(name+":complete_multiplet",len(states)==int(2*j+1))
        for m2,state in states.items():
            gate(name+f":M{m2}:canonical_norm",abs(np.linalg.norm(state)-1),1e-8)
            gate(name+f":M{m2}:J2",np.linalg.norm(angular.j_squared(state,m2/2)-j*(j+1)*state),1e-7)
            gate(name+f":M{m2}:all191880_R3_equations_Ha",np.max(abs(op.attachment_action(core,state)-energy*state)),1e-10)
            partners[name+f"_M{m2}"]=state
        np.savez(out/(name+"_root.npz"),energy=energy,vector=vector,indices=indices)
        targets[name]=dict(EA_Ha=energy,EA_cm1=energy*HARTREE_CM1,total69_relative_Ha=core.total_energy_Ha+energy,
            total69_absolute_Ha=ledger["absolute_energy_origin_Ha"]+core.total_energy_Ha+energy,
            bare_R1_weight=history[-1]["bare_R1_weight"],magnetic_partners=len(states),matvec_calls=len(history))
        save(out/"target_progress.json",targets)
    np.savez(out/"all_target_partners.npz",**partners)
    report["targets"],report["EA_history"]=targets,progress
    report["lines_cm1"]={name:(targets[name]["EA_Ha"]-targets["S"]["EA_Ha"])*HARTREE_CM1 for name in ("D3","D5")}
    old=json.loads((BASE/"actual001/result.json").read_text())
    report["core_closure_shifts_cm1"]={name:report["lines_cm1"][name]-old["lines_cm1"][name] for name in ("D3","D5")}
    report["EA_changes_cm1"]={name:targets[name]["EA_cm1"]-old["targets"][name]["EA_cm1"] for name in targets}
    gate("probe_artifacts_unchanged",all(sha(ROOT/n)==h for n,h in solve_inputs.items()))
    report["status"]="PASS_COMPLETE_5P_NATIVE_CCSDT_R3"


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","probe","solve"))
    args=parser.parse_args()
    prepare() if args.phase=="prepare" else run(args.phase)
