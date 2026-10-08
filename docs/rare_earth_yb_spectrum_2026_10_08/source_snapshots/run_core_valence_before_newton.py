"""Ab initio Yb II frozen [Xe]4f14 core + one-electron Dirac-Fock spectrum."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import resource
import signal
import threading
import time
import traceback
import numpy as np
from scipy import linalg
from run_yb import ROOT, CORE, CONFIGS, save, snapshot, clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--he", type=float, default=.25)
    p.add_argument("--hmax", type=float, default=1.)
    p.add_argument("--rmax", type=float, default=60.)
    args = p.parse_args()
    out = ROOT / "results" / args.label
    out.mkdir(parents=True, exist_ok=False)
    contract = {"hypothesis": "A monovalent Yb II frozen-core V^(N-1) Dirac-Fock model supplies resolved lines with independently verifiable core SCF, valence eigenpair and radial gates.",
        "fixed_conditions": "Z70; self-consistent 68-electron Yb III [Xe]4f14 closed core; exactly one active electron, no spectator fractional populations; Fermi nucleus; Dirac-Coulomb; nonlocal HF exchange; no experimental fit; one BLAS thread",
        "implementation_convention": "build_virtual_basis(potential_mode='VN') retains the full N=68 reference core, which is V^(N-1) relative to the 69-electron Yb II target. Selecting 'VN-1' here would incorrectly remove another core electron.",
        "acceptance": {"core_DF_converged": True, "core_DF_tolerance": 1e-8,
            "valence_projected_eigenpair_residual_Ha": 1e-7,
            "orbital_metric_max_error": 1e-8, "full_CI_minus_core_plus_valence_Ha": 1e-8,
            "radial_transition_drift_cm1": 1., "experimental_excitation_relative_error_goal": .05},
        "observable": "69-electron frozen-core energies E(core)+epsilon(valence), crosschecked against the full 69-electron CSF Hamiltonian; native finite-wavelength E1/E2/M1 tensors",
        "resource_estimate": "19 occupied core orbitals + 5 active orbitals; five one-dimensional CSF blocks; <3 GiB anticipated RSS, no 69-electron SD expansion",
        "stopping_rule": "900 seconds per run; retain and stop on core SCF, eigenpair, metric or Hamiltonian consistency failure; no relaxed gates",
        "limits": "Core orbital relaxation between states, core polarization and core-valence correlation omitted. Not five-level EOL stationarity or crystal-host spectroscopy. Isotope shifts, hyperfine and recoil absent."}
    save(out / "experiment_contract.json", contract)
    save(out / "inputs.json", dict(vars(args), configs=CONFIGS, contract=contract, protocol="core_valence"))
    snapshot(out)
    t0 = time.monotonic()
    done = threading.Event()
    def heartbeat():
        while not done.wait(30):
            print(f"heartbeat elapsed={time.monotonic()-t0:.1f}s peak_RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f}MiB", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    def alarm(*_):
        raise TimeoutError("Declared 900-second solve cap reached")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(900)
    try:
        from atomx import Atom, Basis
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.constants import HARTREE_CM1, C_SQ
        from atomx.methods.mbpt import build_virtual_basis, _build_closed_shell_fock
        from atomx.methods.csf.solver import VariationalCSFResult
        from atomx.methods.ci import ci_solve
        from atomx.numerics.bspline.convol import ConvolCache
        from atomx.numerics.bspline.dirac_assembly import _get_keep_grid
        from atomx.properties.transitions.calculator import TransitionCalculator
        from atomx.potentials.nuclear import nuclear_profile
        print("START", vars(args), flush=True)
        core_atom = Atom(70, charge=2, config=CORE+" 4f14", corrections="none",
            engine="bspline", nuclear="fermi", basis=Basis(ks=9, r_max=args.rmax, grid_he=args.he, grid_hmax=args.hmax))
        core_atom.solve(tol=1e-8, max_iter=160)
        core = core_atom.result
        save(out / "core.json", {"converged": core.converged, "iterations": core.iterations,
            "residual": core.residual, "total_energy_hartree": core.total_energy,
            "solver_diagnostics": core.solver_diagnostics,
            "nuclear_profile": nuclear_profile(70, "fermi"),
            "calculation_spec": core.calculation_spec.to_dict()})
        if not core.converged:
            save(out / "status.json", {"passed": False, "reason": "closed-core SCF not converged"})
            return
        data = core._bsp_data
        cache = ConvolCache(data)
        orbitals = {key:(pq[0].copy(),pq[1].copy()) for key,pq in core.orbitals_by_subshell().items()}
        valence = {(6,-1):"6s_1/2", (6,1):"6p_1/2", (6,-2):"6p_3/2", (5,2):"5d_3/2", (5,-3):"5d_5/2"}
        eps, residuals = {}, {}
        metric_error = 0.
        for key, label in valence.items():
            _, states = build_virtual_basis(core, key[1], cache=cache, e_max=0.,
                potential_mode="VN", orthogonal_to_occupied=True)
            candidates = [(P,Q,e) for P,Q,e in states if -C_SQ < e < 0.]
            if not candidates:
                raise ValueError("no physical negative-energy unoccupied root for "+label)
            P,Q,e = candidates[0]
            orbitals[key] = (P,Q)
            eps[key] = e
            H,S = _build_closed_shell_fock(core,key[1],cache=cache,potential_mode="VN")
            nsp,nsq,ns = data["nsp"],data["nsq"],data["ns"]
            keep, grid = _get_keep_grid(ns,nsp,nsq,1,2)
            Hr = H[grid]; Hr = np.tril(Hr)+np.tril(Hr,-1).T
            Sr = S[grid]
            occupied = [np.r_[p,q] for k,(p,q) in orbitals.items() if k[1]==key[1] and k!=key]
            if occupied:
                normals=Sr@np.column_stack(occupied)[keep,:]
                basis,_=linalg.qr(normals,mode="full")
                complement=basis[:,len(occupied):]
            else:
                complement=np.eye(len(keep))
            v=np.r_[P,Q][keep]
            defect=complement.T@(Hr@v-e*Sr@v)
            sm=complement.T@Sr@complement
            residuals[label]=float(np.sqrt(abs(defect@linalg.solve(sm,defect,assume_a="pos"))))
            metric_error=max(metric_error,abs(float(np.r_[P,Q]@S@np.r_[P,Q])-1.))
            for u in occupied:
                metric_error=max(metric_error,abs(float(u@S@np.r_[P,Q])))
            print("VALENCE",label,e,"projected_residual_Ha",residuals[label],flush=True)
        csfs=[c for cfg in CONFIGS for c in enumerate_jj_csfs(cfg)]
        ci=ci_solve(csfs,orbitals,sorted(orbitals),data,70,data["ns"],
            nuclear_potential=core._nuclear_potential)
        levels=[]
        discrepancies=[]
        for i,energy in enumerate(ci.state_energies):
            dom=int(np.argmax(abs(ci.mixing_coefficients[:,i])))
            c=csfs[dom]
            active=[(s.n,s.kappa) for s in c.subshells if s.occupation==1]
            assert len(active)==1
            key=active[0]
            discrepancies.append(float(energy-core.total_energy-eps[key]))
            levels.append({"index":i,"label":valence[key],"two_J":c.total_J2,"parity":c.parity,
                "dominant_CSF_index":dom,"dominant_CSF":c.label,"dominant_weight":1.,
                "energy_hartree":float(energy),"excitation_cm1":float((energy-ci.ground_energy)*HARTREE_CM1),
                "valence_energy_hartree":eps[key]})
        passed=(max(residuals.values())<1e-7 and metric_error<1e-8 and max(abs(x) for x in discrepancies)<1e-8)
        diagnostics={"physical_method":"closed 68-electron core DF + one-electron V68 frozen-core RCI",
            "core_DF_converged":True,"core_DF_residual":core.residual,"core_total_energy_hartree":core.total_energy,
            "projected_valence_residuals_Ha":residuals,"Hamiltonian_consistency_errors_Ha":discrepancies,
            "full_MCDHF_orbital_optimization":False}
        payload={"converged":passed,"levels":levels,"solver_diagnostics":diagnostics,
            "orbital_optimization":diagnostics,"actual_ns":data["ns"],"actual_ks":data["ks"],
            "orbital_metric_max_error":metric_error,"orbital_energies_hartree":core.orbital_energies,
            "elapsed_seconds_uncontrolled":time.monotonic()-t0,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        save(out / "structure.json",payload)
        result=VariationalCSFResult(converged=passed,iterations=core.iterations,
            state_energies=ci.state_energies,mixing_coefficients=ci.mixing_coefficients,
            ground_energy=ci.ground_energy,orbital_energies=core.orbital_energies,csfs=csfs,
            orbitals_dict=orbitals,bsp_data=data,Z=70,solver_diagnostics=diagnostics,
            occupations={(s.n,s.kappa):s.occupation for s in csfs[0].subshells})
        result._nuclear_potential=core._nuclear_potential
        checkpoint=copy.copy(result);checkpoint._nuclear_potential=None
        with (out / "result.pkl").open("wb") as f:
            pickle.dump(checkpoint,f,protocol=5)
        arrays={"orbital_keys":np.array(sorted(orbitals)),"P":np.array([orbitals[k][0] for k in sorted(orbitals)]),
            "Q":np.array([orbitals[k][1] for k in sorted(orbitals)]),"energies":ci.state_energies,"mixing":ci.mixing_coefficients}
        arrays.update({"basis_"+k:v for k,v in data.items() if isinstance(v,(np.ndarray,int,float))})
        np.savez_compressed(out / "orbitals_workspace.npz",**arrays)
        print("STRUCTURE",json.dumps(clean(payload)),flush=True)
        if not passed:
            save(out / "status.json",{"passed":False,"reason":"valence/metric/Hamiltonian gate failed"})
            return
        from types import SimpleNamespace
        holder=SimpleNamespace(converged=True,Z=70,_csf_result=result)
        lines=TransitionCalculator(holder).compute_all(multipoles=["E1","E2","M1"],gauge="both",line_resolution="ci")
        save(out / "lines.json",[dict(asdict(x),wavelength_vacuum_nm=x.wavelength_nm,
            wavenumber_cm1=x.energy_cm1,gauge_ratio=x.gauge_ratio) for x in lines])
        save(out / "status.json",{"passed":True,"scope":"declared frozen-core model; radial and experimental gates evaluated separately","n_lines":len(lines)})
        print("DONE",args.label,len(lines),flush=True)
    except Exception as exc:
        save(out / "failure.json",{"exception":repr(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-t0})
        print(traceback.format_exc(),flush=True)
        raise
    finally:
        signal.alarm(0);done.set()


if __name__=="__main__":
    main()
