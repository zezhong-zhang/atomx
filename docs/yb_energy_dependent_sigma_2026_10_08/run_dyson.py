"""Exact native pole capture, frozen-orbital Dyson roots and matched tail controls."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from array import array
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import resource
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "yb_diagnosis_repair_2026_10_08"
sys.path.insert(0, str(PRIOR))
from common import KEYS, EXPERIMENT, load_reference, save, sha, source_identity


def main():
    from atomx.constants import HARTREE_CM1
    from atomx.methods.mbpt_transitions import DiagonalSelfEnergy, build_mbpt2_self_energy
    out = ROOT / "results" / "l6_e50_all"
    out.mkdir(parents=True, exist_ok=False)
    save(out/"preflight.json", dict(contract_sha256=sha(ROOT/"contract.json"),
        driver_sha256=sha(__file__), loader_sha256=sha(PRIOR/"common.py"),
        machine_load=os.getloadavg(), cap_seconds=1000, threads=1))
    t0 = time.monotonic()
    def alarm(*_):
        raise TimeoutError("1000-second declared full capture cap reached")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1000)
    try:
        core, result, levels, identity = load_reference()
        save(out/"identity.json", identity)
        old = {r["label"]:r for r in json.loads(
            (PRIOR/"results/l6_e50_all/results.json").read_text())["records"]}
        old4 = {r["label"]:r for r in json.loads(
            (PRIOR/"results/l4_e50_all/results.json").read_text())["records"]}
        kappas = [-1] + [k for l in range(1,7) for k in (l,-l-1)]
        records = []
        for level in levels:
            label = level["label"]
            key = KEYS[label]
            P, Q = result.orbitals_dict[key]
            start = time.monotonic()
            groups_l, groups_e, diagrams = array("B"), array("d"), array("B")
            virtuals = {}
            counts = {"2p1h":0, "2h1p":0}
            def record(term):
                counts[term["diagram"]] += 1
                virtuals.update(zip(term["virtual_indices"], term["virtual"], strict=True))
                if term["numerator"] != 0:
                    groups_l.append(max(k if k>0 else -k-1 for k,e in term["virtual"]))
                    groups_e.append(max(e for k,e in term["virtual"]))
                    diagrams.append(0 if term["diagram"]=="2p1h" else 1)
            snapshot_identity = dict(reference=identity, label=label, key=key,
                epsilon_DF_Ha=level["valence_energy_hartree"], virtual_kappas=kappas,
                upper_ceiling_Ha=50., root_cap=None, hamiltonian="Dirac-Coulomb",
                scope="frozen-orbital diagonal second-order attachment self-energy")
            resolved_id = hashlib.sha256(json.dumps(snapshot_identity,sort_keys=True).encode()).hexdigest()
            print("START", label, "elapsed", time.monotonic()-t0, flush=True)
            model = build_mbpt2_self_energy(core, label, reference_identity=resolved_id,
                valence_orbital=(P,Q,key[1],level["valence_energy_hartree"]),
                virtual_kappas=kappas, n_virt_max=None, energy_window_au=50., term_recorder=record)
            gl, ge, dg = np.asarray(groups_l), np.asarray(groups_e), np.asarray(diagrams)
            np.savez_compressed(out/f"{label.replace('/','_')}_poles.npz",
                poles_Ha=model.poles_Ha, weights_Ha2=model.weights_Ha2,
                virtual_lmax=gl, virtual_emax_Ha=ge, diagram=dg,
                epsilon_DF_Ha=model.reference_energy_Ha, reference_identity=resolved_id)
            save(out/f"{label.replace('/','_')}_identity.json", snapshot_identity)
            eps = model.reference_energy_Ha
            bracket = (eps-.1,eps)
            variants = {}
            for lmax in (4,6):
                keep = gl <= lmax
                sub = DiagonalSelfEnergy(eps, model.poles_Ha[keep], model.weights_Ha2[keep],
                    resolved_id+f":lmax={lmax}")
                on_shell = sub.evaluate(eps)
                matched = old[label] if lmax==6 else old4[label]
                error = on_shell-matched["Sigma2_Ha"]
                if abs(error)>1e-10:
                    raise ValueError(f"on-shell replay failed {label} l{lmax}: {error}")
                root = sub.solve_dyson(bracket)
                step=1e-5
                fd=(sub.evaluate(root.energy_Ha+step)-sub.evaluate(root.energy_Ha-step))/(2*step)
                analytic=sub.derivative(root.energy_Ha)
                if abs(fd-analytic)>1e-7*max(1.,abs(analytic)):
                    raise ValueError("finite-difference derivative gate failed")
                terms = sub.weights_Ha2/(root.energy_Ha-sub.poles_Ha)
                energy_groups = {}
                for title,lo,hi in (("bound",-np.inf,0.),("0_5_Ha",0.,5.),
                                    ("5_20_Ha",5.,20.),("20_50_Ha",20.,50.)):
                    select = (ge[keep]>=lo)&(ge[keep]<hi)
                    energy_groups[title] = float(np.sum(terms[select])*HARTREE_CM1)
                variants[str(lmax)] = dict(on_shell_Ha=on_shell,
                    matched_old_on_shell_error_Ha=error, dyson=asdict(root),
                    analytic_derivative=analytic, finite_difference_derivative=fd,
                    energy_groups_Dyson_cm1=energy_groups,
                    diagram_groups_Dyson_cm1={str(d):float(np.sum(terms[dg[keep]==d])*HARTREE_CM1)
                                             for d in (0,1)}, retained_nonzero_terms=int(sum(keep)))
            records.append(dict(label=label, key=key, epsilon_DF_Ha=eps,
                variants=variants, virtual_manifest=virtuals,
                evaluated_candidate_counts=counts, elapsed_seconds_uncontrolled=time.monotonic()-start))
            save(out/"partial.json",records)
            print("DONE",label,"Dyson shift_cm1",variants["6"]["dyson"]["shift_Ha"]*HARTREE_CM1,
                  "residue",variants["6"]["dyson"]["residue"],
                  "seconds",time.monotonic()-start,flush=True)
        for lmax in ("4","6"):
            ground_df=records[0]["epsilon_DF_Ha"]
            ground_on=ground_df+records[0]["variants"][lmax]["on_shell_Ha"]
            ground_dy=records[0]["variants"][lmax]["dyson"]["energy_Ha"]
            for r in records:
                v=r["variants"][lmax]
                v["excitation_on_shell_cm1"]=(r["epsilon_DF_Ha"]+v["on_shell_Ha"]-ground_on)*HARTREE_CM1
                v["excitation_Dyson_cm1"]=(v["dyson"]["energy_Ha"]-ground_dy)*HARTREE_CM1
                v["experiment_cm1"]=EXPERIMENT[r["label"]]
                v["residual_Dyson_cm1"]=v["excitation_Dyson_cm1"]-v["experiment_cm1"]
                v["energy_dependence_excitation_cm1"]=v["excitation_Dyson_cm1"]-v["excitation_on_shell_cm1"]
        source_ok=source_identity()==identity["source_sha256"]
        if not source_ok:
            raise ValueError("production source changed during measurement")
        errors=[abs(r["variants"]["6"]["residual_Dyson_cm1"])/EXPERIMENT[r["label"]] for r in records[1:]]
        tail=[r["variants"]["6"]["excitation_Dyson_cm1"]-r["variants"]["4"]["excitation_Dyson_cm1"] for r in records[1:]]
        save(out/"results.json",dict(records=records,source_unchanged=source_ok,
            elapsed_seconds_uncontrolled=time.monotonic()-t0,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        save(out/"status.json",dict(completed=True, on_shell_replay_passed=True,
            dyson_residual_and_residue_passed=True, experimental_5percent_goal_passed=max(errors)<.05,
            max_relative_excitation_error=max(errors), l4_to_l6_Dyson_increments_cm1=tail,
            partial_wave_1cm1_gate_passed=max(abs(t) for t in tail)<1.,
            radial_convergence="not_tested",higher_order_coverage="not_completed",
            radiative_operators_dressed=False,scope="diagonal frozen-orbital Dyson only"))
        print("COMPLETE",time.monotonic()-t0,"seconds",flush=True)
    except Exception as exc:
        save(out/"failure.json",dict(exception=repr(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-t0))
        print(traceback.format_exc(),flush=True)
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    main()
