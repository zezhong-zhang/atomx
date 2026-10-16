"""Bounded monovalent on-shell Sigma(2); no fitting or operator dressing."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from collections import defaultdict
import resource
import signal
import time
import traceback
import numpy as np
from common import ROOT, KEYS, EXPERIMENT, load_reference, save, sha, source_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--nvirt", type=int, required=True)
    p.add_argument("--lmax", type=int, required=True)
    p.add_argument("--emax", type=float, default=50.)
    p.add_argument("--base", default="v68_default")
    args = p.parse_args()
    out = ROOT/"results"/args.label
    out.mkdir(parents=True, exist_ok=False)
    kappas = [-1] + [k for l in range(1, args.lmax+1) for k in (l, -l-1)]
    n_virtual = len(kappas)*args.nvirt
    contract = dict(hypothesis="Core-valence attachment self-energy accounts for the DF excitation discrepancy",
        scope="On-shell diagonal Sigma2, both 2p1h and 2h1p, all 19 occupied core orbitals; no Breit, QED, RPA, off-diagonal Sigma or experimental shifts",
        reference="Exactly 68 electrons [Xe]4f14 V68 with one external valence, unchanged core density and orbitals",
        virtual_kappas=kappas, roots_per_kappa=args.nvirt, upper_energy_ceiling_Ha=args.emax,
        bound_virtuals_included=True, radial_baseline=args.base,
        radial_scope="Retained numerically passed DF; MBPT radial convergence evaluated separately",
        denominator_floor_Ha=1e-8, near_zero_policy="raise for nonzero numerator",
        accepted_api_gate="Independent explicit spin-orbital nonzero self-energy sum for occupied and external valence: rel1e-11 abs1e-13",
        resource=dict(candidate_2p1h_terms_per_state=19*n_virtual*(n_virtual+1)//2,
            candidate_2h1p_terms_per_state=n_virtual*19*20//2, fixed_threads=1,
            load_average=os.getloadavg(), cap_seconds=900, anticipated_RSS_MiB=1024),
        experimental_goal_relative=.05, convergence_goal_cm1=1.,
        script_sha256=sha(__file__), common_script_sha256=sha(ROOT/"common.py"))
    save(out/"contract.json", contract)
    t0 = time.monotonic()
    def alarm(*_): raise TimeoutError("900-second declared self-energy cap reached")
    signal.signal(signal.SIGALRM, alarm); signal.alarm(900)
    try:
        from atomx.constants import HARTREE_CM1
        from atomx.methods.mbpt_transitions import mbpt2_orbital_energy_shift
        core, result, levels, identity = load_reference(args.base)
        save(out/"identity.json", identity)
        records = []
        for level in levels:
            key = KEYS[level["label"]]
            P, Q = result.orbitals_dict[key]
            diagrams = defaultdict(float)
            shell_sum = defaultdict(float)
            counts = defaultdict(int)
            virtual_manifest = {}
            min_denominator = float("inf")
            def record(term):
                nonlocal min_denominator
                diagram = term["diagram"]
                diagrams[diagram] += term["contribution_Ha"]
                counts[diagram] += 1
                virtual_manifest.update(zip(term["virtual_indices"], term["virtual"], strict=True))
                if term["numerator"]:
                    min_denominator = min(min_denominator, abs(term["denominator_Ha"]))
                    # Pair diagrams retain both occupied identities as a group.
                    shell_sum[diagram+":"+str(term["core"])] += term["contribution_Ha"]
            start = time.monotonic()
            print("START", level["label"], "nv", n_virtual, "elapsed", time.monotonic()-t0, flush=True)
            shift = mbpt2_orbital_energy_shift(core, level["label"],
                valence_orbital=(P,Q,key[1],level["valence_energy_hartree"]),
                virtual_kappas=kappas, n_virt_max=args.nvirt,
                energy_window_au=args.emax, term_recorder=record)
            records.append(dict(label=level["label"], key=key,
                epsilon_DF_Ha=level["valence_energy_hartree"], Sigma2_Ha=shift,
                Sigma2_cm1=shift*HARTREE_CM1, diagrams_Ha=dict(diagrams),
                shell_groups_Ha=dict(shell_sum), candidate_counts=dict(counts),
                virtual_manifest=virtual_manifest,
                min_nonzero_denominator_Ha=min_denominator,
                elapsed_seconds_uncontrolled=time.monotonic()-start))
            save(out/"partial.json", records)
            print("DONE", level["label"], "shift_cm1", shift*HARTREE_CM1,
                  "seconds", time.monotonic()-start, flush=True)
        ground = records[0]["epsilon_DF_Ha"] + records[0]["Sigma2_Ha"]
        ground_df = records[0]["epsilon_DF_Ha"]
        for r in records:
            r.update(excitation_DF_cm1=(r["epsilon_DF_Ha"]-ground_df)*HARTREE_CM1,
                     excitation_Sigma2_cm1=(r["epsilon_DF_Ha"]+r["Sigma2_Ha"]-ground)*HARTREE_CM1,
                     experiment_cm1=EXPERIMENT[r["label"]])
        save(out/"results.json", dict(records=records, elapsed_seconds_uncontrolled=time.monotonic()-t0,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            source_unchanged=source_identity()==identity["source_sha256"]))
        save(out/"status.json", dict(completed=True, numerical_sum_finite=all(np.isfinite(r["Sigma2_Ha"]) for r in records),
            correlation_converged=False, radiative_operators_dressed=False,
            scope="finite-space on-shell monovalent Sigma2 only"))
    except Exception as exc:
        save(out/"failure.json", dict(exception=repr(exc), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-t0))
        print(traceback.format_exc(), flush=True); raise
    finally: signal.alarm(0)


if __name__ == "__main__": main()
