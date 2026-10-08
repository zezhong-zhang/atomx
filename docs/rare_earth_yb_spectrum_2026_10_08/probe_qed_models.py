"""Fixed-orbital sensitivity of approximate self-energy models, not a QED error bar."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import pickle
import signal
import time
import traceback
import numpy as np
from run_yb import ROOT, save, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="v68_default")
    args = parser.parse_args()
    out = ROOT / "results" / (args.base + "_qed_models")
    out.mkdir(exist_ok=False)
    contract = {
        "hypothesis": "Different approximate SE operators can distinguish a robust radiative scale from sensitivity to the default high-n parameterization.",
        "fixed_conditions": "same passed V68 orbitals, five valence states, Z70; first-order expectations only; common closed-core terms cancel exactly in excitation differences",
        "models": ["indelicato", "shabaev_local", "flambaum_ginges"],
        "observable": "valence VP and SE expectation, then difference from 6s; NOT orbital reoptimization or full screened QED",
        "limits": "Indelicato/Mohr n=6 uses native extrapolation; Shabaev here is local-only, FG electric model omits magnetic form factor. Model spread is NOT a physical uncertainty bound.",
        "acceptance": {"finite": True, "VP_plus_default_SE_excitation_replay_cm1": 1e-5},
        "stopping_rule": "300 seconds, retain exceptions; do not substitute a failed backend",
        "resource_estimate": "reuse 24-orbital workspace, evaluate five valence matrix elements per model; <2GiB anticipated"
    }
    save(out / "experiment_contract.json", contract)
    snapshot(out)
    def stop(*_):
        raise TimeoutError("300-second QED sensitivity cap reached")
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300)
    started = time.monotonic()
    try:
        from atomx.config.subshell import Subshell
        from atomx.constants import HARTREE_CM1
        from atomx.corrections.qed import short_range_radial_matrix, self_energy_radial_matrix, uehling_potential
        base = ROOT / "results" / args.base
        with (base / "result.pkl").open("rb") as handle:
            r = pickle.load(handle)
        keys = [(6,-1),(6,1),(6,-2),(5,2),(5,-3)]
        labels = ["6s_1/2","6p_1/2","6p_3/2","5d_3/2","5d_5/2"]
        orbitals = [(*r.orbitals_dict[k], k[1]) for k in keys]
        vp = np.diag(short_range_radial_matrix(orbitals, r.bsp_data, lambda radius: uehling_potential(radius,70)))
        labeled = [(*r.orbitals_dict[k], k[1], Subshell(k[0],k[1],1).label) for k in keys]
        records = []
        for model in contract["models"]:
            se = np.diag(self_energy_radial_matrix(labeled,70,r.bsp_data,method=model))
            for i, label in enumerate(labels):
                records.append({"model":model,"label":label,"valence_VP_Ha":float(vp[i]),
                    "valence_SE_Ha":float(se[i]),"QED_excitation_shift_cm1":float((se[i]+vp[i]-se[0]-vp[0])*HARTREE_CM1)})
        from run_yb import clean
        import json
        expected = {x["label"]:x["QED_excitation_shift_cm1"] for x in json.loads((ROOT/"results"/(args.base+"_corrections")/"factors.json").read_text())["records"]}
        errors = [abs(x["QED_excitation_shift_cm1"]-expected[x["label"]]) for x in records if x["model"]=="indelicato"]
        passed = all(np.isfinite(x["valence_SE_Ha"]) for x in records) and max(errors)<1e-5
        save(out / "factors.json", {"records":records,"default_replay_max_cm1":max(errors),
            "base_orbitals_sha256":hashlib.sha256((base/"orbitals_workspace.npz").read_bytes()).hexdigest(),
            "elapsed_seconds_uncontrolled":time.monotonic()-started})
        save(out / "status.json", {"passed":bool(passed),"scope":"approximate operator sensitivity only"})
        print(clean(records), flush=True)
    except Exception as exc:
        save(out / "failure.json", {"exception":repr(exc),"traceback":traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":
    main()
