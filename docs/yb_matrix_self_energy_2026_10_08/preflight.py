"""Verify full native virtual and external spaces before ten bounded captures."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import json
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import KEYS, load_reference, save, sha


def main():
    from scipy.linalg import eigh
    from atomx.methods.mbpt import build_virtual_basis, _build_closed_shell_fock
    from atomx.numerics.bspline.convol import ConvolCache
    from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays
    out = ROOT/"preflight"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    contract = json.loads((ROOT/"contract.json").read_text())
    cases = []
    for case in contract["captures"]:
        core, result, levels, identity = load_reference(case["reference"])
        data, cache = core._bsp_data, ConvolCache(core._bsp_data)
        _, w = workspace_arrays(data)
        w = w.ravel()
        kappas = [-1]+[k for l in range(1, case["lmax"]+1) for k in (l, -l-1)]
        arrays, records, total = {}, [], 0
        for kappa in kappas:
            _, states = build_virtual_basis(core, kappa, cache=cache, e_max=case["emax_Ha"],
                                             potential_mode="VN", orthogonal_to_occupied=True)
            total += len(states)
            P, Q, e = np.array([s[0] for s in states]), np.array([s[1] for s in states]), np.array([s[2] for s in states])
            p, q = zip(*(eval_orbital_pair(pp, qq, data) for pp, qq, _ in states))
            p, q = np.array(p).reshape(len(states), -1), np.array(q).reshape(len(states), -1)
            gram = (p*w)@p.T+(q*w)@q.T
            metric = float(np.max(abs(gram-np.eye(len(states)))))
            core_overlap = 0.
            for pc, qc, occ in core.orbitals_raw.get(kappa, []):
                pc, qc = eval_orbital_pair(pc, qc, data)
                pc, qc = pc.ravel(), qc.ravel()
                core_overlap = max(core_overlap, float(np.max(abs(p@(w*pc)+q@(w*qc)))))
            if max(metric, core_overlap) > 1e-7:
                raise ValueError("virtual metric gate failed")
            arrays.update({f"P_{kappa}": P, f"Q_{kappa}": Q, f"energy_{kappa}": e})
            record = dict(kappa=kappa, count=len(states), metric_error=metric, core_overlap=core_overlap)
            label = next((label for label, key in KEYS.items() if key[1] == kappa), None)
            if label is not None:
                key = KEYS[label]
                oldP, oldQ = result.orbitals_dict[key]
                pp, qq = eval_orbital_pair(oldP, oldQ, data)
                pp, qq = pp.ravel(), qq.ravel()
                overlap = float(np.sum(w*(p[0]*pp+q[0]*qq)))
                old_eps = next(s["valence_energy_hartree"] for s in levels if s["label"] == label)
                H, S = _build_closed_shell_fock(core, kappa, cache=cache, potential_mode="VN")
                rawP, rawQ, rawE = P[:3].copy(), Q[:3].copy(), e[:3].copy()
                rawP[0], rawQ[0], rawE[0] = oldP, oldQ, old_eps
                X = np.r_[rawP.T, rawQ.T]
                h0_raw_error = float(np.max(abs(X.T@H@X-np.diag(rawE))))
                # Refined Rydberg orbitals can differ slightly from their
                # returned diagonal energies. Canonicalize the actual Fock
                # projection in the unchanged span instead of weakening H0.
                ee, rotation = eigh(X.T@H@X, X.T@S@X)
                for i in range(3):
                    if rotation[i, i] < 0:
                        rotation[:, i] *= -1
                XX = X@rotation
                h0_error = float(np.max(abs(XX.T@H@XX-np.diag(ee))))
                if abs(abs(overlap)-1.) > 1e-7 or h0_error > 1e-10:
                    save(out/(label.replace("/", "_")+"_H0_failure.json"),
                         dict(overlap=overlap, old_energy=old_eps, new_energy=ee[0], error=h0_error))
                    raise ValueError("DF external reference identity/H0 gate failed")
                arrays.update({f"external_P_{kappa}": XX[:data["ns"]].T,
                               f"external_Q_{kappa}": XX[data["ns"]:].T,
                               f"external_energy_{kappa}": ee,
                               f"external_raw_P_{kappa}": rawP,
                               f"external_raw_Q_{kappa}": rawQ,
                               f"external_raw_energy_{kappa}": rawE})
                record.update(label=label, external_energies_Ha=ee.tolist(),
                              old_target_overlap=overlap, old_energy_difference_Ha=float(rawE[0]-old_eps),
                              virtual_lowest_energy_difference_from_old_Ha=float(e[0]-old_eps),
                              canonical_energy_difference_from_old_Ha=float(ee[0]-old_eps),
                              raw_Fock_projection_error_Ha=h0_raw_error,
                               external_rotation=rotation.tolist(),
                              external_H0_matrix_error_Ha=h0_error)
            records.append(record)
            print(case["name"], kappa, len(states), "metric", metric, flush=True)
        path = out/(case["name"]+"_virtuals.npz")
        np.savez_compressed(path, **arrays)
        count2p1h = 19*total*(total+1)//2
        count2h1p = total*19*20//2
        bytes_upper = (count2p1h+count2h1p)*80  # pole plus full 3x3 residue, including exact zeros.
        # Four live buffers in immutable validation/copy plus kernels. No pair-list allocation.
        rss_upper_MiB = (4*bytes_upper)/(1024**2)+4096
        case_record = dict(case=case, reference_identity=identity, blocks=records,
                           total_virtuals=total, candidate_2p1h=count2p1h, candidate_2h1p=count2h1p,
                           storage_upper_bytes=bytes_upper, conservative_RSS_MiB=rss_upper_MiB,
                           virtual_snapshot_sha256=sha(path))
        if rss_upper_MiB > contract["resource_caps"]["RSS_MiB_per_state"]:
            raise ValueError("declared per-state memory cap fails preflight")
        cases.append(case_record)
    storage = sum(c["storage_upper_bytes"]*5 for c in cases)
    if storage > contract["resource_caps"]["disk_GiB"]*1024**3 or storage > shutil.disk_usage(ROOT).free:
        raise ValueError("disk cap fails preflight")
    save(out/"results.json", dict(cases=cases, maximum_raw_storage_bytes=storage,
                                  contract_sha256=sha(ROOT/"contract.json"),
                                  script_sha256=sha(__file__), load=os.getloadavg(),
                                  seconds=time.monotonic()-started, passed=True))


if __name__ == "__main__":
    main()
