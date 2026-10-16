"""Ten complete matched state captures; freeze source during the experiment."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import resource
import signal
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import KEYS, load_reference, save, sha, source_identity


def worker(job):
    from atomx.methods.mbpt_matrix import MatrixSelfEnergy, build_mbpt2_matrix_self_energy
    from atomx.methods.mbpt import build_virtual_basis
    case, label = job
    out = ROOT/"results"/case["name"]/label.replace("/", "_")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    contract = json.loads((ROOT/"contract.json").read_text())
    cap = contract["resource_caps"]["seconds_per_state"]
    def alarm(*_):
        raise TimeoutError(f"{cap}-second declared cap")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(cap)
    try:
        core, r, levels, identity = load_reference(case["reference"])
        kappa = KEYS[label][1]
        z = np.load(ROOT/"preflight"/(case["name"]+"_virtuals.npz"))
        targets = [(P.copy(), Q.copy(), kappa, float(e)) for P, Q, e in zip(
            z[f"external_P_{kappa}"], z[f"external_Q_{kappa}"], z[f"external_energy_{kappa}"]) ]
        rawtargets = [(P.copy(), Q.copy(), kappa, float(e)) for P, Q, e in zip(
            z[f"external_raw_P_{kappa}"], z[f"external_raw_Q_{kappa}"], z[f"external_raw_energy_{kappa}"]) ]
        kappas = [-1]+[k for l in range(1, case["lmax"]+1) for k in (l, -l-1)]
        physical_id = dict(reference=identity, case=case, label=label, kappa=kappa,
                           external_energies_Ha=[t[3] for t in targets],
                           virtual_snapshot_sha256=sha(ROOT/"preflight"/(case["name"]+"_virtuals.npz")),
                           Hamiltonian="Dirac-Coulomb", metric="native radial quadrature",
                           core_zero="Yb III [Xe]4f14 ground; frozen reference",
                           scope="frozen complete second-order matrix; finite 3-function external space")
        resolved = hashlib.sha256(json.dumps(physical_id, sort_keys=True).encode()).hexdigest()
        save(out/"identity.json", physical_id)
        counts = {"2p1h": 0, "2h1p": 0}
        last_heartbeat = time.monotonic()
        def record(pole, matrix, diagram):
            nonlocal last_heartbeat
            counts[diagram] += 1
            if time.monotonic()-last_heartbeat >= 30:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                if rss > contract["resource_caps"]["RSS_MiB_per_state"]:
                    raise MemoryError("declared RSS cap exceeded")
                save(out/"progress.json", dict(counts=counts, seconds=time.monotonic()-started, RSS_MiB=rss))
                print("PROGRESS", case["name"], label, counts, round(time.monotonic()-started), flush=True)
                last_heartbeat = time.monotonic()
        print("START", case["name"], label, "pid", os.getpid(), flush=True)
        checked_virtual_blocks = []
        def audited_basis(reference, kap, **kwargs):
            occupied, states = build_virtual_basis(reference, kap, **kwargs)
            # Transparent observer: eigensystems are recomputed by the native
            # provider, then compared with preflight before they enter sums.
            for name, values in (("P", [s[0] for s in states]), ("Q", [s[1] for s in states]),
                                 ("energy", [s[2] for s in states])):
                if not np.array_equal(np.array(values), z[f"{name}_{kap}"]):
                    raise ValueError(f"actual virtual snapshot differs from preflight: {kap}/{name}")
            checked_virtual_blocks.append(kap)
            return occupied, states
        with patch("atomx.methods.mbpt.build_virtual_basis", side_effect=audited_basis):
            rawmodel = build_mbpt2_matrix_self_energy(core, rawtargets, reference_identity=resolved+":raw",
                                                   virtual_kappas=kappas, energy_window_au=case["emax_Ha"],
                                                   n_virt_max=None, term_recorder=record)
        scalar = rawmodel.diagonal(0)
        eps = rawtargets[0][3]
        bracket = (eps-.1, eps+.02)
        scalar_root = scalar.solve_dyson(bracket)
        oldpath = ROOT.parent/"yb_correlation_tail_2026_10_08/results"/(
            "coarse_l8_e200" if case["reference"] == "v68_default" else "fine_l6_e200")/label.replace("/", "_")/"results.json"
        old = json.loads(oldpath.read_text())
        old_variant = next(x for x in old["variants"] if x["lmax"] == case["lmax"] and x["emax_Ha"] == case["emax_Ha"])
        replay_error = scalar_root.energy_Ha-old_variant["dyson"]["energy_Ha"]
        onshell_error = scalar.evaluate(eps)-old_variant["on_shell_Ha"]
        if max(abs(replay_error), abs(onshell_error)) > 1e-10:
            raise ValueError("matched old diagonal replay gate failed")
        pre = json.loads((ROOT/"preflight/results.json").read_text())
        entry = next(c for c in pre["cases"] if c["case"]["name"] == case["name"])
        if (counts["2p1h"] != entry["candidate_2p1h"] or counts["2h1p"] != entry["candidate_2h1p"]
                or checked_virtual_blocks != kappas):
            raise ValueError("complete physical pair/virtual coverage gate failed")
        rotation = np.array(next(b for b in entry["blocks"] if b["kappa"] == kappa)["external_rotation"])
        model = MatrixSelfEnergy(np.array([t[3] for t in targets]), rawmodel.poles_Ha,
                                 np.einsum("ia,tij,jb->tab", rotation, rawmodel.residues_Ha2, rotation, optimize=True),
                                 resolved+":canonical")
        del rawmodel, scalar
        np.savez_compressed(out/"matrix_poles.npz", eps_Ha=model.reference_energies_Ha,
                            poles_Ha=model.poles_Ha, residues_Ha2=model.residues_Ha2,
                            reference_identity=model.reference_identity)
        eps = targets[0][3]
        bracket = (eps-.1, eps+.02)
        variants = []
        for n in (1, 2, 3):
            sub = model if n == 3 else model.subspace(tuple(range(n)))
            root = sub.solve_dyson(bracket)
            p = sum(root.coefficients[i]*targets[i][0] for i in range(n))
            q = sum(root.coefficients[i]*targets[i][1] for i in range(n))
            np.savez_compressed(out/f"mixed_orbital_P{n}.npz", P=p, Q=q, energy_Ha=root.energy_Ha,
                                coefficients=root.coefficients, residue=root.residue, kappa=kappa)
            variants.append(dict(external_dimension=n, root=asdict(root),
                                 sigma_at_root_Ha=sub.evaluate(root.energy_Ha)))
            if n != 3:
                del sub
        if source_identity() != json.loads((ROOT/"capture_source_sha256.json").read_text()):
            raise ValueError("production source changed during capture")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if rss > contract["resource_caps"]["RSS_MiB_per_state"]:
            raise MemoryError("declared RSS cap exceeded")
        row = dict(case=case["name"], label=label, variants=variants, candidate_counts=counts,
                   actual_virtual_snapshots_match_preflight=True,
                   retained_nonzero_residues=len(model.poles_Ha), scalar_root=asdict(scalar_root),
                   old_diagonal_root_replay_error_Ha=replay_error, old_onshell_replay_error_Ha=onshell_error,
                   canonical_scalar_reference_change_Ha=variants[0]["root"]["energy_Ha"]-scalar_root.energy_Ha,
                   peak_process_RSS_MiB=rss, seconds=time.monotonic()-started,
                   source_unchanged=True, matrix_poles_sha256=sha(out/"matrix_poles.npz"))
        save(out/"results.json", row)
        save(out/"status.json", dict(completed=True, numerical_gates_passed=True))
        print("DONE", case["name"], label, "seconds", round(row["seconds"], 1), flush=True)
        return dict(case=case["name"], label=label, completed=True, seconds=row["seconds"])
    except Exception as exc:
        failure = dict(exception=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic()-started)
        save(out/"failure.json", failure)
        print("FAILED", case["name"], label, repr(exc), flush=True)
        return dict(case=case["name"], label=label, completed=False, failure=failure)
    finally:
        signal.alarm(0)
        gc.collect()


def main():
    (ROOT/"results").mkdir(exist_ok=False)
    preflight = json.loads((ROOT/"preflight/results.json").read_text())
    assert preflight["passed"]
    contract = json.loads((ROOT/"contract.json").read_text())
    source = source_identity()
    save(ROOT/"capture_source_sha256.json", source)
    save(ROOT/"launch.json", dict(source_sha256=source, load=os.getloadavg(),
                                  workers=contract["resource_caps"]["workers"],
                                  script_sha256=sha(__file__), contract_sha256=sha(ROOT/"contract.json")))
    # Fine model is the retained authority, so start its five states together.
    jobs = [(case, label) for case in reversed(contract["captures"]) for label in KEYS]
    started, status = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=contract["resource_caps"]["workers"]) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            status.append(future.result())
            save(ROOT/"capture_status.json", dict(jobs=status, seconds=time.monotonic()-started,
                                                   completed=len(status) == len(jobs) and all(x["completed"] for x in status)))
    print("ALL", all(x["completed"] for x in status), round(time.monotonic()-started), flush=True)


if __name__ == "__main__":
    main()
