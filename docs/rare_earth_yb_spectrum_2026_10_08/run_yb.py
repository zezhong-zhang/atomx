"""Declared Yb II free-ion calculation; no fitted experimental energy shifts."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import platform
import resource
import signal
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CORE = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6"
CONFIGS = [CORE + " " + tail for tail in ("4f14 6s1", "4f14 6p1", "4f14 5d1")]
CONTRACT = {
    "hypothesis": "A common-orbital five-level relativistic mean-field model resolves Yb II 6s/6p/5d lines; agreement with experiment is an independent test of missing correlation, not a convergence criterion.",
    "fixed_conditions": {"Z": 70, "charge": 1, "electrons": 69,
        "configs": CONFIGS, "engine": "bspline", "exchange": "hf",
        "Hamiltonian": "finite-Fermi-nucleus no-pair Dirac-Coulomb; fixed nuclear positive-energy projector",
        "orbital_objective": "equal-weight average of all five J/parity resolved levels",
        "corrections": "none", "BLAS_threads": 1},
    "observable": "resolved many-electron energy differences and finite-wavelength E1/E2/M1 tensors in the same common orbital basis",
    "acceptance": {"solver_converged": True, "requested_stationarity_tolerance": 1e-7,
        "orbital_metric_max_error": 1e-8, "radial_transition_drift_cm1": 1.0,
        "experimental_excitation_relative_error_goal": 0.05},
    "distinguishing_outcomes": "SCF/metric failure blocks production rates; radial drift >1 cm^-1 blocks a radial-convergence claim; experiment >5% fails the mean-field accuracy goal and motivates correlation/configuration expansion.",
    "resource_estimate": "5 CSFs, 5 one-dimensional J/parity blocks, 24 radial orbitals; spline matrices a few hundred squared; anticipated <3 GiB RSS. No 69-electron SD expansion requested.",
    "stopping_rule": "900 seconds per solve; stop on exception or unconverged orbital solution, retaining diagnostics; never change state weights or insert empirical offsets to pass.",
    "scope_exclusions": ["crystal field or host", "hyperfine/isotope-resolved centers", "full core-valence correlation", "all-order QED", "experimental line intensity/population simulation"],
}


def clean(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    if isinstance(x, (str, float, int, bool)) or x is None:
        return x
    return repr(x)


def save(path, x):
    path.write_text(json.dumps(clean(x), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot(out):
    files = sorted((REPO / "src/atomx").rglob("*.py"))
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(out / "source_identity.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "source_scope": "all Atomx Python source files, including pre-existing dirty/untracked source; HEAD is not the source identity",
        "source_sha256": hashes,
        "source_map_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "threads": 1,
        "load_average": os.getloadavg(), "logical_CPUs": os.cpu_count(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--he", type=float, default=.25)
    parser.add_argument("--hi", type=float, default=.25)
    parser.add_argument("--hmax", type=float, default=1.)
    parser.add_argument("--rmax", type=float, default=60.)
    parser.add_argument("--ks", type=int, default=9)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--protocol", choices=("eol", "frozen"), default="eol")
    args = parser.parse_args()
    out = ROOT / "results" / args.label
    out.mkdir(parents=True, exist_ok=False)
    contract = copy.deepcopy(CONTRACT)
    if args.protocol == "frozen":
        contract["hypothesis"] = "Ground-field frozen-orbital Dirac-Fock/RCI supplies an independently defined free-ion baseline when five-level EOL orbital optimization fails its stationarity gate. Experiment tests this approximation separately."
        contract["fixed_conditions"]["Hamiltonian"] = "finite-Fermi-nucleus Dirac-Coulomb on shared frozen Dirac-Fock orbitals"
        contract["fixed_conditions"]["orbital_objective"] = "native fac_frozen protocol: optimize first 4f14 6s configuration; four spectator orbitals have 0.001 initializer populations each, but all physical CI states contain exactly 69 electrons"
        contract["acceptance"]["requested_stationarity_tolerance"] = None
        contract["acceptance"]["requested_DF_inner_tolerance"] = args.tol
        contract["interpretation_limit"] = "Ground-field SCF convergence and CI matrix convergence do not certify five-level EOL or frozen excited-orbital stationarity. The failed EOL results remain failed."
    if not (ROOT / "experiment_contract.json").exists():
        save(ROOT / "experiment_contract.json", CONTRACT)
    save(out / "experiment_contract.json", contract)
    save(out / "inputs.json", dict(vars(args), configs=CONFIGS, contract=contract))
    snapshot(out)
    from atomx import Atom, Basis, Correlation
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.constants import HARTREE_CM1
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import nuclear_profile
    import numpy as np
    csfs = [c for cfg in CONFIGS for c in enumerate_jj_csfs(cfg)]
    save(out / "preflight.json", {"n_CSFS": len(csfs),
        "orbital_keys": sorted({(s.n, s.kappa) for c in csfs for s in c.subshells}),
        "electron_counts": [sum(s.occupation for s in c.subshells) for c in csfs],
        "blocks": [(c.total_J2, c.parity) for c in csfs],
        "nucleus": nuclear_profile(70, "fermi")})
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
    print("START", args.label, vars(args), flush=True)
    try:
        correlation = (Correlation(j_resolved_csfs=True, state_average=CIStateAverage())
            if args.protocol == "eol" else
            Correlation(j_resolved_csfs=True, radial_optimization="fac_frozen"))
        atom = Atom(70, charge=1, config=CONFIGS, nuclear="fermi", exchange="hf",
            corrections="none", engine="bspline",
            basis=Basis(ks=args.ks, r_max=args.rmax, grid_hi=args.hi, grid_he=args.he, grid_hmax=args.hmax),
            correlation=correlation)
        atom.solve(tol=args.tol, max_iter=args.iterations)
        result = atom.result
        checkpoint = copy.copy(result)
        checkpoint._nuclear_potential = None
        checkpoint.calculation_spec = None  # mappingproxy metadata is retained as JSON below
        try:
            with (out / "result.pkl").open("wb") as f:
                pickle.dump(checkpoint, f, protocol=5)
        except Exception as exc:
            save(out / "pickle_failure.json", {"exception": repr(exc),
                "disposition": "portable orbital/workspace arrays retained separately"})
        arrays = {"orbital_keys": np.array(sorted(result.orbitals_dict)),
            "P": np.array([result.orbitals_dict[k][0] for k in sorted(result.orbitals_dict)]),
            "Q": np.array([result.orbitals_dict[k][1] for k in sorted(result.orbitals_dict)]),
            "energies": result.state_energies, "mixing": result.mixing_coefficients}
        arrays.update({"basis_"+k:v for k,v in result.bsp_data.items()
            if isinstance(v, (np.ndarray, int, float))})
        np.savez_compressed(out / "orbitals_workspace.npz", **arrays)
        levels = []
        for index, energy in enumerate(result.state_energies):
            dom = int(np.argmax(np.abs(result.mixing_coefficients[:, index])))
            c = result.csfs[dom]
            levels.append({"index": index, "energy_hartree": float(energy),
                "excitation_cm1": float((energy-result.state_energies[0])*HARTREE_CM1),
                "two_J": c.total_J2, "parity": c.parity,
                "dominant_CSF_index": dom, "dominant_CSF": c.label,
                "dominant_weight": float(abs(result.mixing_coefficients[dom,index])**2)})
        from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
        _, s = build_dirac_matrices(result.bsp_data["ns"], -1, 70,
            result.bsp_data, result._nuclear_potential)
        metric_errors = []
        keys = sorted(result.orbitals_dict)
        for a in keys:
            p, q = result.orbitals_dict[a]
            for b in keys:
                if a[1] != b[1]:
                    continue
                pb, qb = result.orbitals_dict[b]
                metric_errors.append(abs(float(np.concatenate((p,q))@s@np.concatenate((pb,qb)))-(1. if a==b else 0.)))
        payload = {"converged": bool(result.converged), "iterations": result.iterations,
            "levels": levels, "orbital_optimization": result.orbital_optimization,
            "solver_diagnostics": result.solver_diagnostics,
            "calculation_spec": result.calculation_spec.to_dict() if result.calculation_spec is not None else None,
            "orbital_energies_hartree": result.orbital_energies,
            "occupations": result.occupations,
            "actual_ns": result.bsp_data["ns"], "actual_ks": result.bsp_data["ks"],
            "orbital_metric_max_error": max(metric_errors),
            "elapsed_seconds_uncontrolled": time.monotonic()-t0,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        save(out / "structure.json", payload)
        print("STRUCTURE", json.dumps(clean({k: payload[k] for k in ("converged", "iterations", "levels", "orbital_metric_max_error", "elapsed_seconds_uncontrolled")})), flush=True)
        if not result.converged:
            save(out / "status.json", {"passed": False, "reason": "orbital solve unconverged; no production rates"})
            return
        lines = atom.transitions(multipoles=["E1", "E2", "M1"], gauge="both", line_resolution="ci")
        save(out / "lines.json", [dict(asdict(x), wavelength_vacuum_nm=x.wavelength_nm,
            wavenumber_cm1=x.energy_cm1, gauge_ratio=x.gauge_ratio) for x in lines])
        save(out / "status.json", {"passed": True, "scope": "stationary declared model only; radial/experimental gates evaluated separately", "n_lines":len(lines)})
        print("DONE", args.label, "n_lines", len(lines), flush=True)
    except Exception as exc:
        save(out / "failure.json", {"exception": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic()-t0})
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        signal.alarm(0)
        done.set()


if __name__ == "__main__":
    main()
