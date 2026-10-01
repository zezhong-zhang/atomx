"""Live FAC comparison of the native public DR driver for Ne-like ions."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import atomx.scattering.autoionization.mscheme_ci as ci
from atomx import Atom, Basis, Sampling
from atomx.scattering.autoionization import dr_calculate, dr_rate_coefficient
from atomx.scattering.autoionization.auger import AutoionizationState
from benchmarks.bench_reference_dr_radiative import deck, fac_reference, run_reference


def main():
    parser = argparse.ArgumentParser(
        description="Matched public DR/FAC panel, local orbitals and adaptive continuum grids"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fac", type=Path, required=True)
    parser.add_argument("--z", type=int, nargs="+", default=[26, 82])
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=False)
    report = {
        "cases": [],
        "scope": "Public dr_calculate; separate dr33 hole/parity spaces; point, infinite mass, local Gu, corrections off; independent FAC ground potential and all n=3 final E1 levels.",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/atomx").rglob("*.py"))
        },
        "fac_executable_sha256": hashlib.sha256(args.fac.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    factory = ci.manifold_from_configs
    for z in args.z:
        folder = out / f"Z{z:03d}"
        folder.mkdir(exist_ok=False)
        (folder / "run.sf").write_text(deck(z, 16000, 8.0, include_auger=True))
        tf = run_reference([str(args.fac.resolve()), "run.sf"], folder, timeout=600)
        log = (folder / "stdout.txt").read_text()
        loops = [
            line.split(":", 1)[1].split()
            for line in log.splitlines()
            if line.startswith("optimize loop:")
        ]
        if (
            not loops
            or max(int(row[0]) for row in loops) >= 150
            or "Maximum iteration reached" in log
        ):
            raise RuntimeError("FAC SCF convergence not established")
        ref = fac_reference(folder)
        fr = []
        for group, levels in ref["groups"].items():
            for lv in levels:
                i = lv["ILEV"]
                fr.append(
                    AutoionizationState(
                        f"{group}:{i}",
                        lv["ENERGY_eV"] - ref["ground"]["ENERGY_eV"],
                        lv["TWOJ"] / 2,
                        sum(r["rate_per_s"] for r in ref["auger"] if r["upper"] == i),
                        sum(
                            r["A_per_s"]
                            for r in ref["lines"]["finite_wavelength"]
                            if r["upper"] == i
                        ),
                    )
                )
        a = Atom(
            z,
            charge=z - 10,
            config="1s2 2s2 2p6",
            engine="shooting",
            exchange="gu",
            corrections="none",
            nuclear="point",
            basis=Basis(sampling=Sampling(n_grid=4000)),
        )
        t = time.perf_counter()
        a.solve(tol=1e-10, max_iter=150)
        scf = time.perf_counter() - t
        if not a.converged:
            raise RuntimeError("Atomx ground SCF did not converge")
        calls = []

        def measured(result, configs, *, calls=calls, **kw):
            m, w = factory(result, configs, **kw)
            calls.append(
                dict(
                    configs=configs,
                    energy_limit_eV=kw.get("e_max_eV"),
                    points=len(m.r_fine),
                    r_max=float(m.r_fine[-1]),
                )
            )
            return m, w

        ci.manifold_from_configs = measured
        # Temperatures in eV; retain Fe standard MK and scale by Z² for heavy ion.
        temperatures = np.array([1, 5, 10, 50]) * 1e6 * 8.617333262e-5 * (z / 26) ** 2
        t = time.perf_counter()
        try:
            r = dr_calculate(a, "1s2 2s2 2p6 3s1", T_eV=temperatures)
        finally:
            ci.manifold_from_configs = factory
        elapsed = time.perf_counter() - t
        f = dr_rate_coefficient(fr, temperatures, initial_state_g=1.0)
        def key(r):
            return (r.label.split(":")[0], r.J, r.E_resonance_eV)
        pairs = []
        for x, y in zip(
            sorted(r.resonances, key=key), sorted(fr, key=key), strict=True
        ):
            if key(x)[:2] != key(y)[:2]:
                raise ValueError("FAC and Atomx symmetry inventories differ")
            pairs.append(
                dict(
                    label=x.label,
                    J=x.J,
                    atomx_energy_eV=x.E_resonance_eV,
                    fac_energy_eV=y.E_resonance_eV,
                    atomx_auger_per_s=x.gamma_auger_per_s,
                    fac_auger_per_s=y.gamma_auger_per_s,
                    atomx_radiative_per_s=x.a_radiative_per_s,
                    fac_radiative_per_s=y.a_radiative_per_s,
                )
            )
        case = dict(
            z=z,
            levels=len(r.resonances),
            fac_s=tf,
            atomx_scf_s=scf,
            atomx_dr_s=elapsed,
            grid_calls=calls,
            T_eV=temperatures.tolist(),
            atomx_alpha_cm3_s=r.alpha_cm3_s.tolist(),
            fac_alpha_cm3_s=f.alpha_cm3_s.tolist(),
            alpha_ratios=(r.alpha_cm3_s / f.alpha_cm3_s).tolist(),
            levels_comparison=pairs,
            raw_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in folder.iterdir()
                if p.is_file()
            },
        )
        report["cases"].append(case)
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in case.items()
                    if k not in ["levels_comparison", "raw_sha256"]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
