"""Full retained Yb V68 / l<=3 / <5 Ha native reduced core-SD workload."""

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "yb_full_run003"
REFERENCE = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docs/yb_diagnosis_repair_2026_10_08"))
from build_pci_sd_backend import build
from common import load_reference

from atomx.methods.atomic_space import build_atomic_space
from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    HERE.mkdir(exist_ok=False)
    began = time.perf_counter()
    cap = 32 * 1024**3
    contract = dict(
        reference="original native V68, Fermi nucleus, ns123",
        occupied="all 68 core electrons, no frozen subshell",
        virtual_lmax=3,
        virtual_energy_max_Ha=5,
        memory_bytes=cap,
        residual_tolerance_Ha=1e-10,
        gmres_restart=30,
        maxiter=100,
        load=os.getloadavg(),
        threads=1,
        hypothesis="native reduced pCI diagrams make the complete retained Yb band practical without magnetic quartic storage",
        stop="reject complete overbudget request or failed equation residual; no space reduction or fitted shifts",
        performance_scope="stage times and peak process RSS; no speedup or experimental accuracy claim",
    )
    save("yb_full_contract.json", contract)
    save(
        "yb_full_source_sha256.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/atomx").rglob("*.py"))
        },
    )
    core, *_, reference_identity = load_reference()
    space = build_atomic_space(core, virtual_lmax=3, virtual_energy_max_Ha=5)
    feature = CellCoulombFeatures(core._bsp_data)
    plan = plan_core_sd(space.occupied_keys, space.virtual_keys, feature.identity)
    assert len(space.occupied_keys) == 19 and len(space.virtual_keys) == 414
    assert plan["independent_scalar_amplitudes"] == 13165103
    save("yb_full_plan.json", plan)
    print(
        json.dumps(
            dict(
                stage="complete space and plan",
                coordinates=plan["raw_amplitudes"],
                payload=plan["complete_payload_bound_bytes"],
                seconds=time.perf_counter() - began,
            )
        ),
        flush=True,
    )
    if plan["complete_payload_bound_bytes"] + 40 * plan["krylov_vector_bytes"] > cap:
        raise MemoryError("complete Yb core SD plus Krylov exceeds explicit 32 GiB cap")
    build(
        REFERENCE / "pCI",
        HERE / "yb_full_backend",
        plan["dimensions"],
        REFERENCE / "gfortran",
        runtime_directory=REFERENCE / "toolchain/usr/lib/x86_64-linux-gnu",
    )
    arrays = {f"P_{n}_{k}": v[0] for (n, k), v in space.orbitals.items()}
    arrays.update({f"Q_{n}_{k}": v[1] for (n, k), v in space.orbitals.items()})
    arrays.update(
        {
            f"workspace_{k}": v
            for k, v in core._bsp_data.items()
            if isinstance(v, (np.ndarray, int, float, bool))
        }
    )
    from atomx.potentials.nuclear import nuclear_profile

    arrays["nuclear_on_outer_gauss"] = np.vectorize(
        core._nuclear_potential, otypes=[float]
    )(core._bsp_data["gr"])
    arrays["nuclear_on_auxiliary_nodes"] = np.vectorize(
        core._nuclear_potential, otypes=[float]
    )(feature.radii)
    save("nuclear_model.json", nuclear_profile(70, "fermi"))
    np.savez_compressed(HERE / "yb_full_inputs.npz", **arrays)
    print(
        json.dumps(
            dict(
                stage="prepare full native operator and original diagrams",
                seconds=time.perf_counter() - began,
            )
        ),
        flush=True,
    )
    preparing = time.perf_counter()
    problem = CoreSD(
        space.orbitals,
        core._bsp_data,
        core.Z,
        core._bsp_data["ns"],
        occupied_keys=space.occupied_keys,
        virtual_keys=space.virtual_keys,
        nuclear_potential=core._nuclear_potential,
        backend_directory=HERE / "yb_full_backend",
        memory_bytes=cap,
    )
    prepare_seconds = time.perf_counter() - preparing
    print(
        json.dumps(
            dict(
                stage="prepared",
                prepare_seconds=prepare_seconds,
                seconds=time.perf_counter() - began,
                peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            )
        ),
        flush=True,
    )
    initial = problem.source / problem.denominators
    start = time.perf_counter()
    residual = problem.action(initial) - problem.source
    action_seconds = time.perf_counter() - start
    save(
        "yb_full_first_action.json",
        dict(
            prepare_seconds=prepare_seconds,
            action_seconds=action_seconds,
            initial_residual_max_Ha=float(np.max(abs(residual))),
            provenance=problem.provenance,
            reference_archive={
                k: v for k, v in reference_identity.items() if k != "source_sha256"
            },
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
    )
    print(
        json.dumps(
            dict(
                stage="first full action",
                action_seconds=action_seconds,
                initial_residual_max_Ha=float(np.max(abs(residual))),
            )
        ),
        flush=True,
    )
    del initial, residual
    result = problem.solve(restart=30, maxiter=100)
    np.savez_compressed(
        HERE / "yb_full_solution.npz",
        amplitudes=result.amplitudes,
        source=problem.source,
        denominators=problem.denominators,
    )
    outcome = dict(
        reference_energy_Ha=result.reference_energy_Ha,
        correlation_energy_Ha=result.correlation_energy_Ha,
        total_energy_Ha=result.total_energy_Ha,
        residual_max_Ha=result.residual_max_Ha,
        residual_l2_Ha=result.residual_l2_Ha,
        matvecs=result.matrix_vector_products,
        solve_seconds=result.wall_seconds,
        prepare_seconds=prepare_seconds,
        first_action_seconds=action_seconds,
        full_seconds=time.perf_counter() - began,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        provenance=dict(result.provenance),
        optical_spectrum="not calculated: valence effective Hamiltonian remains separate",
        convergence="raw equations only; radial/virtual/partial-wave and experimental gates not passed",
    )
    save("yb_full_result.json", outcome)
    print(
        json.dumps({k: v for k, v in outcome.items() if k != "provenance"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
