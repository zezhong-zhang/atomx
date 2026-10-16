"""Complete 68-core/414-virtual Yb fixed-energy SD and directed screening.

The external model space contains declared 6s/6p/5d radial columns. Internal
core and virtual intermediates retain the full original run008 space. This
does not identify a physical CI metric or claim an experimental spectrum.
"""

import argparse
import faulthandler
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from build_pci_sigma2_backend import build_sigma2  # noqa: E402
from restart_storage import RestartStorage  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.sigma2_sd import Sigma2SD, plan_sigma2_sd  # noqa: E402
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd  # noqa: E402
from atomx.methods.valence_sd import (  # noqa: E402
    _source_identity as valence_source_identity,
)
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"))


def main(args):
    faulthandler.enable()
    if args.private_seed_handoff and not args.cuda_term2:
        raise ValueError("private numerical handoff requires explicit CUDA phases")
    if args.cuda_krylov and not args.private_seed_handoff:
        raise ValueError("resident GPU Krylov requires actual public handoff")
    private_variant = (
        args.private_fused_angular
        or args.private_term41_blas
        or args.private_terma4_blas
    )
    if private_variant and (not args.cuda_term2 or args.private_seed_handoff):
        raise ValueError(
            "explicit private CUDA variant requires public GMRES only; combined numerical handoff needs its own complete resource proof"
        )
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)
    if args.cuda_term2 and (
        not args.valence_term2_blas or args.valence_term2_threads != 8
    ):
        raise ValueError(
            "explicit complete eight-worker CPU companion required for CUDA"
        )

    def save(name, value):
        (out / name).write_text(json.dumps(plain(value), indent=2) + "\n")

    started = time.perf_counter()
    source = OWNER / "yb_full_run008"
    inputs = source / "yb_full_inputs.npz"
    solution = source / "yb_full_solution.npz"
    old = json.loads((source / "yb_full_result.json").read_text())
    p = json.loads((source / "yb_full_plan.json").read_text())
    occupied, virtual = (
        tuple(map(tuple, p[k])) for k in ("occupied_keys", "virtual_keys")
    )
    keys = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
    cap = 32 * 1024**3
    contract = dict(
        hypothesis="Full original Yb core solution remains a solution of the public exact optimized operator; complete fixed-energy valence/screening equations can be solved without dropping internal intermediates",
        fixed="Z70/V68 Fermi/ns123/l<=3 and all414 bound+positive-energy virtual radial roots below5Ha; all68core electrons, real native Dirac-Coulomb, original redundant signed rho normalization",
        external_columns=keys,
        external_scope="finite declared 6s/6p/5d P columns; every414virtual/core intermediate retained",
        energy_convention="lowest declared virtual Fock diagonal for each kappa, no fitted shift",
        input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        seed_file_sha256=hashlib.sha256(solution.read_bytes()).hexdigest(),
        raw_residual_gate_Ha=1e-10,
        source_gate_Ha=1e-11,
        core_energy_gate_Ha=1e-11,
        core_workers=8,
        blas_inner_threads=1,
        cap_bytes=cap,
        load=os.getloadavg(),
        stop="Any changed physical input, complete-space mismatch, failed full source/denominator/energy/raw gate or cap overrun rejects; no tolerance/space alteration",
        scope="fixed-energy directed operators and numerical residuals; no Hermitian CI substitution, radial convergence, precision or experimental claim",
        screening_requested=args.screening,
        valence_term2_blas=args.valence_term2_blas,
        valence_term2_threads=args.valence_term2_threads,
        private_cuda_term2_requested=args.cuda_term2,
        private_numerical_seed_handoff=args.private_seed_handoff,
        private_resident_cuda_krylov=args.cuda_krylov,
        private_fused_angular=args.private_fused_angular,
        private_term41_blas=args.private_term41_blas,
        private_terma4_blas=args.private_terma4_blas,
        device_pool_cap_bytes=(10 if args.cuda_krylov else 2) * 1024**3
        if args.cuda_term2
        else 0,
        preflight_only_requested=args.plan_only,
        seed_producer_policy="private complete MGS2 GCROT for new columns, then fresh actual public GMRES/source/root/response certification; never relabel private result or use it in Sigma2"
        if args.private_seed_handoff
        else "saved complete certified columns or source-bound iteration guesses only",
        partial_parent_policy="different-driver partial generations are retained without import; complete certified parent columns still revalidate"
        if args.private_seed_handoff
        else "exact current full source binding required",
        column_solver=dict(restart=30, maxiter=100, inner_atol_Ha=1e-14),
        resume_from=args.resume_from,
        checkpoint_policy="complete saved numerators are initial guesses only; fixed physical/input/energy/channel/denominator identity and fresh full signed/raw gates required",
        iteration_checkpoint_policy="each GMRES restart stores a complete uncertified numerical guess as an immutable generation with an atomic head; exact actual-source/physical/coordinate binding required",
        iteration_checkpoint_cap_bytes=8 * 1024**3,
    )
    resume = None
    if args.resume_from is not None:
        resume = OWNER / args.resume_from
        parent = json.loads((resume / "contract.json").read_text())
        if any(
            canonical(parent[field]) != canonical(contract[field])
            for field in (
                "fixed",
                "external_columns",
                "energy_convention",
                "input_sha256",
                "seed_file_sha256",
                "raw_residual_gate_Ha",
                "source_gate_Ha",
            )
        ):
            raise ValueError("checkpoint parent complete physical contract differs")
        contract["checkpoint_parent_contract_sha256"] = hashlib.sha256(
            (resume / "contract.json").read_bytes()
        ).hexdigest()
    save("contract.json", contract)
    paths = (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + sorted((ROOT / "src/atomx").rglob("*.f90"))
        + sorted((ROOT / "tools").glob("build_pci_*.py"))
        + [Path(__file__).resolve(), OWNER / "restart_storage.py"]
    )
    if args.cuda_term2:
        paths += [
            OWNER / name
            for name in (
                "cuda_valence_bridge.py",
                "cuda_full_term2_control.py",
                "cuda_density_control.py",
            )
        ]
    if args.private_seed_handoff:
        paths += [
            OWNER / name
            for name in (
                "private_seed_handoff.py",
                "private_recycling_control.py",
                "reorthogonal_gcrot.py",
                "reorthogonal_gcrot_source_binding.json",
                "scipy_private_gcrot_LICENSE.txt",
            )
        ]
    if args.cuda_krylov:
        paths.append(OWNER / "cuda_krylov_control.py")
    if args.private_fused_angular:
        paths += [
            OWNER / "cuda_batched_projection_control.py",
            OWNER / "cuda_fused_angular_control.py",
        ]
    if args.private_term41_blas:
        paths.append(OWNER / "valence_term41_blas_control.py")
    if args.private_terma4_blas:
        paths.append(OWNER / "valence_terma4_blas_control.py")
    hashes = {}
    for path in paths:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    save("source_sha256.json", hashes)
    with np.load(inputs, allow_pickle=False) as data:
        workspace = {
            k.removeprefix("workspace_"): data[k].copy()
            for k in data
            if k.startswith("workspace_")
        }
        workspace = {k: v.item() if v.ndim == 0 else v for k, v in workspace.items()}
        waves = {
            key: (
                data[f"P_{key[0]}_{key[1]}"].copy(),
                data[f"Q_{key[0]}_{key[1]}"].copy(),
            )
            for key in occupied + virtual
        }
        nuclear = make_nuclear_potential(70, "fermi")
        feature = PolynomialCellCoulombFeatures(workspace)
        for grid, expected in (
            (workspace["gr"], data["nuclear_on_outer_gauss"]),
            (feature.radii, data["nuclear_on_auxiliary_nodes"]),
        ):
            if (
                np.max(abs(np.vectorize(nuclear, otypes=[float])(grid) - expected))
                > 1e-11
            ):
                raise ValueError("original native Fermi nucleus differs")
    plan = plan_core_sd(
        occupied,
        virtual,
        feature.identity,
        term21_threads=8,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
    )
    if len(virtual) != 414 or plan["raw_amplitudes"] != 28218500:
        raise AssertionError("complete original Yb space differs")
    valence_plan = plan_valence_sd(
        plan,
        keys,
        valence_term2_blas=args.valence_term2_blas,
        valence_term2_threads=args.valence_term2_threads,
    )
    screening_plan = plan_sigma2_sd(
        plan,
        keys,
        valence_term2_blas=args.valence_term2_blas,
        valence_term2_threads=args.valence_term2_threads,
    )
    max_column = max(valence_plan["raw_amplitudes"].values())
    solve_vectors = 196 if args.private_seed_handoff else 56
    budget = max(
        plan["complete_payload_bound_bytes"] + 40 * plan["krylov_vector_bytes"],
        screening_plan["complete_payload_bound_bytes"] + 8 * max_column * solve_vectors,
    )
    if args.cuda_term2:
        budget += 16 * plan["dimensions"]["NXX"] ** 2 * valence_plan["nvh"]
    if args.private_seed_handoff:
        budget += 3 * plan["krylov_vector_bytes"]
    singles_scratch = (
        32 * plan["dimensions"]["NHF"] * plan["dimensions"]["NX"]
        if args.private_terma4_blas
        else 0
    )
    fused_host_plan = 128 * 1024**2 if args.private_fused_angular else 0
    budget += singles_scratch + fused_host_plan
    fused_resource = None
    if args.private_fused_angular:
        evidence = OWNER / "fused_full_fields001"
        measured = json.loads((evidence / "result.json").read_text())
        if measured["status"] != "pass" or [
            tuple(r["key"]) for r in measured["columns"]
        ] != list(keys):
            raise ValueError("complete declared fused field resource evidence required")
        bounds = []
        for key in keys:
            path = evidence / f"{key[0]}_{key[1]}" / "fused_gpu/fused_angular.json"
            row = json.loads(path.read_text())
            if (
                row["source_sha256"]
                != hashlib.sha256(
                    (OWNER / "cuda_fused_angular_control.py").read_bytes()
                ).hexdigest()
            ):
                raise ValueError("actual fused resource/source differs")
            bounds.append(row["full_device_payload_bound_bytes"])
        fused_resource = dict(
            max_complete_context_device_bound_bytes=max(bounds),
            original_diagram_pool_cap_bytes=2 * 1024**3,
            extra_host_plan_cap_bytes=fused_host_plan,
            evidence_result_sha256=hashlib.sha256(
                (evidence / "result.json").read_bytes()
            ).hexdigest(),
            scope="complete frozen same-dimension/state field bounds; each fresh native context independently rechecks exact reachable rows, disjointness and actual device cap; planning only is not a root/map gate",
        )
        if max(bounds) > 2 * 1024**3:
            raise MemoryError("complete fused context exceeds original diagram pool")
    gpu_plan = None
    if args.cuda_krylov:
        from private_seed_handoff import device_payload_bound

        operator_bound = device_payload_bound(
            feature, plan["dimensions"], valence_plan["nvh"]
        )
        resident_vectors = 2 * (30 + 10 + 1) + 6 * (10 + 1) + 24 + 16
        resident_bound = 8 * max_column * resident_vectors + operator_bound
        gpu_plan = dict(
            operator_device_bound_bytes=operator_bound,
            resident_vectors=resident_vectors,
            resident_numerical_pool_bound_bytes=resident_bound,
            resident_numerical_pool_cap_bytes=8 * 1024**3,
            original_diagram_pool_cap_bytes=2 * 1024**3,
            total_pool_cap_bytes=10 * 1024**3,
            runtime_free_guard="before each complete resident inverse allocation",
        )
        if resident_bound > 8 * 1024**3 or operator_bound > 2 * 1024**3:
            raise MemoryError("complete resident device payload exceeds declared caps")
    save(
        "resource.json",
        dict(
            core_plan=plan,
            valence_plan=valence_plan,
            screening_plan=screening_plan,
            peak_payload_bound_bytes=budget,
            solver_reserved_complete_vectors=solve_vectors,
            private_resident_gpu_plan=gpu_plan,
            private_fused_resource=fused_resource,
            extra_complete_singles_scratch_bytes=singles_scratch,
            extra_complete_fused_host_plan_bytes=fused_host_plan,
        ),
    )
    if budget > cap:
        raise MemoryError(
            "complete unchanged core/valence/screening and Krylov exceed cap"
        )
    if args.plan_only:
        save(
            "preflight_only.json",
            dict(
                status="complete resource plan passed",
                production_started=False,
                raw_or_signed_gate_executed=False,
                scope="complete fixed input/space/source/resource planning only; not a numerical solve",
            ),
        )
        return
    if private_variant:
        proof = OWNER / (
            "cuda_valence006" if args.private_fused_angular else "cuda_valence005"
        )
        proved = json.loads((proof / "result.json").read_text())
        proof_contract = json.loads((proof / "contract.json").read_text())
        proof_sources = json.loads((proof / "source_sha256.json").read_text())
        if proved[
            "status"
        ] != "all complete additive map/mode and declared solve gates passed" or [
            tuple(r["key"]) for r in proved["columns"]
        ] != list(keys):
            raise ValueError("all five complete actual-source map gates required")
        for name, enabled in (
            ("private_term41_blas", args.private_term41_blas),
            ("private_terma4_blas", args.private_terma4_blas),
            ("private_fused_angular", args.private_fused_angular),
        ):
            if proof_contract.get(name, False) != enabled:
                raise ValueError("complete compiled/GPU proof variant differs")
        for path in paths:
            relative = str(path.relative_to(ROOT))
            if (
                path != Path(__file__).resolve()
                and path != OWNER / "restart_storage.py"
                and hashes[relative] != proof_sources[relative]
            ):
                raise ValueError("complete map actual-source identity differs")
        save(
            "private_variant_binding.json",
            dict(
                proof=str(proof.relative_to(ROOT)),
                result_sha256=hashlib.sha256(
                    (proof / "result.json").read_bytes()
                ).hexdigest(),
                source_manifest_sha256=hashlib.sha256(
                    (proof / "source_sha256.json").read_bytes()
                ).hexdigest(),
                role="private complete-map coverage only; fresh public root/raw/response and screening still required; no performance/public adoption",
            ),
        )
    build_sigma2(
        REF / "pCI",
        out / "backend",
        plan["dimensions"],
        REF / "gfortran",
        sizes=dict(screening_plan["sizes"]),
        runtime_directory=REF / "toolchain/usr/lib/x86_64-linux-gnu",
        blas_library=OWNER
        / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
        term21_threads=8,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
        valence_term2_blas=args.valence_term2_blas,
        valence_term2_threads=args.valence_term2_threads,
    )
    if args.cuda_term2:
        from cuda_valence_bridge import install

        install(out / "backend")
    if args.private_term41_blas:
        from valence_term41_blas_control import install as install_term41

        install_term41(out / "backend")
    if args.private_terma4_blas:
        from valence_terma4_blas_control import install as install_terma4

        install_terma4(out / "backend")
    before = time.perf_counter()
    core = CoreSD(
        waves,
        workspace,
        70,
        workspace["ns"],
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=nuclear,
        backend_directory=out / "backend",
        memory_bytes=cap,
        term21_threads=8,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
        radial_representation="polynomial",
    )
    save(
        "prepare.json",
        dict(seconds=time.perf_counter() - before, provenance=core.provenance),
    )
    print(
        json.dumps(
            dict(
                stage="complete core/valence/screening prepared",
                seconds=time.perf_counter() - before,
            )
        ),
        flush=True,
    )
    physical_keys = (
        "hamiltonian",
        "binding",
        "physical_source_sha256",
        "occupied_keys",
        "virtual_keys",
        "frozen_keys",
        "feature_identity",
        "method",
        "equation",
        "amplitude_representation",
        "backend_pin",
    )
    mismatched = [
        k
        for k in physical_keys
        if canonical(old["provenance"][k]) != canonical(core.provenance[k])
    ]
    if mismatched:
        save("physical_identity_failure.json", mismatched)
        raise ValueError("original complete physical/equation identity differs")
    with np.load(solution, allow_pickle=False) as data:
        source_error = float(np.max(abs(data["source"] - core.source)))
        denominator_equal = np.array_equal(data["denominators"], core.denominators)
        seed = data["amplitudes"].copy()
    if source_error > 1e-11 or not denominator_equal:
        raise AssertionError("original complete source/denominators differ")

    # Observe complete replay gates without another action or altered arithmetic.
    original_core_action = core.action

    def observed_core_action(amplitudes):
        begin = time.perf_counter()
        load_before = os.getloadavg()
        print(
            json.dumps(dict(stage="enter_complete_core_action", matvecs=core._matvecs)),
            flush=True,
        )
        value = original_core_action(amplitudes)
        entry = dict(
            stage="complete_core_action",
            seconds=time.perf_counter() - begin,
            matvecs=core._matvecs,
            load_before=load_before,
            load_after=os.getloadavg(),
            numerical_role="original full source-bound action; observation only",
        )
        with (out / "core_map_observations.jsonl").open("a") as log:
            log.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        return value

    core.action = observed_core_action
    result = core.solve(initial_amplitudes=seed)
    energy_error = abs(result.correlation_energy_Ha - old["correlation_energy_Ha"])
    if (
        energy_error > 1e-11
        or abs(result.reference_energy_Ha - old["reference_energy_Ha"]) > 1e-11
    ):
        raise AssertionError("same-physics core replay energy differs")
    save(
        "core_replay.json",
        dict(
            source_error_Ha=source_error,
            denominator_bitwise_equal=denominator_equal,
            correlation_energy_error_Ha=energy_error,
            residual_max_Ha=result.residual_max_Ha,
            matvecs=result.matrix_vector_products,
            seconds=result.wall_seconds,
            provenance=result.provenance,
            parent_result=old,
            parent_solution_sha256=contract["seed_file_sha256"],
        ),
    )
    del seed
    valence = ValenceSD(core, result, valence_keys=keys)
    cuda_control = None
    if args.cuda_term2:
        from cuda_valence_bridge import BoundCUDAValenceControl

        if args.private_fused_angular:
            from cuda_fused_angular_control import BoundFusedCUDAControl

            cuda_control = BoundFusedCUDAControl(
                valence, out / "cuda", projection_policy="reachable"
            )
        else:
            cuda_control = BoundCUDAValenceControl(valence, out / "cuda")
        cuda_control.select("cuda")
        save(
            "private_cuda_scope.json",
            dict(
                manifest=core._manifest["private_cuda_term2"],
                device_cap_bytes=cuda_control.cap,
                numerical_role="same complete additive native FP64 term2; fresh source/raw/signed/response gates required",
                timing_role="production diagnostic under recorded load; no controlled gain claim",
                public_cuda_adopted=False,
            ),
        )
    handoff = None
    if args.private_seed_handoff:
        from private_seed_handoff import PrivateSeedHandoff

        handoff = PrivateSeedHandoff(
            valence,
            result,
            cuda_control,
            out / "private_seed",
            cuda_krylov=args.cuda_krylov,
        )
    # Observe the actual unchanged maps rather than infer progress from energy
    # stability. Complete solved vectors are retained after each certified column.
    original_evaluate = valence._evaluate
    original_value = valence._evaluate_value
    original_initialize = valence._initialize
    map_calls = 0
    current_column = None

    def observed_initialize(key, energy):
        begin = time.perf_counter()
        print(
            json.dumps(dict(stage="enter_complete_valence_initialization", key=key)),
            flush=True,
        )
        context = original_initialize(key, energy)
        entry = dict(
            stage="complete_valence_initialization",
            key=key,
            energy_Ha=energy,
            seconds=time.perf_counter() - begin,
            load_after=os.getloadavg(),
            numerical_role="original full COMMON/coordinates/channel/denominator initialization; observation only",
        )
        with (out / "initialization_observations.jsonl").open("a") as log:
            log.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        return context

    def record_map(energy, begin, *, linear, response):
        nonlocal map_calls
        map_calls += 1
        entry = dict(
            stage="complete_native_valence_map",
            column=current_column,
            map_call=map_calls,
            linear=linear,
            response_requested=response,
            energy_Ha=energy,
            seconds=time.perf_counter() - begin,
            matvecs=valence._matvecs,
            load_after=os.getloadavg(),
        )
        with (out / "map_observations.jsonl").open("a") as log:
            log.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def observed_evaluate(energy, vector, prime, context, *, linear=False):
        begin = time.perf_counter()
        mapped = original_evaluate(energy, vector, prime, context, linear=linear)
        record_map(energy, begin, linear=linear, response=True)
        return mapped

    def observed_value(energy, vector, context):
        begin = time.perf_counter()
        mapped = original_value(energy, vector, context)
        record_map(energy, begin, linear=True, response=False)
        return mapped

    valence._evaluate = observed_evaluate
    valence._evaluate_value = observed_value
    valence._initialize = observed_initialize
    rows = core.plan["groups"]
    columns = []
    for key in keys:
        current_column = key
        partial = next(i for i, g in enumerate(rows) if key in g)
        anchor = next(k for k in rows[partial] if k in virtual)
        label_index = core.integrals.labels.index((*anchor, -2 * abs(anchor[1]) + 1))
        e = float(core.integrals.fock_Ha[label_index, label_index].real)
        print(
            json.dumps(dict(stage="enter_full_valence_column", key=key, energy_Ha=e)),
            flush=True,
        )
        seeds, parent_arrays = {}, None
        label = f"{key[0]}_{key[1]}"
        restart_binding = dict(
            input_sha256=contract["input_sha256"],
            seed_file_sha256=contract["seed_file_sha256"],
            actual_source_manifest_sha256=hashlib.sha256(
                (out / "source_sha256.json").read_bytes()
            ).hexdigest(),
            physical={k: core.provenance[k] for k in physical_keys},
        )
        restart_store = RestartStorage(
            out / f"iterations_{label}",
            restart_binding,
            cap_bytes=contract["iteration_checkpoint_cap_bytes"],
        )
        if resume is not None and (resume / f"valence_{label}.json").exists():
            parent_record = json.loads((resume / f"valence_{label}.json").read_text())
            parent_provenance = parent_record["provenance"]
            if tuple(parent_record["column"]) != key or parent_record["energy_Ha"] != e:
                raise ValueError("saved column/energy identity differs")
            if any(
                canonical(parent_provenance[field]) != canonical(core.provenance[field])
                for field in physical_keys
                if field != "method"
            ):
                raise ValueError(
                    "saved complete core/physical/equation identity differs"
                )
            with core._lock:
                context = valence._initialize(key, e)
            if canonical(parent_record["channels"]) != canonical(context[1]):
                raise ValueError("saved complete signed channel order differs")
            group = rows[context[0]]
            expected_rows = tuple(k for k in group if k in virtual)
            if canonical(parent_record["row_keys"]) != canonical(expected_rows):
                raise ValueError("saved complete Sigma1 row identity differs")
            with np.load(resume / f"valence_{label}.npz", allow_pickle=False) as data:
                parent_arrays = {name: data[name].copy() for name in data.files}
            if not np.array_equal(parent_arrays["denominators"], context[4]):
                raise ValueError("saved complete denominator identity differs")
            seeds = dict(
                initial_numerators=parent_arrays["numerators"],
                initial_numerator_derivative=parent_arrays["numerator_derivative"],
            )
            del context
        elif (
            resume is not None
            and (resume / f"iterations_{label}").exists()
            and not args.private_seed_handoff
        ):
            with core._lock:
                context = valence._initialize(key, e)
            identity = dict(
                key=key,
                energy_Ha=e,
                core_identity_sha256=valence._core_identity,
                core_amplitudes_sha256=valence._core_amplitudes_sha256,
                valence_source_sha256=tuple(valence_source_identity.items()),
                packed_coordinates_sha256=tuple(
                    hashlib.sha256(a.tobytes()).hexdigest() for a in context[3]
                ),
                denominators_sha256=hashlib.sha256(context[4].tobytes()).hexdigest(),
            )
            previous = RestartStorage(
                resume / f"iterations_{label}",
                restart_binding,
                cap_bytes=contract["iteration_checkpoint_cap_bytes"],
            )
            count = valence.plan["raw_amplitudes"][key]
            for stage, argument in (
                ("root", "initial_numerators"),
                ("response", "initial_numerator_derivative"),
            ):
                guess = previous.load(stage, identity, count)
                if guess is not None:
                    seeds[argument] = guess
            save(
                f"iteration_replay_{label}.json",
                dict(
                    parent=args.resume_from,
                    stages_loaded=tuple(seeds),
                    identity=identity,
                    certified=False,
                    role="complete initial guesses only; fresh source/affine/raw/response gates required",
                ),
            )
            del context
        candidate = None
        if handoff is not None and not seeds:
            candidate_store = RestartStorage(
                out / f"candidate_iterations_{label}",
                restart_binding,
                cap_bytes=contract["iteration_checkpoint_cap_bytes"],
            )
            print(
                json.dumps(
                    dict(
                        stage="enter_private_complete_seed_producer",
                        key=key,
                        energy_Ha=e,
                    )
                ),
                flush=True,
            )
            candidate = handoff.guess(key, e, iteration_callback=candidate_store)
            seeds = dict(
                initial_numerators=candidate.numerators,
                initial_numerator_derivative=candidate.numerator_derivative,
            )
            print(
                json.dumps(
                    dict(
                        stage="private_complete_seed_ready",
                        key=key,
                        raw_root_Ha=candidate.residual_max_Ha,
                        raw_prime_Ha=candidate.derivative_residual_max_Ha,
                        matvecs=candidate.matrix_vector_products,
                        seconds=candidate.wall_seconds,
                        public_certified=False,
                    )
                ),
                flush=True,
            )
        column = valence.solve(
            key,
            e,
            solve_response=True,
            iteration_callback=restart_store,
            **seeds,
        )
        if candidate is not None:
            errors = {
                name: float(
                    np.max(abs(getattr(candidate, name) - getattr(column, name)))
                )
                for name in (
                    "numerators",
                    "numerator_derivative",
                    "sigma1_Ha",
                    "sigma1_derivative",
                )
            }
            save(
                f"private_handoff_{label}.json",
                dict(
                    full_signed_errors_Ha=errors,
                    public_root_Ha=column.residual_max_Ha,
                    public_prime_Ha=column.derivative_residual_max_Ha,
                    public_source_provenance=column.provenance,
                    role="new public result from actual full solver; private arrays were numerical guesses only",
                ),
            )
            if max(errors.values()) > 1e-11:
                raise AssertionError(
                    "full private/public signed numerical handoff fails"
                )
            del candidate
        if parent_arrays is not None:
            comparison = dict(
                parent_json_sha256=hashlib.sha256(
                    (resume / f"valence_{label}.json").read_bytes()
                ).hexdigest(),
                parent_arrays_sha256=hashlib.sha256(
                    (resume / f"valence_{label}.npz").read_bytes()
                ).hexdigest(),
                parent_source_sha256=hashlib.sha256(
                    (resume / "source_sha256.json").read_bytes()
                ).hexdigest(),
                full_signed_errors={
                    name: float(np.max(abs(parent_arrays[name] - current)))
                    for name, current in (
                        ("numerators", column.numerators),
                        ("numerator_derivative", column.numerator_derivative),
                        ("sigma1", column.sigma1_Ha),
                        ("sigma1_derivative", column.sigma1_derivative),
                    )
                },
                fresh_raw_root_Ha=column.residual_max_Ha,
                fresh_raw_response_Ha=column.derivative_residual_max_Ha,
                scope="source changed; same complete physics is re-certified, saved arrays are never trusted as a current solved result",
            )
            save(f"checkpoint_{label}.json", comparison)
            if max(comparison["full_signed_errors"].values()) > 1e-11:
                raise AssertionError("fresh complete checkpoint signed replay differs")
        columns.append(column)
        np.savez_compressed(
            out / f"valence_{label}.npz",
            numerators=column.numerators,
            denominators=column.denominators_Ha,
            sigma1=column.sigma1_Ha,
            numerator_derivative=column.numerator_derivative,
            sigma1_derivative=column.sigma1_derivative,
        )
        record = dict(
            column=key,
            energy_Ha=e,
            row_keys=column.row_keys,
            channels=column.channels,
            raw_residual_Ha=column.residual_max_Ha,
            response_residual_Ha=column.derivative_residual_max_Ha,
            matvecs=column.matrix_vector_products,
            seconds=column.wall_seconds,
            provenance=column.provenance,
        )
        save(f"valence_{label}.json", record)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("provenance", "channels", "row_keys")
                }
            ),
            flush=True,
        )
    if args.screening:
        screened = Sigma2SD(valence).evaluate(columns, joint_response=True)
        np.savez_compressed(
            out / "sigma2.npz",
            radial_Ha=screened.radial_Ha,
            joint_derivative=screened.joint_derivative,
            energy_parameters_Ha=screened.energy_parameters_Ha,
        )
        save(
            "sigma2.json",
            dict(
                provenance=dict(screened.provenance),
                channels=screened.channels,
                virtual_keys=screened.virtual_keys,
                valence_keys=screened.valence_keys,
                groups=screened.groups,
            ),
        )
    save(
        "result.json",
        dict(
            status="full fixed-energy columns passed, directed screening computed"
            if args.screening
            else "full fixed-energy columns passed; screening not requested",
            seconds=time.perf_counter() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            space_converged=False,
            physical_metric_certified=False,
            experiment_validated=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--screening", action="store_true")
    parser.add_argument("--valence-term2-blas", action="store_true")
    parser.add_argument("--valence-term2-threads", type=int, default=1)
    parser.add_argument("--cuda-term2", action="store_true")
    parser.add_argument("--private-seed-handoff", action="store_true")
    parser.add_argument("--cuda-krylov", action="store_true")
    parser.add_argument("--private-fused-angular", action="store_true")
    parser.add_argument("--private-term41-blas", action="store_true")
    parser.add_argument("--private-terma4-blas", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resume-from",
        help="owning run with complete certified column arrays; always fresh raw/signed replay",
    )
    args = parser.parse_args()
    if (
        not args.run_name.startswith("yb_full_valence")
        or not args.run_name[15:].isdigit()
    ):
        raise ValueError("fresh explicit yb_full_valenceNNN name required")
    try:
        main(args)
    except BaseException as error:
        out = OWNER / args.run_name
        if out.exists() and not (out / "failure.json").exists():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
