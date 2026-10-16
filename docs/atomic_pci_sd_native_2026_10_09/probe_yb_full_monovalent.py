"""Complete original Yb space: standalone SD maps and first feedback root.

Historical closed-core amplitudes are only a numerical guess. The current
public CoreSD operator must re-certify them before any standalone consumer.
Every internal orbital and direct rho channel remains in each declared column.
"""

import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from build_pci_monovalent_backend import build_monovalent  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.reduced_monovalent_sd import (  # noqa: E402
    ReducedMonovalentSD,
    plan_reduced_monovalent_sd,
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    faulthandler.enable()
    if args.all_roots and not args.davidson:
        raise ValueError(
            "all-root diagnostic requires explicitly validated private Davidson"
        )
    if args.davidson and not args.term31_blas:
        raise ValueError(
            "private Davidson large-space run requires explicit term31 control"
        )
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)
    started = time.perf_counter()

    def save(name, value):
        (out / name).write_text(json.dumps(plain(value), indent=2) + "\n")

    def observe(value):
        row = dict(value, elapsed_seconds=time.perf_counter() - started)
        with (out / "observations.jsonl").open("a") as log:
            log.write(json.dumps(plain(row)) + "\n")
        print(json.dumps(plain(row)), flush=True)

    parent = OWNER / "yb_full_run008"
    inputs, solution = parent / "yb_full_inputs.npz", parent / "yb_full_solution.npz"
    previous = json.loads((parent / "yb_full_result.json").read_text())
    old_plan = json.loads((parent / "yb_full_plan.json").read_text())
    occupied, virtual = (
        tuple(map(tuple, old_plan[k])) for k in ("occupied_keys", "virtual_keys")
    )
    keys = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
    cap = 32 * 1024**3
    options = dict(
        term21_threads=8,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
    )
    contract = dict(
        hypothesis="Current full-space standalone diagrams remain affine and retain the unchanged double-only fixed-energy map at bare ev; the reference-connected 6s coupled SD root passes actual full raw/energy gates",
        fixed="Z70/V68/Fermi/ns123/l<=3; original all68 core electrons and all414 radial virtual roots below5Ha, real DC and complete signed redundant direct rho",
        input_sha256=digest(inputs),
        seed_sha256=digest(solution),
        valence_keys=keys,
        root_keys=keys if args.all_roots else ((6, -1),),
        root_method="Private complete-image directed Davidson,16-vector subspace/MGS2/restart; actual public full raw/energy each8couplings and final affine"
        if args.davidson
        else "Current public ReducedMonovalentSD.solve; eight explicit .125..1 continuation steps, fixed closed-core denominators, no fitted shift, small-amplitude pruning or changed space",
        raw_residual_gate_Ha=1e-10,
        signed_map_gate_Ha=1e-11,
        replay_core_energy_gate_Ha=1e-11,
        historical_complete_core_action_gate_Ha=1e-11,
        source_transition="Only the four previously validated shared CSF weak-amplitude filters may differ from historical physical sources. Require exact audited text replacements, unchanged other physical source hashes, unchanged physical/input/denominator identity, complete stored signed action comparison and actual new CoreSD raw/energy certification; historical results are never restamped",
        numerical_inverse_gate_Ha=None if args.davidson else 1e-12,
        restart=20,
        krylov_vectors=40,
        max_outer_iterations=16,
        max_inverse_restarts=200,
        diagnostic_map_cap=1600,
        diagnostic_wall_cap_seconds=12000,
        cap_bytes=cap,
        core_workers=8,
        valence_term2_workers=8,
        inner_blas_threads=1,
        private_term31_blas=args.term31_blas,
        private_davidson=args.davidson,
        private_large_root_actions=160 if args.davidson else None,
        load=os.getloadavg(),
        stop="Failed source/physical identity, signed map, raw/energy or denominator gate, cap overrun, or declared diagnostic work/time exhaustion rejects the run; retain full arrays and failure, never reduce physics or relax tolerance",
        scope="Complete original finite radial/partial-wave window and numerical equations only; no radial convergence, independent large-H oracle, production normalized SD state/property, experimental, GPU or performance certificate",
    )
    save("contract.json", contract)
    paths = (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + sorted((ROOT / "src/atomx").rglob("*.f90"))
        + sorted((ROOT / "tools").glob("build_pci_*.py"))
        + [Path(__file__).resolve()]
    )
    if args.term31_blas:
        paths += [OWNER / "monovalent_term31_blas_control.py"]
    if args.davidson:
        paths += [
            OWNER / "monovalent_davidson_control.py",
            OWNER / "probe_monovalent_term31_blas.py",
        ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    save(
        "machine.json",
        dict(
            load=os.getloadavg(),
            cpu_count=os.cpu_count(),
            meminfo=Path("/proc/meminfo").read_text(),
            threading={
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        ),
    )
    try:
        with np.load(inputs, allow_pickle=False) as data:
            workspace = {
                k.removeprefix("workspace_"): data[k].copy()
                for k in data
                if k.startswith("workspace_")
            }
            workspace = {
                k: v.item() if v.ndim == 0 else v for k, v in workspace.items()
            }
            waves = {
                key: (
                    data[f"P_{key[0]}_{key[1]}"].copy(),
                    data[f"Q_{key[0]}_{key[1]}"].copy(),
                )
                for key in occupied + virtual
            }
            nuclear = make_nuclear_potential(70, "fermi")
            features = PolynomialCellCoulombFeatures(workspace)
            for grid, expected in (
                (workspace["gr"], data["nuclear_on_outer_gauss"]),
                (features.radii, data["nuclear_on_auxiliary_nodes"]),
            ):
                if (
                    np.max(abs(np.vectorize(nuclear, otypes=[float])(grid) - expected))
                    > 1e-11
                ):
                    raise ValueError("original Fermi nuclear potential differs")
        core_plan = plan_core_sd(occupied, virtual, features.identity, **options)
        plan = plan_reduced_monovalent_sd(
            core_plan, keys, valence_term2_blas=True, valence_term2_threads=8
        )
        if len(virtual) != 414 or core_plan["raw_amplitudes"] != 28218500:
            raise AssertionError("complete original Yb space differs")
        bounds = {
            key: plan["complete_payload_bound_bytes"] + 8 * (40 * count + 4 * 21**2)
            for key, count in plan["amplitudes"].items()
        }
        budget = max(
            max(bounds.values()),
            core_plan["complete_payload_bound_bytes"]
            + 40 * core_plan["krylov_vector_bytes"],
        )
        private_extra = 0
        if args.term31_blas:
            from monovalent_term31_blas_control import install, scratch_bytes

            private_extra = scratch_bytes(core_plan["dimensions"])
            budget += private_extra
        if args.davidson:
            from monovalent_davidson_control import solve as davidson_solve

            proof = OWNER / "monovalent_davidson002"
            proved = json.loads((proof / "result.json").read_text())
            archived = json.loads((proof / "source_sha256.json").read_text())
            if proved["status"] != "pass" or proved["independent_gates"] != 84:
                raise ValueError(
                    "actual full finite independent private control required"
                )
            for name in (
                "monovalent_davidson_control.py",
                "monovalent_term31_blas_control.py",
            ):
                relative = str((OWNER / name).relative_to(ROOT))
                if archived[relative] != digest(OWNER / name):
                    raise ValueError("private solver/kernel proof source differs")
            save(
                "finite_private_binding.json",
                dict(
                    proof=str(proof.relative_to(ROOT)),
                    result_sha256=digest(proof / "result.json"),
                    source_manifest_sha256=digest(proof / "source_sha256.json"),
                ),
            )
        # Diagnostic retained complete map arrays also fit within the root bound.
        if budget > cap:
            raise MemoryError(
                "complete current Yb native/Krylov payload exceeds original cap"
            )
        save(
            "resource.json",
            dict(
                core_plan=core_plan,
                standalone_plan=plan,
                solve_bound_bytes=bounds,
                maximum_bound_bytes=budget,
                private_extra_payload_bytes=private_extra,
            ),
        )
        observe(dict(stage="complete_space_preflight_pass", maximum_bound_bytes=budget))
        if args.plan_only:
            save(
                "result.json",
                dict(
                    status="preflight only; no backend loaded or action/root",
                    resources_pass=True,
                ),
            )
            return
        build_monovalent(
            REF / "pCI",
            out / "backend",
            core_plan["dimensions"],
            REF / "gfortran",
            nxv=plan["nxv"],
            nvh=plan["nvh"],
            runtime_directory=REF / "toolchain/usr/lib/x86_64-linux-gnu",
            blas_library=OWNER
            / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
            valence_term2_blas=True,
            valence_term2_threads=8,
            **options,
        )
        if args.term31_blas:
            install(out / "backend", memory_bytes=private_extra)
        begin = time.perf_counter()
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
            radial_representation="polynomial",
            **options,
        )
        observe(dict(stage="core_prepared", seconds=time.perf_counter() - begin))
        for field in (
            "hamiltonian",
            "binding",
            "occupied_keys",
            "virtual_keys",
            "frozen_keys",
            "feature_identity",
            "method",
            "equation",
            "amplitude_representation",
            "backend_pin",
        ):
            if plain(core.provenance[field]) != plain(previous["provenance"][field]):
                raise ValueError("original complete core identity differs: " + field)
        prior_sources = previous["provenance"]["physical_source_sha256"]
        current_sources = core.provenance["physical_source_sha256"]
        allowed_source = "csf_hamiltonian.py"
        if set(prior_sources) != set(current_sources) or any(
            prior_sources[name] != current_sources[name]
            for name in prior_sources
            if name != allowed_source
        ):
            raise ValueError(
                "historical physical source transition exceeds audited CSF change"
            )
        prior_csf = (
            OWNER / "yb_full_valence012/source/src/atomx/angular/csf_hamiltonian.py"
        )
        current_csf = ROOT / "src/atomx/angular/csf_hamiltonian.py"
        if (
            digest(prior_csf) != prior_sources[allowed_source]
            or digest(current_csf) != current_sources[allowed_source]
        ):
            raise ValueError("historical/current shared CSF source binding differs")
        prior_text = prior_csf.read_text()
        replacements = (
            ("if abs(amp) <= 1e-14:", "if amp == 0.0:"),
            ("if abs(amplitude) > 1e-14", "if amplitude != 0.0"),
            ("if abs(coefficient) > 1e-14", "if coefficient != 0.0"),
            ("if abs(value) > 1e-14", "if value != 0.0"),
        )
        for before, after in replacements:
            if prior_text.count(before) != 1:
                raise ValueError(
                    "exact historical weak-amplitude source replacement required"
                )
            prior_text = prior_text.replace(before, after, 1)
        if prior_text != current_csf.read_text():
            raise ValueError("shared CSF transition contains an unaudited change")
        save(
            "physical_source_transition.json",
            dict(
                previous=prior_sources,
                current=current_sources,
                exact_four_replacements=True,
                prior_source=str(prior_csf.relative_to(ROOT)),
                finite_evidence="native_bloch_transitions004/240 affected passing checks",
                complete_action_gate_status="pending",
            ),
        )
        with np.load(solution, allow_pickle=False) as data:
            source_error = float(np.max(abs(data["source"] - core.source)))
            same_diagonal = np.array_equal(data["denominators"], core.denominators)
            seed = data["amplitudes"].copy()
        if source_error > 1e-11 or not same_diagonal:
            raise AssertionError("original complete core source/denominators differ")
        original_core_action = core.action

        def observed_core_action(values):
            begin = time.perf_counter()
            observe(dict(stage="enter_complete_core_action", matvecs=core._matvecs))
            result = original_core_action(values)
            observe(
                dict(
                    stage="complete_core_action",
                    seconds=time.perf_counter() - begin,
                    matvecs=core._matvecs,
                )
            )
            return result

        core.action = observed_core_action
        reference_file = parent / "yb_full_reference_action.npz"
        with np.load(reference_file, allow_pickle=False) as data:
            reference_vector = data["amplitudes"].copy()
            previous_action = data["action"].copy()
        current_action = core.action(reference_vector)
        action_error = float(np.max(abs(current_action - previous_action)))
        np.savez(
            out / "historical_complete_core_action.npz",
            amplitudes=reference_vector,
            previous_action=previous_action,
            current_action=current_action,
        )
        save(
            "historical_complete_core_action.json",
            dict(
                previous_file=str(reference_file.relative_to(ROOT)),
                previous_sha256=digest(reference_file),
                maximum_error_Ha=action_error,
                gate_Ha=1e-11,
                current_physical_sources=current_sources,
                scope="complete stored signed vector/action regression; not an independent large-H oracle",
            ),
        )
        del reference_vector, previous_action, current_action
        if action_error > 1e-11:
            raise AssertionError(
                "current complete core action differs from historical stored signed action"
            )
        save(
            "physical_source_transition.json",
            dict(
                previous=prior_sources,
                current=current_sources,
                exact_four_replacements=True,
                prior_source=str(prior_csf.relative_to(ROOT)),
                complete_action_gate_status="pass",
                complete_action_error_Ha=action_error,
            ),
        )
        core_result = core.solve(initial_amplitudes=seed)
        del seed
        energy_error = abs(
            core_result.correlation_energy_Ha - previous["correlation_energy_Ha"]
        )
        if (
            max(
                energy_error,
                abs(core_result.reference_energy_Ha - previous["reference_energy_Ha"]),
            )
            > 1e-11
        ):
            raise AssertionError("same-physics core replay energy differs")
        save(
            "core_replay.json",
            dict(
                source_error_Ha=source_error,
                denominator_bitwise_equal=same_diagonal,
                correlation_energy_error_Ha=energy_error,
                residual_max_Ha=core_result.residual_max_Ha,
                matvecs=core_result.matrix_vector_products,
                provenance=core_result.provenance,
            ),
        )
        problem = ReducedMonovalentSD(core, core_result, valence_keys=keys)
        original_evaluate, original_prepare = problem._evaluate, problem._prepare

        def observed_prepare(key):
            begin = time.perf_counter()
            result = original_prepare(key)
            observe(
                dict(
                    stage="standalone_prepared",
                    key=key,
                    seconds=time.perf_counter() - begin,
                    amplitudes=len(result[2]),
                )
            )
            return result

        def observed_evaluate(vector, prepared, *, linear=False):
            if (
                problem._matvecs >= contract["diagnostic_map_cap"]
                or time.perf_counter() - started
                > contract["diagnostic_wall_cap_seconds"]
            ):
                np.savez(
                    out / "last_uncertified_iterate.npz",
                    amplitudes=vector,
                    diagonal_Ha=prepared[2],
                )
                raise RuntimeError(
                    "declared diagnostic map/wall cap reached; current vector is uncertified"
                )
            begin = time.perf_counter()
            result = original_evaluate(vector, prepared, linear=linear)
            observe(
                dict(
                    stage="complete_standalone_map",
                    map_call=problem._matvecs,
                    linear=linear,
                    seconds=time.perf_counter() - begin,
                    reference_Ha=result[0],
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    load_after=os.getloadavg(),
                )
            )
            return result

        problem._prepare, problem._evaluate = observed_prepare, observed_evaluate
        controls = []
        roots = []
        root_keys = keys if args.all_roots else ((6, -1),)

        def solve_root(key):
            begin = time.perf_counter()
            result = (
                davidson_solve(problem, key) if args.davidson else problem.solve(key)
            )
            prefix = f"root_{key[0]}_{key[1]}"
            np.savez(
                out / (prefix + ".npz"),
                amplitudes=result.amplitudes,
                denominators_Ha=result.denominators_Ha,
            )
            root = {
                field: getattr(result, field)
                for field in result.__dataclass_fields__
                if field not in ("amplitudes", "denominators_Ha")
            }
            root["solve_seconds"] = time.perf_counter() - begin
            save(prefix + ".json", root)
            row = dict(
                stage="complete_feedback_root_pass",
                key=key,
                correlation_energy_Ha=result.correlation_energy_Ha,
                attachment_energy_Ha=result.attachment_energy_Ha,
                raw_residual_Ha=result.residual_max_Ha,
                energy_residual_Ha=result.energy_residual_Ha,
                matvecs=result.matrix_vector_products,
                solve_seconds=root["solve_seconds"],
            )
            roots.append(row)
            save("roots.json", roots)
            observe(row)

        for key in keys:
            with core._lock:
                prepared = problem._prepare(key)
                count, ns = len(prepared[2]), len(prepared[1])
                zero = np.zeros(count)
                constant, source = problem._evaluate(zero, prepared)
                rng = np.random.default_rng(710 + key[0] * 10 + key[1])
                vector = rng.normal(0, 3e-5, count)
                component_error = None
                if args.term31_blas:
                    native = core._backend.atomx_monovalent_term31_control
                    pointer = np.ctypeslib.ndpointer(
                        dtype=np.float64, flags="F_CONTIGUOUS"
                    )
                    native.argtypes = [ctypes.c_int, pointer, pointer]
                    native.restype = None
                    s = np.zeros(core_plan["dimensions"]["NX"])
                    s[prepared[1]] = vector[:ns]
                    a, b = (np.zeros(prepared[0][2], order="F") for _ in range(2))
                    begin_component = time.perf_counter()
                    native(0, s, a)
                    original_seconds = time.perf_counter() - begin_component
                    begin_component = time.perf_counter()
                    native(1, s, b)
                    optimized_seconds = time.perf_counter() - begin_component
                    component_error = float(np.max(abs(a - b)))
                    np.savez(
                        out / f"term31_{key[0]}_{key[1]}.npz",
                        physical_singles=s,
                        original=a,
                        blas=b,
                    )
                    observe(
                        dict(
                            stage="complete_term31_component_control",
                            key=key,
                            maximum_error_Ha=component_error,
                            original_seconds=original_seconds,
                            blas_seconds=optimized_seconds,
                            timing_scope="single observations under live load; no alternating warm controlled speedup",
                        )
                    )
                    if component_error > 1e-11:
                        raise AssertionError(
                            "complete original/BLAS standalone term31 component gate failed"
                        )
                energy_linear, rhs_linear = problem._evaluate(
                    vector, prepared, linear=True
                )
                energy_full, rhs_full = problem._evaluate(vector, prepared)
                affine_error = max(
                    abs(energy_full - constant - energy_linear),
                    float(np.max(abs(rhs_full - source - rhs_linear))),
                )
                # Double-only fixed-energy equations are a regression anchor,
                # not an independent physical oracle for the new singles terms.
                double_vector = vector.copy()
                double_vector[:ns] = 0
                _, double_rhs = problem._evaluate(double_vector, prepared, linear=True)
                context = prepared[0]
                fixed_s, fixed_d = problem.valence._evaluate_value(
                    prepared[-1], prepared[2][ns:] * double_vector[ns:], context
                )
                fixed_error = max(
                    float(np.max(abs(double_rhs[ns:] - fixed_d))),
                    float(np.max(abs(double_rhs[:ns] - fixed_s[prepared[1]]))),
                )
                np.savez(
                    out / f"maps_{key[0]}_{key[1]}.npz",
                    source=source,
                    vector=vector,
                    vector_double_only=double_vector,
                    rhs_linear=rhs_linear,
                    rhs_affine=rhs_full,
                    rhs_double_only=double_rhs,
                    rhs_fixed_double=fixed_d,
                    rhs_fixed_single=fixed_s,
                    diagonal_Ha=prepared[2],
                    energy_maps_Ha=np.array([constant, energy_linear, energy_full]),
                )
                if max(affine_error, fixed_error) > 1e-11:
                    raise AssertionError(
                        "full signed affine/double-only regression gate failed"
                    )
                row = dict(
                    key=key,
                    amplitudes=count,
                    singles=ns,
                    affine_error_Ha=affine_error,
                    fixed_double_only_error_Ha=fixed_error,
                    term31_component_error_Ha=component_error,
                )
                controls.append(row)
                save("map_controls.json", controls)
                observe(dict(stage="complete_signed_map_controls_pass", **row))
            if key == (6, -1) and not args.maps_only and not args.all_roots:
                solve_root(key)
        if args.all_roots and not args.maps_only:
            for key in root_keys:
                solve_root(key)
        changed = [path for path, sha in hashes.items() if digest(ROOT / path) != sha]
        if changed:
            raise ValueError(
                "actual loaded source changed during diagnostic: " + str(changed)
            )
        save("source_postcheck.json", dict(status="pass", files=len(hashes)))
        save(
            "result.json",
            dict(
                status="complete five-column signed maps passed"
                + (
                    "; all requested complete feedback roots passed"
                    if not args.maps_only
                    else "; roots not requested"
                ),
                controls=controls,
                roots=roots,
                seconds=time.perf_counter() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                highest_level_reached=False,
            ),
        )
    except Exception as error:
        save(
            "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
                seconds=time.perf_counter() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--term31-blas", action="store_true")
    parser.add_argument("--davidson", action="store_true")
    parser.add_argument("--all-roots", action="store_true")
    main(parser.parse_args())
