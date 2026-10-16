"""Public full noncanonical valence API against independent magnetic equations."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from build_pci_valence_backend import build_valence  # noqa: E402
from finite_direct_coulomb import direct_coulomb  # noqa: E402
from finite_valence_direct_solver import magnetic_solver  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402
from general_rho_control import core_numerators, expand_valence  # noqa: E402
from valence_allorder_oracle import action  # noqa: E402
from valence_energy_response_oracle import energy_derivative  # noqa: E402
from valence_fock_oracle import magnetic_fock  # noqa: E402

from atomx.methods.valence_sd import ValenceSD  # noqa: E402


def main(
    run_name,
    core_virtual_rotation,
    compare_case,
    valence_term2_blas=False,
    valence_term2_threads=1,
    cuda_term2=False,
    recycled_control=False,
    certify_private_seed=False,
    cuda_krylov=False,
    profile_linear_map=False,
    projection_policy=None,
    term41_blas=False,
    terma4_blas=False,
    fused_angular=False,
):
    if fused_angular and projection_policy != "reachable":
        raise ValueError("explicit exactly reachable fused CUDA control required")
    if projection_policy is not None and (
        projection_policy not in ("all", "reachable")
        or not cuda_term2
        or certify_private_seed
    ):
        raise ValueError(
            "batched projection control requires explicit CUDA public-GMRES validation without a mixed private producer"
        )
    if cuda_krylov and not certify_private_seed:
        raise ValueError("resident GPU Krylov requires fresh public certification")
    if certify_private_seed and not (cuda_term2 and recycled_control):
        raise ValueError("explicit private recycled CUDA producer required for handoff")
    if cuda_term2 and (not valence_term2_blas or valence_term2_threads != 8):
        raise ValueError("private CUDA requires an explicit eight-worker CPU companion")
    if not run_name.startswith("public_valence") or not run_name[14:].isdigit():
        raise ValueError("explicit fresh public_valenceNNN run required")
    if compare_case is not None and core_virtual_rotation:
        raise ValueError(
            "changed occupied projector is not a bitwise old-case comparison"
        )
    here = OWNER / run_name
    here.mkdir(exist_ok=False)
    archive = here / "source"
    names = [
        Path(__file__).resolve(),
        OWNER / "finite_direct_coulomb.py",
        OWNER / "finite_valence_direct_solver.py",
        OWNER / "finite_valence_fixture.py",
        OWNER / "general_rho_control.py",
        OWNER / "valence_allorder_oracle.py",
        OWNER / "valence_energy_response_oracle.py",
        OWNER / "valence_fock_oracle.py",
        ROOT / "src/atomx/methods/valence_sd.py",
        ROOT / "src/atomx/methods/_pci_valence.f90",
        ROOT / "src/atomx/methods/_pci_valence_term2_streamed.f90",
        ROOT / "src/atomx/methods/core_sd.py",
        ROOT / "tools/build_pci_sd_backend.py",
        ROOT / "tools/build_pci_valence_backend.py",
    ]
    if cuda_term2:
        names += [
            OWNER / name
            for name in (
                "cuda_valence_bridge.py",
                "cuda_full_term2_control.py",
                "cuda_density_control.py",
            )
        ]
    if recycled_control:
        import scipy.sparse.linalg._isolve._gcrotmk as gcrot_source

        names.append(OWNER / "private_recycling_control.py")
        names.extend(
            OWNER / name
            for name in (
                "reorthogonal_gcrot.py",
                "reorthogonal_gcrot_source_binding.json",
                "scipy_private_gcrot_LICENSE.txt",
            )
        )
        dependency = here / "dependency_source"
        dependency.mkdir()
        shutil.copy2(gcrot_source.__file__, dependency / "_gcrotmk.py")
    if certify_private_seed:
        names.append(OWNER / "private_seed_handoff.py")
    if cuda_krylov:
        names.append(OWNER / "cuda_krylov_control.py")
    if profile_linear_map:
        names.append(OWNER / "valence_stage_profile.py")
    if projection_policy is not None:
        names.append(OWNER / "cuda_batched_projection_control.py")
    if term41_blas:
        names.append(OWNER / "valence_term41_blas_control.py")
    if terma4_blas:
        names.append(OWNER / "valence_terma4_blas_control.py")
    if fused_angular:
        names.append(OWNER / "cuda_fused_angular_control.py")
    identity = {}
    if recycled_control:
        identity["dependency_source/_gcrotmk.py"] = hashlib.sha256(
            (dependency / "_gcrotmk.py").read_bytes()
        ).hexdigest()
    for path in names:
        relative = path.relative_to(ROOT)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        identity[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (here / "source_sha256.json").write_text(json.dumps(identity, indent=2) + "\n")
    (here / "contract.json").write_text(
        json.dumps(
            dict(
                hypothesis="Public complete fixed-energy noncanonical SD reproduces independent long-double magnetic roots/actions/implicit responses with matched native DC integrals. Optional same-input comparison requires bitwise equality.",
                fixed="C2+ all1s2/2s2 core, six declared s/p radial virtuals and all16magnetic states, Z6/ns40/ks7; complete native physicalFock, .37rad rotation of each two-virtual partial-wave pair; no synthetic operator or fitted energy",
                energy="each actual Fock diagonal eps_v+.075, core numerators fixed for derivative",
                core_virtual_rotation_rad=core_virtual_rotation,
                occupied_projector="rotated 2s/3s before full native integral construction; explicit changed reference"
                if core_virtual_rotation
                else "original DHF occupied projector",
                bitwise_comparison_case=compare_case,
                valence_term2_blas=valence_term2_blas,
                valence_term2_threads=valence_term2_threads,
                private_cuda_term2=cuda_term2,
                private_recycled_control=recycled_control,
                fresh_public_seed_certification=certify_private_seed,
                private_resident_cuda_krylov=cuda_krylov,
                complete_homogeneous_stage_profile=profile_linear_map,
                private_projection_policy=projection_policy,
                private_term41_blas=term41_blas,
                private_terma4_blas=terma4_blas,
                private_fused_angular=fused_angular,
                resident_device_policy="owned8GiB Krylov pool plus original2GiB diagram pool; complete native actions/FP64/full gates, explicit total10GiB"
                if cuda_krylov
                else "original diagram2GiB pool only when CUDA selected",
                inverse="private GCROT(m=30,k=10), two complete MGS passes and actual C=A U refresh before each inverse, original100 outer cycles/three refinement budgets"
                if recycled_control
                else "certified public GMRES",
                numerical_scope="private follow-up to012/023; same complete signed and native raw gates; no public solver adoption or controlled speedup claim"
                if recycled_control
                else "current public numerical path",
                gates_Ha=dict(
                    action=1e-11,
                    signed_root=1e-11,
                    signed_prime=1e-11,
                    sigma1_prime=1e-11,
                    actual_core_and_valence_residual=1e-10,
                ),
                resource="512MiB complete core/valence/Krylov preflight plus bounded64MiB finite reference; two heap core workers and explicitly declared heap term2 workers when streamed, otherwise original one worker",
                load=os.getloadavg(),
                stop="failure retains full arrays/source; no gate or space reduction",
                scope="private complete CUDA candidate against independent magnetic equations; public CUDA not adopted"
                if cuda_term2
                else "public equation and physical identity proof in declared finite space; not Yb/experimental convergence",
            ),
            indent=2,
        )
        + "\n"
    )

    def builder(
        source, directory, dimensions, compiler, *, nxv, nvh, runtime, **core_options
    ):
        manifest = build_valence(
            source,
            directory,
            dimensions,
            compiler,
            nxv=nxv,
            nvh=nvh,
            runtime_directory=runtime,
            valence_term2_blas=valence_term2_blas,
            valence_term2_threads=valence_term2_threads,
            blas_library=OWNER
            / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0"
            if valence_term2_blas
            else None,
            **core_options,
        )
        if cuda_term2:
            from cuda_valence_bridge import install

            manifest = install(directory)
        if term41_blas:
            from valence_term41_blas_control import install as install_term41

            manifest = install_term41(directory)
        if terma4_blas:
            from valence_terma4_blas_control import install as install_terma4

            # This complete finite fixture's four new static buffers are
            # counted before the shared library can be loaded.
            extra = 32 * dimensions["NHF"] * dimensions["NX"]
            if extra + 64 * 1024**2 > 512 * 1024**2:
                raise MemoryError("complete finite singles scratch exceeds cap")
            manifest = install_terma4(directory)
        return manifest

    core, orbitals, plan, rows, _, _, problem, fock = prepare(
        here,
        REF,
        backend_builder=builder,
        fock_probe=False,
        virtual_rotation=0.37,
        core_virtual_rotation=core_virtual_rotation,
        core_options=dict(
            term21_threads=2,
            term21_mode="streamed",
            core_singles_blas=True,
            core_single_to_doubles_blas=True,
        )
        if valence_term2_blas
        else None,
    )
    core_result = problem.solve(residual_tolerance_Ha=1e-12)
    solver_class = ValenceSD
    if recycled_control and not certify_private_seed:
        from private_recycling_control import RecycledValenceSD

        solver_class = RecycledValenceSD
    valence = solver_class(problem, core_result, valence_keys=plan["virtual_keys"])
    cuda_control = None
    if cuda_term2:
        from cuda_valence_bridge import BoundCUDAValenceControl

        if projection_policy is None:
            cuda_control = BoundCUDAValenceControl(valence, here / "cuda")
        else:
            from cuda_batched_projection_control import BoundBatchedCUDAControl

            projection_class = BoundBatchedCUDAControl
            if fused_angular:
                from cuda_fused_angular_control import BoundFusedCUDAControl

                projection_class = BoundFusedCUDAControl
            cuda_control = projection_class(
                valence, here / "cuda", projection_policy=projection_policy
            )
        cuda_control.select("cuda")
    handoff = None
    if certify_private_seed:
        from private_seed_handoff import PrivateSeedHandoff

        handoff = PrivateSeedHandoff(
            valence,
            core_result,
            cuda_control,
            here / "private_seed",
            cuda_krylov=cuda_krylov,
        )
    if (
        valence.plan["complete_payload_bound_bytes"]
        + 64 * 1024**2
        + (
            32 * problem.plan["dimensions"]["NHF"] * problem.plan["dimensions"]["NX"]
            if terma4_blas
            else 0
        )
        > problem.memory_bytes
    ):
        raise MemoryError(
            "public core/valence plus complete finite reference exceeds cap"
        )
    core_s, core_d = core_numerators(problem, plan, core_result.amplitudes)
    direct, direct_error = direct_coulomb(
        problem, orbitals, core._bsp_data, radial_representation="bound_features"
    )
    direct, eps, core_s, core_d, fock = (
        np.asarray(a, dtype=np.longdouble)
        for a in (direct, np.diag(fock), core_s, core_d, fock)
    )
    labels, nc = problem.integrals.labels, problem.integrals.noccupied
    physical_fov = float(np.max(abs(fock[:nc, nc:])))
    (here / "reference_change.json").write_text(
        json.dumps(
            dict(
                core_virtual_rotation_rad=core_virtual_rotation,
                actual_fov_max_Ha=physical_fov,
                physical_identity=dict(problem.integrals.identity),
                meaning="changed occupied reference projector; no basis-invariance or orbital-optimization claim",
            ),
            indent=2,
        )
        + "\n"
    )
    if core_virtual_rotation and physical_fov < 1e-3:
        raise AssertionError(
            "predeclared strong native occupied-virtual probe is ineffective"
        )
    np.savez_compressed(
        here / "inputs.npz",
        direct=direct,
        eps=eps,
        core_s=core_s,
        core_d=core_d,
        fock=fock,
        labels=np.asarray(labels),
        core_amplitudes=core_result.amplitudes,
    )
    rng = np.random.default_rng(207451)
    records = []
    for key in plan["virtual_keys"]:
        count = valence.plan["raw_amplitudes"][key]
        energy = float(eps[labels.index((*key, -2 * abs(key[1]) + 1))] + 0.075)
        numerical_options = {"inverse_method": "gcrotmk"} if recycled_control else {}
        candidate = None
        if handoff is not None:
            candidate = handoff.guess(key, energy)
            numerical_options = dict(
                initial_numerators=candidate.numerators,
                initial_numerator_derivative=candidate.numerator_derivative,
            )
        result = valence.solve(
            key,
            energy,
            residual_tolerance_Ha=1e-11,
            **numerical_options,
        )
        if candidate is not None:
            errors = {
                name: float(
                    np.max(abs(getattr(result, name) - getattr(candidate, name)))
                )
                for name in (
                    "numerators",
                    "numerator_derivative",
                    "sigma1_Ha",
                    "sigma1_derivative",
                )
            }
            if max(errors.values()) > 1e-11:
                raise AssertionError(
                    "private guess/fresh public full signed handoff failed"
                )
            (here / f"handoff_{key[0]}_{key[1]}.json").write_text(
                json.dumps(
                    dict(
                        full_signed_errors_Ha=errors,
                        public_matvecs=result.matrix_vector_products,
                        public_source_provenance=dict(result.provenance),
                    ),
                    indent=2,
                )
                + "\n"
            )
        random = rng.normal(size=count) * 0.001
        if profile_linear_map:
            from valence_stage_profile import profile

            profile(valence, key, energy, random, here / f"profile_{key[0]}_{key[1]}")
        native_action = valence.evaluate(key, energy, random)
        shape = (plan["dimensions"]["NXX"],) * 2 + (valence.plan["nvh"],)
        coordinates = tuple(
            (m, n, c)
            for c, (_, im, jn, _) in enumerate(rows[key])
            for n, nk in enumerate(plan["groups"][jn])
            for m, mk in enumerate(plan["groups"][im])
            if mk in plan["virtual_keys"] and nk in plan["virtual_keys"]
        )
        index = tuple(np.asarray(coordinates).T)

        def unpack(vector, index=index, shape=shape):
            full = np.zeros(shape)
            full[index] = vector
            return full

        partial = next(i for i, group in enumerate(plan["groups"]) if key in group)
        vectors = dict(
            numerators=result.numerators,
            prime=result.numerator_derivative,
            random=random,
            native_action=native_action[1],
            sigma1=result.sigma1_Ha,
            sigma1_prime=result.sigma1_derivative,
        )
        per_m = []
        for mj in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
            column = labels.index((*key, mj))

            def oracle(E, sigma, col, *, homogeneous=False):
                sigma = np.asarray(sigma, dtype=np.longdouble)
                cs, cd = (
                    (np.zeros_like(core_s), np.zeros_like(core_d))
                    if homogeneous
                    else (core_s, core_d)
                )
                rs, rd = action(
                    direct,
                    eps,
                    cs,
                    cd,
                    col,
                    np.longdouble(E),
                    sigma,
                    include_source=not homogeneous,
                )
                fs, fd, _, _ = magnetic_fock(
                    fock,
                    eps,
                    cs,
                    cd,
                    col,
                    np.longdouble(E),
                    sigma,
                    np.zeros_like(sigma),
                )[0]
                return rs + fs, rd + fd

            def linear(E, sigma, col, oracle=oracle):
                return oracle(E, sigma, col, homogeneous=True)

            independent, residual, op, inverse, conditioning = magnetic_solver(
                labels, nc, column, energy, oracle, linear=linear
            )

            def derivative(sigma, prime, column=column, energy=energy):
                rs, rd = energy_derivative(
                    direct,
                    eps,
                    core_s,
                    core_d,
                    column,
                    np.longdouble(energy),
                    sigma,
                    prime,
                )
                _, _, fs, fd = magnetic_fock(
                    fock,
                    eps,
                    core_s,
                    core_d,
                    column,
                    np.longdouble(energy),
                    sigma,
                    prime,
                )[0]
                return rs + fs, rd + fd

            rhs = derivative(independent, np.zeros_like(independent))[1]
            prime = inverse(rhs)
            sigma1_prime = derivative(independent, prime)[0]
            response_residual = op @ prime.reshape(-1) - rhs.reshape(-1)
            root = expand_valence(
                problem, plan, rows[key], unpack(result.numerators), column
            )
            native_prime = expand_valence(
                problem, plan, rows[key], unpack(result.numerator_derivative), column
            )
            random_m = expand_valence(problem, plan, rows[key], unpack(random), column)
            os_, od = oracle(energy, random_m, column)
            nd = expand_valence(
                problem, plan, rows[key], unpack(native_action[1]), column
            )
            row_keys = result.row_keys
            s_indices = [plan["groups"][partial].index(k) for k in row_keys]
            m_indices = [labels.index((*k, mj)) for k in row_keys]
            action_error = max(
                float(np.max(abs(nd - od))),
                max(
                    float(abs(native_action[0][i] - os_[p]))
                    for i, p in zip(s_indices, m_indices, strict=True)
                ),
            )
            record = dict(
                two_m=mj,
                action_Ha=action_error,
                root_Ha=float(np.max(abs(root - independent))),
                prime_Ha=float(np.max(abs(native_prime - prime))),
                sigma1_prime_Ha=float(
                    np.max(abs(result.sigma1_derivative - sigma1_prime[m_indices]))
                ),
                oracle_residual_Ha=float(np.max(abs(residual))),
                oracle_prime_residual_Ha=float(np.max(abs(response_residual))),
                conditioning=conditioning,
            )
            per_m.append(record)
            vectors[f"oracle_root_{mj}"] = independent
            vectors[f"oracle_prime_{mj}"] = prime
        np.savez_compressed(here / f"vectors_{key[0]}_{key[1]}.npz", **vectors)
        if compare_case is not None:
            reference = np.load(OWNER / compare_case / f"vectors_{key[0]}_{key[1]}.npz")
            for field in (
                "numerators",
                "prime",
                "sigma1",
                "sigma1_prime",
                "native_action",
            ):
                if not np.array_equal(vectors[field], reference[field]):
                    raise AssertionError(
                        f"same-input packed/buffer action changed {field}"
                    )
        records.append(
            dict(
                key=key,
                energy_Ha=energy,
                matrix_vector_products=result.matrix_vector_products,
                raw_residual_Ha=result.residual_max_Ha,
                response_residual_Ha=result.derivative_residual_max_Ha,
                provenance=dict(result.provenance),
                multiplets=per_m,
            )
        )
        (here / "comparison.json").write_text(
            json.dumps(
                dict(
                    core_residual_Ha=core_result.residual_max_Ha,
                    direct_error_Ha=direct_error,
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )
        if any(
            max(
                m[name]
                for name in ("action_Ha", "root_Ha", "prime_Ha", "sigma1_prime_Ha")
            )
            > 1e-11
            for m in per_m
        ):
            raise AssertionError(
                "public complete signed root/action/response gate failed"
            )
    (here / "result.json").write_text(
        json.dumps(
            dict(status="pass", columns=6, magnetic_columns=16, source_sha256=identity),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="public_valence002")
    parser.add_argument("--core-virtual-rotation", type=float, default=0.0)
    parser.add_argument("--compare-case", default=None)
    parser.add_argument("--valence-term2-blas", action="store_true")
    parser.add_argument("--valence-term2-threads", type=int, default=1)
    parser.add_argument("--cuda-term2", action="store_true")
    parser.add_argument("--recycled-control", action="store_true")
    parser.add_argument("--certify-private-seed", action="store_true")
    parser.add_argument("--cuda-krylov", action="store_true")
    parser.add_argument("--profile-linear-map", action="store_true")
    parser.add_argument("--projection-policy", choices=("all", "reachable"))
    parser.add_argument("--term41-blas", action="store_true")
    parser.add_argument("--terma4-blas", action="store_true")
    parser.add_argument("--fused-angular", action="store_true")
    args = parser.parse_args()
    try:
        main(
            args.run_name,
            args.core_virtual_rotation,
            args.compare_case,
            args.valence_term2_blas,
            args.valence_term2_threads,
            args.cuda_term2,
            args.recycled_control,
            args.certify_private_seed,
            args.cuda_krylov,
            args.profile_linear_map,
            args.projection_policy,
            args.term41_blas,
            args.terma4_blas,
            args.fused_angular,
        )
    except Exception as exc:
        directory = OWNER / args.run_name
        if directory.exists():
            (directory / "failure.json").write_text(
                json.dumps(
                    dict(exception=type(exc).__name__, message=str(exc)), indent=2
                )
                + "\n"
            )
        raise
