"""Matched .61rad native mixed-width contraction control; unchanged raw gates."""

import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(OWNER)]
from build_pci_valence_backend import build_valence  # noqa: E402
from build_wide_valence_control import build_wide  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402

from atomx.methods.valence_sd import ValenceSD  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run_name, *, matrix_free=False, reference_quad=False):
    faulthandler.enable(all_threads=True)
    out = OWNER / run_name
    out.mkdir(exist_ok=False)
    if reference_quad and not matrix_free:
        raise ValueError("quad forward reference applies only to matrix-free control")
    reference_directory = OWNER / (
        "valence_native_quad005" if reference_quad else "valence_native_wide002"
    )
    contract = dict(
        hypothesis="Amplified double contraction rounding contributes to the retained .61rad native response floor; widen actual valence-owned/local/density contractions and evaluate the same raw operator residual",
        fixed="all six C2+ radial columns/16magnetic states/ns40/point/DC, actual .37virtual and .61occupied-virtual rotation, same complete core result and energy eps_v+.075; original diagrams/no screening/no energy fit",
        scope="private finite mixed-width native diagnosis; shared core fields/angular helpers and some wave-only products retain original double precision; not a uniformly extended-precision physical Hamiltonian",
        cap_bytes=512 * 1024**2,
        matrix_reference_cap_bytes=64 * 1024**2,
        gates_Ha=dict(
            normal_signed_map=1e-11,
            native_root=1e-11,
            native_response=1e-11,
            full_vs_separated_response=1e-11,
        ),
        baseline="fresh public double solve, failures retained; original006 source/input convention retained",
        inverse="70digit inverse of actual mixed-width native basis-action coefficients, returned as longdouble then actual native raw residual reevaluated",
        stop="any source/ABI/model/resource/nonfinite or unchanged gate failure rejects; no automatic public/full-scale adoption, no precision or performance claim",
        load=os.getloadavg(),
    )
    if matrix_free:
        contract.update(
            hypothesis="Double inner matrix-free Krylov corrections plus actual extended native residuals remove the same strong-reference floor without assembling a solver matrix",
            inverse="double public homogeneous GMRES with normalized RHS, extended-precision outer iterates and up to5actual-native correction steps; restart30/maxiter100/inneratol1e-14",
            reference="same complete physical inputs and actual wide equations as native_wide002; retained70digit coefficient inverse is independent of this matrix-free solve",
            forward_gates_Ha=dict(root=1e-11, prime=1e-11),
        )
        if reference_quad:
            contract["reference"] = (
                "same complete physical inputs asquad005; actual113-significant-bit contraction/coefficient reference replaces the numerically failed80-bit coefficient reference, without changing the1e-11signed gate or any archived failure"
            )
            contract["reference_precision_scope"] = (
                "same native equations/mixed base fields, not an independent magnetic or physical Hamiltonian oracle"
            )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    paths = (
        list((ROOT / "src/atomx").rglob("*.py"))
        + list((ROOT / "src/atomx/methods").glob("_pci*.f90"))
        + list((ROOT / "tools").glob("build_pci*.py"))
        + [
            Path(__file__),
            OWNER / "build_wide_valence_control.py",
            OWNER / "finite_valence_fixture.py",
        ]
    )
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")

    def builder(source, directory, dimensions, compiler, *, nxv, nvh, runtime):
        build_valence(
            source,
            directory,
            dimensions,
            compiler,
            nxv=nxv,
            nvh=nvh,
            runtime_directory=runtime,
        )
        build_wide(directory)

    atom, waves, plan, _, _, _, core, _ = prepare(
        out,
        REF,
        backend_builder=builder,
        fock_probe=False,
        virtual_rotation=0.37,
        core_virtual_rotation=0.61,
    )
    core_result = core.solve(residual_tolerance_Ha=1e-12)
    public = ValenceSD(core, core_result, valence_keys=plan["virtual_keys"])
    dims = plan["dimensions"]
    shape = (dims["NXX"], dims["NXX"], public.plan["nvh"])
    count = max(public.plan["raw_amplitudes"].values())
    extra = 16 * (
        dims["NXX"] ** 2 * (32 * public.plan["nvh"] + 2 * public.plan["nxv"])
        + 24 * dims["NHF"] * dims["NX"] ** 2
        + count * count * 12
    )
    if extra + public.plan["complete_payload_bound_bytes"] > contract["cap_bytes"]:
        raise MemoryError("complete finite double/wide images and work exceed cap")
    if count * count * 512 > contract["matrix_reference_cap_bytes"]:
        raise MemoryError("complete high-precision coefficient solve exceeds cap")
    (out / "resource.json").write_text(
        json.dumps(
            dict(
                public_bound_bytes=public.plan["complete_payload_bound_bytes"],
                wide_extra_bytes=extra,
                matrix_coordinate_max=count,
                dimensions=dict(dims),
                scope="conservative finite native/work/matrix payload; excludes interpreter overhead",
            ),
            indent=2,
        )
        + "\n"
    )
    inputs = dict(
        core_amplitudes=core_result.amplitudes, physical_fock=core.integrals.fock_Ha
    )
    for (n, k), (p, q) in waves.items():
        inputs[f"P_{n}_{k}"], inputs[f"Q_{n}_{k}"] = p, q
    for name, value in atom._bsp_data.items():
        if isinstance(value, np.ndarray) or np.isscalar(value):
            inputs["workspace_" + name] = np.asarray(value)
    np.savez_compressed(out / "physical_inputs.npz", **inputs)
    if matrix_free:
        reference_path = reference_directory / "physical_inputs.npz"
        with np.load(reference_path, allow_pickle=False) as retained:
            if set(retained.files) != set(inputs) or not all(
                np.array_equal(retained[name], value) for name, value in inputs.items()
            ):
                raise AssertionError("complete wide reference physical inputs differ")
        (out / "reference_binding.json").write_text(
            json.dumps(
                dict(
                    physical_inputs_sha256=digest(reference_path),
                    reference_result_sha256=digest(reference_directory / "result.json"),
                    scope="same all physical radial/workspace/core/Fock input arrays; solver changes only",
                ),
                indent=2,
            )
            + "\n"
        )
    lib = core._backend
    ptr = np.ctypeslib.ndpointer(dtype=np.longdouble, flags="F_CONTIGUOUS")
    ptr64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    initialize = lib.atomx_wide_val_init
    initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ptr64]
    initialize.restype = None
    precision = lib.atomx_wide_val_precision
    precision.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    precision.restype = None
    bits, significant = ctypes.c_int(), ctypes.c_int()
    precision(ctypes.byref(bits), ctypes.byref(significant))
    if bits.value != np.dtype(np.longdouble).itemsize * 8 or significant.value <= 53:
        raise ValueError("actual compatible extended C/Fortran/NumPy ABI required")
    (out / "precision.json").write_text(
        json.dumps(
            dict(
                storage_bits=bits.value,
                significant_bits=significant.value,
                numpy_eps=str(np.finfo(np.longdouble).eps),
            ),
            indent=2,
        )
        + "\n"
    )
    mapping, linear = (
        lib.atomx_wide_val_map_full_fock,
        lib.atomx_wide_val_linear_value_fock,
    )
    mapping.argtypes = [ctypes.c_longdouble] + [ptr] * 8
    linear.argtypes = [ctypes.c_longdouble] + [ptr] * 4
    mapping.restype = linear.restype = None
    records = []
    rng = np.random.default_rng(910064)
    with core._lock:
        for key in public.plan["valence_keys"]:
            i = next(i for i, k in enumerate(core.integrals.labels) if k[:2] == key)
            energy = float(core.integrals.fock_Ha[i, i].real + 0.075)
            context = public._initialize(key, energy)
            index = context[3]
            count = len(index[0])
            slot = next(group.index(key) for group in plan["groups"] if key in group)
            initialize(
                slot + 1,
                key[1],
                len(core_result.amplitudes),
                np.asfortranarray(core_result.amplitudes),
            )
            channels = np.zeros((4, shape[2]), dtype=np.int32, order="F")
            source = np.zeros(shape, dtype=np.longdouble, order="F")
            orbital = np.zeros(1, dtype=np.longdouble)
            info = lib.atomx_wide_val_info
            info.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS"),
                ptr,
                ptr,
            ]
            info.restype = None
            info(channels, source, orbital)
            expected = np.asarray(public.plan["channels"][key]).T + 1
            expected[3] -= 1
            if (
                not np.array_equal(channels[:, : expected.shape[1]], expected)
                or float(orbital[0]) != context[5]
            ):
                raise ValueError("wide complete channel/energy identity differs")
            work = [np.zeros(shape, dtype=np.longdouble, order="F") for _ in range(4)]
            s, rs, ps = [np.zeros(dims["NX"], dtype=np.longdouble) for _ in range(3)]

            def evaluate(
                vector,
                prime=None,
                *,
                homogeneous=False,
                work=work,
                index=index,
                energy=energy,
                s=s,
                rs=rs,
                ps=ps,
            ):
                d, dd, rd, pd = work
                d[index] = vector
                dd[index] = 0 if prime is None else prime
                if homogeneous:
                    linear(energy, s, d, rs, rd)
                    outputs = rs.copy(), rd[index]
                else:
                    mapping(energy, s, d, s, dd, rs, rd, ps, pd)
                    outputs = rs.copy(), rd[index], ps.copy(), pd[index]
                if not all(np.all(np.isfinite(a)) for a in outputs):
                    raise RuntimeError("nonfinite actual native wide action")
                return outputs

            numerator, derivative = rng.normal(size=(2, count)) * 0.01
            baseline = public._evaluate(energy, numerator, derivative, context)
            widened = evaluate(numerator, derivative)
            ordinary_error = max(
                float(np.max(abs(a.astype(np.longdouble) - b)))
                for a, b in zip(baseline, widened, strict=True)
            )
            if ordinary_error > contract["gates_Ha"]["normal_signed_map"]:
                raise AssertionError("wide/double normal signed action differs")
            zero = np.zeros(count, dtype=np.longdouble)
            constant = evaluate(zero)[1]

            def action(v, evaluate=evaluate):
                return v - evaluate(v, homogeneous=True)[1]

            refinement = []
            if matrix_free:

                def inner_action(v, energy=energy, context=context):
                    return v - public._evaluate_value(energy, v, context)[1]

                operator = LinearOperator(
                    (count, count), matvec=inner_action, dtype=float
                )

                def inverse(
                    rhs,
                    action=action,
                    operator=operator,
                    count=count,
                    refinement=refinement,
                ):
                    def correction(target):
                        scale = np.max(abs(target))
                        if scale == 0:
                            return np.zeros(count, dtype=np.longdouble), 0
                        step, info = gmres(
                            operator,
                            np.asarray(target / scale, dtype=float),
                            atol=1e-14,
                            rtol=0,
                            restart=30,
                            maxiter=100,
                        )
                        return step.astype(np.longdouble) * scale, info

                    solution, info = correction(rhs)
                    for iteration in range(5):
                        defect = rhs - action(solution)
                        refinement.append(
                            dict(
                                stage=len(refinement),
                                iteration=iteration,
                                inner_info=int(info),
                                actual_defect_Ha=float(np.max(abs(defect))),
                            )
                        )
                        if np.max(abs(defect)) <= 1e-14:
                            break
                        delta, info = correction(defect)
                        solution += delta
                    return solution

                root = inverse(constant)
                mapped = evaluate(root)
                rhs = mapped[3]
                prime = inverse(rhs)
                with np.load(
                    reference_directory / f"column_{key[0]}_{key[1]}.npz",
                    allow_pickle=False,
                ) as retained:
                    matrix = retained["matrix"].astype(np.longdouble)
                    root_error = float(
                        np.max(
                            abs(root - retained["root"].astype(np.longdouble).ravel())
                        )
                    )
                    prime_error = float(
                        np.max(
                            abs(prime - retained["prime"].astype(np.longdouble).ravel())
                        )
                    )
            else:
                matrix = np.column_stack(
                    [action(v) for v in np.eye(count, dtype=np.longdouble)]
                )
                root_error, prime_error = None, None
                with mp.workdps(70):
                    mm = mp.matrix([[mp.mpf(str(x)) for x in row] for row in matrix])

                    def inverse(rhs, mm=mm):
                        solution = mp.lu_solve(
                            mm, mp.matrix([mp.mpf(str(x)) for x in rhs])
                        )
                        return np.asarray([np.longdouble(str(x)) for x in solution])

                    root = inverse(constant)
                    mapped = evaluate(root)
                    rhs = mapped[3]
                    prime = inverse(rhs)
            raw = action(root) - constant
            raw_affine = root - mapped[1]
            raw_response = action(prime) - rhs
            full_response = evaluate(root, prime)[2]
            separated = mapped[2] + evaluate(prime, homogeneous=True)[0]
            singular = np.linalg.svd(matrix.astype(float), compute_uv=False)
            # Separate public double re-solve: never turn its failure into a
            # wide pass or discard it. Both spaces/energy contracts are equal.
            try:
                public_result = public.solve(key, energy, residual_tolerance_Ha=1e-11)
                double_status = dict(
                    status="passed",
                    raw=public_result.residual_max_Ha,
                    response=public_result.derivative_residual_max_Ha,
                )
            except RuntimeError as error:
                double_status = dict(
                    status="failed", type=type(error).__name__, message=str(error)
                )
            record = dict(
                key=key,
                energy_Ha=energy,
                count=count,
                ordinary_signed_error_Ha=ordinary_error,
                condition_number=float(singular[0] / singular[-1]),
                max_prime=str(np.max(abs(prime))),
                wide_raw_root_Ha=float(max(np.max(abs(raw)), np.max(abs(raw_affine)))),
                wide_raw_response_Ha=float(np.max(abs(raw_response))),
                wide_response_decomposition_Ha=float(
                    np.max(abs(full_response - separated))
                ),
                public_double=double_status,
                matrix_free=matrix_free,
                signed_root_reference_error_Ha=root_error,
                signed_prime_reference_error_Ha=prime_error,
                refinement=refinement,
            )
            np.savez_compressed(
                out / f"column_{key[0]}_{key[1]}.npz",
                matrix=matrix,
                constant=constant,
                root=root,
                rhs=rhs,
                prime=prime,
                native_root_residual=raw,
                native_affine_residual=raw_affine,
                native_response_residual=raw_response,
                full_response=full_response,
                separated_response=separated,
            )
            records.append(record)
            (out / "partial_result.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
            if (
                max(
                    record["wide_raw_root_Ha"],
                    record["wide_raw_response_Ha"],
                    record["wide_response_decomposition_Ha"],
                )
                > 1e-11
            ):
                raise AssertionError("actual wide native unchanged raw gate failed")
            if matrix_free and max(root_error, prime_error) > 1e-11:
                raise AssertionError(
                    "matrix-free native wide signed reference gate failed"
                )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="all six native mixed-width raw gates passed",
                records=records,
                scope="finite mixed-width native diagnosis; no scalable solver/public/fullYb/physical precision claim",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--matrix-free", action="store_true")
    parser.add_argument("--reference-quad", action="store_true")
    args = parser.parse_args()
    try:
        main(
            args.run_name,
            matrix_free=args.matrix_free,
            reference_quad=args.reference_quad,
        )
    except BaseException as error:
        out = OWNER / args.run_name
        if out.exists():
            (out / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
