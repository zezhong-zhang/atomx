"""Same-input binary128 native coefficient/action control of the .61rad floor."""

import argparse
import ctypes
import faulthandler
import json
import os
import shutil
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(OWNER)]
from build_pci_valence_backend import build_valence  # noqa: E402
from build_quad_valence_control import build_quad  # noqa: E402
from build_wide_valence_control import digest  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402

from atomx.methods.valence_sd import ValenceSD  # noqa: E402


def main(run_name):
    faulthandler.enable(all_threads=True)
    out = OWNER / run_name
    out.mkdir(exist_ok=False)
    contract = dict(
        hypothesis="The native80 basis-coefficient reference and matrix-free evaluation have different amplified rounding errors; actual binary128 contractions distinguish them at unchanged signed/raw gates",
        fixed="all6C2+ radial columns/16magnetic states/ns40/point/DC/.37virtual/.61occupied-reference rotation, same complete physical fields/core result, E=eps+.075; no screening/fit",
        precision="actual binary128 valence-owned/local/density contractions; base physical core fields and angular helpers remain double, no uniform Hamiltonian/physical-precision claim",
        solver="70digit inverse of actual binary128 complete basis actions, followed by actual binary128 root/response reevaluation",
        raw_gates_Ha=dict(root=1e-11, response=1e-11, decomposition=1e-11),
        normal_signed_map_gate_Ha=1e-11,
        old_vector_diagnostic_gate_Ha=1e-11,
        reference="wide002 coefficient inverse and matrixfree003 retained complete vectors are diagnostics against this fresh arithmetic control; their original pass/fail records remain unchanged",
        cap_bytes=512 * 1024**2,
        reference_cap_bytes=64 * 1024**2,
        stop="any source/input/ABI/nonfinite/raw gate failure rejects this experiment; no public or fullYb adoption and no performance claim",
        load=os.getloadavg(),
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    paths = list((ROOT / "src/atomx").rglob("*.py"))
    paths += list((ROOT / "src/atomx/methods").glob("_pci*.f90"))
    paths += list((ROOT / "tools").glob("build_pci*.py"))
    paths += [
        Path(__file__),
        OWNER / "build_quad_valence_control.py",
        OWNER / "build_wide_valence_control.py",
        OWNER / "finite_valence_fixture.py",
    ]
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
        build_quad(directory)

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
    count_max = max(public.plan["raw_amplitudes"].values())
    # Three complete images, transport/work and high-precision matrix/inverse.
    extra = 32 * (
        dims["NXX"] ** 2 * (32 * public.plan["nvh"] + 2 * public.plan["nxv"])
        + 24 * dims["NHF"] * dims["NX"] ** 2
        + count_max**2 * 12
    )
    if extra + public.plan["complete_payload_bound_bytes"] > contract["cap_bytes"]:
        raise MemoryError("complete double/80/128 images/work exceed cap")
    if count_max**2 * 512 > contract["reference_cap_bytes"]:
        raise MemoryError("complete coefficient inverse exceeds cap")
    (out / "resource.json").write_text(
        json.dumps(
            dict(
                public_bound_bytes=public.plan["complete_payload_bound_bytes"],
                extra_bytes=extra,
                count_max=count_max,
                dimensions=dict(dims),
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
    reference = OWNER / "valence_native_wide002/physical_inputs.npz"
    with np.load(reference, allow_pickle=False) as retained:
        if set(retained.files) != set(inputs) or not all(
            np.array_equal(retained[name], value) for name, value in inputs.items()
        ):
            raise AssertionError("complete reference physical inputs differ")
    (out / "reference_binding.json").write_text(
        json.dumps(
            dict(
                physical_inputs_sha256=digest(reference),
                wide_result_sha256=digest(OWNER / "valence_native_wide002/result.json"),
                matrixfree_failure_sha256=digest(
                    OWNER / "valence_native_wide_matrixfree003/failure.json"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    lib = core._backend
    raw_dtype = np.dtype("V16")
    ptr = np.ctypeslib.ndpointer(dtype=raw_dtype, flags="F_CONTIGUOUS")
    chars = np.ctypeslib.ndpointer(dtype="S64", flags="F_CONTIGUOUS")
    pack_native, unpack_native = lib.atomx_quad_pack, lib.atomx_quad_unpack
    pack_native.argtypes = [ctypes.c_int, ctypes.c_int, chars, ptr]
    unpack_native.argtypes = [ctypes.c_int, ctypes.c_int, ptr, chars]
    pack_native.restype = unpack_native.restype = None

    def pack(values):
        # Python float conversion to mp is exact; do not round energy/input
        # via its short decimal display or confuse NumPy80 with binary128.
        text = np.asarray([mp.nstr(mp.mpf(v), 55) for v in values], dtype="S64")
        text = np.char.ljust(text, 64)
        raw = np.zeros(len(text), dtype=raw_dtype)
        pack_native(len(text), 64, text, raw)
        return raw

    def unpack(raw):
        words = np.empty(raw.size, dtype="S64")
        unpack_native(
            raw.size, 64, np.asfortranarray(raw).reshape(-1, order="F"), words
        )
        return mp.matrix([mp.mpf(word.decode().strip()) for word in words])

    def maximum(values):
        return max(abs(v) for v in values)

    initialize, info = lib.atomx_quad_val_init, lib.atomx_quad_val_info
    initialize.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    ]
    info.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS"),
        ptr,
        ptr,
    ]
    initialize.restype = info.restype = None
    mapping, linear = (
        lib.atomx_quad_val_map_full_fock,
        lib.atomx_quad_val_linear_value_fock,
    )
    mapping.argtypes, linear.argtypes = [ptr] * 9, [ptr] * 5
    mapping.restype = linear.restype = None
    precision = lib.atomx_quad_val_precision
    precision.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
    precision.restype = None
    bits, significant = ctypes.c_int(), ctypes.c_int()
    precision(ctypes.byref(bits), ctypes.byref(significant))
    if (bits.value, significant.value) != (128, 113):
        raise ValueError("actual113-significant-bit binary128 ABI required")
    (out / "precision.json").write_text(
        json.dumps(
            dict(
                storage_bits=bits.value,
                significant_bits=significant.value,
                transport="opaque16byte storage plus checked explicit decimal conversion",
            ),
            indent=2,
        )
        + "\n"
    )
    rng, records = np.random.default_rng(910064), []
    with mp.workdps(70), core._lock:
        transport = pack([0, 1, mp.mpf(2) ** -100, -mp.pi])
        if maximum(
            unpack(transport) - mp.matrix([0, 1, mp.mpf(2) ** -100, -mp.pi])
        ) > mp.mpf("1e-32"):
            raise AssertionError("binary128 transport precision gate failed")
        for key in public.plan["valence_keys"]:
            i = next(i for i, k in enumerate(core.integrals.labels) if k[:2] == key)
            energy = float(core.integrals.fock_Ha[i, i].real + 0.075)
            context = public._initialize(key, energy)
            index, count = context[3], len(context[3][0])
            slot = next(group.index(key) for group in plan["groups"] if key in group)
            initialize(
                slot + 1,
                key[1],
                len(core_result.amplitudes),
                np.asfortranarray(core_result.amplitudes),
            )
            channels = np.zeros((4, shape[2]), dtype=np.int32, order="F")
            source, orbital = (
                np.zeros(shape, dtype=raw_dtype, order="F"),
                np.zeros(1, dtype=raw_dtype),
            )
            info(channels, source, orbital)
            expected = np.asarray(public.plan["channels"][key]).T + 1
            expected[3] -= 1
            if (
                not np.array_equal(channels[:, : expected.shape[1]], expected)
                or float(unpack(orbital)[0]) != context[5]
            ):
                raise AssertionError(
                    "complete binary128 channel/energy identity differs"
                )
            energy_raw = pack([energy])
            work = [np.zeros(shape, dtype=raw_dtype, order="F") for _ in range(4)]
            s, rs, ps = [np.zeros(dims["NX"], dtype=raw_dtype) for _ in range(3)]

            def evaluate(
                vector,
                prime=None,
                *,
                homogeneous=False,
                work=work,
                s=s,
                rs=rs,
                ps=ps,
                index=index,
                energy_raw=energy_raw,
            ):
                d, dd, rd, pd = work
                d[index] = pack(vector)
                dd[index] = pack([0] * len(vector) if prime is None else prime)
                if homogeneous:
                    linear(energy_raw, s, d, rs, rd)
                    return unpack(rs), unpack(np.asfortranarray(rd[index]))
                mapping(energy_raw, s, d, s, dd, rs, rd, ps, pd)
                return (
                    unpack(rs),
                    unpack(np.asfortranarray(rd[index])),
                    unpack(ps),
                    unpack(np.asfortranarray(pd[index])),
                )

            numerator, derivative = rng.normal(size=(2, count)) * 0.01
            baseline, candidate = (
                public._evaluate(energy, numerator, derivative, context),
                evaluate(numerator, derivative),
            )
            normal_error = max(
                maximum(mp.matrix([mp.mpf(float(v)) for v in a]) - b)
                for a, b in zip(baseline, candidate, strict=True)
            )
            if normal_error > contract["normal_signed_map_gate_Ha"]:
                raise AssertionError(
                    "complete ordinary binary128/double signed gate failed"
                )
            zero = mp.zeros(count, 1)
            constant = evaluate(zero)[1]

            def action(vector, evaluate=evaluate):
                return vector - evaluate(vector, homogeneous=True)[1]

            matrix = mp.zeros(count, count)
            for j in range(count):
                unit = mp.zeros(count, 1)
                unit[j] = 1
                matrix[:, j] = action(unit)
            root = mp.lu_solve(matrix, constant)
            mapped = evaluate(root)
            rhs = mapped[3]
            prime = mp.lu_solve(matrix, rhs)
            raw, affine = action(root) - constant, root - mapped[1]
            response = action(prime) - rhs
            full = evaluate(root, prime)[2]
            separated = mapped[2] + evaluate(prime, homogeneous=True)[0]
            old = {}
            for label, dirname in (
                ("coefficient80", "valence_native_wide002"),
                ("matrixfree80", "valence_native_wide_matrixfree003"),
            ):
                path = OWNER / dirname / f"column_{key[0]}_{key[1]}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as retained:
                        old_root = mp.matrix([mp.mpf(str(x)) for x in retained["root"]])
                        old_prime = mp.matrix(
                            [mp.mpf(str(x)) for x in retained["prime"]]
                        )
                    root_error, prime_error = (
                        maximum(old_root - root),
                        maximum(old_prime - prime),
                    )
                    old[label] = dict(
                        root_error_Ha=float(root_error),
                        prime_error_Ha=float(prime_error),
                        unchanged_signed_gate_passed=max(root_error, prime_error)
                        <= mp.mpf("1e-11"),
                    )
            record = dict(
                key=key,
                energy_Ha=energy,
                count=count,
                ordinary_signed_error_Ha=float(normal_error),
                raw_root_Ha=float(max(maximum(raw), maximum(affine))),
                raw_response_Ha=float(maximum(response)),
                decomposition_Ha=float(maximum(full - separated)),
                old_vectors=old,
            )
            arrays = dict(
                matrix=matrix,
                constant=constant,
                root=root,
                rhs=rhs,
                prime=prime,
                raw_root=raw,
                raw_affine=affine,
                raw_response=response,
                full_response=full,
                separated_response=separated,
            )
            encoded = {
                name: np.asarray([mp.nstr(x, 55) for x in value], dtype="S80").reshape(
                    value.rows, value.cols
                )
                for name, value in arrays.items()
            }
            np.savez_compressed(out / f"column_{key[0]}_{key[1]}.npz", **encoded)
            records.append(record)
            (out / "partial_result.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
            if not all(mp.isfinite(v) for value in arrays.values() for v in value):
                raise AssertionError("nonfinite complete binary128 result")
            if (
                max(
                    record["raw_root_Ha"],
                    record["raw_response_Ha"],
                    record["decomposition_Ha"],
                )
                > 1e-11
            ):
                raise AssertionError("actual binary128 unchanged raw gate failed")
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="complete6column binary128 arithmetic control passed",
                records=records,
                scope="same native equations arithmetic diagnosis, no independent magnetic/physical/scalable/public claim",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    try:
        main(args.run_name)
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
