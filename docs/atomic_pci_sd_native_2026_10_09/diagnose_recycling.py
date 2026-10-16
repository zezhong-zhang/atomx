"""Finite complete4s operator diagnosis of withdrawn GCROT012, not production."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(OWNER)]
from build_pci_valence_backend import build_valence  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402
from reorthogonal_gcrot import gcrotmk  # noqa: E402

from atomx.methods.valence_sd import ValenceSD  # noqa: E402


def main():
    out = OWNER / "valence_recycling_reorthogonal027"
    out.mkdir(exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save(
        "contract.json",
        dict(
            hypothesis="Distinguish finite complete-operator conditioning, matrix-free rounding and approximate recycled C versus actual A*U in the withdrawn strong4s case",
            fixed="same C2+ all16magnetic/sixvirtual/ns40/point/nativeDC/.37virtual/.37occupied-reference,4s E=Fvv+.075; current certified CPU operator, no truncation",
            gate_Ha=1e-11,
            matrix_cap_bytes=64 * 1024**2,
            solver_budget=dict(
                m=30, k=10, maxiter=100, refinements=3, inner_atol_Ha=1e-14
            ),
            variants="private two-pass full MGS GCROT, actualC refresh: complete dense and native operators; original failed026 retained",
            stop="one matched4s problem and three variants only; retain all failures; no threshold/budget/physical input change or public solver adoption",
            load=os.getloadavg(),
            scope="finite diagnostic complete matrix only; no full-space performance/spectral claim",
            observation="observe complete C-AU and C Gram at outer boundaries without modifying CU; archive the actual root-dependent implicit rhs per variant (025 used the dense-root rhs and did not reproduce023)",
        ),
    )
    files = [
        Path(__file__),
        OWNER / "reorthogonal_gcrot.py",
        OWNER / "reorthogonal_gcrot_source_binding.json",
        OWNER / "scipy_private_gcrot_LICENSE.txt",
        OWNER / "finite_valence_fixture.py",
        ROOT / "tools/build_pci_valence_backend.py",
        ROOT / "src/atomx/methods/valence_sd.py",
        ROOT / "src/atomx/methods/core_sd.py",
    ]
    hashes = {}
    for p in files:
        relative = p.relative_to(ROOT)
        q = out / "source" / relative
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        hashes[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
    save("source_sha256.json", hashes)

    def builder(
        source, directory, dimensions, compiler, *, nxv, nvh, runtime, **options
    ):
        return build_valence(
            source,
            directory,
            dimensions,
            compiler,
            nxv=nxv,
            nvh=nvh,
            runtime_directory=runtime,
            valence_term2_blas=True,
            valence_term2_threads=8,
            blas_library=OWNER
            / "openblas_openmp_runtime/root/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
            **options,
        )

    _, _, plan, _, _, _, core, _ = prepare(
        out,
        REF,
        backend_builder=builder,
        fock_probe=False,
        virtual_rotation=0.37,
        core_virtual_rotation=0.37,
        core_options=dict(
            term21_threads=2,
            term21_mode="streamed",
            core_singles_blas=True,
            core_single_to_doubles_blas=True,
        ),
    )
    result = core.solve(residual_tolerance_Ha=1e-12)
    problem = ValenceSD(core, result, valence_keys=plan["virtual_keys"])
    key = (4, -1)
    index = next(i for i, k in enumerate(core.integrals.labels) if k[:2] == key)
    energy = float(core.integrals.fock_Ha[index, index].real + 0.075)
    baseline = json.loads((OWNER / "public_valence011/comparison.json").read_text())[
        "records"
    ][1]
    if energy != baseline["energy_Ha"]:
        raise ValueError("matched4s energy differs")
    with np.load(OWNER / "public_valence011/inputs.npz", allow_pickle=False) as d:
        if not np.array_equal(result.amplitudes, d["core_amplitudes"]):
            raise ValueError("matched complete core root differs")
    with np.load(OWNER / "public_valence011/vectors_4_-1.npz", allow_pickle=False) as d:
        reference_root = d["numerators"].copy()
        reference_prime = d["prime"].copy()
    with core._lock:
        context = problem._initialize(key, energy)
        count = problem.plan["raw_amplitudes"][key]
        if (
            problem.plan["complete_payload_bound_bytes"] + 64 * 1024**2
            > core.memory_bytes
            or 8 * count * count * 8 > 64 * 1024**2
        ):
            raise MemoryError("complete finite diagnostic exceeds declared cap")
        zero = np.zeros(count)
        source = problem._evaluate(energy, zero, None, context)[1]

        def native(v):
            return v - problem._evaluate_value(energy, v, context)[1]

        matrix = np.column_stack([native(v) for v in np.eye(count)])
        singular = np.linalg.svd(matrix, compute_uv=False)
        dense_root = np.linalg.solve(matrix, source)
        mapped = problem._evaluate(energy, dense_root, None, context)
        rhs = mapped[3]
        dense_prime = np.linalg.solve(matrix, rhs)
        np.savez_compressed(
            out / "complete_operator.npz",
            matrix=matrix,
            source=source,
            rhs=rhs,
            dense_root=dense_root,
            dense_prime=dense_prime,
            reference_root=reference_root,
            reference_prime=reference_prime,
            singular_values=singular,
        )
        save(
            "operator.json",
            dict(
                count=count,
                energy_Ha=energy,
                condition_2=float(singular[0] / singular[-1]),
                min_singular_value=float(singular[-1]),
                dense_root_raw_Ha=float(np.max(abs(native(dense_root) - source))),
                dense_prime_raw_Ha=float(np.max(abs(native(dense_prime) - rhs))),
                complete_dense_native_binding_error_Ha=float(
                    np.max(abs(matrix @ reference_root - native(reference_root)))
                ),
            ),
        )
        records = []
        for variant, matvec, refresh in [
            ("dense_reorthogonal", lambda v: matrix @ v, True),
            ("native_reorthogonal", native, True),
        ]:
            cu = []
            calls = 0
            observations = []

            def action(v, matvec=matvec):
                nonlocal calls
                calls += 1
                return matvec(v)

            operator = LinearOperator((count, count), matvec=action, dtype=float)

            def inverse(
                b,
                stage,
                operator=operator,
                cu=cu,
                refresh=refresh,
                action=action,
                observations=observations,
            ):
                refinement = [0]

                def observe(x, b=b, stage=stage, cu=cu, matrix=matrix):
                    present = [(c, u) for c, u in cu if c is not None]
                    with np.errstate(over="ignore", invalid="ignore"):
                        raw = float(np.max(abs(matrix @ x - b)))
                        if present:
                            c = np.column_stack([pair[0] for pair in present])
                            u = np.column_stack([pair[1] for pair in present])
                            image_error = float(np.max(abs(c - matrix @ u)))
                            gram_error = float(
                                np.max(abs(c.T @ c - np.eye(len(present))))
                            )
                        else:
                            image_error, gram_error = 0.0, 0.0
                    observations.append(
                        dict(
                            stage=stage,
                            refinement=refinement[0],
                            images=len(present),
                            true_raw_Ha=raw,
                            image_error=image_error,
                            gram_error=gram_error,
                        )
                    )

                x, info = gcrotmk(
                    operator,
                    b,
                    m=30,
                    k=10,
                    maxiter=100,
                    atol=1e-14,
                    rtol=0,
                    CU=cu,
                    discard_C=refresh,
                    callback=observe,
                )
                for _ in range(3):
                    defect = b - action(x)
                    if np.max(abs(defect)) <= 1e-14:
                        break
                    refinement[0] = _ + 1
                    base = x.copy()
                    delta, info = gcrotmk(
                        operator,
                        defect,
                        m=30,
                        k=10,
                        maxiter=100,
                        atol=1e-15,
                        rtol=0,
                        CU=cu,
                        discard_C=refresh,
                        callback=lambda delta, base=base: observe(base + delta),
                    )
                    x += delta
                return x, info

            root, info = inverse(source, "root")
            raw = float(np.max(abs(native(root) - source)))
            record = dict(
                variant=variant,
                refresh_C=refresh,
                root_info=info,
                root_raw_Ha=raw,
                root_signed_error_Ha=float(np.max(abs(root - reference_root))),
                calls=calls,
            )
            vectors = dict(root=root)
            if np.isfinite(raw) and raw <= 1e-11:
                implicit_rhs = problem._evaluate(energy, root, None, context)[3]
                vectors["rhs"] = implicit_rhs
                prime, pi = inverse(implicit_rhs, "response")
                vectors["prime"] = prime
                record.update(
                    prime_info=pi,
                    prime_raw_Ha=float(
                        np.max(
                            abs(
                                native(prime)
                                - problem._evaluate(energy, root, None, context)[3]
                            )
                        )
                    ),
                    prime_signed_error_Ha=float(np.max(abs(prime - reference_prime))),
                    calls=calls,
                )
            np.savez_compressed(out / f"{variant}.npz", **vectors)
            save(f"{variant}_observations.json", observations)
            records.append(record)
            save("partial.json", records)
            print(json.dumps(record), flush=True)
        save(
            "result.json",
            dict(
                records=records,
                scope="diagnosis only; no public inverse change or causal claim beyond measured operator/variant distinction",
            ),
        )


if __name__ == "__main__":
    main()
