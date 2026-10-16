"""Full finite023 matrix audit of C=A U and C orthogonality; diagnostic only."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse.linalg._isolve._gcrotmk as scipy_source
from scipy.sparse.linalg import LinearOperator, gcrotmk

OWNER = Path(__file__).resolve().parent


def audit():
    parent = OWNER / "valence_recycling_diagnosis023"
    out = OWNER / "valence_recycling_images025"
    out.mkdir(exist_ok=False)
    paths = [
        Path(__file__),
        Path(scipy_source.__file__),
        parent / "complete_operator.npz",
    ]
    identity = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out / "source").mkdir()
    for p in paths:
        shutil.copy2(p, out / "source" / p.name)
    (out / "contract.json").write_text(
        json.dumps(
            dict(
                hypothesis="Retained GCROT image/orthogonality drift distinguishes the023 response failure; observe every outer boundary without changing equations or algorithm",
                fixed="All64 physical native coordinates, exact retained023 matrix/source/response rhs, m30/k10/maxiter100 and three actual-residual refinements",
                matrix_scope="complete finite actual native operator; no synthetic truncation or production adoption",
                identity=identity,
                cap_bytes=64 * 1024**2,
                gate_Ha=1e-11,
                stop="preserve each failed response; no tolerance or budget alteration; diagnostics do not change CU",
            ),
            indent=2,
        )
        + "\n"
    )
    with np.load(parent / "complete_operator.npz", allow_pickle=False) as data:
        matrix, source, rhs = (data[k].copy() for k in ("matrix", "source", "rhs"))
    if (
        matrix.shape != (64, 64)
        or sum(a.nbytes for a in (matrix, source, rhs)) > 64 * 1024**2
    ):
        raise MemoryError("complete declared finite operator/cap required")
    records = []
    for refresh in (False, True):
        cu, observations = [], []
        calls = [0]

        def action(v, matrix=matrix, calls=calls):
            calls[0] += 1
            return matrix @ v

        operator = LinearOperator(matrix.shape, matvec=action, dtype=np.float64)

        def inverse(
            b,
            stage,
            cu=cu,
            observations=observations,
            refresh=refresh,
            action=action,
            operator=operator,
        ):
            def notify(x, b=b, cu=cu):
                present = [(c, u) for c, u in cu if c is not None]
                raw = float(np.max(abs(matrix @ x - b)))
                if present:
                    c = np.asarray([pair[0] for pair in present]).T
                    u = np.asarray([pair[1] for pair in present]).T
                    image_error = float(np.max(abs(c - matrix @ u)))
                    orthogonality = float(np.max(abs(c.T @ c - np.eye(len(present)))))
                else:
                    image_error, orthogonality = 0.0, 0.0
                observations.append(
                    dict(
                        stage=stage,
                        refinement=refinement[0],
                        images=len(present),
                        true_raw_Ha=raw,
                        image_error=image_error,
                        gram_error=orthogonality,
                    )
                )

            refinement = [0]
            x, info = gcrotmk(
                operator,
                b,
                atol=1e-14,
                rtol=0,
                m=30,
                k=10,
                maxiter=100,
                CU=cu,
                discard_C=refresh,
                callback=notify,
            )
            for i in range(3):
                defect = b - action(x)
                if np.max(abs(defect)) <= 1e-14:
                    break
                refinement[0] = i + 1
                delta, info = gcrotmk(
                    operator,
                    defect,
                    atol=1e-15,
                    rtol=0,
                    m=30,
                    k=10,
                    maxiter=100,
                    CU=cu,
                    discard_C=refresh,
                    callback=notify,
                )
                x += delta
            return x, info, float(np.max(abs(action(x) - b)))

        with np.errstate(over="ignore", invalid="ignore"):
            root, root_info, root_raw = inverse(source, "root")
            prime, prime_info, prime_raw = inverse(rhs, "response")
        tag = "refresh" if refresh else "retain"
        np.savez_compressed(out / f"{tag}.npz", root=root, prime=prime)
        (out / f"{tag}_observations.json").write_text(
            json.dumps(observations, indent=2) + "\n"
        )
        records.append(
            dict(
                refresh_C=refresh,
                calls=calls[0],
                root_info=root_info,
                prime_info=prime_info,
                root_raw_Ha=root_raw,
                prime_raw_Ha=prime_raw,
                pass_raw_gate=bool(
                    np.isfinite(prime_raw) and max(root_raw, prime_raw) <= 1e-11
                ),
            )
        )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                records=records,
                scope="finite invariant observations only; neither public change nor causal/performance claim",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    audit()
