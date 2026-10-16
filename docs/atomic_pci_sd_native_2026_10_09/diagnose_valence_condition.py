"""Diagnose the retained .61 rad finite response failure without changing gates."""

import hashlib
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
from finite_valence_fixture import prepare  # noqa: E402

from atomx.methods.valence_sd import ValenceSD  # noqa: E402


def main():
    out = OWNER / "valence_condition006"
    out.mkdir(exist_ok=False)
    (out / "contract.json").write_text(
        json.dumps(
            dict(
                hypothesis="Strong .61rad changed occupied projector causes near-singular fixed-energy response or a floating-point residual floor; compare original matrix-free map, its complete basis matrix and 70digit solve of that matrix",
                fixed="complete same C2+16magnetic/ns40/point/DC/source/.37virtual and .61occupied-virtual rotation as failed public004, all six valence columns, E=eps_v+.075",
                acceptance_Ha=1e-11,
                stop="keep actual raw native defect; no relaxed gate or altered model/energy; diagnostic dense matrix stays finite-only",
                cap_bytes=512 * 1024**2,
                reference_bytes=64 * 1024**2,
                load=os.getloadavg(),
                limit="high precision inversion of double matrix coefficients is diagnostic, not a high precision physical native operator",
            ),
            indent=2,
        )
        + "\n"
    )
    for path in [
        Path(__file__),
        *sorted((ROOT / "src/atomx").rglob("*.py")),
        *sorted((ROOT / "src/atomx/methods").glob("_pci_*.f90")),
        OWNER / "finite_valence_fixture.py",
        ROOT / "tools/build_pci_valence_backend.py",
    ]:
        dest = out / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (out / "source_sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(out / "source")): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (out / "source").rglob("*")
                if p.is_file()
            },
            indent=2,
        )
        + "\n"
    )

    def builder(source, directory, dimensions, compiler, *, nxv, nvh, runtime):
        return build_valence(
            source,
            directory,
            dimensions,
            compiler,
            nxv=nxv,
            nvh=nvh,
            runtime_directory=runtime,
        )

    _, _, plan, _, _, _, core, _ = prepare(
        out,
        REF,
        backend_builder=builder,
        fock_probe=False,
        virtual_rotation=0.37,
        core_virtual_rotation=0.61,
    )
    core_result = core.solve(residual_tolerance_Ha=1e-12)
    problem = ValenceSD(core, core_result, valence_keys=plan["virtual_keys"])
    records = []
    with core._lock:
        for key in problem.plan["valence_keys"]:
            index = next(i for i, k in enumerate(core.integrals.labels) if k[:2] == key)
            energy = float(core.integrals.fock_Ha[index, index].real + 0.075)
            context = problem._initialize(key, energy)
            count = problem.plan["raw_amplitudes"][key]
            if 8 * count * count * 8 > 64 * 1024**2:
                raise MemoryError("finite diagnostic matrix exceeds predeclared cap")
            zero = np.zeros(count)
            constant = problem._evaluate(energy, zero, None, context)[1]

            def action(v, energy=energy, context=context):
                return v - problem._evaluate(energy, v, None, context, linear=True)[1]

            matrix = np.column_stack([action(v) for v in np.eye(count)])
            with mp.workdps(70):
                mm = mp.matrix(matrix.tolist())
                root = np.array(
                    list(mp.lu_solve(mm, mp.matrix(constant.tolist()))), dtype=float
                ).ravel()
                mapped = problem._evaluate(energy, root, None, context)
                rhs = mapped[3]
                prime = np.array(
                    list(mp.lu_solve(mm, mp.matrix(rhs.tolist()))), dtype=float
                ).ravel()
            residual = action(root) - constant
            response = action(prime) - rhs
            sigma_response = problem._evaluate(energy, root, prime, context)[2]
            source_free_response = (
                mapped[2]
                + problem._evaluate(energy, prime, None, context, linear=True)[0]
            )
            singular = np.linalg.svd(matrix, compute_uv=False)
            record = dict(
                key=key,
                energy_Ha=energy,
                count=count,
                min_abs_denominator_Ha=float(np.min(abs(context[4]))),
                condition_number=float(singular[0] / singular[-1]),
                min_singular_value=float(singular[-1]),
                max_root=float(np.max(abs(root))),
                max_prime=float(np.max(abs(prime))),
                raw_native_root_defect_Ha=float(np.max(abs(residual))),
                raw_native_response_defect_Ha=float(np.max(abs(response))),
                matrix_response_defect_Ha=float(np.max(abs(matrix @ prime - rhs))),
                response_decomposition_error_Ha=float(
                    np.max(abs(sigma_response - source_free_response))
                ),
            )
            np.savez_compressed(
                out / f"column_{key[0]}_{key[1]}.npz",
                matrix=matrix,
                constant=constant,
                root=root,
                rhs=rhs,
                prime=prime,
                native_root_residual=residual,
                native_response_residual=response,
                sigma_response=sigma_response,
                source_free_response=source_free_response,
            )
            records.append(record)
            (out / "partial_result.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="diagnosis completed; original gate unchanged", records=records
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
