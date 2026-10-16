"""Complete multi-core noncanonical core/valence roots and state response."""

import ctypes
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from finite_direct_coulomb import direct_coulomb  # noqa: E402
from finite_valence_direct_solver import magnetic_solver, reduced_solver  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402
from general_rho_control import core_numerators, expand_valence  # noqa: E402
from valence_allorder_oracle import action as magnetic_action  # noqa: E402
from valence_energy_response_oracle import energy_derivative  # noqa: E402
from valence_fock_oracle import magnetic_fock  # noqa: E402

OWNER = Path(__file__).resolve().parent
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")


def main():
    here = OWNER / "valence_fock_allorder006"
    here.mkdir(exist_ok=False)
    archive = here / "source"
    archive.mkdir()
    identity = {}
    for name in (
        Path(__file__).name,
        "finite_direct_coulomb.py",
        "finite_valence_fixture.py",
        "finite_valence_direct_solver.py",
        "general_rho_control.py",
        "build_valence_fock_control.py",
        "valence_fock_control.f90",
        "valence_fock_oracle.py",
        "build_valence_control.py",
        "valence_control.f90",
        "valence_allorder_oracle.py",
        "valence_energy_response_oracle.py",
        "probe_sigma2_control003.py",
    ):
        p = OWNER / name
        shutil.copy2(p, archive / name)
        identity[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (here / "source_sha256.json").write_text(json.dumps(identity, indent=2) + "\n")
    core, orbitals, plan, rows_by_key, nvh, extra, problem, fock = prepare(here, REF)
    (here / "plan.json").write_text(
        json.dumps(dict(core=plan, extra_bytes=extra), indent=2) + "\n"
    )
    (here / "native_base_identity.json").write_text(
        json.dumps(problem.provenance, indent=2) + "\n"
    )
    # Refresh the core source after the explicit operator perturbation.
    source, denominator = (
        np.zeros_like(problem.source),
        np.zeros_like(problem.denominators),
    )
    problem._vector_call("atomx_sd_vector_info", [source, denominator])
    kop = LinearOperator((len(source),) * 2, matvec=problem.action, dtype=float)
    pre = LinearOperator(kop.shape, matvec=lambda x: x / denominator, dtype=float)
    physical, info = gmres(
        kop, source, M=pre, atol=1e-12, rtol=0, restart=30, maxiter=100
    )
    core_residual = problem.action(physical) - source
    np.savez_compressed(
        here / "inputs.npz",
        core_physical=physical,
        core_residual=core_residual,
        core_source=source,
        core_denominator=denominator,
        target_fock=fock,
        base_fock=problem.integrals.fock_Ha,
        labels=np.asarray(problem.integrals.labels),
    )
    if (
        info != 0
        or not np.all(np.isfinite(core_residual))
        or np.max(abs(core_residual)) > 1e-10
    ):
        raise AssertionError("complete perturbed core root residual failed")
    core_s, core_d = core_numerators(problem, plan, physical)
    direct, direct_error = direct_coulomb(problem, orbitals, core._bsp_data)
    eps = np.diag(fock)
    labels = problem.integrals.labels
    dims = plan["dimensions"]
    virtual = plan["virtual_keys"]
    init = problem._backend.atomx_val_init
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ptr]
    init.restype = None
    mapping = problem._backend.atomx_val_map_full_fock
    mapping.argtypes = [ctypes.c_double] + [ptr] * 8
    mapping.restype = None
    records = []
    rng = np.random.default_rng(451277)
    for key in virtual:
        partial = next(i for i, g in enumerate(plan["groups"]) if key in g)
        slot = plan["groups"][partial].index(key)
        init(slot + 1, key[1], len(physical), np.asfortranarray(physical))
        rows = rows_by_key[key]
        coordinates = [
            (m, n, c)
            for c, (_, im, jn, _) in enumerate(rows)
            for n, nk in enumerate(plan["groups"][jn])
            for m, mk in enumerate(plan["groups"][im])
            if mk in virtual and nk in virtual
        ]
        index = np.asarray(coordinates).T
        coordinate_count = len(coordinates)
        shape = (dims["NXX"], dims["NXX"], nvh)
        zs = np.zeros(dims["NX"], order="F")
        value = float(eps[labels.index((*key, -2 * abs(key[1]) + 1))] + 0.075)

        def unpack(vector, shape=shape, index=index):
            d = np.zeros(shape, order="F")
            d[tuple(index)] = vector
            return d

        def evaluate(
            energy,
            vector,
            prime=None,
            *,
            unpack=unpack,
            size=coordinate_count,
            zs=zs,
            shape=shape,
            index=index,
        ):
            d = unpack(vector)
            dd = unpack(np.zeros(size) if prime is None else prime)
            rs, rd, ps, pd = (
                np.zeros_like(zs),
                np.zeros(shape, order="F"),
                np.zeros_like(zs),
                np.zeros(shape, order="F"),
            )
            mapping(energy, zs, d, zs, dd, rs, rd, ps, pd)
            if not all(np.all(np.isfinite(a)) for a in (rs, rd, ps, pd)):
                raise ValueError("nonfinite complete native valence map")
            return rs, rd[tuple(index)], ps, pd[tuple(index)]

        def solve_native(energy, evaluate=evaluate, size=coordinate_count, key=key):
            solved, op, residual, inverse, matrix = reduced_solver(
                energy, size, evaluate
            )
            if not np.all(np.isfinite(solved)) or np.max(abs(residual)) > 1e-10:
                np.savez_compressed(
                    here / f"failure_native_{key[0]}_{key[1]}.npz",
                    energy=energy,
                    vector=solved,
                    residual=residual,
                )
                raise AssertionError("complete noncanonical valence residual failed")
            return solved, op, residual, inverse, matrix

        def oracle(energy, sigma, column):
            os_, od = magnetic_action(
                direct, eps, core_s, core_d, column, energy, sigma
            )
            fs, fd, _, _ = magnetic_fock(
                fock, eps, core_s, core_d, column, energy, sigma, np.zeros_like(sigma)
            )[0]
            return os_ + fs, od + fd

        def solve_oracle(energy, column, oracle=oracle, key=key):
            solved, residual, op, inverse, conditioning = magnetic_solver(
                labels, problem.integrals.noccupied, column, energy, oracle
            )
            if not np.all(np.isfinite(solved)) or np.max(abs(residual)) > 1e-10:
                np.savez_compressed(
                    here / f"failure_oracle_{key[0]}_{key[1]}_{column}.npz",
                    energy=energy,
                    vector=solved,
                    residual=residual,
                )
                raise AssertionError(
                    "complete independent magnetic valence residual failed"
                )
            return solved, residual, op, inverse, conditioning

        random = rng.normal(size=len(coordinates)) * 0.001
        action = evaluate(value, random)
        root, op, residual, native_inverse, native_matrix = solve_native(value)
        rhs = evaluate(value, root)[3]
        prime = native_inverse(rhs)
        prime_residual = op @ prime - rhs
        if not np.all(np.isfinite(prime)) or np.max(abs(prime_residual)) > 1e-10:
            raise AssertionError("complete native numerator-response equation failed")
        total = evaluate(value, root, prime)[2]
        row = dict(
            key=key,
            energy_Ha=value,
            raw_residual_Ha=float(np.max(abs(residual))),
            derivative_residual_Ha=float(np.max(abs(prime_residual))),
            multiplets=[],
        )
        vectors = dict(
            direct=direct,
            core_s=core_s,
            core_d=core_d,
            random=random,
            root=root,
            prime=prime,
            residual=residual,
            derivative_residual=prime_residual,
            sigma1=evaluate(value, root)[0],
            sigma1_prime=total,
            native_matrix=native_matrix,
        )
        for mj in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
            column = labels.index((*key, mj))
            os_, od = oracle(
                value,
                expand_valence(problem, plan, rows, unpack(random), column),
                column,
            )
            nd = expand_valence(problem, plan, rows, unpack(action[1]), column)
            singles = [
                (i, labels.index((*k, mj)))
                for i, k in enumerate(plan["groups"][partial])
                if k in virtual
            ]
            action_error = max(
                float(np.max(abs(nd - od))),
                *(abs(action[0][i] - os_[p]) for i, p in singles),
            )
            native = expand_valence(problem, plan, rows, unpack(root), column)
            independent, ores, oracle_op, oracle_inverse, conditioning = solve_oracle(
                value, column
            )
            root_error = float(np.max(abs(native - independent)))
            zero_m = np.zeros_like(independent)

            def derivative_action(sigma, direction, column=column, value=value):
                es, ed = energy_derivative(
                    direct, eps, core_s, core_d, column, value, sigma, direction
                )
                _, _, fs, fd = magnetic_fock(
                    fock, eps, core_s, core_d, column, value, sigma, direction
                )[0]
                return es + fs, ed + fd

            derivative_rhs = derivative_action(independent, zero_m)[1]
            oracle_prime = oracle_inverse(derivative_rhs)
            oracle_prime_residual = oracle_op @ oracle_prime.reshape(
                -1
            ) - derivative_rhs.reshape(-1)
            oracle_prime = oracle_prime.reshape(independent.shape)
            native_prime = expand_valence(problem, plan, rows, unpack(prime), column)
            prime_error = float(np.max(abs(native_prime - oracle_prime)))
            analytic_single = derivative_action(independent, oracle_prime)[0]
            analytic_error = max(abs(total[i] - analytic_single[p]) for i, p in singles)
            vectors[f"native_prime_{mj}"] = native_prime
            vectors[f"oracle_prime_{mj}"] = oracle_prime
            vectors[f"oracle_prime_residual_{mj}"] = oracle_prime_residual
            vectors[f"oracle_sigma1_prime_{mj}"] = analytic_single
            derivatives = []
            for step in (1e-3, 3e-4, 1e-4):
                plus = solve_oracle(value + step, column)[0]
                minus = solve_oracle(value - step, column)[0]
                coarse = (
                    oracle(value + step, plus, column)[0]
                    - oracle(value - step, minus, column)[0]
                ) / (2 * step)
                levels = [coarse]
                for level in (1, 2, 3):
                    h = step / 2**level
                    lp = solve_oracle(value + h, column)[0]
                    lm = solve_oracle(value - h, column)[0]
                    estimate = (
                        oracle(value + h, lp, column)[0]
                        - oracle(value - h, lm, column)[0]
                    ) / (2 * h)
                    row_levels = [estimate]
                    for order in range(1, level + 1):
                        row_levels.append(
                            (4**order * row_levels[-1] - levels[order - 1])
                            / (4**order - 1)
                        )
                    levels = row_levels
                difference = levels[-1]
                errors = np.asarray([abs(total[i] - difference[p]) for i, p in singles])
                tolerance = np.asarray(
                    [1e-9 + 1e-6 * abs(total[i]) for i, _ in singles]
                )
                derivatives.append(
                    dict(
                        step_Ha=step,
                        max_error=float(np.max(errors)),
                        second_order_error=float(
                            max(abs(total[i] - coarse[p]) for i, p in singles)
                        ),
                        passed=bool(np.all(errors <= tolerance)),
                    )
                )
                vectors[f"derivative_{mj}_{step}"] = difference
                vectors[f"second_order_derivative_{mj}_{step}"] = coarse
            vectors[f"oracle_root_{mj}"] = independent
            vectors[f"native_root_{mj}"] = native
            vectors[f"oracle_residual_{mj}"] = ores
            row["multiplets"].append(
                dict(
                    two_m=mj,
                    signed_action_error_Ha=float(action_error),
                    signed_root_error_Ha=root_error,
                    oracle_residual_Ha=float(np.max(abs(ores))),
                    signed_response_error_Ha=prime_error,
                    sigma1_analytic_derivative_error=float(analytic_error),
                    oracle_response_residual_Ha=float(
                        np.max(abs(oracle_prime_residual))
                    ),
                    conditioning=conditioning,
                    derivatives=derivatives,
                )
            )
            np.savez_compressed(here / f"vectors_{key[0]}_{key[1]}.npz", **vectors)
            records_current = records + [row]
            (here / "comparison.json").write_text(
                json.dumps(
                    dict(
                        core_residual_Ha=float(np.max(abs(core_residual))),
                        direct_error_Ha=direct_error,
                        records=records_current,
                    ),
                    indent=2,
                )
                + "\n"
            )
            if (
                max(action_error, root_error, prime_error, analytic_error) > 1e-11
                or np.max(abs(oracle_prime_residual)) > 1e-10
                or not all(d["passed"] for d in derivatives)
            ):
                raise AssertionError(
                    "complete signed noncanonical action/root/response gate failed; evidence retained"
                )
        records.append(row)
    print(
        json.dumps(
            dict(
                passed=True,
                core_residual_Ha=float(np.max(abs(core_residual))),
                records=records,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
