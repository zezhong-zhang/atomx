"""Public full noncanonical valence API against independent magnetic equations."""

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


def main():
    here = OWNER / "public_valence001"
    here.mkdir(exist_ok=False)
    archive = here / "source"
    names = (
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
        ROOT / "src/atomx/methods/core_sd.py",
        ROOT / "tools/build_pci_sd_backend.py",
        ROOT / "tools/build_pci_valence_backend.py",
    )
    identity = {}
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
                hypothesis="Public complete fixed-energy noncanonical SD reproduces independent long-double magnetic roots/actions/implicit responses with matched native DC integrals.",
                fixed="C2+ all1s2/2s2 core, six declared s/p radial virtuals and all16magnetic states, Z6/ns40/ks7; complete native physicalFock, .37rad rotation of each two-virtual partial-wave pair; no synthetic operator or fitted energy",
                energy="each actual Fock diagonal eps_v+.075, core numerators fixed for derivative",
                gates_Ha=dict(
                    action=1e-11,
                    signed_root=1e-11,
                    signed_prime=1e-11,
                    sigma1_prime=1e-11,
                    actual_core_and_valence_residual=1e-10,
                ),
                resource="512MiB complete core/valence/Krylov preflight plus bounded64MiB finite reference, one worker",
                load=os.getloadavg(),
                stop="failure retains full arrays/source; no gate or space reduction",
                scope="public equation and physical identity proof in declared finite space; not Yb/experimental convergence",
            ),
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

    core, orbitals, plan, rows, _, _, problem, fock = prepare(
        here, REF, backend_builder=builder, fock_probe=False, virtual_rotation=0.37
    )
    core_result = problem.solve(residual_tolerance_Ha=1e-12)
    valence = ValenceSD(problem, core_result, valence_keys=plan["virtual_keys"])
    if (
        valence.plan["complete_payload_bound_bytes"] + 64 * 1024**2
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
        result = valence.solve(key, energy, residual_tolerance_Ha=1e-11)
        random = rng.normal(size=count) * 0.001
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
    main()
