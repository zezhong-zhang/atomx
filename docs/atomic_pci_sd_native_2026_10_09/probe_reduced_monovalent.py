"""Source-bound complete finite-M validation of the reduced standalone SD API."""

import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
OUT = OWNER / "reduced_monovalent008"


def freeze():
    OUT.mkdir(exist_ok=False)
    contract = json.loads((OWNER / "reduced_monovalent005/contract.json").read_text())
    contract.update(
        panel="four original/streamed/virtual-rotated/occupied-rotated native twelve-spinor fixtures; four valence columns and all ten M per fixture",
        angular_gate=1e-9,
        normalized_state_gate=1e-11,
        numerical_solve_tolerance_Ha=1e-14,
        retained_failure="006/007 passed original raw/amplitude gates but unrestricted-M normalized-state comparison failed at 1.3624e-11 even after tightening the numerical solve; retain that negative result and compare an independently declared complete exact-total-J finite oracle",
        angular_oracle="Complete fixed-M determinant excitation space is decomposed by independently constructed J^2; retain every eigenvector in the exact target J irrep. No small-amplitude selection or Hamiltonian symmetrization. Retain full unprojected raw residual and unrestricted-state discrepancy separately.",
        resources="256MiB complete per-fixture bound; 40 retained Krylov vectors; exact scalar equations, no magnetic tensor in production",
        load=os.getloadavg(),
        scope="finite signed diagram/root/state/J certification; no full Yb, production normalized state/property, GPU or experimental precision claim",
    )
    (OUT / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    files = (
        list((ROOT / "src/atomx").rglob("*.py"))
        + list((ROOT / "src/atomx/methods").glob("_pci_*.f90"))
        + list((ROOT / "tools").glob("build_pci_*.py"))
        + [
            ROOT / "tests/test_monovalent_sd.py",
            ROOT / "tests/test_reduced_monovalent_sd.py",
            Path(__file__).resolve(),
        ]
    )
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = OUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def independent_j_root(helper, finite, jp, masks, m):
    """Independent dense diagram pencil, decomposed by exact angular symmetry."""
    count = finite.plan["amplitudes"]
    reference = masks.index(finite.reference_mask)
    basis = np.zeros((len(masks), count + 1))
    basis[reference, 0] = 1
    for i, (_, _, operations) in enumerate(finite.excitations):
        target, sign = helper["dense_linked"].__globals__["elementary"](
            finite.reference_mask, operations
        )
        basis[masks.index(target), i + 1] = sign
    j, mj = abs(finite.integrals.labels[finite.valence_index][1]) - 0.5, m / 2
    angular = basis.T @ jp.T @ jp @ basis + mj * (mj + 1) * np.eye(count + 1)
    eig, vectors = np.linalg.eigh(angular)
    subspace = vectors[:, abs(eig - j * (j + 1)) < 1e-10]
    if not subspace.shape[1]:
        raise AssertionError("complete target J irrep not found")
    constant, _ = helper["dense_linked"](finite, np.zeros(count))
    linear = np.empty((count + 1, count))
    for i in range(count):
        unit = np.zeros(count)
        unit[i] = 1
        value, _ = helper["dense_linked"](finite, unit)
        linear[:, i] = value - constant
    pencil = np.column_stack((constant, linear))
    pencil[0, 0] -= finite.orbital_energy_Ha
    diagonal = np.diag(np.r_[0, finite.diagonal_gaps_Ha])
    previous = subspace.T[:, 0]
    previous /= np.linalg.norm(previous)
    for coupling in np.linspace(0.125, 1, 8):
        matrix = subspace.T @ (diagonal + coupling * (pencil - diagonal)) @ subspace
        energies, roots = np.linalg.eig(matrix)
        selected = int(np.argmax(abs(previous @ roots) / np.linalg.norm(roots, axis=0)))
        delta, root = energies[selected], roots[:, selected]
        if abs(delta.imag) > 1e-12 or np.max(abs(root.imag)) > 1e-12:
            raise AssertionError("complex independent angular root")
        previous = root.real / np.linalg.norm(root.real)
    physical = subspace @ root.real
    physical /= physical[0]
    raw = pencil @ physical - delta.real * physical
    return float(delta.real), physical[1:], basis[:, 1:] @ physical[1:], raw, subspace


def main():
    sys.path.insert(0, str(ROOT / "tests"))
    helper = runpy.run_path(str(ROOT / "tests/test_reduced_monovalent_sd.py"))
    records, arrays, resources = [], {}, {}

    class Factory:
        def mktemp(self, name):
            path = OUT / name
            path.mkdir(exist_ok=False)
            return path

    for variant in ("original", "streamed", "rotated", "occupied_rotated"):
        core, core_result, problem = helper["reduced_core"].__wrapped__(
            Factory(), SimpleNamespace(param=variant)
        )
        labels = core.integrals.labels
        arrays[variant + "_labels"] = np.asarray(labels)
        arrays[variant + "_core_amplitudes"] = core_result.amplitudes
        arrays[variant + "_fock_Ha"] = core.integrals.fock_Ha
        arrays[variant + "_one_body_Ha"] = core.integrals.one_body_Ha
        resources[variant] = dict(
            complete_payload_bound_bytes=problem.plan["complete_payload_bound_bytes"],
            native_static_bytes=problem.plan["standalone_static_bytes"],
            columns={str(k): v for k, v in problem.plan["amplitudes"].items()},
            provenance=dict(core_result.provenance),
        )
        for key in problem.plan["valence_keys"]:
            reduced = problem.solve(key, residual_tolerance_Ha=1e-14)
            for m in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                index = labels.index((*key, m))
                finite = helper["FiniteMonovalentSD"](
                    core, core_result, index, work_cap=200_000_000
                )
                reference = finite.solve()
                physical = helper["expanded"](problem, key, reduced.amplitudes, finite)
                independent, (masks, h, tc, sv) = helper["dense_linked"](
                    finite, physical
                )
                energy, rhs = problem.evaluate(key, reduced.amplitudes)
                projected = (
                    helper["expanded"](problem, key, rhs, finite)
                    + finite.diagonal_gaps_Ha * physical
                )
                raw = independent[1:] - reduced.correlation_energy_Ha * physical
                random = (
                    np.random.default_rng(962341).normal(size=len(reduced.amplitudes))
                    * 0.013
                )
                random_physical = helper["expanded"](problem, key, random, finite)
                random_energy, random_rhs = problem.evaluate(key, random)
                random_independent, _ = helper["dense_linked"](finite, random_physical)
                random_native = (
                    helper["expanded"](problem, key, random_rhs, finite)
                    + finite.diagonal_gaps_Ha * random_physical
                )
                psi = (np.eye(len(masks)) + tc + sv)[
                    :, masks.index(finite.reference_mask)
                ]
                psi /= np.linalg.norm(psi)
                raising = []
                for p, (nr, k, m2) in enumerate(labels):
                    target = (nr, k, m2 + 2)
                    if target in labels:
                        j, mj = abs(k) - 0.5, m2 / 2
                        raising.append(
                            (
                                np.sqrt(j * (j + 1) - mj * (mj + 1)),
                                ((labels.index(target), True), (p, False)),
                            )
                        )
                _, jp = helper["full_operator"](
                    len(labels), core.integrals.noccupied + 1, raising
                )
                j, mj = abs(key[1]) - 0.5, m / 2
                jdefect = jp.T @ jp @ psi + (mj * (mj + 1) - j * (j + 1)) * psi
                actual = dict(
                    zip(reference.determinants, reference.coefficients, strict=True)
                )
                reconstructed = np.asarray([actual.get(mask, 0) for mask in masks])
                jdelta, jphysical, jexcited, jraw, jsubspace = independent_j_root(
                    helper, finite, jp, masks, m
                )
                jstate = (np.eye(len(masks)) + tc)[
                    :, masks.index(finite.reference_mask)
                ] + jexcited
                jstate /= np.linalg.norm(jstate)
                errors = dict(
                    random_map_Ha=float(
                        np.max(abs(random_native - random_independent[1:]))
                    ),
                    random_energy_Ha=float(
                        abs(
                            random_energy
                            + finite.orbital_energy_Ha
                            - random_independent[0]
                        )
                    ),
                    map_at_root_Ha=float(np.max(abs(projected - independent[1:]))),
                    root_Ha=float(
                        abs(
                            reduced.correlation_energy_Ha
                            - reference.correlation_energy_Ha
                        )
                    ),
                    physical_amplitudes=float(
                        np.max(abs(physical - reference.amplitudes))
                    ),
                    independent_raw_Ha=float(np.max(abs(raw))),
                    independent_energy_Ha=float(
                        abs(independent[0] - reduced.attachment_energy_Ha)
                    ),
                    native_raw_Ha=reduced.residual_max_Ha,
                    native_energy_Ha=reduced.energy_residual_Ha,
                    normalized_state=float(np.max(abs(psi - reconstructed))),
                    exact_J_state=float(np.max(abs(psi - jstate))),
                    exact_J_root_Ha=float(abs(reduced.correlation_energy_Ha - jdelta)),
                    exact_J_amplitudes=float(np.max(abs(physical - jphysical))),
                    exact_J_full_raw_Ha=float(np.max(abs(jraw))),
                    J2=float(np.max(abs(jdefect))),
                )
                passed = (
                    max(
                        errors[k]
                        for k in (
                            "random_map_Ha",
                            "random_energy_Ha",
                            "map_at_root_Ha",
                            "root_Ha",
                            "exact_J_state",
                            "exact_J_root_Ha",
                        )
                    )
                    < 1e-11
                    and max(
                        errors[k]
                        for k in (
                            "physical_amplitudes",
                            "independent_raw_Ha",
                            "independent_energy_Ha",
                            "exact_J_amplitudes",
                            "exact_J_full_raw_Ha",
                        )
                    )
                    < 1e-10
                    and max(errors[k] for k in ("native_raw_Ha", "native_energy_Ha"))
                    < 1e-12
                    and errors["J2"] < 1e-9
                )
                records.append(
                    dict(
                        variant=variant,
                        key=key,
                        m2=m,
                        errors=errors,
                        passed=bool(passed),
                        unrestricted_normalized_state_gate_passed=errors[
                            "normalized_state"
                        ]
                        < 1e-11,
                        attachment_energy_Ha=reduced.attachment_energy_Ha,
                        finite_attachment_energy_Ha=reference.attachment_energy_Ha,
                        full_eigenstate_residual_Ha=reference.full_eigenstate_residual_Ha,
                        continuation=reduced.continuation_path,
                        matvecs=reduced.matrix_vector_products,
                    )
                )
                tag = f"{variant}_{key[0]}_{key[1]}_{m}_"
                for name, value in dict(
                    reduced=reduced.amplitudes,
                    denominators=reduced.denominators_Ha,
                    physical=physical,
                    finite=reference.amplitudes,
                    independent=independent,
                    native=projected,
                    raw=raw,
                    masks=masks,
                    state=psi,
                    J2=jdefect,
                    exact_J_state=jstate,
                    exact_J_physical=jphysical,
                    exact_J_full_raw=jraw,
                    exact_J_subspace=jsubspace,
                    random=random,
                    random_physical=random_physical,
                    random_independent=random_independent,
                    random_native=random_native,
                    finite_source=finite.source_Ha,
                    finite_linear=finite.linear_map_Ha,
                ).items():
                    arrays[tag + name] = np.asarray(value)
                arrays[variant + "_full_H"] = h
                arrays[variant + "_core_T"] = tc
                if not passed:
                    np.savez_compressed(OUT / "arrays.npz", **arrays)
                    (OUT / "result.json").write_text(
                        json.dumps(
                            dict(records=records, resources=resources, passed=False),
                            indent=2,
                        )
                        + "\n"
                    )
                    raise AssertionError(
                        "unchanged reduced standalone full-M gates failed"
                    )
        np.savez_compressed(OUT / "arrays.npz", **arrays)
        (OUT / "result.json").write_text(
            json.dumps(
                dict(records=records, resources=resources, passed=True), indent=2
            )
            + "\n"
        )
    print(
        json.dumps(dict(states=len(records), arrays=len(arrays), passed=True)),
        flush=True,
    )


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    else:
        main()
