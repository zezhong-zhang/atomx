"""Independent complete finite-space core-cluster/Bloch operator diagnosis.

An exponential of the exact two-electron core cluster removes disconnected
core excitations. Intermediate normalization then fixes the eight-column
attachment wave operator. No Fov deletion or Hamiltonian change is permitted.
This determinant oracle is deliberately finite, not a scalable production CC.
"""

import argparse
import hashlib
import itertools
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
from test_transverse_spinor_integrals import apply, determinant_matrix  # noqa: E402

from atomx.methods.fscc import load_integrals  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excitation_matrix(masks, operators):
    index = {mask: i for i, mask in enumerate(masks)}
    result = np.zeros((len(masks), len(masks)), complex)
    for operations, coefficient in operators:
        for column, mask in enumerate(masks):
            target, sign = apply(mask, operations)
            if sign:
                result[index[target], column] += coefficient * sign
    return result


def diagnose(integrals, target=None):
    if integrals.noccupied != 2 or len(integrals.labels) > 12:
        raise ValueError(
            "oracle requires two occupied spinors and at most twelve total"
        )
    masks2, h2 = determinant_matrix(integrals, 2)
    masks3, h3 = determinant_matrix(integrals, 3)
    eigen2, vectors2 = np.linalg.eigh(h2)
    reference = 3
    ref_index = masks2.index(reference)
    ground = vectors2[:, 0] / vectors2[ref_index, 0]
    if abs(vectors2[ref_index, 0]) < 0.5:
        raise ValueError("reference overlap too small for declared core branch")
    singles = []
    for i in range(2):
        for p in range(2, len(integrals.labels)):
            operations = ((p, True), (i, False))
            mask, sign = apply(reference, operations)
            singles.append((operations, ground[masks2.index(mask)] / sign))
    t1 = excitation_matrix(masks2, singles)
    double_remainder = (
        ground
        - np.eye(len(masks2))[:, ref_index]
        - t1[:, ref_index]
        - (t1 @ t1)[:, ref_index] / 2
    )
    doubles = []
    for p, q in itertools.combinations(range(2, len(integrals.labels)), 2):
        operations = ((p, True), (q, True), (1, False), (0, False))
        mask, sign = apply(reference, operations)
        doubles.append((operations, double_remainder[masks2.index(mask)] / sign))
    tcore2 = excitation_matrix(masks2, singles + doubles)
    exp2 = expm(tcore2)
    core_reconstruction = float(np.max(abs(exp2[:, ref_index] - ground)))
    tcore3 = excitation_matrix(masks3, singles + doubles)
    exp3, inv3 = expm(tcore3), expm(-tcore3)
    hbar = inv3 @ h3 @ exp3
    roots, vectors = np.linalg.eigh(h3)
    active = list(range(2, len(integrals.labels)))
    selected = [masks3.index(reference | (1 << v)) for v in active]
    transformed = inv3 @ vectors[:, : len(active)]
    projection = transformed[selected, :]
    condition = float(np.linalg.cond(projection))
    if condition > 1e6:
        raise ValueError("low-root model-space projector is ill-conditioned")
    omega = transformed @ np.linalg.inv(projection)
    heff = (hbar @ omega)[selected, :] - eigen2[0] * np.eye(len(active))
    residual = hbar @ omega - omega @ (heff + eigen2[0] * np.eye(len(active)))
    normalization = float(np.max(abs(omega[selected, :] - np.eye(len(active)))))
    operator_residual = float(np.max(abs(residual)))
    defect = float(
        np.max(
            abs(
                np.sort(np.linalg.eigvals(heff).real + eigen2[0]) - roots[: len(active)]
            )
        )
    )
    metrics = dict(
        core_energy_Ha=float(eigen2[0]),
        core_reconstruction_max=core_reconstruction,
        model_projector_condition=condition,
        normalization_max=normalization,
        bloch_equation_max_Ha=operator_residual,
        fci_root_max_Ha=defect,
        fov_max_Ha=float(np.max(abs(integrals.fock_Ha[:2, 2:]))),
        oracle_gate_pass=bool(
            max(operator_residual, defect) <= 1e-11
            and max(core_reconstruction, normalization) <= 1e-12
        ),
        scope="Complete finite determinant operator diagnosis; no production amplitude repair or experimental accuracy claim",
        highest_level_reached=False,
    )
    arrays = dict(
        full_h2_Ha=h2,
        full_h3_Ha=h3,
        core_cluster_2=tcore2,
        core_cluster_3=tcore3,
        hbar3_Ha=hbar,
        omega=omega,
        exact_heff_Ha=heff,
        bloch_residual_Ha=residual,
        exact_low_roots_Ha=roots[: len(active)],
        model_determinant_indices=selected,
    )
    if target is not None:
        difference = target - heff
        metrics.update(
            backend_heff_max_Ha=float(np.max(abs(difference))),
            backend_diagonal_defects_Ha=difference.diagonal().real.tolist(),
            backend_strict_1e10_gate_pass=bool(np.max(abs(difference)) <= 1e-10),
        )
        arrays["backend_minus_exact_heff_Ha"] = difference
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    owner = Path(__file__).resolve().parent
    source = (
        owner
        / "transverse_integrals004/test_tmp/test_live_finite_frequency_fsc0/fsccsdt"
    )
    target = owner / "transverse_precision007/tight/model_space_matrix.npz"
    paths = [
        Path(__file__),
        REPO / "tests/test_transverse_spinor_integrals.py",
        target,
        *[p for p in source.iterdir() if p.is_file()],
    ]
    binding = {str(p.relative_to(REPO)): sha(p) for p in paths}
    contract = dict(
        hypothesis="A complete independent Bloch effective operator distinguishes a model-Hamiltonian defect from state sorting and iterative residuals, while keeping nonzero Fov unchanged.",
        fixed="Saved complete signed 10-spinor 2-core/8-valence Hamiltonian; exact core exponential; all 120 three-electron determinants and 8 low roots.",
        gates=dict(
            oracle_Ha=1e-11, core_and_normalization=1e-12, backend_strict_Ha=1e-10
        ),
        stop="One complete unchanged-H diagnosis; 2 GiB address space; no backend source edits or fitted shift.",
        source_and_inputs=binding,
        load=os.getloadavg(),
    )
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    try:
        integrals = load_integrals(source)
        matrix = np.load(target)["matrix_Ha"]
        metrics, arrays = diagnose(integrals, matrix)
        assert metrics["oracle_gate_pass"]
        assert all(sha(REPO / p) == expected for p, expected in binding.items())
        np.savez_compressed(args.output / "full_arrays.npz", **arrays)
        metrics.update(
            inputs_unchanged=True, arrays_sha256=sha(args.output / "full_arrays.npz")
        )
        (args.output / "result.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(dict(error=type(error).__name__, message=str(error)))
        )
        raise


if __name__ == "__main__":
    main()
