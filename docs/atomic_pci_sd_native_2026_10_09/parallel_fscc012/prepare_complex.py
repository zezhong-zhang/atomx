"""Prepare a declared complex occupied/virtual unitary representation control."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rotation", choices=("complex_block", "axial_phase"), required=True
    )
    args = parser.parse_args()
    from atomx.methods.fscc import export_integrals, load_integrals
    from atomx.physics.csf.spinor_integrals import SpinorIntegrals

    a = load_integrals(probe.DEFAULT_CASE)
    u = np.eye(10, dtype=complex)
    if args.rotation == "complex_block":
        rng = np.random.default_rng(83)
        for begin, end in ((0, 2), (2, 10)):
            z = rng.normal(size=(end - begin, end - begin)) + 1j * rng.normal(
                size=(end - begin, end - begin)
            )
            u[begin:end, begin:end] = np.linalg.qr(z)[0]
    else:
        u = np.diag(np.exp(1j * np.linspace(0.1, 2.3, 10)))
    dense = np.array(
        [a.eri(*idx) for idx in itertools.product(range(10), repeat=4)]
    ).reshape((10,) * 4)
    rotated = np.einsum(
        "pa,qb,pqrs,rc,sd->abcd", u.conj(), u.conj(), dense, u, u, optimize=True
    )
    packed = np.array(
        [
            rotated[p, q, r, s]
            for p in range(9)
            for q in range(p + 1, 10)
            for r in range(p, 9)
            for s in range(r + 1, 10)
        ]
    )

    def hermitian(x):
        y = u.conj().T @ x @ u
        return 0.5 * (y + y.conj().T)

    b = SpinorIntegrals(
        a.labels,
        2,
        hermitian(a.one_body_Ha),
        hermitian(a.fock_Ha),
        packed,
        a.frozen_energy_Ha,
        a.reference_energy_Ha,
        dict(
            a.identity,
            representation="explicit complex occupied/virtual rotation",
            rotation=args.rotation,
            parent_integral_identity=dict(a.identity),
        ),
    )
    export_integrals(b, args.output)
    np.savez_compressed(args.output / "rotation.npz", unitary=u)
    oracle = probe.load_oracle()
    masks, h = oracle.determinant_matrix(a, 3)
    _, rotated_h = oracle.determinant_matrix(b, 3)
    occupied = [tuple(i for i in range(10) if mask >> i & 1) for mask in masks]
    induced = np.array(
        [
            [np.linalg.det(u[np.ix_(left, right)]) for right in occupied]
            for left in occupied
        ]
    )
    residual = float(np.max(abs(rotated_h - induced.conj().T @ h @ induced)))
    result = dict(
        rotation=args.rotation,
        same_operator_unitary=True,
        complete_120_by_120_unitary_transform_max_Ha=residual,
        input_files_sha256={
            name: probe.sha(args.output / name)
            for name in ("ATOMX_ONE", "ATOMX_TWO", "rotation.npz")
        },
        parent_input_files_sha256={
            name: probe.sha(probe.DEFAULT_CASE / name)
            for name in ("ATOMX_ONE", "ATOMX_TWO")
        },
        source_sha256=probe.sha(Path(__file__)),
    )
    (args.output / "rotation_identity.json").write_text(json.dumps(result, indent=2))
    if residual > 1e-11:
        raise ValueError("complex representation changed the complete Hamiltonian")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
