"""Exact same-model redundant-rho pair-exchange domain and quotient audit."""

import json
import resource
import sys

import numpy as np
from probe import OUT, ROOT, digest, terms_from_integrals

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from test_monovalent_sd import full_operator  # noqa: E402

from atomx.methods.core_sd import CoreSD  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.methods.monovalent_sd import _core_terms  # noqa: E402


def main():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    contract = json.loads((OUT / "quotient_contract.json").read_text())
    assert all(
        digest((OUT if p == "probe.py" else ROOT) / p) == h
        for p, h in contract["source_sha256"].items()
    )
    radial = DFSolver(
        Z=6, subshells=[Subshell(1, -1, 2), Subshell(2, -1, 2)], ns=30, ks=7
    ).solve()
    occupied = ((1, -1), (2, -1))
    virtual = ((3, -1), (4, -1), (2, -2))
    orbitals = {key: radial.orbitals_by_subshell()[key][:2] for key in occupied}
    for k in (-1, -2):
        _, states = build_virtual_basis(
            radial, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[: 2 if k == -1 else 1]):
            orbitals[(i + 3 if k == -1 else i + 2), k] = state[:2]
    core = CoreSD(
        orbitals,
        radial._bsp_data,
        6,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=OUT / "four-core-run002/backend",
        nuclear_potential=radial._nuclear_potential,
        radial_representation="polynomial",
        memory_bytes=256 * 1024**2,
    )
    solved = core.solve(residual_tolerance_Ha=1e-12)
    n, o = len(core.integrals.labels), core.integrals.noccupied
    masks, h = full_operator(n, o, terms_from_integrals(core.integrals))
    ref = (1 << o) - 1
    idx = masks.index(ref)
    qsd = np.array([0 < (ref & ~mask).bit_count() <= 2 for mask in masks])
    count = len(core.source)
    K = np.empty((count, count))
    B = np.empty((len(masks), count))
    L = np.empty_like(B)
    for i in range(count):
        unit = np.eye(count)[:, i]
        K[:, i] = core.action(unit)
        _, T = full_operator(n, o, _core_terms(core, unit))
        B[:, i] = T[:, idx]
        L[:, i] = h @ T[:, idx] - T @ h[:, idx]
    P = np.eye(count)
    cursor = 0
    meta = []
    for a in core.plan["occupied_keys"]:
        group = next(g for g in core.plan["groups"] if a in g)
        for m in group:
            if m in virtual:
                meta.append(dict(index=cursor, single=[list(m), list(a)]))
                cursor += 1
    for a, b, im, jn, rank in core.plan["channels"]:
        ms = [key for key in core.plan["groups"][im] if key in virtual]
        ns = [key for key in core.plan["groups"][jn] if key in virtual]
        indices = np.arange(cursor, cursor + len(ms) * len(ns)).reshape(
            (len(ms), len(ns)), order="F"
        )
        for j in range(len(ns)):
            for i in range(len(ms)):
                ix = int(indices[i, j])
                meta.append(
                    dict(
                        index=ix,
                        holes=[a, b],
                        particles=[list(ms[i]), list(ns[j])],
                        groups=[im, jn],
                        rank=rank,
                    )
                )
                if a == b and im == jn:
                    P[ix, :] = 0
                    P[ix, ix] += 0.5
                    P[ix, int(indices[j, i])] += 0.5
        cursor += len(ms) * len(ns)
    assert cursor == count
    raw = np.load(OUT / "localization_arrays.npz")["random_redundant_t"]
    mismatch = B @ K + L
    S = B @ P
    _, singular, vh = np.linalg.svd(S, full_matrices=True)
    threshold = max(S.shape) * np.finfo(float).eps * np.max(singular, initial=0)
    rank = int(np.sum(singular > threshold))
    null = P @ vh[rank:].T
    result = dict(
        source_sha256=digest(__file__),
        raw_amplitudes=count,
        physical_lift_rank=rank,
        singular_values=singular.tolist(),
        rank_threshold=threshold,
        pair_constraint_rank=int(round(np.trace(P))),
        raw_signed_linear_error_Ha=float(np.max(abs(mismatch[qsd]))),
        constrained_complete_signed_linear_error_Ha=float(
            np.max(abs((mismatch @ P)[qsd]))
        ),
        raw_random_error_Ha=float(np.max(abs((mismatch @ raw)[qsd]))),
        symmetrized_random_error_Ha=float(np.max(abs((mismatch @ P @ raw)[qsd]))),
        constrained_nullspace_action_error_Ha=float(
            np.max(abs((B @ K @ null)[qsd]), initial=0)
        ),
        source_pair_constraint_error_Ha=float(
            np.max(abs(core.source - P @ core.source))
        ),
        solved_pair_constraint_error=float(
            np.max(abs(solved.amplitudes - P @ solved.amplitudes))
        ),
        operator_pair_constraint_error_Ha=float(
            np.max(abs((np.eye(count) - P) @ K @ P))
        ),
        backend_sha256=core._manifest["files"]["native_core_sd.so"],
        scope="complete constrained finite34coordinate map and null directions; not Yb/full-rank physics completeness",
    )
    np.savez_compressed(
        OUT / "quotient_arrays.npz",
        K_Ha=K,
        B=B,
        L_Ha=L,
        P=P,
        H_Ha=h,
        source_Ha=core.source,
        denominators_Ha=core.denominators,
        solved_t=solved.amplitudes,
        raw_random_t=raw,
        masks=np.array(masks, dtype=np.uint64),
        qsd=qsd,
    )
    (OUT / "quotient_metadata.json").write_text(
        json.dumps(dict(labels=core.integrals.labels, coordinates=meta), indent=2)
        + "\n"
    )
    (OUT / "quotient.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
