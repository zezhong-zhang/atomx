"""Retained four-core mismatch localization without production mutation."""

import json
import resource
import sys
import time

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
    start = time.perf_counter()
    contract = json.loads((OUT / "localization_contract.json").read_text())
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
    ranks = np.array([(ref & ~mask).bit_count() for mask in masks])
    rows = []
    arrays = {
        "H_Ha": h,
        "core_source_Ha": core.source,
        "core_denominators_Ha": core.denominators,
        "masks": np.array(masks, dtype=np.uint64),
    }
    for name, t in [
        (
            "random_redundant",
            np.random.default_rng(751386).normal(size=core.source.shape) * 0.013,
        ),
        ("physical_first_Born", core.source / core.denominators),
        ("actual_solved", solved.amplitudes),
    ]:
        _, tc = full_operator(n, o, _core_terms(core, t))
        expected = h[:, idx] + h @ tc[:, idx] - tc @ h[:, idx]
        native = core.action(t) - core.source
        _, native_op = full_operator(n, o, _core_terms(core, native))
        delta = expected + native_op[:, idx]
        selected = np.flatnonzero((ranks > 0) & (ranks <= 2))
        worst = selected[np.argmax(abs(delta[selected]))]
        row = dict(
            direction=name,
            complete_signed_action_error_Ha=float(max(abs(delta[selected]))),
            independent_QSD_residual_Ha=float(max(abs(expected[selected]))),
            native_raw_residual_Ha=float(max(abs(native))),
            single_error_Ha=float(np.max(abs(delta[ranks == 1]), initial=0)),
            double_error_Ha=float(np.max(abs(delta[ranks == 2]), initial=0)),
            energy_error_Ha=abs(core.energy(t) - float(h[idx] @ tc[:, idx])),
            worst_determinant=int(masks[worst]),
            worst_labels=[
                list(core.integrals.labels[i])
                for i in range(n)
                if masks[worst] & (1 << i)
            ],
            worst_expected_Ha=float(expected[worst]),
            worst_encoded_native_residual_Ha=float(native_op[worst, idx]),
        )
        rows.append(row)
        arrays[name + "_t"] = t
        arrays[name + "_expected_Ha"] = expected
        arrays[name + "_encoded_native_Ha"] = native_op[:, idx]
    np.savez_compressed(OUT / "localization_arrays.npz", **arrays)
    result = dict(
        rows=rows,
        source_sha256=digest(__file__),
        original_backend_sha256=core._manifest["files"]["native_core_sd.so"],
        wall_seconds=time.perf_counter() - start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_unchanged=all(
            digest((OUT if p == "probe.py" else ROOT) / p) == h
            for p, h in contract["source_sha256"].items()
        ),
        operator_scope="same complete12spinor DC linearized core projected commutator; no physical-space expansion, no production patch",
    )
    (OUT / "localization.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
