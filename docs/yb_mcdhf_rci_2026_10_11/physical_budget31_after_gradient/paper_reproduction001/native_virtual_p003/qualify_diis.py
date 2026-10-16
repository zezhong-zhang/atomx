"""Independent finite fermion/FCI and previous actual-Yb DIIS qualification."""

import json
import sys

import numpy as np
from diis_core import solve
from prepare_virtuals import HERE, ROOT, save

sys.path.insert(0, str(ROOT / "tests"))
from test_eom_attachment import independent_hamiltonian, integrals  # noqa: E402

from atomx.methods.tensor_coupled_cluster import CCSDTTensorOperator  # noqa: E402
from atomx.physics.csf.spinor_integrals import SpinorIntegrals  # noqa: E402


def main():
    controls = []

    def gate(name, value, cap):
        controls.append(
            dict(name=name, measured=float(value), cap=cap, passed=bool(value <= cap))
        )
        assert value <= cap, controls[-1]

    obj, h, eri = integrals()
    op = CCSDTTensorOperator(obj, projection_storage="canonical")
    core, finite_history = solve(op, scalar_sector=False)
    _, hc = independent_hamiltonian(obj, h, eri, 3)
    gate("finite_complete_SDT", np.max(abs(op.core_residual(core.amplitudes))), 1e-10)
    gate(
        "finite_independent_FCI_energy",
        abs(core.total_energy_Ha - np.linalg.eigvalsh(hc)[0]),
        1e-10,
    )
    old = HERE.parent / "native_tensor_ccsdt001/actual001"
    ledger = json.loads((old / "operator.json").read_text())
    with np.load(old / "integrals.npz") as z:
        obj = SpinorIntegrals(
            tuple(map(tuple, ledger["labels"])),
            4,
            z["h"],
            z["fock"],
            z["g"],
            ledger["shifted_frozen_energy_Ha"],
            0.0,
            ledger["identity"],
        )
    op = CCSDTTensorOperator(obj, projection_storage="canonical")
    core, actual_history = solve(op)
    with np.load(old / "CCSDT_core.npz") as z:
        gate(
            "previous_Yb_all1324_amplitudes",
            np.max(abs(core.amplitudes - z["amplitudes"])),
            1e-9,
        )
    gate(
        "previous_Yb_all1324_equations",
        np.max(abs(op.core_residual(core.amplitudes))),
        1e-10,
    )
    expected = json.loads((old / "result.json").read_text())["cores"]["CCSDT"][
        "energy_Ha"
    ]
    gate("previous_Yb_energy", abs(core.total_energy_Ha - expected), 1e-10)
    save(
        HERE / "diis_qualification.json",
        dict(
            status="PASS_INDEPENDENT_FINITE_AND_OLD_YB_DIIS",
            controls=controls,
            finite_history=finite_history,
            actual_history=actual_history,
            scope="Numerical solver qualification at unchanged equations, no large-p spectral or speedup certificate",
        ),
    )
    print("PASS_DIIS", len(finite_history), len(actual_history), flush=True)


if __name__ == "__main__":
    main()
