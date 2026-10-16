"""Fresh original-equation, radial-pencil, symmetry and provenance readback."""

import hashlib
import json
import os
import resource
import signal
import tarfile
import time
import traceback

import numpy as np
from prepare_virtuals import (
    DHF,
    HERE,
    OLD,
    ROOT,
    AngularCoordinates,
    load_operator,
    save,
    sha,
)
from scipy import linalg

from atomx.constants import HARTREE_CM1
from atomx.methods.tensor_coupled_cluster import CCSDTTensorOperator, TensorCoreResult
from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def main():
    os.sched_setaffinity(0, {27})
    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))

    def alarm(*_):
        raise TimeoutError("declared240second saved readback cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(240)
    started = time.monotonic()
    report = dict(
        status="RUNNING",
        controls=[],
        resources=dict(
            cpu=27,
            threads=1,
            AS_bytes=24 * 1024**3,
            wall_seconds=240,
            complete_core_calls=1,
            complete_EA_partner_calls=12,
        ),
        scope="Saved projectedSDT/R3 states and original radial pencils; replay is not an independent full18e FCI or exp(T) norm certificate.",
    )

    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(
            dict(name=name, measured=measured, cap=cap, passed=bool(passed))
        )
        if not passed:
            raise AssertionError((name, measured, cap))

    try:
        actual = json.loads((HERE / "actual001/result.json").read_text())
        probe = json.loads((HERE / "probe001/result.json").read_text())
        gate("completed_actual", actual["status"] == "PASS_COMPLETE_6P_NATIVE_CCSDT_R3")
        gate("all_live_equation_controls", all(x["passed"] for x in actual["controls"]))
        gate(
            "all_live_probe_controls",
            probe["status"] == "PASS_COMPLETE_6P_OFFSHELL_PROBE"
            and all(x["passed"] for x in probe["controls"]),
        )
        source = json.loads((HERE / "source.json").read_text())
        gate(
            "all_current_source_bytes",
            all(sha(ROOT / n) == h for n, h in source["files"].items()),
        )
        with tarfile.open(HERE / "source.tar.gz") as a:
            gate(
                "all362_archive_members",
                set(a.getnames()) == set(source["files"]) and len(a.getnames()) == 362,
            )
            gate(
                "all_archive_bytes",
                all(
                    hashlib.sha256(a.extractfile(n).read()).hexdigest() == h
                    for n, h in source["files"].items()
                ),
            )
        for name in (
            "probe_contract.json",
            "diis_qualification_contract.json",
            "solve_contract.json",
            "corner_contract.json",
        ):
            c = json.loads((HERE / name).read_text())
            gate(
                name + ":all_inputs_unchanged",
                all(sha(ROOT / n) == h for n, h in c["inputs"].items()),
            )
        gate(
            "actual_contract_binding",
            actual["contract_sha256"] == sha(HERE / "solve_contract.json"),
        )
        gate(
            "probe_contract_binding",
            probe["contract_sha256"] == sha(HERE / "probe_contract.json"),
        )
        owner = json.loads((OLD / "increment_receipt.json").read_text())
        gate(
            "previous_all_artifacts_preserved",
            all(sha(ROOT / n) == h for n, h in owner["artifacts"].items()),
        )
        gate(
            "same_previous_source_epoch",
            owner["source_map_sha256"] == source["source_map_sha256"],
        )
        qualification = json.loads((HERE / "diis_qualification.json").read_text())
        gate(
            "independent_finite_DIIS_qualification",
            qualification["status"] == "PASS_INDEPENDENT_FINITE_AND_OLD_YB_DIIS"
            and all(x["passed"] for x in qualification["controls"]),
        )
        with (
            np.load(HERE / "probe001/virtuals.npz") as z,
            np.load(DHF / "actual004/blocks_ns180.npz") as origin,
        ):
            for row in probe["new_radial_roots"]:
                k = row["key"][1]
                h, s, c, b, full = [
                    z[f"{n}_k{k}"]
                    for n in ("H", "S", "occupied", "complement", "selected")
                ]
                for name, value in (("H", h), ("S", s), ("occupied", c)):
                    gate(
                        f"p{k}:same_original_{name}",
                        np.array_equal(value, origin[f"{name}_k{k}"]),
                    )
                x = full[origin["active_coordinates"]]
                wide = np.longdouble
                r = np.asarray(
                    b.astype(wide).T
                    @ (
                        h.astype(wide) @ x.astype(wide)
                        - wide(row["energy_Ha"]) * s.astype(wide) @ x.astype(wide)
                    ),
                    dtype=float,
                )
                sp = b.T @ s @ b
                sp = np.tril(sp) + np.tril(sp, -1).T
                dual = np.linalg.norm(
                    linalg.solve_triangular(
                        linalg.cholesky(sp, lower=True), r, lower=True
                    )
                )
                gate(f"p{k}:original_radial_dual_Ha", dual, 1e-8)
                gate(
                    f"p{k}:original_metric_norm",
                    abs(float(x.astype(wide) @ s.astype(wide) @ x.astype(wide)) - 1),
                    1e-10,
                )
                gate(
                    f"p{k}:original_occupied_overlap",
                    np.max(abs(c.astype(wide).T @ s.astype(wide) @ x.astype(wide))),
                    1e-10,
                )
        op, ledger = load_operator(HERE)
        gate("loaded_operator_identity", op.operator_sha256 == probe["operator_sha256"])
        for row in probe["new_radial_roots"]:
            indices = [
                i
                for i, label in enumerate(op.labels)
                if tuple(label[:2]) == tuple(row["key"])
            ]
            gate(
                str(row["key"]) + ":native_Fock_vs_original_radial_root",
                np.max(abs(np.diag(op.f)[indices] - row["energy_Ha"])),
                1e-7,
            )
        with np.load(HERE / "actual001/core.npz") as z:
            amplitudes = z["amplitudes"]
        residual = op.core_residual(amplitudes)
        energy = op.core_energy(amplitudes)
        gate("all689589_core_equations", np.max(abs(residual)), 1e-10)
        gate("core_energy", abs(energy - actual["core"]["energy_relative_Ha"]), 1e-12)
        active = op.scalar_core_indices()
        gate("all37533_exact_scalar_coordinates", len(active) == 37533)
        outside = np.setdiff1d(np.arange(op.core.size), active)
        gate("exact_forbidden_core_zeros", np.all(amplitudes[outside] == 0))
        angular_core = AngularCoordinates(op, False)
        gate("core_Jplus", np.linalg.norm(angular_core.ladder(amplitudes, 1)), 1e-7)
        core = TensorCoreResult(
            amplitudes,
            energy,
            energy,
            float(np.max(abs(residual))),
            0,
            op.operator_sha256,
        )
        angular = AngularCoordinates(op, True)
        maximum = 0.0
        with np.load(HERE / "actual001/all_target_partners.npz") as partners:
            gate("all12_target_arrays", len(partners.files) == 12)
            for name, key, j in [
                ("S", (6, -1), 0.5),
                ("D3", (5, 2), 1.5),
                ("D5", (5, -3), 2.5),
            ]:
                expected = actual["targets"][name]["EA_Ha"]
                with np.load(HERE / "actual001" / (name + "_root.npz")) as z:
                    gate(
                        name + ":complete_highestM_sector",
                        np.array_equal(z["indices"], angular.sector(int(2 * j))),
                    )
                    gate(
                        name + ":same_root_as_partner",
                        np.array_equal(z["vector"], partners[name + f"_M{int(2 * j)}"]),
                    )
                    gate(
                        name + ":same_root_energy",
                        abs(float(z["energy"]) - expected),
                        1e-12,
                    )
                    bare = [
                        a
                        for a, label in enumerate(op.labels[op.o :])
                        if tuple(label[:2]) == key
                    ]
                    weight = float(np.sum(abs(z["vector"][bare]) ** 2))
                    gate(name + ":prescribed_bare_R1_identity", weight >= 0.5)
                keys = [k for k in partners.files if k.startswith(name + "_M")]
                gate(name + ":all2Jplus1_partners", len(keys) == int(2 * j + 1))
                for key in keys:
                    m2 = int(key.split("_M")[1])
                    vector = partners[key]
                    gate(
                        key + ":exact_other_sector_zeros",
                        np.all(vector[(angular.m2 != m2) | (angular.parity != 0)] == 0),
                    )
                    gate(
                        key + ":canonical_R_norm", abs(np.linalg.norm(vector) - 1), 1e-8
                    )
                    gate(
                        key + ":J2",
                        np.linalg.norm(
                            angular.j_squared(vector, m2 / 2) - j * (j + 1) * vector
                        ),
                        1e-7,
                    )
                    error = float(
                        np.max(
                            abs(op.attachment_action(core, vector) - expected * vector)
                        )
                    )
                    maximum = max(maximum, error)
                    gate(key + ":all127620_original_R3_equations", error, 1e-10)
        lines = {
            name: (actual["targets"][name]["EA_Ha"] - actual["targets"]["S"]["EA_Ha"])
            * HARTREE_CM1
            for name in ("D3", "D5")
        }
        previous = json.loads((OLD / "actual003/result.json").read_text())
        for name in lines:
            gate(
                name + ":same_state_line",
                abs(lines[name] - actual["lines_cm1"][name]),
                1e-9,
            )
            gate(
                name + ":same_state_change",
                abs(
                    lines[name]
                    - previous["lines_cm1"][name]
                    - actual["newp_line_changes_cm1"][name]
                ),
                1e-9,
            )
        gate(
            "absolute_energy_origin",
            abs(
                ledger["absolute_energy_origin_Ha"]
                - json.loads((OLD / "operator.json").read_text())[
                    "absolute_energy_origin_Ha"
                ]
            ),
            1e-10,
        )
        corner = json.loads((HERE / "corner001/result.json").read_text())
        gate(
            "corner_completed",
            corner["status"] == "PASS_FROZEN4F_COMPLETE_6P_CORNER"
            and all(x["passed"] for x in corner["controls"]),
        )
        gate(
            "corner_contract_binding",
            corner["contract_sha256"] == sha(HERE / "corner_contract.json"),
        )
        cl = json.loads((HERE / "corner001/operator.json").read_text())
        with np.load(HERE / "corner001/integrals.npz") as z:
            ci = SpinorIntegrals(
                tuple(map(tuple, cl["labels"])),
                4,
                z["h"],
                z["fock"],
                z["g"],
                cl["shifted_frozen_energy_Ha"],
                0.0,
                cl["identity"],
            )
        co = CCSDTTensorOperator(ci, memory_MiB=512, projection_storage="canonical")
        picked = list(range(4)) + list(range(18, 36))
        gate(
            "corner_all_retained_ERI",
            np.max(abs(co.g - op.g[np.ix_(picked, picked, picked, picked)])),
            1e-10,
        )
        gate(
            "corner_all_retained_Fock",
            np.max(abs(co.f - op.f[np.ix_(picked, picked)])),
            1e-10,
        )
        with np.load(HERE / "integrals.npz") as z:
            expected_h = z["h"][np.ix_(picked, picked)].copy()
            expected_frozen = (
                ledger["shifted_frozen_energy_Ha"]
                + float(np.trace(z["h"][4:18, 4:18]).real)
                + 0.5
                * sum(op.g[a, b, a, b] for a in range(4, 18) for b in range(4, 18))
            )
            gate(
                "corner_frozen_constant",
                abs(expected_frozen - cl["shifted_frozen_energy_Ha"]),
                1e-10,
            )
            for a, p in enumerate(picked):
                for b, q in enumerate(picked):
                    expected_h[a, b] += sum(op.g[p, f, q, f] for f in range(4, 18))
        gate("corner_frozen_one_body", np.max(abs(expected_h - ci.one_body_Ha)), 1e-10)
        with np.load(HERE / "corner001/core.npz") as z:
            ct = z["amplitudes"]
        cr = co.core_residual(ct)
        ce = co.core_energy(ct)
        gate("corner_all4254_SDT_equations", np.max(abs(cr)), 1e-10)
        gate(
            "corner_core_energy", abs(ce - corner["core"]["energy_relative_Ha"]), 1e-12
        )
        cc = TensorCoreResult(ct, ce, ce, float(np.max(abs(cr))), 0, co.operator_sha256)
        ca = AngularCoordinates(co, True)
        with np.load(HERE / "corner001/all_target_partners.npz") as z:
            gate("corner_all12_partners", len(z.files) == 12)
            for key in z.files:
                name, m = key.split("_M")
                m2 = int(m)
                j = {"S": 0.5, "D3": 1.5, "D5": 2.5}[name]
                v = z[key]
                gate(
                    key + ":corner_J2",
                    np.linalg.norm(ca.j_squared(v, m2 / 2) - j * (j + 1) * v),
                    1e-7,
                )
                gate(
                    key + ":corner_original_R3_equations",
                    np.max(
                        abs(
                            co.attachment_action(cc, v)
                            - corner["targets"][name]["EA_Ha"] * v
                        )
                    ),
                    1e-10,
                )
        report.update(
            status="PASS_SAVED_COMPLETE_6P_PROJECTED_REVIEW",
            maximum_partner_residual_Ha=maximum,
            lines_cm1=lines,
        )
    except Exception:
        report.update(status="FAILED", exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_RSS_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        save(HERE / "saved_data_review.json", report)
        print(report["status"], len(report["controls"]), flush=True)


if __name__ == "__main__":
    main()
