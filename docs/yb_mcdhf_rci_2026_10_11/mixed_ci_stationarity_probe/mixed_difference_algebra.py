#!/usr/bin/env python3
"""Read-only saved-matrix check of finite-norm mixed-CI difference algebra.

This does not implement a radial polynomial, solve CI, evaluate a gradient or
accept any orbital step. The prospective contract states those boundaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402 — pin BLAS threads before importing NumPy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDE = np.longdouble


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    contract = json.loads((HERE / "successor_assessment_contract.json").read_text())
    cap = int(contract["resource_caps"]["address_space_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.alarm(contract["resource_caps"]["wall_timeout_s"])
    checks = []

    def check(name, value, limit=0.0):
        checks.append({"name": name, "error": float(value), "limit": limit,
                       "passed": bool(np.isfinite(value) and value <= limit)})

    for path, digest in contract["inputs_sha256"].items():
        check("input_SHA/" + path, 0 if sha(HERE / path) == digest else 1)
    for path, digest in contract["external_inputs_sha256"].items():
        check("external_SHA/" + path, 0 if sha(ROOT / path) == digest else 1)

    result = json.loads((HERE / "result.json").read_text())
    basis = json.loads((ROOT / "docs/yb_mcdhf_rci_2026_10_11/"
                        "balanced_cv_replay/basis_identity.json").read_text())
    binding = json.loads((ROOT / "docs/yb_mcdhf_rci_2026_10_11/"
                          "balanced_cv_replay/stage_4f5p5s/target_binding.json").read_text())
    baseline = np.load(HERE / "baseline_arrays.npz")
    h0 = baseline["matrix_Ha"].astype(WIDE)
    v0 = baseline["mixing"].astype(WIDE)
    e0 = baseline["energies_Ha"].astype(WIDE)
    symmetry = np.array([(x["twoJ"], x["parity"]) for x in basis["CSFs"]])
    targets = binding["targets"]
    records = []
    identity_errors = []
    quotient_errors = []

    def trial_column(vectors, energies, target):
        allowed = np.all(symmetry == (target["twoJ"], target["parity"]), axis=1)
        leakage = np.sum(vectors[~allowed] ** 2, axis=0)
        columns = np.flatnonzero(leakage < 1e-20)
        return int(columns[np.argsort(energies[columns])[0]]), allowed

    for trial in result["trials"]:
        artifact = trial["artifact"]
        saved = np.load(HERE / artifact)
        h1 = saved["matrix_Ha"].astype(WIDE)
        v1 = saved["mixing"].astype(WIDE)
        e1 = saved["energies_Ha"].astype(WIDE)
        dh = h1 - h0
        rows = []
        for track in trial["physical_tracking"]:
            check(artifact + "/physical_UID_retained",
                  0 if track["reference_levels"] == track["selected_levels"] else 1)
            check(artifact + "/physical_overlap_squared",
                  0.5 - min(track["principal_overlap_squared"]), 0.0)
        for target in targets:
            label = target["label"]
            col0 = target["models"]["DC"]["column"]
            col1, allowed = trial_column(v1, e1, target)
            c0, c1 = v0[:, col0].copy(), v1[:, col1].copy()
            if c0 @ c1 < 0:
                c1 *= -1
            n0, n1 = c0 @ c0, c1 @ c1
            check(artifact + "/" + label + "/same_exact_symmetry",
                  max(np.sum(c0[~allowed] ** 2), np.sum(c1[~allowed] ** 2)), 1e-20)
            d = c1 - c0
            a0 = h0.copy()
            a1 = h1.copy()
            diagonal = np.diag_indices_from(a0)
            a0[diagonal] -= e0[col0]
            a1[diagonal] -= e1[col1]
            r0, r1 = a0 @ c0, a1 @ c1
            radial = c1 @ dh @ c1 / n1
            response = d @ a0 @ d / n1
            cross = 2 * (d @ r0) / n1
            ref_offset = c0 @ r0 / n1
            trial_offset = -(c1 @ r1) / n1
            formula = radial + response + cross + ref_offset + trial_offset
            reported = e1[col1] - e0[col0]
            error = abs(formula - reported)
            check(artifact + "/" + label + "/finite_reported_eigenvalue_identity_Ha",
                  error, 5e-12)
            identity_errors.append(float(error))

            q0 = e0[col0] + (c0 @ r0) / n0
            q1 = e1[col1] + (c1 @ r1) / n1
            aq = h0.copy()
            aq[diagonal] -= q0
            rq = aq @ c0
            quotient_formula = (c1 @ dh @ c1 + d @ aq @ d
                                + 2 * (d @ rq) + c0 @ rq) / n1
            qerror = abs(quotient_formula - (q1 - q0))
            check(artifact + "/" + label + "/finite_Rayleigh_quotient_identity_Ha",
                  qerror, 5e-12)
            quotient_errors.append(float(qerror))
            rows.append({"label": label, "reference_column": col0,
                         "trial_column": col1, "norm_reference": float(n0),
                         "norm_trial": float(n1), "aligned_coefficient_dot": float(c0 @ c1),
                         "reported_energy_delta_Ha": float(reported),
                         "trial_RDM_deltaH_Ha": float(radial),
                         "quadratic_CI_response_Ha": float(response),
                         "reference_residual_cross_Ha": float(cross),
                         "reference_residual_offset_Ha": float(ref_offset),
                         "trial_residual_offset_Ha": float(trial_offset),
                         "full_formula_Ha": float(formula), "identity_error_Ha": float(error),
                         "quotient_delta_Ha": float(q1 - q0),
                         "quotient_identity_error_Ha": float(qerror),
                         "frozen_reference_RDM_deltaH_Ha": float(c0 @ dh @ c0 / n0),
                         "raw_CI_norm_bias_if_trial_deltaH_unscaled_Ha":
                             float((c1 @ dh @ c1) - radial)})
        fields = ("reported_energy_delta_Ha", "trial_RDM_deltaH_Ha",
                  "quadratic_CI_response_Ha", "reference_residual_cross_Ha",
                  "reference_residual_offset_Ha", "trial_residual_offset_Ha",
                  "full_formula_Ha", "quotient_delta_Ha",
                  "frozen_reference_RDM_deltaH_Ha")
        records.append({"artifact": artifact, "rows": rows,
                        "equal_weight_EOL": {f: sum(x[f] for x in rows) / 5 for f in fields}})
    signal.alarm(0)
    output = {"status": "saved_matrix_algebra_verified_only",
              "all_passed": all(x["passed"] for x in checks), "checks_count": len(checks),
              "checks": checks, "matched_target_pairs": sum(len(x["rows"]) for x in records),
              "maximum_finite_eigenvalue_identity_error_Ha": max(identity_errors),
              "maximum_quotient_identity_error_Ha": max(quotient_errors),
              "records": records,
              "contract_sha256": sha(HERE / "successor_assessment_contract.json"),
              "script_sha256": sha(__file__),
              "same_frozen_probe_result_sha256": sha(HERE / "result.json"),
              "arithmetic": {"dtype": str(np.dtype(WIDE)),
                             "mantissa_bits": int(np.finfo(WIDE).nmant),
                             "matrix_source": "existing stored rounded full DC matrices",
                             "radial_polynomial_implemented": False,
                             "new_CI_solves": 0, "new_orbital_gradients": 0,
                             "new_optimization_steps": 0},
              "resource_uncontrolled": {"elapsed_s": time.monotonic() - start,
                                       "max_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}}
    (HERE / "mixed_difference_algebra.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("all_passed", "checks_count", "matched_target_pairs",
                     "maximum_finite_eigenvalue_identity_error_Ha",
                     "maximum_quotient_identity_error_Ha", "resource_uncontrolled")}))
    if not output["all_passed"]:
        raise RuntimeError("preserved failed saved-matrix mixed-CI algebra verification")


if __name__ == "__main__":
    main()
