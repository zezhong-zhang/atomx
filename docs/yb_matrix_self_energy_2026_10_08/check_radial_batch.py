"""Same-input scalar/GEMM radial equivalence and bounded timing control."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import load_reference, save, sha


def main():
    from atomx.numerics.bspline.convol import (
        build_slater_rk_bilinear, eval_slater_rk_bilinear, compute_slater_rk_table,
    )
    save(ROOT/"radial_batch_contract.json", dict(fixed="Fine DF reference, target 5d5/2 and core 4f7/2, rank 2, same 65 mixed-kappa virtual spinors, all 4225 ordered pairs",
                                                 hypothesis="The existing shared GEMM table agrees with scalar bilinear contractions and reduces radial-stage work",
                                                 gates=dict(absolute_Ha=1e-11, relative=1e-11),
                                                 repeats="warm then two alternating pairs; no end-to-end speed claim",
                                                 caps=dict(seconds=60, memory_MiB=64), source_mutations=False))
    started = time.monotonic()
    core, _, _, identity = load_reference("v68_refined_newton")
    z = np.load(ROOT/"preflight/fine_l6_e200_virtuals.npz")
    kappas = [-1]+[k for l in range(1, 7) for k in (l, -l-1)]
    states = [(P, Q) for k in kappas for P, Q in zip(z[f"P_{k}"][:5], z[f"Q_{k}"][:5])]
    P, Q = np.array([s[0] for s in states]).T, np.array([s[1] for s in states]).T
    aP, aQ = z["external_raw_P_-3"][0], z["external_raw_Q_-3"][0]
    bP, bQ, _ = core.orbitals_raw[-4][0]
    M = build_slater_rk_bilinear(aP, aQ, bP, bQ, core._bsp_data, 2)
    def scalar():
        return np.array([[eval_slater_rk_bilinear(M, *left, *right) for right in states] for left in states])
    def batch():
        return compute_slater_rk_table(M, P, Q, P, Q)
    a, b = scalar(), batch()
    error = float(np.max(abs(a-b)))
    assert error <= 1e-11
    np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-11)
    times = {"scalar": [], "batch": []}
    for order in (("scalar", "batch"), ("batch", "scalar")):
        for name in order:
            t = time.monotonic()
            value = scalar() if name == "scalar" else batch()
            times[name].append(time.monotonic()-t)
            assert np.max(abs(value-a)) <= 1e-11
    assert time.monotonic()-started < 60
    save(ROOT/"radial_batch_check.json", dict(reference_identity=identity, actual_pairs=len(states)**2,
                                              max_abs_error_Ha=error, raw_stage_seconds=times,
                                              machine_load=os.getloadavg(), source_script_sha256=sha(__file__),
                                              elapsed_seconds=time.monotonic()-started, passed=True,
                                              scope="Numerical radial-stage anchor only; no matrix-capture end-to-end performance claim"))
    print(error, times)


if __name__ == "__main__":
    main()
