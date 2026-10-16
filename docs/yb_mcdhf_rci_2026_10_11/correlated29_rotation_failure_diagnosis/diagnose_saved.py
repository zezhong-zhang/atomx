"""Bounded exact saved-array diagnosis; no physical producers or eigensolves."""
import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    authority = json.loads(Path(args.authorization).read_text())
    if (authority.get("readonly_execution_authorized_by_root") is not True
            or authority.get("contract_sha256") != sha(HERE / "contract.json")
            or authority.get("driver_sha256") != sha(__file__)):
        raise ValueError("Missing exact root read-only authorization")
    for role in contract["inputs"].values():
        if sha(REPO / role["path"]) != role["sha256"]:
            raise ValueError("Changed saved input")
    if any(os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise ValueError("Single-thread environment required")
    accounting = sorted(os.sched_getaffinity(0))
    if 31 not in accounting:
        raise ValueError("Declared CPU31 unavailable")
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    os.sched_setaffinity(0, {31})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("90s saved-only cap")))
    signal.alarm(90)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    begin = time.monotonic()
    products = 0
    result = {"original_rotation002_passed": False, "physical_producer_calls": 0,
              "eigen_SVD_calls": 0, "two_body_cells_or_potentials": 0,
              "new_public_RDM_contracts": 0, "optimizer_authorized": False,
              "no_new_direction_or_step": True}

    def flush(phase):
        write(output / "progress.json", {"phase": phase, "exact_products": products,
              "elapsed_s": time.monotonic() - begin, "partial": result})

    try:
        import numpy as np

        def path(name):
            return REPO / contract["inputs"][name]["path"]

        def load(name):
            return json.loads(path(name).read_text())

        def arrays(name, wanted):
            with np.load(path(name), allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in wanted}

        cache = {}

        def f(value):
            value = float(value)
            if value not in cache:
                cache[value] = Fraction.from_float(value)
            return cache[value]

        def mul(a, b):
            nonlocal products
            products += 1
            if products > contract["resources"]["max_exact_dyadic_scalar_products"]:
                raise RuntimeError("Exact scalar product cap")
            return a * b

        def dot(a, b):
            return sum((mul(x, y) for x, y in zip(a, b, strict=True)), Fraction(0))

        def sparse(matrix):
            return [[(int(j), f(row[j])) for j in np.flatnonzero(row)] for row in matrix]

        def action(rows, vector):
            return [sum((mul(value, vector[j]) for j, value in row), Fraction(0)) for row in rows]

        def key(value):
            return tuple(key(v) for v in value) if isinstance(value, list) else value

        def vector(values):
            return [f(v) for v in values]

        base = arrays("base_waves", ("keys", "P", "Q"))
        keys = [tuple(map(int, k)) for k in base["keys"]]
        if len(keys) != 29 or len(set(keys)) != 29:
            raise ValueError("Wrong all29 base keys")
        c = {k: vector(np.concatenate((p, q))) for k, p, q in zip(keys, base["P"], base["Q"], strict=True)}
        gdata = arrays("saved_gradient", ("x", "raw_coordinate_gradient", "tangent_gradient", "radial_gradient",
                  *[f"projector_{i}" for i in range(7)]))
        direction = arrays("direction", ("direction",))["direction"]
        layout = load("saved_layout")
        if direction.shape != (3451,) or gdata["radial_gradient"].shape != (29, 246):
            raise ValueError("Wrong gradient/direction shapes")
        h = f(contract["fixed"]["original_rotation_h"])
        direction_f = vector(direction)
        actual, increments, trial_x = [], [], []
        for sign in ("plus", "minus"):
            trial = arrays(sign + "_waves", ("keys", "P", "Q"))
            coord = arrays(sign + "_coordinates", ("candidate_x", "explicit_increment_P", "explicit_increment_Q"))
            if not np.array_equal(trial["keys"], base["keys"]):
                raise ValueError("Trial key/order mismatch")
            actual.append({k: vector(np.concatenate((p, q))) for k, p, q in zip(keys, trial["P"], trial["Q"], strict=True)})
            increments.append({k: vector(np.concatenate((p, q))) for k, p, q in zip(keys, coord["explicit_increment_P"], coord["explicit_increment_Q"], strict=True)})
            trial_x.append(vector(coord["candidate_x"]))
        modes = {"ideal_P_times_seed": {}, "explicit_polar_odd": {}, "rounded_physical_odd": {}}
        for i, row in enumerate(layout):
            group, shape = list(map(tuple, row["keys"])), row["shape"]
            start, stop = row["slice"]
            subdirection = direction_f[start:stop]
            projector = sparse(gdata[f"projector_{i}"])
            for j, k in enumerate(group):
                column = subdirection[j::shape[1]]
                modes["ideal_P_times_seed"][k] = action(projector, column)
        for k in keys:
            modes["explicit_polar_odd"][k] = [(a - b) / (2 * h) for a, b in zip(increments[0][k], increments[1][k], strict=True)]
            modes["rounded_physical_odd"][k] = [(a - b) / (2 * h) for a, b in zip(actual[0][k], actual[1][k], strict=True)]
        coordinate_odd = [(a - b) / (2 * h) for a, b in zip(trial_x[0], trial_x[1], strict=True)]
        density = load("signed_density")
        full = {key(k): f(v) for k, v in density["unfiltered"]}
        retained = {key(k): f(v) for k, v in density["actual_gradient_retained"]}
        hkeys = sorted(k for k in full.keys() | retained.keys() if k[0] == "h")
        rkeys = {k for k in full.keys() | retained.keys() if k[0] == "r"}
        result["table_comparison"] = {
            "full_terms": len(full), "retained_terms": len(retained), "h_keys": len(hkeys),
            "missing_h_keys": [list(k) for k in hkeys if k not in retained],
            "changed_h_coefficients": sum(full.get(k, 0) != retained.get(k, 0) for k in hkeys),
            "changed_r_coefficients": sum(full.get(k, 0) != retained.get(k, 0) for k in rkeys),
            "missing_r_keys": sum(k in full and k not in retained for k in rkeys),
            "not_a_pure_cutoff_difference": True}
        flush("first_h_table_comparison")
        geometry = arrays("logical_geometry", tuple(f"h_{k[1]}" for k in keys))
        operators, matrix_seams = {}, []
        for i in range(7):
            data = arrays(f"raw_hS_{i}", ("kappa", "raw_h", "S"))
            kappa, raw, s = int(data["kappa"]), data["raw_h"], data["S"]
            logical = np.tril(raw) + np.tril(raw, -1).T
            if not np.array_equal(logical, geometry[f"h_{kappa}"]):
                raise ValueError("Saved public logical h mismatch")
            operators[kappa] = (sparse(logical), sparse(raw), sparse(raw.T), sparse(s))
            matrix_seams.append({"kappa": kappa, "raw_minus_logical_max": float(np.max(abs(raw - logical))),
                                 "raw_asymmetry_max": float(np.max(abs(raw - raw.T))),
                                 "max_abs_raw_entry": float(np.max(abs(raw)))})
        result["saved_h_representation_seams"] = matrix_seams
        actions = {}
        for k in keys:
            logical, raw, transposed, s = operators[k[1]]
            actions[k] = tuple(action(rows, c[k]) for rows in (logical, raw, transposed, s))
        flush("exact_one_body_and_metric_actions")

        def slope(weights, mode, raw=False, normalized=False):
            value = Fraction(0)
            pieces = []
            for term in hkeys:
                coefficient = weights.get(term, Fraction(0))
                if not coefficient:
                    continue
                _, na, kappa, nb = term
                a, b = (na, kappa), (nb, kappa)
                hb = actions[b][1 if raw else 0]
                ha = actions[a][2 if raw else 0]
                numerator = dot(mode[a], hb) + dot(mode[b], ha)
                if normalized:
                    norm = dot(c[b], actions[b][3])
                    if norm <= 0:
                        raise ValueError("Nonpositive archived exact ket norm")
                    old_numerator = dot(c[a], hb)
                    norm_prime = 2 * dot(mode[b], actions[b][3])
                    numerator = numerator / norm - old_numerator * norm_prime / (norm * norm)
                contribution = mul(coefficient, numerator)
                value += contribution
                pieces.append((float(abs(contribution)), list(term), float(contribution)))
            pieces.sort(reverse=True)
            return value, [{"key": k, "contribution": v} for _, k, v in pieces[:5]]

        gradient = {k: vector(row) for k, row in zip(keys, gdata["radial_gradient"], strict=True)}
        delta_weights = {k: full.get(k, Fraction(0)) - retained.get(k, Fraction(0)) for k in hkeys}
        outputs = {}
        for name, mode in modes.items():
            logical_full, _ = slope(full, mode)
            logical_retained, _ = slope(retained, mode)
            logical_difference, largest = slope(delta_weights, mode)
            raw_retained, _ = slope(retained, mode, raw=True)
            quotient_retained, _ = slope(retained, mode, raw=True, normalized=True)
            quotient_full, _ = slope(full, mode, raw=True, normalized=True)
            full_radial = sum((dot(gradient[k], mode[k]) for k in keys), Fraction(0))
            conditional = full_radial + quotient_retained - logical_retained
            outputs[name] = {"logical_h_full_slope": float(logical_full),
                "logical_h_retained_slope": float(logical_retained),
                "global_h_coefficient_difference_slope": float(logical_difference),
                "largest_h_difference_contributions": largest,
                "raw_h_retained_slope": float(raw_retained),
                "raw_h_ket_normalized_retained_slope": float(quotient_retained),
                "raw_h_ket_normalized_full_slope": float(quotient_full),
                "raw_minus_logical_h_slope": float(raw_retained - logical_retained),
                "ket_norm_derivative_change": float(quotient_retained - raw_retained),
                "full_saved_raw_radial_gradient_action": float(full_radial),
                "conditional_saved_full_gradient_plus_onebody_seam": float(conditional),
                "conditional_two_body_coefficients_and_curvature_unmeasured": True}
            result["direction_modes"] = outputs
            flush("exact_slopes_" + name)
        result["saved_coordinate_gradient_actions"] = {
            "raw_seed": float(dot(vector(gdata["raw_coordinate_gradient"]), direction_f)),
            "tangent_seed": float(dot(vector(gdata["tangent_gradient"]), direction_f)),
            "raw_polar_odd": float(dot(vector(gdata["raw_coordinate_gradient"]), coordinate_odd)),
            "tangent_polar_odd": float(dot(vector(gdata["tangent_gradient"]), coordinate_odd))}
        measured = load("failed_ledger")
        diagnostics = load("failed_diagnostics")
        result["already_measured_failed_ledger"] = measured
        result["already_measured_central_pieces"] = {name: (diagnostics[0][name] - diagnostics[1][name]) / 2e-6
            for name in ("explicit_radial", "reconstruction_radial", "rayleigh_response", "reported_response")}
        predicted = outputs["rounded_physical_odd"]["conditional_saved_full_gradient_plus_onebody_seam"]
        result["causal_scale_comparison"] = {
            "measured_Rayleigh_error": measured["Rayleigh_central"] - measured["analytic_saved_g"],
            "conditional_linearized_roundoff_shift": predicted - outputs["ideal_P_times_seed"]["conditional_saved_full_gradient_plus_onebody_seam"],
            "measured_Rayleigh_minus_conditional_linearization": measured["Rayleigh_central"] - predicted,
            "not_a_full_finite_difference_or_missing_physics_bound": True}
        if any(n == "atomx" or n.startswith(("atomx.", "scipy")) for n in sys.modules):
            raise ValueError("Forbidden physical or eigensolver module")
        if any(sha(REPO / role["path"]) != role["sha256"] for role in contract["inputs"].values()):
            raise ValueError("Input changed during saved audit")
        result.update(saved_only_diagnosis_completed=True, exact_products=products,
                      contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
                      authorization_sha256=sha(Path(args.authorization)), elapsed_s=time.monotonic() - begin,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, CPU=31,
                      accounting_CPUs=accounting, known_unresolved=[
                      "two-body global density-table difference has no directional bound here",
                      "trial-density response and complete finite-cell one/two-body increment terms are not reconstructed",
                      "linearization does not bound finite polar curvature or public gradient arithmetic error"],
                      no_uniform_error_bound=True, original_failed_gate_unchanged=True)
        write(output / "result.json", result)
        print(json.dumps({k: result[k] for k in ("saved_only_diagnosis_completed", "exact_products", "causal_scale_comparison", "elapsed_s")}))
    except BaseException as error:
        import traceback
        flush("failed")
        write(output / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
              "partial": result, "exact_products": products, "physical_calls": 0,
              "original_failed_gate_unchanged": True})
        raise


if __name__ == "__main__":
    main()
