"""Prospective fixed-multiplier saved-point diagnosis, never a production adapter."""
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
        raise ValueError("Exact root authorization required before numerical imports")
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
    result = {"physical_producer_calls": 0, "eigen_SVD_calls": 0,
              "two_body_cells_or_potentials": 0, "new_public_RDM_contracts": 0,
              "original_rotation002_passed": False, "optimizer_authorized": False,
              "production_callback": False, "no_uniform_error_bound": True,
              "scope": "conditional fixed-Lambda same-saved-point constrained-gradient control"}

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
            if products > contract["resources"]["max_exact_scalar_products"]:
                raise RuntimeError("Exact scalar-product cap")
            return a * b

        def dot(a, b):
            return sum((mul(x, y) for x, y in zip(a, b, strict=True)), Fraction(0))

        def sparse(matrix):
            return [[(int(j), f(row[j])) for j in np.flatnonzero(row)] for row in matrix]

        def action(rows, vector):
            return [sum((mul(value, vector[j]) for j, value in row), Fraction(0)) for row in rows]

        def vector(values):
            return [f(v) for v in values]

        def key(value):
            return tuple(key(v) for v in value) if isinstance(value, list) else value

        def difference(a, b):
            return [x - y for x, y in zip(a, b, strict=True)]

        def float_matrix(matrix):
            return [[float(x) for x in row] for row in matrix]

        base = arrays("base_waves", ("keys", "P", "Q"))
        keys = [tuple(map(int, k)) for k in base["keys"]]
        if len(keys) != 29 or len(set(keys)) != 29:
            raise ValueError("Wrong complete29 key set")
        c = {k: vector(np.concatenate((p, q))) for k, p, q in zip(keys, base["P"], base["Q"], strict=True)}
        gd = arrays("saved_gradient", ("radial_gradient", "tangent_gradient", *[f"projector_{i}" for i in range(7)]))
        if gd["radial_gradient"].shape != (29, 246):
            raise ValueError("Wrong complete radial gradient shape")
        raw_g = {k: vector(v) for k, v in zip(keys, gd["radial_gradient"], strict=True)}
        g = {k: list(v) for k, v in raw_g.items()}
        layout = load("saved_layout")
        seed = vector(arrays("direction", ("direction",))["direction"])
        if len(seed) != 3451 or len(layout) != 7:
            raise ValueError("Wrong original layout or seed")
        groups, ideal = [], {}
        for i, row in enumerate(layout):
            group = [tuple(map(int, k)) for k in row["keys"]]
            start, stop = row["slice"]
            shape = row["shape"]
            projector = sparse(gd[f"projector_{i}"])
            if shape != [119, len(group)] or stop - start != 119 * len(group):
                raise ValueError("Changed original positive-coordinate layout")
            for j, k in enumerate(group):
                ideal[k] = action(projector, seed[start:stop][j::shape[1]])
            groups.append(group)
        if set(ideal) != set(keys) or len({k for group in groups for k in group}) != 29:
            raise ValueError("Incomplete same-kappa groups")

        logical_saved = arrays("logical_geometry", tuple(f"h_{k[1]}" for k in keys))
        operators = {}
        for i in range(7):
            data = arrays(f"raw_hS_{i}", ("kappa", "raw_h", "S"))
            kappa, raw, s = int(data["kappa"]), data["raw_h"], data["S"]
            logical = np.tril(raw) + np.tril(raw, -1).T
            if not np.array_equal(logical, logical_saved[f"h_{kappa}"]):
                raise ValueError("Changed logical/raw h correspondence")
            if not np.array_equal(s, s.T):
                raise ValueError("Saved S must be exactly symmetric")
            operators[kappa] = tuple(sparse(m) for m in (logical, raw, raw.T, s))
        ac = {k: tuple(action(rows, c[k]) for rows in operators[k[1]]) for k in keys}
        density = load("signed_density")
        retained = {key(k): f(v) for k, v in density["actual_gradient_retained"]}
        h_terms = [k for k in retained if k[0] == "h"]
        if len(h_terms) != 51:
            raise ValueError("Changed signed retained one-body table")
        for term in h_terms:
            weight = retained[term]
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            norm = dot(c[b], ac[b][3])
            if norm <= 0:
                raise ValueError("Nonpositive actual ket norm")
            numerator = dot(c[a], ac[b][1])
            for j in range(246):
                g[a][j] += mul(weight, ac[b][1][j] / norm - ac[b][0][j])
                g[b][j] += mul(weight, ac[a][2][j] / norm - ac[a][0][j]
                              - mul(2 * numerator / (norm * norm), ac[b][3][j]))
        flush("fixed_actual_quotient_ambient_gradient")

        h = f(contract["fixed"]["h"])
        points, explicit = [], []
        for sign in ("plus", "minus"):
            trial = arrays(sign + "_waves", ("keys", "P", "Q"))
            delta = arrays(sign + "_coordinates", ("explicit_increment_P", "explicit_increment_Q"))
            if not np.array_equal(trial["keys"], base["keys"]):
                raise ValueError("Changed trial keys/order")
            points.append({k: vector(np.concatenate((p, q))) for k, p, q in zip(keys, trial["P"], trial["Q"], strict=True)})
            explicit.append({k: vector(np.concatenate((p, q))) for k, p, q in zip(keys, delta["explicit_increment_P"], delta["explicit_increment_Q"], strict=True)})
        total = [{k: difference(point[k], c[k]) for k in keys} for point in points]
        recon = [{k: difference(total[i][k], explicit[i][k]) for k in keys} for i in range(2)]
        multipliers, normal, normal_exp, normal_ideal = [], [Fraction(0)] * 2, [Fraction(0)] * 2, Fraction(0)
        raw_normal = [Fraction(0)] * 2
        ideal_g_action = sum((dot(g[k], ideal[k]) for k in keys), Fraction(0))
        for group in groups:
            kappa = group[0][1]
            if any(k[1] != kappa for k in group):
                raise ValueError("Non-kappa-homogeneous group")
            sc = {k: ac[k][3] for k in group}
            cg = [[dot(c[a], g[b]) for b in group] for a in group]
            crg = [[dot(c[a], raw_g[b]) for b in group] for a in group]
            lam = [[(cg[i][j] + cg[j][i]) / 4 for j in range(len(group))] for i in range(len(group))]
            raw_lam = [[(crg[i][j] + crg[j][i]) / 4 for j in range(len(group))] for i in range(len(group))]
            base_gram = [[dot(c[a], sc[b]) for b in group] for a in group]
            base_defect = max(abs(value - (i == j)) for i, row in enumerate(base_gram) for j, value in enumerate(row))
            if base_defect > f(contract["fixed"]["actual_Gram_gate"]):
                raise ValueError("Baseline original Gram gate failed")
            ideal_dg = [[dot(ideal[a], sc[b]) + dot(ideal[b], sc[a]) for b in group] for a in group]
            ni = sum((mul(lam[i][j], ideal_dg[i][j]) for i in range(len(group)) for j in range(len(group))), Fraction(0))
            normal_ideal += ni
            block = {"kappa": kappa, "keys": [list(k) for k in group], "Lambda": float_matrix(lam),
                     "raw_gradient_Lambda": float_matrix(raw_lam), "baseline_Gram": float_matrix(base_gram),
                     "baseline_Gram_defect": float(base_defect), "ideal_Gram_derivative": float_matrix(ideal_dg),
                     "ideal_normal_derivative": float(ni), "trials": []}
            for t in range(2):
                st = {k: action(operators[kappa][3], total[t][k]) for k in group}
                se = {k: action(operators[kappa][3], explicit[t][k]) for k in group}
                sr = {k: action(operators[kappa][3], recon[t][k]) for k in group}
                linear = [[dot(total[t][a], sc[b]) + dot(total[t][b], sc[a]) for b in group] for a in group]
                quadratic = [[dot(total[t][a], st[b]) for b in group] for a in group]
                dg = [[linear[i][j] + quadratic[i][j] for j in range(len(group))] for i in range(len(group))]
                de = [[dot(explicit[t][a], sc[b]) + dot(explicit[t][b], sc[a]) + dot(explicit[t][a], se[b])
                       for b in group] for a in group]
                dr = [[dot(recon[t][a], sc[b]) + dot(recon[t][b], sc[a])
                       + dot(explicit[t][a], sr[b]) + dot(recon[t][a], se[b]) + dot(recon[t][a], sr[b])
                       for b in group] for a in group]
                trial_gram = [[dot(points[t][a], [sc[b][j] + st[b][j] for j in range(246)])
                               for b in group] for a in group]
                if any(dg[i][j] != de[i][j] + dr[i][j] or dg[i][j] != trial_gram[i][j] - base_gram[i][j]
                       or dg[i][j] != dg[j][i] for i in range(len(group)) for j in range(len(group))):
                    raise ValueError("Exact full-Gram/quadratic/split identity failed")
                defect = max(abs(value - (i == j)) for i, row in enumerate(trial_gram) for j, value in enumerate(row))
                if defect > f(contract["fixed"]["actual_Gram_gate"]):
                    raise ValueError("Trial original Gram gate failed")
                n = sum((mul(lam[i][j], dg[i][j]) for i in range(len(group)) for j in range(len(group))), Fraction(0))
                ne = sum((mul(lam[i][j], de[i][j]) for i in range(len(group)) for j in range(len(group))), Fraction(0))
                nl = sum((mul(lam[i][j], linear[i][j]) for i in range(len(group)) for j in range(len(group))), Fraction(0))
                nq = sum((mul(lam[i][j], quadratic[i][j]) for i in range(len(group)) for j in range(len(group))), Fraction(0))
                ndiag = sum((mul(lam[i][i], dg[i][i]) for i in range(len(group))), Fraction(0))
                nr = sum((mul(raw_lam[i][j], dg[i][j]) for i in range(len(group)) for j in range(len(group))), Fraction(0))
                normal[t] += n
                normal_exp[t] += ne
                raw_normal[t] += nr
                block["trials"].append({"sign": ["plus", "minus"][t], "Delta_Gram": float_matrix(dg),
                    "Delta_Gram_linear": float_matrix(linear), "Delta_Gram_quadratic": float_matrix(quadratic),
                    "Delta_Gram_explicit": float_matrix(de), "Delta_Gram_reconstruction": float_matrix(dr),
                    "trial_Gram_defect": float(defect), "normal_full": float(n), "normal_linear": float(nl),
                    "normal_quadratic": float(nq), "normal_diagonal": float(ndiag), "normal_offdiagonal": float(n - ndiag),
                    "normal_explicit": float(ne), "normal_reconstruction": float(n - ne),
                    "raw_gradient_normal": float(nr), "all_exact_Gram_identities_passed": True})
            multipliers.append(block)
            result["complete_same_kappa_blocks"] = multipliers
            flush("complete_Gram_kappa_" + str(kappa))
        diagnostics = load("failed_diagnostics")
        measured = load("failed_ledger")
        trials, centered_values = [], []
        for i, d in enumerate(diagnostics):
            if d["optimizer_adapter"] or d["certified_error_bound"] or not d["diagnostic_only"]:
                raise ValueError("Prior finite increment scope changed")
            for target in d["targets"]:
                if target["weight"] != 0.2 or target["selected_reference"] != target["selected_trial"]:
                    raise ValueError("Prior matched target/weight changed")
            radial = f(d["explicit_radial"]) + f(d["reconstruction_radial"])
            rayleigh = radial + f(d["rayleigh_response"])
            centered = rayleigh - normal[i]
            centered_values.append(centered)
            trials.append({"sign": ["plus", "minus"][i], "rayleigh_change": float(rayleigh),
                "finite_CI_Rayleigh_response_retained": d["rayleigh_response"],
                "finite_CI_reported_response_retained": d["reported_response"],
                "fixed_Lambda_full_normal": float(normal[i]), "raw_gradient_normal": float(raw_normal[i]),
                "centered_rayleigh_change": float(centered),
                "centered_explicit_radial": float(f(d["explicit_radial"]) - normal_exp[i]),
                "centered_reconstruction_radial": float(f(d["reconstruction_radial"]) - normal[i] + normal_exp[i]),
                "component_sum_minus_saved_rayleigh_change": float(rayleigh - f(d["rayleigh_change"])),
                "original_direct_retained_quotient_difference": d["direct_retained_quotient_difference"],
                "original_direct_reported_root_difference": d["direct_reported_root_difference"]})
        central = (centered_values[0] - centered_values[1]) / (2 * h)
        analytic_centered = ideal_g_action - normal_ideal
        analytic_original = f(measured["analytic_saved_g"])
        threshold = f(contract["fixed"]["unchanged_derivative_gate"])
        result.update(trials=trials, original_analytic=float(analytic_original),
            conditional_actual_ambient_analytic=float(ideal_g_action), ideal_normal_derivative=float(normal_ideal),
            conditional_centered_analytic=float(analytic_centered), centered_Rayleigh_central=float(central),
            centered_error_vs_original=float(central - analytic_original),
            centered_error_vs_conditional_analytic=float(central - analytic_centered),
            unchanged_derivative_gate=float(threshold),
            same_saved_point_conditional_gate_passed=bool(abs(central - analytic_original) <= threshold
                and abs(central - analytic_centered) <= threshold),
            original_failed_Rayleigh_central=measured["Rayleigh_central"],
            original_direct_stored_H_quotient_central=measured["direct_retained_quotient_central"],
            known_unresolved=["baseline mixed CI weights/norm response not newly replayed or fully normalized",
                "unfiltered-minus-retained two-body gradient contribution is not bounded",
                "finite fixed-Lambda centering remainder has no uniform sign/error bound",
                "archived finite radial polynomial is float64 and retains its uncertified closure gap"],
            exact_products=products, elapsed_s=time.monotonic() - begin,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
            authorization_sha256=sha(Path(args.authorization)), CPU=31, accounting_CPUs=accounting,
            saved_only_diagnosis_completed=True)
        if any(n == "atomx" or n.startswith(("atomx.", "scipy")) for n in sys.modules):
            raise ValueError("Forbidden physical/eigensolver import")
        if any(sha(REPO / role["path"]) != role["sha256"] for role in contract["inputs"].values()):
            raise ValueError("Saved input changed during audit")
        write(output / "result.json", result)
        print(json.dumps({k: result[k] for k in ("saved_only_diagnosis_completed", "same_saved_point_conditional_gate_passed",
              "centered_Rayleigh_central", "centered_error_vs_original", "exact_products", "elapsed_s")}))
    except BaseException as error:
        import traceback
        flush("failed")
        write(output / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
              "partial": result, "exact_products": products, "physical_calls": 0,
              "original_failed_gate_unchanged": True})
        raise


if __name__ == "__main__":
    main()
