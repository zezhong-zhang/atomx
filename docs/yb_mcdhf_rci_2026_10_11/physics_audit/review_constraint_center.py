"""Signed finite-cell anchor for a separately constrained energy extension."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from review_stable_difference import independent_density

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.state_average import CIStateAverage

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mp_array(value):
    return mp.matrix([[mp.mpf(str(x)) for x in row] for row in value])


def symmetry(value):
    return (value + value.T) / 2


def max_abs(value):
    return max(abs(x) for x in value)


def compressed_energy(functional, weights, basis):
    """Full physical energy, compressed from actual reference/increment arrays.

    This is independent of production pair potentials, finite differences,
    and gradients. Compression uses the entire four-component cell tensor.
    """
    ns, size = functional.ns, basis.shape[1]
    basis = np.asarray(basis, dtype=np.longdouble)
    orders = (functional.data.get("ksp", functional.data["ks"]),
              functional.data.get("ksq", functional.data["ks"]))
    pairs = tuple(itertools.product(range(size), repeat=2))
    densities = []
    for side in (0, 1):
        part = basis[side * ns:(side + 1) * ns]
        densities.append(np.stack([
            independent_density(part[:, a], part[:, b], ns, orders[side]).T.ravel()
            for a, b in pairs]))
    kernels = {}
    for rank in sorted({term[1] for term in weights if term[0] == "r"}):
        tensor = np.zeros((size**2, size**2), dtype=np.longdouble)
        for left, pl in enumerate(("p", "q")):
            for right, pr in enumerate(("p", "q")):
                bank, extent, _ = functional.cache.get_rkb_t(rank, pl + pr + pl + pr)
                assert extent == ns
                cell = bank.transpose(0, 2, 1, 3).reshape(ns * orders[left], ns * orders[right])
                tensor += densities[left] @ cell.astype(np.longdouble) @ densities[right].T
        kernels[rank] = mp_array(tensor)
    h = basis.T @ functional.one_body[-1].astype(np.longdouble) @ basis
    s = basis.T @ functional.metric[-1].astype(np.longdouble) @ basis
    h, s = mp_array((h + h.T) / 2), mp_array((s + s.T) / 2)
    columns = {key: i for i, key in enumerate(functional.keys)}

    def pair(x, a, b):
        return mp.matrix([x[i, a] * x[j, b] for i, j in pairs])

    def energy(x):
        total = mp.mpf(0)
        for term, weight in weights.items():
            coefficient = mp.mpf(float(weight))
            if term[0] == "h":
                _, na, kappa, nb = term
                a, b = columns[na, kappa], columns[nb, kappa]
                va, vb = x[:, a], x[:, b]
                total += coefficient * (va.T * h * vb)[0] / (vb.T * s * vb)[0]
            else:
                _, rank, a, b, c, d = term
                pa, pb = pair(x, columns[a], columns[b]), pair(x, columns[c], columns[d])
                total += coefficient * (pa.T * kernels[rank] * pb)[0]
        return total

    def polar(x):
        values, vectors = mp.eigsy(x.T * s * x)
        return x * vectors * mp.diag([1 / mp.sqrt(v) for v in values]) * vectors.T

    return energy, polar, s


def main():
    source_paths = (
        "src/atomx/methods/csf/variational.py",
        "src/atomx/physics/csf/radial_density.py",
        "src/atomx/numerics/bspline/convol.py",
        "src/atomx/physics/csf/matrix.py",
        "src/atomx/numerics/bspline/orbital_optimization.py",
    )
    before = {p: sha(ROOT / p) for p in source_paths}
    ns, dimension = 35, 4
    data = prepare_bspline_data(make_knots(ns, 9, .005, 30), ns, 9)
    csfs = enumerate_jj_csfs("1s1 2s1")
    functional = CSFOrbitalFunctional(csfs, data, 4, state_average=CIStateAverage())
    nuclear = solve_dirac(functional.one_body[-1], functional.metric[-1], ns, -1)
    radial = nuclear.coefficients[:, :dimension]
    wide = radial.astype(np.longdouble)
    reference = {key: (radial[:ns, i], radial[ns:, i])
                 for i, key in enumerate(functional.keys)}
    _, ci = functional.evaluate(reference, gradient=False)
    weights = functional.radial_density.contract(ci, functional.state_average.weights(csfs, ci))
    orders = data.get("ksp", data["ks"]), data.get("ksq", data["ks"])
    pairs = tuple(itertools.product(range(dimension), repeat=2))
    densities = []
    for side in (0, 1):
        component = wide[side * ns:(side + 1) * ns]
        densities.append(np.stack([
            independent_density(component[:, a], component[:, b], ns, orders[side]).T.ravel()
            for a, b in pairs]))
    ranks = sorted({term[1] for term in weights if term[0] == "r"})
    tensors = {}
    for rank in ranks:
        tensor = np.zeros((dimension**2, dimension**2), dtype=np.longdouble)
        for left, label_left in enumerate(("p", "q")):
            for right, label_right in enumerate(("p", "q")):
                bank, extent, _ = functional.cache.get_rkb_t(
                    rank, label_left + label_right + label_left + label_right)
                assert extent == ns
                kernel = bank.transpose(0, 2, 1, 3).reshape(ns * orders[left], ns * orders[right])
                tensor += densities[left] @ kernel.astype(np.longdouble) @ densities[right].T
        tensors[rank] = tensor
    one_body = wide.T @ functional.one_body[-1].astype(np.longdouble) @ wide
    metric = wide.T @ functional.metric[-1].astype(np.longdouble) @ wide
    compression_asymmetry = {
        "one_body_Ha": str(np.max(abs(one_body - one_body.T))),
        "metric": str(np.max(abs(metric - metric.T))),
    }
    # H and S are physically Hermitian. Accumulating their subspace
    # projections in long double introduces last-bit asymmetry. Restore
    # the same mathematical operator before exact mp algebra checks.
    one_body, metric = (one_body + one_body.T) / 2, (metric + metric.T) / 2

    with mp.workdps(80):
        h, s = mp_array(one_body), mp_array(metric)
        kernel = {rank: mp_array(tensor) for rank, tensor in tensors.items()}
        columns = {key: i for i, key in enumerate(functional.keys)}
        angular = [(term, mp.mpf(float(coefficient))) for term, coefficient in weights.items()]

        def pair(x, a, b):
            return mp.matrix([x[i, a] * x[j, b] for i, j in pairs])

        def energy(x):
            total = mp.mpf(0)
            for term, coefficient in angular:
                if term[0] == "h":
                    _, na, kappa, nb = term
                    a, b = columns[na, kappa], columns[nb, kappa]
                    va, vb = x[:, a], x[:, b]
                    total += coefficient * (va.T * h * vb)[0] / (vb.T * s * vb)[0]
                else:
                    _, rank, a, b, c, d = term
                    pa = pair(x, columns[a], columns[b])
                    pb = pair(x, columns[c], columns[d])
                    total += coefficient * (pa.T * kernel[rank] * pb)[0]
            return total

        def actual_gradient(x):
            result = mp.zeros(dimension, 2)
            for term, coefficient in angular:
                if term[0] == "h":
                    _, na, kappa, nb = term
                    a, b = columns[na, kappa], columns[nb, kappa]
                    va, vb = x[:, a], x[:, b]
                    norm = (vb.T * s * vb)[0]
                    numerator = (va.T * h * vb)[0]
                    result[:, a] += coefficient * h * vb / norm
                    result[:, b] += coefficient * (h * va / norm - 2 * numerator * s * vb / norm**2)
                else:
                    _, rank, *keys = term
                    roles = [columns[key] for key in keys]
                    for indices in itertools.product(range(dimension), repeat=4):
                        i, j, k, ell = indices
                        value = coefficient * kernel[rank][i * dimension + j, k * dimension + ell]
                        for position in range(4):
                            product = value
                            for other in range(4):
                                if other != position:
                                    product *= x[indices[other], roles[other]]
                            result[indices[position], roles[position]] += product
            return result

        def polar(x):
            values, vectors = mp.eigsy(x.T * s * x)
            inverse = vectors * mp.diag([1 / mp.sqrt(v) for v in values]) * vectors.T
            return x * inverse

        base = mp.matrix([[mp.cos(.08), -mp.sin(.08)],
                          [mp.sin(.08), mp.cos(.08)], [mp.mpf('.03'), 0],
                          [0, mp.mpf('.02')]])
        ideal_base = polar(base)
        # Stored float coordinates determine the actual finite endpoint.
        base = mp.matrix(np.asarray(ideal_base.tolist(), dtype=float).tolist())
        base_energy = energy(base)
        gradient = actual_gradient(base)
        automatic = mp.zeros(dimension, 2)
        for i, j in itertools.product(range(dimension), range(2)):
            def varying(value, i=i, j=j):
                point = base.copy()
                point[i, j] = value
                return energy(point)
            automatic[i, j] = mp.diff(varying, base[i, j])
        gradient_error = max_abs(gradient - automatic)
        multiplier = symmetry(base.T * gradient) / 2

        physical = radial @ np.array(base.tolist(), dtype=float)
        waves = {key: (physical[:ns, i], physical[ns:, i])
                 for i, key in enumerate(functional.keys)}
        native = functional.gradient_from_weights(waves, weights)
        for term, coefficient in weights.items():
            if term[0] != "h":
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            va, vb = np.concatenate(waves[a]), np.concatenate(waves[b])
            full_h, full_s = functional.one_body[kappa], functional.metric[kappa]
            norm, numerator = vb @ full_s @ vb, va @ full_h @ vb
            native[a] += coefficient * (1 / norm - 1) * (full_h @ vb)
            native[b] += coefficient * ((1 / norm - 1) * (full_h @ va)
                                       - 2 * numerator / norm**2 * (full_s @ vb))
        projected = radial.T @ np.column_stack([native[key] for key in functional.keys])
        native_error = float(np.max(abs(projected - np.array(gradient.tolist(), dtype=float))))

        raw = mp.matrix([[0, 0], [0, 0], [1, mp.mpf('.3')], [mp.mpf('-.2'), mp.mpf('.8')]])
        direction = raw - ideal_base * symmetry(ideal_base.T * s * raw)
        direction /= mp.sqrt(mp.fsum(direction[i, j] * (s * direction)[i, j]
                                     for i, j in itertools.product(range(dimension), range(2))))
        normal = mp.matrix([[mp.mpf('.6'), mp.mpf('.3')],
                            [mp.mpf('.3'), mp.mpf('-.4')]])
        controls = [("true_finite_polar_curvature", mp.mpf('1e-3'), mp.mpf('0')),
                    ("tiny_step_positive_normal", mp.mpf('1e-12'), mp.mpf('1e-15')),
                    ("tiny_step_negative_normal", mp.mpf('1e-12'), mp.mpf('-1e-15')),
                    ("smaller_step_normal", mp.mpf('1e-13'), mp.mpf('1e-15')),
                    ("pure_normal", mp.mpf('0'), mp.mpf('1e-15')),
                    ("finite_constraint_defect", mp.mpf('1e-12'), mp.mpf('1e-8'))]
        rows = []
        api_rows = []
        gate = 64 * np.finfo(float).eps
        for name, step, defect in controls:
            ideal = polar(ideal_base + step * direction)
            raw_trial = ideal * (mp.eye(2) + defect * normal)
            trial = mp.matrix(np.asarray(raw_trial.tolist(), dtype=float).tolist())
            delta = trial - base
            base_gram, trial_gram = base.T * s * base, trial.T * s * trial
            cross = base.T * s * delta
            gram_delta = cross + cross.T + delta.T * s * delta
            gram_identity_error = max_abs(gram_delta - (trial_gram - base_gram))
            normal_change = mp.fsum(multiplier[i, j] * gram_delta[i, j]
                                   for i, j in itertools.product(range(2), repeat=2))
            full_change = energy(trial) - base_energy
            centered = full_change - normal_change
            physical_change = energy(polar(trial)) - energy(polar(base))
            linear_only = mp.fsum(multiplier[i, j] * (cross + cross.T)[i, j]
                                  for i, j in itertools.product(range(2), repeat=2))
            offdiagonal = mp.fsum(multiplier[i, j] * gram_delta[i, j]
                                  for i, j in ((0, 1), (1, 0)))
            base_defect = max_abs(base_gram - mp.eye(2))
            trial_defect = max_abs(trial_gram - mp.eye(2))
            supported = base_defect <= gate and trial_defect <= gate
            row = {
                "control": name, "tangent_step": str(step),
                "base_gram_defect": str(base_defect), "trial_gram_defect": str(trial_defect),
                "full_change_Ha": str(full_change), "full_Gram_normal_change_Ha": str(normal_change),
                "centered_change_Ha": str(centered), "canonical_metric_polar_change_Ha": str(physical_change),
                "centered_minus_canonical_Ha": str(centered - physical_change),
                "wrong_linear_Gram_center_minus_full_center_Ha": str(normal_change - linear_only),
                "offdiagonal_normal_contribution_Ha": str(offdiagonal),
                "Gram_increment_identity_abs": str(gram_identity_error),
                "roundoff_endpoint_guard_supported": bool(supported),
                "passed": bool(gram_identity_error < mp.mpf('1e-60')
                               and ((not supported) if name == "finite_constraint_defect" else supported)),
            }
            rows.append(row)
            # Use the declared linear-map increment, not subtract two
            # separately reconstructed orbital arrays. The oracle span
            # contains these actual float reference/increment arrays.
            change = radial @ np.asarray(delta.tolist(), dtype=float)
            scale = np.longdouble(np.max(abs(change)))
            actual_trial = physical + change
            newer = {key: (actual_trial[:ns, i], actual_trial[ns:, i])
                     for i, key in enumerate(functional.keys)}
            increments = {key: (change[:ns, i], change[ns:, i])
                          for i, key in enumerate(functional.keys)}
            _, newer_ci = functional.evaluate(newer, gradient=False)
            native_delta = functional.constrained_energy_difference(
                waves, newer, ci, newer_ci, increments=increments)
            span = np.column_stack((physical.astype(np.longdouble),
                                    change.astype(np.longdouble) / scale))
            full_energy, normalize, actual_s = compressed_energy(functional, weights, span)
            actual_base = mp.zeros(dimension, 2)
            actual_base[0, 0] = actual_base[1, 1] = 1
            actual_new = actual_base.copy()
            actual_new[2, 0] = actual_new[3, 1] = mp.mpf(str(scale))
            actual_gradient_mp = mp.zeros(dimension, 2)
            for i, j in itertools.product(range(dimension), range(2)):
                def varying_actual(value, i=i, j=j, anchor=actual_base, objective=full_energy):
                    point = anchor.copy()
                    point[i, j] = value
                    return objective(point)
                actual_gradient_mp[i, j] = mp.diff(varying_actual, actual_base[i, j])
            actual_multiplier = symmetry(actual_base.T * actual_gradient_mp) / 2
            actual_dg = actual_new.T * actual_s * actual_new - actual_base.T * actual_s * actual_base
            actual_center = (full_energy(actual_new) - full_energy(actual_base)
                             - mp.fsum(actual_multiplier[i, j] * actual_dg[i, j]
                                       for i, j in itertools.product(range(2), repeat=2)))
            canonical = full_energy(normalize(actual_new)) - full_energy(normalize(actual_base))
            api_error = None if native_delta is None else abs(mp.mpf(native_delta) - actual_center)
            error_gate = mp.mpf('3e-12') if step > mp.mpf('1e-8') else mp.mpf('3e-24')
            accepted = (native_delta is None if name == "finite_constraint_defect"
                        else native_delta is not None and api_error < error_gate)
            api_rows.append({
                "control": name, "actual_constrained_API_difference_Ha": native_delta,
                "independent_full_centered_difference_Ha": str(actual_center),
                "independent_canonical_normalized_difference_Ha": str(canonical),
                "centered_minus_canonical_Ha": str(actual_center - canonical),
                "API_minus_independent_centered_abs_Ha": None if api_error is None else str(api_error),
                "API_absolute_error_gate_Ha": str(error_gate),
                "passed": bool(accepted),
            })
        report = {
            "contract_sha256": sha(OWNER / "constraint_center_contract.json"),
            "actual_API_contract_sha256": sha(OWNER / "constraint_center_api_contract.json"),
            "script_sha256": sha(Path(__file__)),
            "source_hashes": before,
            "source_unchanged_during_run": before == {p: sha(ROOT / p) for p in source_paths},
            "precision_decimal_digits": 80,
            "fixed_cell_compression_mantissa_bits": int(np.finfo(np.longdouble).nmant),
            "pre_symmetrization_compressed_asymmetry": compression_asymmetry,
            "analytic_actual_vs_automatic_gradient_abs": str(gradient_error),
            "native_projected_quotient_gradient_abs_Ha": native_error,
            "same_kappa_offdiagonal_multiplier_Ha": str(multiplier[0, 1]),
            "rows": rows,
            "actual_API_rows": api_rows,
            "passed": bool(gradient_error < mp.mpf('1e-60') and native_error < 8e-12
                           and all(row["passed"] for row in rows)
                           and all(row["passed"] for row in api_rows)
                           and before == {p: sha(ROOT / p) for p in source_paths}),
            "limits": "Independent actual-API review, not optimizer acceptance; finite-defect centering is not literally identical to canonical metric-polar normalized energy. Both endpoint guards and sign-error margin are necessary. Cell/subspace compression is 63-bit; 80 decimal arithmetic resolves subtraction and normalization of those fixed Hermitian operators.",
        }
    (OWNER / "constraint_center_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
