"""Complete-spectrum numerical refinement copied from qualified firstκ method.

Only coordinate algebra on original saved physical h/S/B; no physical builders.
Reference driver SHA ea5287bcbbe316e1fd71096f020b3943b82b1367493cf12cf530c2f880b1f476.
"""
from __future__ import annotations

import time


def refine(np, *, h, s, b, initial, caps, gates, calls, check, save):
    started = time.monotonic()
    h, s, b, initial = [x.astype(np.longdouble) for x in (h, s, b, initial)]
    n = b.shape[1]
    sweep_rows = []
    limits = dict(maximum_complete_sweeps=caps['maximum_complete_sweeps_each_spectrum'],
                  maximum_attempted_pair_rotations=caps['maximum_attempted_pair_rotations'],
                  maximum_saved_residual_evaluations=caps['maximum_saved_residual_evaluations'])
    if calls['complete_spectrum_refinement_attempts'] >= caps['full_spectrum_refinement_attempts']:
        raise RuntimeError('Four complete-spectrum refinement budget exhausted')
    calls['complete_spectrum_refinement_attempts'] += 1
    try:
        projected_h, projected_s = b.T @ h @ b, b.T @ s @ b
        ah, ms = (projected_h + projected_h.T) / 2, (projected_s + projected_s.T) / 2
        gram = initial.T @ ms @ initial
        defect = (gram + gram.T) / 2 - np.eye(n, dtype=np.longdouble)
        check("complete_basis_near_identity_Gram_inf", np.max(np.sum(abs(defect), axis=1)), 1e-10)
        calls["near_identity_Gram_whitenings"] += 1
        d2, d3 = defect @ defect, defect @ defect @ defect
        d4 = d2 @ d2
        whitening = np.eye(n, dtype=np.longdouble) - defect / 2 + 3 * d2 / 8 - 5 * d3 / 16 + 35 * d4 / 128
        coefficient_basis = initial @ whitening
        check("whitened_actual_Gram", np.max(abs(coefficient_basis.T @ projected_s @ coefficient_basis - np.eye(n))),
              gates["screened_full_spectrum_SGram_error"])
        reduced = coefficient_basis.T @ ah @ coefficient_basis
        reduced = (reduced + reduced.T) / 2
        original_reduced = reduced.copy()
        rotations = np.eye(n, dtype=np.longdouble)
        save("initial_complete_representation", projected_h_longdouble=projected_h, projected_s_longdouble=projected_s,
             symmetric_projected_h_longdouble=ah, symmetric_projected_s_longdouble=ms,
             initial_complete_coefficients=initial, initial_Gram=gram, whitening=whitening,
             coefficient_basis=coefficient_basis, initial_reduced_operator=original_reduced,
             original_h=h, original_s=s, basis=b)

        def physical_state():
            calls["physical_residual_evaluations"] += 1
            if calls["physical_residual_evaluations"] > limits["maximum_saved_residual_evaluations"]:
                raise RuntimeError("Predeclared full residual evaluation budget exhausted")
            coeff = coefficient_basis @ rotations
            vectors = np.asarray(b @ coeff, dtype=float)
            vv = vectors.astype(np.longdouble)
            norms = np.sum(vv * (s @ vv), axis=0)
            check("complete_physical_norms_positive/" + str(calls["physical_residual_evaluations"]),
                  np.any(norms <= 0) or np.any(~np.isfinite(norms)))
            vectors /= np.asarray(np.sqrt(norms), dtype=float)
            vv = vectors.astype(np.longdouble)
            hv, sv = h @ vv, s @ vv
            norms_after = np.sum(vv * sv, axis=0)
            rayleigh = np.sum(vv * hv, axis=0) / norms_after
            constrained = b.T @ (hv - sv * rayleigh)
            absolute = np.sqrt(np.sum(constrained**2, axis=0))
            relative = absolute / np.maximum(1., np.maximum(abs(rayleigh), np.sqrt(np.sum((b.T @ hv)**2, axis=0))))
            gram_after = vv.T @ sv
            off = reduced - np.diag(np.diag(reduced))
            off_relative = np.sqrt(np.sum(off**2, axis=0)) / np.maximum(1., abs(np.diag(reduced)))
            return dict(coefficients=coeff, normalized_physical_vectors=vectors, normalization_factors=np.sqrt(norms), norm_after=norms_after,
                        actual_Rayleigh_Ha=rayleigh, actual_full_SGram=gram_after,
                        full_constrained_residual=constrained, full_constrained_residual_norm_Ha=absolute,
                        full_constrained_residual_relative=relative, reduced_offdiagonal_relative=off_relative)

        before = physical_state()
        save("before_refinement", **before)
        converged = False
        final = None
        for sweep in range(1, limits["maximum_complete_sweeps"] + 1):
            if time.monotonic() - started > caps["remaining4_refinement_wall_cap_each_s"]:
                raise TimeoutError("Remainingκ numerical refinement120s cap")
            calls["attempted_complete_sweeps"] += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    calls["attempted_pair_rotations"] += 1
                    if calls["attempted_pair_rotations"] > limits["maximum_attempted_pair_rotations"]:
                        raise RuntimeError("Predeclared complete Jacobi pair budget exhausted")
                    cross = reduced[i, j]
                    if cross == 0:
                        calls["skipped_exact_zero_pairs"] += 1
                        continue
                    difference = (reduced[j, j] - reduced[i, i]) / 2
                    t = (np.copysign(np.longdouble(1), cross) if difference == 0 else
                         np.copysign(np.longdouble(1), difference) * cross /
                         (abs(difference) + np.hypot(difference, cross)))
                    cosine = 1 / np.sqrt(1 + t * t)
                    sine = t * cosine
                    if not np.isfinite(t):
                        raise ValueError("Nonfinite Jacobi rotation")
                    left, right = reduced[:, i].copy(), reduced[:, j].copy()
                    ii, jj = reduced[i, i], reduced[j, j]
                    rotated_left, rotated_right = cosine * left - sine * right, sine * left + cosine * right
                    reduced[:, i], reduced[i, :] = rotated_left, rotated_left
                    reduced[:, j], reduced[j, :] = rotated_right, rotated_right
                    reduced[i, i], reduced[j, j] = ii - t * cross, jj + t * cross
                    reduced[i, j], reduced[j, i] = 0, 0
                    left, right = rotations[:, i].copy(), rotations[:, j].copy()
                    rotations[:, i], rotations[:, j] = cosine * left - sine * right, sine * left + cosine * right
                    calls["completed_pair_rotations"] += 1
            calls["completed_complete_sweeps"] += 1
            final = physical_state()
            row = dict(sweep=sweep, full_root_count=n,
                       original_full_physical_residual_relative_max=float(np.max(final["full_constrained_residual_relative"])),
                       full_Gram_error=float(np.max(abs(final["actual_full_SGram"] - np.eye(n)))),
                       complete_reduced_offdiagonal_relative_max=float(np.max(final["reduced_offdiagonal_relative"])))
            sweep_rows.append(row)
            save("complete_sweep_" + str(sweep).zfill(2), **final, jacobi_matrix=reduced, full_rotations=rotations)
            if (row["original_full_physical_residual_relative_max"] <= gates["screened_full_spectrum_constrained_residual_relative"]
                    and row["full_Gram_error"] <= gates["screened_full_spectrum_SGram_error"]
                    and row["complete_reduced_offdiagonal_relative_max"] <= caps["Jacobi_reduced_offdiagonal_relative_norm"]):
                converged = True
                break
        check("complete_spectrum_refinement_converged_within_budget", not converged)
        calls["completed_complete_spectrum_refinements"] += 1

        order = np.argsort(final["actual_Rayleigh_Ha"], kind="stable")
        return dict(physical=final["normalized_physical_vectors"][:, order],
                    coefficients=final["coefficients"][:, order],
                    coordinate_diagonal_Ha=np.diag(reduced)[order],
                    actual_Rayleigh_Ha=final["actual_Rayleigh_Ha"][order],
                    normalization_factors=final["normalization_factors"][order],
                    original_to_sorted_indices=order, sweeps=sweep_rows,
                    full_final_rotations=rotations, final_reduced_operator=reduced,
                    numerical_method="same qualified firstκ complete fullJacobi")
    except Exception:
        if "reduced" in locals() and "rotations" in locals():
            save("failed_partial_iteration", jacobi_matrix=reduced, full_rotations=rotations,
                 coefficient_basis=coefficient_basis)
        raise
