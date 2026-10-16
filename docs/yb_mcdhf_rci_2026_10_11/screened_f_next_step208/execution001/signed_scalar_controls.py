"""Public closed-shell scalar contraction versus native field bilinears.

No imports or numerical execution at import. The admitted caller supplies
the original workspace and native radial/angular functions. Component probes
are unnormalized signed numerical vectors, not physical orbital states.
"""


def validate(np, compute_slater_rk_cell, closed_core_one_body, *, core_rows,
             waves, probe_seeds, fields, workspace, cache, caps, gates, invoke,
             check, archive, write):
    core = {tuple(row["key"]): row["occupation"] for row in core_rows}
    probes, descriptors = {}, []
    for new in ((5, 3), (5, -4)):
        for origin, pair in (("literal208_original4f", waves[4, new[1]]),
                             ("saved_qualified_nuclear5f", probe_seeds[new])):
            for component in ("P", "Q"):
                key = (100 + len(descriptors), new[1])
                p, q = pair
                probes[key] = ((p.copy(), np.zeros(123)) if component == "P"
                               else (np.zeros(123), q.copy()))
                descriptors.append(dict(key=key, origin=origin, component=component,
                                        physical_orbital=False, renormalized=False))
    check("scalar8_component_probes_disjoint_closed68", len(probes) != 8 or bool(set(probes) & set(core)))
    all_waves = {key: waves[key] for key in core} | probes
    bare = {(a, b): 0. for a in probes for b in probes if a[1] == b[1]}
    radial_cache = {}

    def radial(key):
        def evaluate():
            if key[0] != "r":
                raise ValueError("Independent scalar oracle admits only two-body radial keys")
            if key not in radial_cache:
                _, rank, a, b, c, d = key
                radial_cache[key] = invoke("scalar_Slater_unique_calls", compute_slater_rk_cell,
                    *all_waves[a], *all_waves[b], *all_waves[c], *all_waves[d],
                    workspace, rank, cache=cache)
            return radial_cache[key]
        return invoke("scalar_radial_callbacks", evaluate)

    analytic = invoke("closed_core_scalar", closed_core_one_body, probes, core,
                      radial, bare_one_body=bare)
    check("all32_ordered_samek_scalar_entries", len(analytic) != 32 or set(analytic) != set(bare))
    rows, bilinears = [], []
    # Save BOTH complete 4x4 matrices (all32 signed entries) before any
    # field-versus-scalar numerical gate. No coefficient magnitude screening.
    for kappa in (3, -4):
        keys = [key for key in probes if key[1] == kappa]
        vectors = np.column_stack([np.concatenate(probes[key]) for key in keys])
        scalar = np.array([[analytic[a, b] for b in keys] for a in keys])
        dense = vectors.T @ fields[kappa]["JK"] @ vectors
        archive("signed_scalar_quadrants_" + str(kappa), probe_keys=np.array(keys),
                probe_vectors=vectors, scalar_closed_core_JK_Ha=scalar, dense_native_JK_Ha=dense)
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                bilinears.append(dict(kappa=kappa, bra=a, ket=b,
                    scalar_Ha=float(scalar[i, j]), native_Ha=float(dense[i, j]),
                    signed_difference_Ha=float(dense[i, j] - scalar[i, j])))
        for quadrant, rr, cc in (("PP", [0, 2], [0, 2]), ("QQ", [1, 3], [1, 3]),
                                 ("PQ", [0, 2], [1, 3]), ("QP", [1, 3], [0, 2])):
            oracle, observed = scalar[np.ix_(rr, cc)], dense[np.ix_(rr, cc)]
            error, magnitude = float(np.max(abs(oracle - observed))), float(np.max(abs(oracle)))
            rows.append(dict(kappa=kappa, quadrant=quadrant, error_Ha=error,
                magnitude_Ha=magnitude, relative_error=error / max(magnitude, 1e-300)))
    summary = dict(rows=rows, signed_bilinears=bilinears, probes=descriptors, ordered_bilinears=32,
        unique_Slater_keys=len(radial_cache), zero_bare_one_body=True,
        internal_retry_actual_count_not_observed=True,
        existing_internal_Slater_retry_upper_bound=3 * len(radial_cache),
        anchor_scope="Original4f plus saved nuclear5f component probes; operator qualification before Fock spectra/CI. Old nuclear late counter FAIL remains.")
    write("signed_scalar_controls", summary)
    for row in bilinears:
        check("signed_scalar_entry_" + str((row["bra"], row["ket"])),
              abs(row["signed_difference_Ha"]), gates["auxiliary_scalar_bilinear_error_Ha"])
    for row in rows:
        suffix = row["quadrant"] + "/" + str(row["kappa"])
        if row["quadrant"] in ("PQ", "QP"):
            check("scalar_cross_nonzero_" + suffix,
                  row["magnitude_Ha"] <= gates["auxiliary_cross_bilinear_magnitude_strict_min_Ha"])
            check("scalar_cross_relative_" + suffix, row["relative_error"],
                  gates["auxiliary_scalar_PQ_QP_relative_error"])
    check("scalar_triangle_predeclared_callback_bound", len(radial_cache), caps["scalar_triangle_only_callback_upper"])
    return summary
