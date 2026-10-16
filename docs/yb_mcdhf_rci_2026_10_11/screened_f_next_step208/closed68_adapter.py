"""Source-only seed-field consumer; no imports, executor or physical kernels.

Arguments provide the separately admitted immutable f043 native classes/workspace.
The caller owns exact source/point admission, resource counters and saved gates.
"""


def build_fields(np, DFSolver, Subshell, *, workspace, cache, core_rows, waves,
                 target_keys, nuclear_h, invoke, check, archive):
    core = {tuple(row["key"]): row["occupation"] for row in core_rows}
    targets = tuple(map(tuple, target_keys))
    check("closed68_count", len(core) != 19 or sum(core.values()) != 68)
    check("closed_shell_capacities", any(q != 2 * abs(key[1]) for key, q in core.items()))
    check("fixed_two_f_targets", targets != ((5, 3), (5, -4)))
    check("zero_target_keys_disjoint_core", bool(set(targets) & set(core)))
    check("all_saved_core_sources_present", not set(core) <= set(waves))
    keys = tuple(sorted(set(core) | set(targets)))
    check("ordered21_field_keys", len(keys) != 21)
    ns = int(workspace["ns"])
    check("saved_ns123", ns != 123)
    check("shared_cache_grid_identity", not cache.same_grid(workspace))
    shells = [Subshell(*key, core.get(key, 0)) for key in keys]
    numerator = np.array([[core.get(key, 0) for key in keys] for _ in keys], dtype=float)
    divisor = np.ones(len(keys))
    check("source_weights_finite", not np.all(np.isfinite(numerator)))
    orbitals = {}
    for key, shell in zip(keys, shells, strict=True):
        p, q = waves[key] if key in core else (np.zeros(ns), np.zeros(ns))
        check("core_source_shape_" + str(key), p.shape != (ns,) or q.shape != (ns,)
              or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)))
        orbitals.setdefault(shell.kappa, []).append((p, q, shell.occupation))
    archive("field_inputs", ordered_keys=np.array(keys), numerator=numerator, divisor=divisor,
            core_keys=np.array(sorted(core)), core_occupations=np.array([core[k] for k in sorted(core)]))
    variants = []
    for scale in (1., 3.):
        # The existing audited helper consumes only these attributes. Constructor
        # or SCF would introduce a new workspace/h/S and is explicitly excluded.
        solver = DFSolver.__new__(DFSolver)
        solver.subshells = shells
        solver.mean_field_coeffs = scale * numerator
        solver.mean_field_normalization = scale * divisor
        solver.bsp_data = workspace
        solver._convol_cache = cache
        invoke("auxiliary_angular_precompute", solver._precompute_angular_coefficients, None)
        variants.append(solver)
    fields = {}
    for key in targets:
        results = [invoke("auxiliary_JK_builder", solver._build_hf_matrix_dbsr,
                          keys.index(key), orbitals, skip_pq_exchange=False) for solver in variants]
        pp, qq, pq, qp = results[0]
        jk = np.block([[pp, pq], [qp, qq]])
        hn = nuclear_h[key[1]]  # Already saved functional L(raw), never raw nonsymmetric h.
        check("saved_logical_h_shape_" + str(key), hn.shape != (2 * ns, 2 * ns)
              or not np.all(np.isfinite(hn)))
        full = hn + jk
        archive("field_" + str(key), primary_PP=pp, primary_QQ=qq, primary_PQ=pq, primary_QP=qp,
                scaled_PP=results[1][0], scaled_QQ=results[1][1],
                scaled_PQ=results[1][2], scaled_QP=results[1][3],
                logical_nuclear_h=hn, JK_Ha=jk, auxiliary_Fock_Ha=full)
        check("field_finite_" + str(key), not np.all(np.isfinite(full)))
        for quadrant, a, b in zip(("PP", "QQ", "PQ", "QP"), results[0], results[1], strict=True):
            check("row_scale_" + quadrant + str(key), np.max(abs(a - b)), 5e-9)
        check("PQ_QP_transpose_" + str(key), np.max(abs(pq - qp.T)), 1e-10)
        check("JK_Hermitian_" + str(key), np.max(abs(jk - jk.T)), 1e-10)
        check("full_Hermitian_" + str(key), np.max(abs(full - full.T)), 1e-10)
        fields[key[1]] = {"F": full, "JK": jk}
    return fields
