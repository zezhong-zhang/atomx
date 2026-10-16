"""Prospective auxiliary closed68 field; no physical imports or calls at import."""


def construct(np, *, here, geometry, waves, old_keys, new_keys, raw, data,
              core_rows, caps, gates, calls, check, save, write, sha, stages, progress):
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.numerics.bspline.convol import compute_slater_rk_cell
    from atomx.physics.csf.frozen_core import closed_core_one_body

    core = {tuple(row['key']): row['occupation'] for row in core_rows}
    check('auxiliary_closed68_count', sum(core.values()) != 68 or len(core) != 19, 0)
    check('auxiliary_closed_subshell_capacities', any(q != 2 * abs(k[1]) for k, q in core.items()), 0)
    check('auxiliary_all_core_waves_present', not set(core) <= set(waves), 0)
    check('auxiliary_zero_targets_disjoint_core', bool(set(new_keys) & set(core)), 0)
    keys = tuple(sorted(set(core) | set(new_keys)))
    subshells = [Subshell(n, k, core.get((n, k), 0)) for n, k in keys]
    coefficients = np.array([[core.get(key, 0) for key in keys] for _ in keys], dtype=float)
    normalizations = np.ones(len(keys))
    check('auxiliary_coefficients_shape_finite', coefficients.shape != (24, 24)
          or not np.all(np.isfinite(coefficients)), 0)
    check('auxiliary_divisors_positive_finite', normalizations.shape != (24,)
          or not np.all(np.isfinite(normalizations)) or np.min(normalizations) < 1, 0)
    check('auxiliary_cache_exact_grid', not geometry.cache.same_grid(data), 0)
    orbitals = {}
    for key, shell in zip(keys, subshells, strict=True):
        pq = waves[key] if key in core else (np.zeros(123), np.zeros(123))
        check('auxiliary_source_shape/' + str(key), any(x.shape != (123,) or not np.all(np.isfinite(x)) for x in pq), 0)
        orbitals.setdefault(shell.kappa, []).append((*pq, shell.occupation))
    check('auxiliary_complete_same_kappa_source_order', sum(map(len, orbitals.values())) != len(keys), 0)
    # Avoid constructor: it would create a second workspace and nuclear matrices.
    # These two audited native methods consume only the explicitly installed attrs.
    solvers = []
    for scale in (1., 3.):
        field = DFSolver.__new__(DFSolver)
        field.subshells = subshells
        field.mean_field_coeffs = scale * coefficients
        field.mean_field_normalization = scale * normalizations
        field.bsp_data = data
        field._convol_cache = geometry.cache
        if calls['auxiliary_angular_attempts'] >= caps['auxiliary_angular_precompute_calls_budget']:
            raise RuntimeError('Two angular precomputation budget exhausted')
        calls['auxiliary_angular_attempts'] += 1
        field._precompute_angular_coefficients(None)
        calls['completed_auxiliary_angular_precomputations'] += 1
        solvers.append(field)
    save(np, here / 'auxiliary_field_inputs.npz', ordered_keys=np.array(keys),
         raw_coefficients=coefficients, normalization=normalizations, scaled_coefficients=3 * coefficients,
         scaled_normalization=3 * normalizations, core_keys=np.array(sorted(core)),
         core_occupations=np.array([core[k] for k in sorted(core)]),
         source_P=np.array([waves[k][0] if k in core else np.zeros(123) for k in keys]),
         source_Q=np.array([waves[k][1] if k in core else np.zeros(123) for k in keys]))
    stages.append(dict(name='auxiliary_field_inputs', path='auxiliary_field_inputs.npz',
                       sha256=sha(here / 'auxiliary_field_inputs.npz')))
    save(np, here / 'auxiliary_angular_coefficients.npz',
         primary_direct=solvers[0]._angular_direct, primary_exchange=solvers[0]._angular_exchange,
         scaled_direct=solvers[1]._angular_direct, scaled_exchange=solvers[1]._angular_exchange)
    stages.append(dict(name='auxiliary_angular_coefficients', path='auxiliary_angular_coefficients.npz',
                       sha256=sha(here / 'auxiliary_angular_coefficients.npz')))
    fields, jk_fields = {}, {}
    for j, key in enumerate(new_keys):
        index = keys.index(key)
        values = []
        for variant, solver in zip(('primary', 'row_scaled'), solvers, strict=True):
            if calls['auxiliary_JK_builder_attempts'] >= caps['screened_JK_builder_calls_budget']:
                raise RuntimeError('Ten native auxiliary J/K builder budget exhausted')
            calls['auxiliary_JK_builder_attempts'] += 1
            progress('before_native_auxiliary_JK/' + variant + '/' + str(key))
            blocks = solver._build_hf_matrix_dbsr(index, orbitals, skip_pq_exchange=False)
            calls['completed_auxiliary_JK_builders'] += 1
            values.append(blocks)
        pp, qq, pq, qp = values[0]
        pp2, qq2, pq2, qp2 = values[1]
        jk = np.block([[pp, pq], [qp, qq]])
        h_nuclear = geometry.one_body[key[1]]
        full = h_nuclear + jk
        path = here / ('virtual_' + str(j) + '_auxiliary_Fock.npz')
        save(np, path, key=np.array(key), pp=pp, qq=qq, pq=pq, qp=qp,
             scaled_pp=pp2, scaled_qq=qq2, scaled_pq=pq2, scaled_qp=qp2,
             JK_Ha=jk, nuclear_h_Ha=h_nuclear, auxiliary_Fock_Ha=full)
        stages.append(dict(name='auxiliary_Fock/' + str(key), path=path.name, sha256=sha(path)))
        progress('auxiliary_Fock_arrays_saved_before_gates/' + str(key))
        check('auxiliary_JK_finite/' + str(key), not np.all(np.isfinite(jk)), 0)
        for quadrant, a, b in zip(('PP', 'QQ', 'PQ', 'QP'), values[0], values[1], strict=True):
            check('auxiliary_row_scaling/' + quadrant + '/' + str(key), np.max(abs(a - b)),
                  gates['auxiliary_normalization_quadrant_error_Ha'])
        check('auxiliary_PQ_QP_transpose/' + str(key), np.max(abs(pq - qp.T)), gates['auxiliary_JK_Hermitian_error_Ha'])
        check('auxiliary_JK_Hermitian/' + str(key), np.max(abs(jk - jk.T)), gates['auxiliary_JK_Hermitian_error_Ha'])
        check('auxiliary_full_Fock_Hermitian/' + str(key), np.max(abs(full - full.T)), gates['auxiliary_JK_Hermitian_error_Ha'])
        fields[key[1]], jk_fields[key[1]] = full, jk
    # Unnormalized numerical probes split components, so all four radial
    # quadrants and signed old/new offdiagonals are independent scalar anchors.
    probes, descriptors = {}, []
    valence = {-3: (5, -3), 2: (5, 2), -2: (6, -2), -1: (6, -1), 1: (6, 1)}
    for j, new in enumerate(new_keys):
        kappa = new[1]
        for origin, (p, q) in zip(('old_valence', 'saved_nuclear_new'),
                                 (waves[valence[kappa]], (raw['P'][j], raw['Q'][j])), strict=True):
            for component in ('P', 'Q'):
                key = (100 + len(descriptors), kappa)
                probes[key] = (p.copy(), np.zeros(123)) if component == 'P' else (np.zeros(123), q.copy())
                descriptors.append(dict(key=key, origin=origin, component=component,
                                        physical_orbital=False, renormalized=False))
    check('auxiliary_probe_count_and_disjoint', len(probes) != caps['synthetic_probe_count'] or bool(set(probes) & set(core)), 0)
    probe_waves = {key: waves[key] for key in core} | probes
    bare = {(a, b): 0. for a in probes for b in probes if a[1] == b[1]}
    scalar_cache = {}

    def radial(key):
        if key[0] != 'r':
            raise ValueError('Scalar auxiliary oracle admits only two-body radial keys')
        if calls['scalar_radial_callback_attempts'] >= caps['scalar_radial_callback_calls_budget']:
            raise RuntimeError('Scalar radial callback budget exhausted')
        calls['scalar_radial_callback_attempts'] += 1
        if key not in scalar_cache:
            if calls['scalar_Slater_attempts'] >= caps['scalar_Slater_calls_budget']:
                raise RuntimeError('Scalar Slater budget exhausted')
            calls['scalar_Slater_attempts'] += 1
            _, rank, a, b, c, d = key
            scalar_cache[key] = compute_slater_rk_cell(*probe_waves[a], *probe_waves[b],
                                                     *probe_waves[c], *probe_waves[d], data, rank,
                                                     cache=geometry.cache)
            calls['completed_scalar_Slater'] += 1
        return scalar_cache[key]

    if calls['closed_core_scalar_attempts'] >= caps['closed_core_one_body_calls_budget']:
        raise RuntimeError('One independent scalar closed-core contraction budget exhausted')
    calls['closed_core_scalar_attempts'] += 1
    analytic = closed_core_one_body(probes, core, radial, bare_one_body=bare)
    calls['completed_closed_core_scalar'] += 1
    check('auxiliary_scalar_exact80_entries', len(analytic) != caps['synthetic_ordered_same_kappa_bilinear_count'], 0)
    vectors = {key: np.concatenate(value) for key, value in probes.items()}
    rows = []
    for kappa in sorted(fields):
        kk = [key for key in probes if key[1] == kappa]
        vectors4 = np.column_stack([vectors[key] for key in kk])
        scalar = np.array([[analytic[a, b] for b in kk] for a in kk])
        dense = vectors4.T @ jk_fields[kappa] @ vectors4
        path = here / ('auxiliary_scalar_quadrants_' + str(kappa) + '.npz')
        save(np, path, probe_keys=np.array(kk), probe_vectors=vectors4,
             scalar_closed_core_JK_Ha=scalar, dense_native_JK_Ha=dense)
        stages.append(dict(name='auxiliary_scalar/' + str(kappa), path=path.name, sha256=sha(path)))
        progress('auxiliary_scalar_arrays_saved_before_gates/' + str(kappa))
        for quadrant, rr, cc in (('PP', [0, 2], [0, 2]), ('QQ', [1, 3], [1, 3]),
                                ('PQ', [0, 2], [1, 3]), ('QP', [1, 3], [0, 2])):
            oracle, observed = scalar[np.ix_(rr, cc)], dense[np.ix_(rr, cc)]
            error, magnitude = float(np.max(abs(oracle - observed))), float(np.max(abs(oracle)))
            row = dict(kappa=kappa, quadrant=quadrant, error_Ha=error, magnitude_Ha=magnitude,
                       relative_error=error / max(magnitude, 1e-300), path=path.name)
            rows.append(row)
            check('auxiliary_scalar/' + quadrant + '/' + str(kappa), error, gates['auxiliary_scalar_bilinear_error_Ha'])
            if quadrant in ('PQ', 'QP'):
                check('auxiliary_measured_cross_anchor/' + quadrant + '/' + str(kappa),
                      magnitude <= gates['auxiliary_cross_bilinear_magnitude_strict_min_Ha'], 0)
                check('auxiliary_scalar_cross_relative/' + quadrant + '/' + str(kappa),
                      row['relative_error'], gates['auxiliary_scalar_PQ_QP_relative_error'])
    write(here / 'auxiliary_scalar_control.json', dict(rows=rows, probes=descriptors,
          exact_ordered_bilinears=80, actual_scalar_callbacks=calls['scalar_radial_callback_attempts'],
          actual_public_Slater_calls=calls['scalar_Slater_attempts'],
          existing_internal_retry_upper_bound=3 * calls['scalar_Slater_attempts'],
          internal_retry_actual_count_not_observed=True, zero_bare_one_body=True,
          synthetic_probes_not_physical_orbitals=True))
    return fields
