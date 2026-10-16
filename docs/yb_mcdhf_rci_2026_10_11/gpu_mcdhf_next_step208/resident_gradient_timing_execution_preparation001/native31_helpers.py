"""Literal qualified native31 setup/validation helper, private timing epoch.

No numerical import at module load. Numerical formulas and public consumer
calls are retained from the frozen successful qualification. No AST execution.
"""

import json
import math
import time
from fractions import Fraction


def prepare_native31_point(runtime):
    np = runtime['np']
    check, begin, finish = runtime['check'], runtime['begin'], runtime['finish']
    load, plain, roles, fixed, gates = (runtime[k] for k in ('load', 'plain', 'roles', 'fixed', 'gates'))
    variational, tracking_module, optimization, ci_module = (runtime[k] for k in ('variational', 'tracking_module', 'optimization', 'ci_module'))
    csf_signature, enumerate_jj_csfs, get_shared_orbital_basis = (runtime[k] for k in ('csf_signature', 'enumerate_jj_csfs', 'get_shared_orbital_basis'))
    CIStateAverage, make_nuclear_potential = runtime['CIStateAverage'], runtime['make_nuclear_potential']
    def archive(role):
        begin("NPZ_load")
        with np.load(roles[role], allow_pickle=False) as arrays:
            value = {k: arrays[k].copy() for k in arrays.files}
        finish("NPZ_load")
        return value
    point, radial, native_hS, reference_ci_arrays = (archive(role) for role in
             ("recovered_point", "workspace", "original_hS", "tracking_CI"))
    saved_ci, saved_g = archive("recovered_CI"), archive("saved_gradient_metric")
    check("saved_public_radial_gradient31_contiguous246", saved_g["full_normalized_radial"].shape != (31, 246))
    for field in ("full_normalized_raw_coordinate", "full_normalized_tangent", "full_normalized_preconditioned_tangent"):
        check("saved_complete3689_" + field, saved_g[field].shape != (3689,))
    data = {k[6:]: v.item() if v.ndim == 0 else v.copy()
            for k, v in radial.items() if k.startswith("basis_")}
    keys = tuple(map(tuple, fixed["keys31"]))
    check("saved_complete31_order", tuple(map(tuple, point["keys"])) != keys)
    models = load(roles["models"])
    sectors = {tuple(v) for v in models["selected_sectors"]}
    parent = [c for row in models["models"]["complete5f"]["configs"]
              for c in enumerate_jj_csfs(row["config"])]
    csfs = [c for c in parent if (c.total_J2, c.parity) in sectors]
    def signature(c):
        s = csf_signature(c)
        return json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
    check("complete3789_parent1595_targets", len(parent) != 3789 or len(csfs) != 1595)
    check("charge69", {c.n_electrons for c in parent} != {69})
    check("signed_CSF_order", [signature(c) for c in csfs] != load(roles["basis"])["signatures"])
    check("public31_orbitals", tuple(sorted(get_shared_orbital_basis(csfs))) != keys)
    reference_models = load(roles["reference_models"])["models"]
    reference_csfs = [c for row in reference_models["baseline"]["configs"]
                     for c in enumerate_jj_csfs(row["config"])]
    targets = load(roles["tracking_binding"])["targets"]
    average = CIStateAverage(tuple((r["twoJ"], r["parity"], r["models"]["DC"]["local_root"], 1.) for r in targets))
    reference_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                       zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
    tracker = tracking_module.CIBasisTracking(average, reference_csfs,
         ci_module.CIResult(reference_ci_arrays["energies_Ha"], reference_ci_arrays["mixing"]),
         reference_waves, {k: native_hS[f"s_{k}"] for k in fixed["kappas"]}, max_determinant_pairs=30000)
    nucleus = make_nuclear_potential(70, "fermi", A=175)
    functional = variational.CSFOrbitalFunctional(csfs, data, 70, nucleus,
         state_average=average, level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
    check("public_functional_keys", functional.keys != keys)
    for kappa in fixed["kappas"]:
        check("same_L_completed_h_" + str(kappa), not np.array_equal(functional.one_body[kappa], native_hS[f"h_{kappa}"]))
        check("same_S_" + str(kappa), not np.array_equal(functional.metric[kappa], native_hS[f"s_{kappa}"]))
    layout = tuple((tuple(map(tuple, row["keys"])), point[f"projector_{i}"],
                    tuple(row["shape"]), slice(*row["slice"]))
                   for i, row in enumerate(load(roles["layout"])))
    inverses = tuple(tuple(blocks) for blocks in (point[f"inverse_blocks_{i}"] for i in range(7)))
    state = optimization.OrbitalOptimizationState.from_arrays(functional, layout=layout,
         coordinates=point["x"], inverse_diagonal=point["inverse_diagonal"], inverse_blocks=inverses,
         projector_source="nuclear", objective_identity=fixed["objective_identity"], occupation_floor=.05)
    state.validate(functional)
    check("native3689_metric_identity", state.residual_metric_sha256 != fixed["new_metric_sha256"])
    check("native3689_geometry_identity", state.geometry_sha256 != fixed["new_geometry_sha256"])
    check("native3689_state_identity", state.state_sha256 != fixed["new_state_sha256"])
    waves = {}
    for group, projector, shape, where in state.layout:
        c = projector @ state.coordinates[where].reshape(shape)
        check("actual_SGram_" + str(group[0][1]), np.max(abs(c.T @ functional.metric[group[0][1]] @ c - np.eye(len(group)))), gates["radial_Gram"])
        for j, key in enumerate(group):
            waves[key] = c[:123, j].copy(), c[123:, j].copy()
            stored = np.concatenate((point["P"][keys.index(key)], point["Q"][keys.index(key)]))
            check("actual_stored_point_" + str(key), not np.array_equal(c[:, j], stored))
            outside = c[:, j] - projector @ (projector.T @ functional.metric[key[1]] @ c[:, j])
            check("actual_originalP_leakage_" + str(key), abs(float(outside @ functional.metric[key[1]] @ outside)), gates["P_leak_S_squared"])


    return dict(point=point, radial=radial, native_hS=native_hS, reference_ci_arrays=reference_ci_arrays,
                saved_ci=saved_ci, saved_g=saved_g, data=data, keys=keys, csfs=csfs,
                tracker=tracker, nucleus=nucleus, functional=functional, state=state, waves=waves)


def fresh_native31_reference(runtime, point):
    np, check = runtime['np'], runtime['check']
    load, plain, roles, fixed, gates = (runtime[k] for k in ('load', 'plain', 'roles', 'fixed', 'gates'))
    output, write, save = runtime['output'], runtime['write'], runtime['save']
    ci_module, ci_level_blocks, adapter_module = (runtime[k] for k in ('ci_module', 'ci_level_blocks', 'adapter_module'))
    csfs, waves, keys, data, nucleus, functional, saved_ci, tracker, state = (point[k] for k in ('csfs','waves','keys','data','nucleus','functional','saved_ci','tracker','state'))
    timer = time.monotonic()
    ci = ci_module.ci_solve(csfs, waves, keys, data, 70, 123,
            nuclear_potential=nucleus, convol_cache=functional.cache, solver="dense",
            n_states=None, include_breit=False, max_determinants=2000, retain_hamiltonian=True)
    public_HCI_s = time.monotonic() - timer
    h = ci.hamiltonian_for(csfs, waves, keys, data, 70, 123, nuclear_potential=nucleus, max_determinants=2000)
    write(output / "actual_owner.json", ci.hamiltonian_snapshot.identity_payload())
    check("saved_actual_owner_header", plain(ci.hamiltonian_snapshot.identity_payload()) != load(roles["recovered_owner"]))
    check("saved_same_epoch_H_bitwise", not np.array_equal(h, saved_ci["H"]))
    check("saved_same_epoch_allroots_bitwise", not np.array_equal(ci.state_energies, saved_ci["energies"]))
    check("complete_H_symmetry", not np.array_equal(h, h.T))
    check("all1595_eigenpair_column2norm", np.max(np.linalg.norm(h @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
    check("all1595_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1595))), gates["CI_Gram"])
    _, blocks_map = ci_level_blocks(csfs, ci, require_full=True)
    for sector, columns in blocks_map.items():
        rows = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
        exterior = [i for i in range(len(csfs)) if i not in set(rows)]
        check("cross_sector_" + str(sector), np.max(abs(h[np.ix_(rows, exterior)])), gates["cross_symmetry_zero_Ha"])
        groups, group = [], []
        for column in columns:
            if group and ci.state_energies[column] - ci.state_energies[group[-1]] > gates["cluster_gap_Ha"]:
                groups.append(group)
                group = []
            group.append(int(column))
        groups.append(group)
        for group in groups:
            a = saved_ci["mixing"][np.ix_(rows, group)]
            b = ci.mixing_coefficients[np.ix_(rows, group)]
            check("full_signed_span_" + str(group), np.linalg.norm(a @ a.T - b @ b.T), gates["signed_span"])
    timer = time.monotonic()
    weights, tracked = tracker.resolve(csfs, ci, waves, functional.metric)
    tracking_s = time.monotonic() - timer
    check("original_equal_five_weights", np.count_nonzero(weights) != 5 or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
    check("same_saved_tracking_UIDs_weights", plain(dict(weights=weights, tracked=tracked)) != load(roles["recovered_tracking"]))
    matched = {tuple(r["reference_levels"][0]): r for r in tracked}
    check("five_original_UIDs", set(matched) != {tuple(v) for _, v in fixed["target_UIDs"]})
    timer = time.monotonic()
    tables, contributions, target_rows = load(roles["signed_tables"]), {}, []
    for label, uid in fixed["target_UIDs"]:
        row = matched[tuple(uid)]
        check("single_same_sector_target_" + label, len(row["reference_levels"]) != 1 or len(row["selected_levels"]) != 1 or row["selected_levels"][0][:2] != tuple(uid[:2]))
        j, parity, root = row["selected_levels"][0]
        column = int(blocks_map[j, parity][root])
        c = ci.mixing_coefficients[:, column]
        norm = sum((Fraction.from_float(float(v))**2 for v in c), Fraction(0))
        if norm <= 0:
            raise ValueError("nonpositive complete target norm")
        table = functional.radial_density.contract_bilinear(c, c)
        for key, value in table.items():
            contributions.setdefault(key, []).append(Fraction.from_float(float(weights[column])) * Fraction.from_float(float(value)) / norm)
        target_rows.append(dict(label=label, UID=uid, column=column,
                exact_CI_norm=(str(norm.numerator), str(norm.denominator)), norm_defect=float(norm - 1),
                unfiltered=[[k, v] for k, v in table.items()]))
    check("all_fresh_unfiltered_target_tables_exact", plain(target_rows) != tables["targets"])
    def represent(value):
        result = float(value)
        if not math.isfinite(result) or (value != 0 and result == 0):
            raise ValueError("Unrepresentable nonzero angular coefficient; no silent omission")
        return result
    exact = {k: sum(v, Fraction(0)) for k, v in contributions.items()}
    full_weights = {k: represent(v) for k, v in exact.items() if v != 0}
    check("fresh_full_normalized_RDM_exact", plain([[k, v] for k, v in full_weights.items()]) != tables["full_normalized"])
    signed_layout = adapter_module.collect_signed_layout(keys, full_weights.items())
    union = load(roles["adapter_union"])
    check("full706_union_order_identity", signed_layout.source_union_sha256 != union["source_union_sha256"] or len(signed_layout.jobs) != 706)
    check("full19589_signed_table_identity", signed_layout.weights_sha256 != union["weights_sha256"] or len(signed_layout.items) != 19589)
    normalized_RDM_s = time.monotonic() - timer
    save("complete_public_point.npz", H=h, energies=ci.state_energies, mixing=ci.mixing_coefficients,
         keys=np.array(keys), P=np.array([waves[k][0] for k in keys]), Q=np.array([waves[k][1] for k in keys]),
         x=state.coordinates, inverse_diagonal=state.inverse_diagonal,
         **{f"projector_{i}": r[1] for i, r in enumerate(state.layout)},
         **{f"inverse_blocks_{i}": np.stack(b) for i, b in enumerate(state.inverse_blocks)})
    owner_identity = ci.hamiltonian_snapshot.identity_payload()
    write(output / "actual_signed_tables.json", dict(targets=target_rows, full_normalized=[[k, v] for k, v in full_weights.items()]))


    # Actual owner/read provenance remains h. Independent bytes-backed arrays
    # make unchanged-H/C gates independent of the later retained owner views.
    h_baseline = np.frombuffer(h.tobytes(), dtype=h.dtype).reshape(h.shape)
    energies_baseline = np.frombuffer(ci.state_energies.tobytes(), dtype=ci.state_energies.dtype).reshape(ci.state_energies.shape)
    mixing_baseline = np.frombuffer(ci.mixing_coefficients.tobytes(), dtype=ci.mixing_coefficients.dtype).reshape(ci.mixing_coefficients.shape)
    return dict(ci=ci, h=h, owner_identity=owner_identity, weights=weights, tracked=tracked,
                full_weights=full_weights, signed_layout=signed_layout,
                public_HCI_s=public_HCI_s, tracking_s=tracking_s, normalized_RDM_s=normalized_RDM_s,
                H_independent_baseline=h_baseline, ci_energies_baseline=energies_baseline, ci_mixing_baseline=mixing_baseline)


def pack_native31_gradient(runtime, point, gradient):
    np, check = runtime['np'], runtime['check']
    functional, state, keys = (point[k] for k in ('functional', 'state', 'keys'))
    def coordinate(radial_gradient):
        result = np.zeros(3689, dtype=radial_gradient.dtype)
        lookup = {k: radial_gradient[i] for i, k in enumerate(functional.keys)}
        for group, projector, _, where in state.layout:
            result[where] = (projector.astype(result.dtype).T @ np.column_stack([lookup[k] for k in group])).ravel()
        return result
    def tangent(vector):
        result = vector.copy()
        for _, _, shape, where in state.layout:
            y, g = state.coordinates[where].reshape(shape).astype(vector.dtype), vector[where].reshape(shape)
            multiplier = y.T @ g
            result[where] = (g - .5 * y @ (multiplier + multiplier.T)).ravel()
        return result
    def metric_action(vector):
        result = vector.copy()
        for (_, _, shape, where), inverse in zip(state.layout, state.inverse_blocks, strict=True):
            g = vector[where].reshape(shape)
            result[where] = np.column_stack([b.astype(vector.dtype) @ g[:, j] for j, b in enumerate(inverse)]).ravel()
        return tangent(result)

    def pack(gradient):
        radial_g = np.array([gradient[k] for k in keys])
        check("full_public_private_gradient31_contiguous246", radial_g.shape != (31, 246))
        raw = coordinate(radial_g)
        tg = tangent(raw)
        return dict(radial=radial_g, raw_coordinate=raw, tangent=tg, pg=metric_action(tg))

    return pack(gradient)


def validate_reference_bytes(runtime, point, reference, name):
    """Same complete CI certificate is applicable only to unchanged inputs."""
    check = runtime['check']
    ci = reference['ci']
    current_h = ci.hamiltonian_for(point['csfs'], point['waves'], point['keys'], point['data'], 70, 123,
                                 nuclear_potential=point['nucleus'], max_determinants=2000)
    for field, actual, expected in (('H', current_h, reference['H_independent_baseline']),
                                  ('allroots', ci.state_energies, reference['ci_energies_baseline']),
                                  ('complete_signed_CI', ci.mixing_coefficients, reference['ci_mixing_baseline'])):
        check(name + '/unchanged_bytes_' + field, actual.dtype != expected.dtype
              or actual.shape != expected.shape or actual.tobytes() != expected.tobytes())
    check(name + '/same_owner_input_matrix_identity', ci.hamiltonian_snapshot.identity_payload() != reference['owner_identity'])
    layout = runtime['adapter_module'].collect_signed_layout(point['keys'], reference['full_weights'].items())
    check(name + '/complete_signed_table_source_order', layout != reference['signed_layout'])


def validate_native31_gradient(runtime, point, reference, name, values, expected, packets, saved_bands):
    """All comparisons are outside scored timers; no new H/CI/gradient calls."""
    np, check, gates = runtime['np'], runtime['check'], runtime['gates']
    for key in values:
        tolerance = gates['pg'] if key == 'pg' else gates['g']
        check(name + '/complete_' + key, np.max(abs(values[key] - expected[key])), tolerance)
    validate_reference_bytes(runtime, point, reference, name)
    cursor, checked, bitwise, maximum = 0, 0, 0, 0.
    for jobs, bands in packets:
        expected_jobs = reference['signed_layout'].jobs[cursor:cursor + len(jobs)]
        check(name + '/packet_job_order_' + str(cursor), jobs != expected_jobs)
        check(name + '/packet_readonly_' + str(cursor), bands.flags.writeable)
        for i in range(len(jobs)):
            for comp in range(4):
                target, value = saved_bands[cursor + i, comp], bands[i, comp]
                error = float(np.max(abs(value - target)) / max(1., float(np.max(abs(target)))))
                check(name + '/band_' + str(checked), error, gates['band_scaled'])
                maximum = max(maximum, error)
                bitwise += int(np.array_equal(value, target))
                checked += 1
        cursor += len(jobs)
    check(name + '/complete706_jobs_2824components', cursor != 706 or checked != 2824 or len(packets) != 6)
    return dict(all_bands=checked, bitwise_bands=bitwise, max_scaled_band_error=maximum)
