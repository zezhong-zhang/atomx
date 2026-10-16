"""Generate the prospective consumer from frozen source text only."""
import ast
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
NATIVE = DIAG.parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
SCREEN = NATIVE.parent
OLD = SCREEN / "constrained_fd_proposal001/representation_repair001"


def sha(path):
    if Path(path).suffix == ".npz":
        raise ValueError("No NPZ reads or hashes in preparation")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role(path, expected=None):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=expected or sha(path),
                deferred_numerical=path.suffix == ".npz")


code = (OLD / "run.py").read_text()
code = code.replace('"""One native31 combined4f/5f constrained FD; no numerical default-mode imports."""',
                    '"""Fresh literal13 native31 directional qualification; source-only default mode."""')
start = code.index('    result, receipt, completion = (load(roles[n])')
end = code.index('    return contract, roles, source, deferred', start)
code = code[:start] + '''    result, receipt = load(roles["native_result"]), load(roles["native_receipt"])
    point, stageA = load(roles["literal13_manifest"]), load(roles["stageA_result"])
    stageA_receipt = load(roles["stageA_receipt"])
    if (result["all_completed_controls_passed"] is not True
            or receipt["status"] != "foreground_closed" or receipt["exit_code"] != 0
            or point["iteration"] != 13 or point["qualified"] is not True
            or stageA["all_declared_replay_controls_passed"] is not True
            or len(stageA["checks"]) != 200 or any(not r["passed"] for r in stageA["checks"])
            or stageA_receipt["status"] != "foreground_closed" or stageA_receipt["exit_code"] != 0
            or stageA["physical_derivative_or_stationarity_certified"] is not False
            or stageA["summary"]["future_stageB_qualified"] is not False):
        raise ValueError("Closed native13 and original replay prerequisites changed")
    for field, saved in (("new_metric_sha256", point["metric_sha256"]),
                         ("new_geometry_sha256", point["geometry_sha256"]),
                         ("objective_identity", point["objective_identity"]),
                         ("source_map_sha256", point["source_map_sha256"])):
        if saved != contract["fixed"][field]:
            raise ValueError("Wrong literal13 physical identity: " + field)
    if (stageA["summary"]["direction_array_sha256"] != contract["input_roles"]["stageA_direction"]["sha256"]
            or point["array_sha256"] != contract["input_roles"]["literal13_arrays"]["sha256"]):
        raise ValueError("Wrong immutable point or reconstructed direction")
''' + code[end:]
code = code.replace('    return contract, roles, source, deferred', '''    census, calls = contract["primitive_census"], contract["resources"]["calls"]
    if (census["gradient_source_pairs_total"] != 3 * 706
            or census["gradient_components_total"] != 3 * 706 * 4
            or census["finite_component_upper"] != 4 * 19589
            or census["finite_potential_upper"] != 4 * 19589 * 3
            or census["finite_density_upper"] != 4 * 19589 * 6
            or census["five_H_C_archives_uncompressed_bytes"] != 5 * 2 * 1595 * 1595 * 8
            or census["gradient_source_pairs_total"] > calls["gradient_pair_potential"]
            or census["gradient_components_total"] > calls["gradient_components"]
            or census["finite_component_upper"] > calls["finite_component_cache_miss"]
            or census["complete_SciPy_eigh_upper"] > calls["complete_SciPy_eigh"]):
        raise ValueError("Before physical loops: complete primitive/resource census mismatch")
    return contract, roles, source, deferred''')
code = code.replace('"finite_density_cache_miss")\n                if not quiet:',
                    '"finite_density_cache_miss", "gradient_pair_potential", "gradient_components")\n                if not quiet:')
code = code.replace('if not quiet or completed[name] % 128 == 0:', 'if not quiet:')
code = code.replace('patch(variational.CSFOrbitalFunctional, "_pair_potential", "gradient_pair_potential")',
                    'patch(variational.CSFOrbitalFunctional, "_pair_potential", "gradient_pair_potential")\n'
                    '        patch(variational, "_convol_case1_t", "gradient_components")\n'
                    '        patch(variational, "_convol_case1_semiseparable_t", "gradient_components")')
start = code.index('        def archive(role):')
end = code.index('        def signature(c):', start)
code = code[:start] + '''        decoded_bytes, archive_loads = [0], [0]

        def archive(role):
            archive_loads[0] += 1
            plan = contract["array_access"][role]
            if archive_loads[0] > 5 or roles[role].stat().st_size > plan["compressed_bytes_max"]:
                raise RuntimeError("Before archive load/byte cap")
            result = {}
            with np.load(roles[role], allow_pickle=False) as a:
                names = list(plan["fields"])
                if plan.get("basis_prefix"):
                    names += [k for k in a.files if k.startswith("basis_")]
                if len(names) > plan["fields_max"] or len(set(names)) != len(names):
                    raise RuntimeError("Before complete field census cap")
                for key in names:
                    value = a[key]
                    spec = plan["fields"].get(key)
                    if spec is not None and (value.shape != tuple(spec["shape"])
                            or value.dtype.str != spec["dtype"]):
                        raise ValueError("Native input dtype/shape: " + role + "/" + key)
                    if value.dtype.hasobject or not np.isfinite(value).all():
                        raise ValueError("Nonfinite/nonnumeric complete input")
                    decoded_bytes[0] += value.nbytes
                    if decoded_bytes[0] > caps["maximum_total_input_decoded_bytes"]:
                        raise RuntimeError("Before complete decoded byte cap")
                    result[key] = value.copy()
            write(output / ("input_" + role + "_decoded.json"), dict(fields=[dict(name=k,
                shape=list(v.shape), dtype=v.dtype.str, bytes=v.nbytes) for k, v in result.items()],
                cumulative_decoded_bytes=decoded_bytes[0]))
            return result

''' + code[end:]
start = code.index('        saved_g, saved_point, saved_CI, radial, original_hS = ')
end = code.index('        data = ', start)
code = code[:start] + '''        saved_point = archive("literal13_arrays")
        saved_g = saved_CI = saved_point
        saved_direction = archive("stageA_direction")
        radial, original_hS = archive("geometry_workspace"), archive("screened_geometry")
''' + code[end:]
code = code.replace('        check("fullgradient_saved31_keys", not np.array_equal(saved_g["keys"], saved_point["keys"]))\n', '')
start = code.index('        for name in ("x", "inverse_diagonal", *[f"projector_{i}"')
end = code.index('        state = optimization.OrbitalOptimizationState.from_arrays', start)
code = code[:start] + code[end:]
code = code.replace('                              ("new_state_sha256", state.state_sha256)):', '                              ):')
code = code.replace('authoritative_point="direct original nuclear P times own recovered3689 x"',
                    'authoritative_point="literal accepted13 coordinates through original nuclear P"')
start = code.index('        check("same_own_baseline_owner",')
end = code.index('        # Consumer-only exact', start)
new = '''        baseline_manifest = load(roles["literal13_manifest"])
        check("same_own_baseline_owner", plain(base_ci.hamiltonian_snapshot.identity_payload())
              != baseline_manifest["owner"])
        check("same_own_baseline_EO", points["baseline"]["EOL_reported_Ha"].hex()
              != baseline_manifest["diagnostics"]["energy_ha"].hex())
        old_rows = {r["label"]: r for r in baseline_manifest["rows"]}
        for row in points["baseline"]["rows"]:
            old_row = old_rows[row["label"]]
            check("same_literal13_UID_selection_" + row["label"], tuple(row["UID"]) != tuple(old_row["UID"])
                  or tuple(row["selected_level"]) != tuple(old_row["selected_level"]) or row["column"] != old_row["column"])

        # One public retained gradient, signed full-minus-retained, independent
        # direct normalized full gradient: three separate same-point controls.
        baseline_energy, retained_g, returned_ci = functional.evaluate(base_waves, ci_result=base_ci)
        check("gradient_owns_literal13_real_CI", returned_ci is not base_ci)
        check("gradient_preserves_reported_EO", baseline_energy.hex() != points["baseline"]["EOL_reported_Ha"].hex())
        retained = np.array([retained_g[k] for k in functional.keys])
        check("fresh_literal13_retained_radial", np.max(abs(retained - saved_g["retained_radial"])), 1e-9)
VECTORS
        raw, g, pg, retained_metrics = vectors(state, retained)
        check("fresh_literal13_rawg", np.max(abs(raw - saved_g["raw_coordinate_gradient"])), 1e-9)
        check("fresh_literal13_pg", np.max(abs(pg - saved_g["preconditioned_gradient"])), 1e-10)
        check("original_metric_residual_scalar", abs(retained_metrics["residual"]
              - baseline_manifest["diagnostics"]["orbital_residual"]), 1e-10)
        check("original_metric_indicator_scalar", abs(retained_metrics["indicator_Ha"]
              - baseline_manifest["diagnostics"]["estimated_energy_error"]), 1e-10)
        rows, weights = points["baseline"]["rows"], functional.last_state_weights
        columns = [row["column"] for row in rows]
        orbitals, ci, prefix = base_waves, base_ci, output / "baseline"
FULL_NORMALIZED
        check("whole_fresh_full_table_identity", plain(dict(targets=tables,
            retained=[[k, v] for k, v in retained_weights.items()],
            full_normalized=[[k, v] for k, v in full.items()],
            full_minus_retained=[[k, v] for k, v in delta.items()], exact_zero_only=True,
            delta_order="full original order then retained-only original order",
            no_gradient_or_energy_replacement=True)) != load(roles["literal13_tables"]))
        for label, values, fields in (("full_normalized", full_vectors,
                ("full_normalized_raw_coordinate", "full_normalized_tangent", "full_normalized_pg")),
                ("raw_h_corrected", corrected_vectors,
                ("raw_h_corrected_raw_coordinate", "raw_h_corrected_tangent", "raw_h_corrected_pg"))):
            for index, field in enumerate(fields):
                check("same_literal13_" + field, np.max(abs(values[index] - saved_g[field])),
                      1e-10 if index == 2 else 1e-9)
            check(label + "_same_retained_tangent", np.max(abs(values[1] - g)), 1e-9)
            check(label + "_same_retained_pg", np.max(abs(values[2] - pg)), 1e-10)
        save("baseline_fullsigned_gradient_metric.npz", keys=np.array(functional.keys), x=state.coordinates,
            retained_radial=retained, full_minus_retained_radial=delta_array,
            full_normalized_radial=full_array, raw_h_quotient_corrected_radial=corrected_array,
            retained_raw_coordinate=raw, retained_tangent=g, retained_pg=pg,
            full_normalized_raw_coordinate=full_vectors[0], full_normalized_tangent=full_vectors[1],
            full_normalized_pg=full_vectors[2], raw_h_corrected_raw_coordinate=corrected_vectors[0],
            raw_h_corrected_tangent=corrected_vectors[1], raw_h_corrected_pg=corrected_vectors[2])
        write(output / "baseline_fullsigned_metrics.json", dict(retained=retained_metrics,
            full_normalized=full_vectors[3], raw_h_quotient_corrected=corrected_vectors[3],
            baseline_only_no_uniform_density_bound=True, HF_partial_not_complete_finite_CI_response=True,
            ordinary_reported_objective_unchanged=True))
        # This transported tangent defines a NEW local retraction centered
        # at13; it is not the derivative of the old alpha1 polar path/Wolfe.
        check("StageA_same_literal13_x", not np.array_equal(saved_direction["literal_x13"], state.coordinates))
        transported = saved_direction["direction12"].copy()
        for _, _, shape, where in state.layout:
            y, v = state.coordinates[where].reshape(shape), transported[where].reshape(shape)
            multiplier = y.T @ v
            transported[where] = (v - y @ (.5 * (multiplier + multiplier.T))).ravel()
        check("StageA_original_transport_exact", not np.array_equal(transported,
              saved_direction["transported_direction13"]))
        norm = float(np.linalg.norm(transported))
        if not math.isfinite(norm) or norm < contract["direction"]["minimum_norm_for_admission"]:
            raise ValueError("Direction is zero/nonfinite; no substitute direction")
        eta = saved_direction["normalized_transported_direction13"].copy()
        check("StageA_normalization_exact", not np.array_equal(eta, transported / norm))
        check("whole_direction_unit_norm", abs(float(np.linalg.norm(eta)) - 1.), 1e-10)
        direction_rows = []
        for keys, _, shape, where in state.layout:
            y, ee = state.coordinates[where].reshape(shape), eta[where].reshape(shape)
            absolute = float(np.max(abs(y.T @ ee + ee.T @ y)))
            input_norm = float(np.linalg.norm(ee))
            check("scaled_tangent_" + str(keys[0][1]), absolute / max(1., input_norm), gates["scaled_tangent_identity"])
            direction_rows.append(dict(kappa=keys[0][1], absolute_tangent_defect=absolute,
                input_G_norm=input_norm, scaled_defect=absolute / max(1., input_norm)))
        save("fixed_StageA_direction.npz", direction=eta, transported=transported,
             norm=np.array(norm), keys=np.array(functional.keys), x=state.coordinates)
        write(output / "direction_identity.json", dict(rows=direction_rows, full3689_transport_norm=norm,
            normalized_direction_exact_StageA_bytes=True, no_direction_or_gap_selection=True,
            old_alpha1_path_derivative_or_Wolfe_qualified=False))

'''
native = (NATIVE / "run.py").read_text()
tree = ast.parse(native)
vectors = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "vectors")
vector_source = '\n'.join(native.splitlines()[vectors.lineno - 1:vectors.end_lineno])
new = new.replace('VECTORS', vector_source)
a = native.index('            contributions, tables = {}, []')
b = native.index('            for name, vector in (("full_normalized", full_vectors)', a)
normalized = textwrap.dedent(native[a:b])
new = new.replace('FULL_NORMALIZED', textwrap.indent(normalized.rstrip(), '        '))
new = new.replace('raise QualificationStop(', 'raise ValueError(')
code = code[:start] + new + code[end:]
code = code.replace('corrected = saved_g["raw_h_quotient_corrected_radial"]', 'corrected = corrected_array')
code = code.replace('vector(saved_g["retained_raw_coordinate"])', 'vector(raw)')
code = code.replace('vector(saved_g["full_normalized_raw_coordinate"])', 'vector(full_vectors[0])')
code = code.replace('("public_reported_energy_evaluate", 5)', '("public_reported_energy_evaluate", 6)')
code = code.replace('("tracker_resolve", 18)', '("tracker_resolve", 19)')
code = code.replace('("tracking_SVD", 90)', '("tracking_SVD", 95)')
code = code.replace('("tracker_overlap", 19)', '("tracker_overlap", 20)')
code = code.replace('("unfiltered_target_RDM", 20)', '("unfiltered_target_RDM", 25)')
code = code.replace('("retained_RDM", 0), ("gradient_from_weights", 0)',
                    '("retained_RDM", 1), ("gradient_from_weights", 3),\n'
                    '                            ("gradient_pair_potential", 2118), ("gradient_components", 8472)')
code = code.replace('status="native31_single_combined_direction_two_h_FD_completed"',
                    'status="literal13_fixed_transported_direction_two_h_FD_completed"')
code = code.replace('            angular_capacity=term_capacity,',
                    '            angular_capacity=term_capacity, baseline_fullsigned_metrics=dict(\n'
                    '                retained=retained_metrics, full_normalized=full_vectors[3], corrected=corrected_vectors[3]),\n'
                    '            input_archive_loads=archive_loads[0], input_decoded_bytes=decoded_bytes[0],')
ast.parse(code)
(HERE / 'run.py').write_text(code)

contract = json.loads((OLD / 'contract.json').read_text())
keep = ('source_identity', 'new_layout', 'new_objective', 'models', 'canonical_basis',
        'reference_models', 'tracking_binding', 'geometry_workspace', 'tracking_ci', 'screened_geometry')
contract['input_roles'] = {k: contract['input_roles'][k] for k in keep}
manifest = json.loads((NATIVE / 'actual/checkpoints/accepted_0013.json').read_text())
stage = DIAG / 'stageA_execution_preparation001'
stage_result = json.loads((stage / 'actual/result.json').read_text())
for name, path, expected in (
        ('native_result', NATIVE/'actual/result.json', None),
        ('native_receipt', NATIVE/'root_execution.json', None),
        ('native_contract', NATIVE/'contract.json', None),
        ('literal13_manifest', NATIVE/'actual/checkpoints/accepted_0013.json', None),
        ('literal13_tables', NATIVE/'actual/checkpoints/accepted_0013_signed_tables.json', None),
        ('literal13_arrays', NATIVE/'actual/checkpoints/accepted_0013.npz', manifest['array_sha256']),
        ('stageA_result', stage/'actual/result.json', None),
        ('stageA_receipt', stage/'root_execution.json', None),
        ('stageA_contract', stage/'contract.json', None),
        ('stageA_direction', stage/'actual/reconstructed_direction.npz', stage_result['summary']['direction_array_sha256']),
        ('original_stageB_proposal', DIAG/'contract.json', None),
        ('frozen_FD_driver', OLD/'run.py', None),
        ('frozen_FD_contract', OLD/'contract.json', None),
        ('frozen_native_driver', NATIVE/'run.py', None)):
    contract['input_roles'][name] = role(path, expected)
contract['status'] = 'SOURCE_ONLY_concrete_literal13_stageB_executor_ready_for_root_review'
contract['hypothesis'] = 'At literal accepted13, test the fresh correlated derivative along the original transported and normalized step12 direction. StageA reproduced numerical history but did not certify this derivative. Two fixed h discriminate ordinary reported energy resolution and independently centered finite correlated response; no physical or optimizer defect is presumed.'
contract['executor_implemented'] = True
contract['execution_authorized'] = False
contract.pop('executor_implemented_or_authorized', None)
contract['fixed'].pop('new_state_sha256')
contract['fixed']['literal_accepted_point'] = 13
contract['fixed']['new_state_sha256_is_runtime_own_literal13_not_old_point0'] = True
contract['gates']['baseline_H_roots_signedC_owner_UID_bitwise_to_own31_recovered'] = 'Exact to literal accepted13, not old canonical31 or recovered point0'
contract['reported_production_objective']['primary_analytic'] = 'Fresh samepoint ordinary retained public radial gradient mapped through original P and dotted with exact StageA normalized transported direction at13.'
contract['reported_production_objective']['additional_analytic'] = 'Fresh independently normalized/unfiltered full coordinate action and corrected ambient quotient action, at the same literal13 point and same direction.'
contract['native_call_chain'] = [
    'Public complete1595 model, original531 physical tracker, native f043 functional; all5 equal UID targets, original seven nuclearP and fixed inverse metric.',
    'from_arrays validates original31-group3689 geometry with7 public Dirac span checks; no P rebuild, coordinate reproject or baseline retraction.',
    'One fresh full1595 retained baseline CI: entireH/roots/signedC/owner/UID/report compared strictly to literal accepted13.',
    'At this same real owner, one ordinary public retained gradient and two extra public gradient_from_weights calls for exact-zero-only full-minus-retained and independent direct full-normalized density. Five signed contract_bilinear tables, exact CI norm/.2 aggregation, raw h/ket-norm ambient correction, g1e-9/pg1e-10 gates.',
    'Four fixed polar points around literal13 in exact transported normalized direction: fresh fullCI/reported energy/physical tracker for each. No direction/h selection or optimization.',
    'Four unchanged public mixed finite diagnostics retain complete radial polynomial, rounded reconstruction, finite CI norm/phase/residual response. Complete signed fixedLambda quadratic actual Gram and offdiagonal centering remains independently labeled.',
    'Both ordinary and four centered domain gates are evaluated after all4 finite ledgers; no callback, prior failure upgrade or automatic continuation.']
contract['counter_call_proof'].update(
    tracker='Five gradient=False reported resolves +five explicit metadata resolves +one fresh baseline gradient resolve +four mixed x2 =19;5 SVD per resolve=95; constructor1+19 overlaps=20.',
    RDM='Five baseline unfiltered normalized target tables +four mixed x5 =25. One retained public contract. Three gradient_from_weights including the public retained evaluation;706 source groups each,2118 potentials/8472 gradient components.',
    physical_evaluate='Five evaluate gradient=False plusone baseline gradient=True, allprovided samepoint real owner, exactly6 reported functional calls; no optimizer/GPU.',
    primitive_finite='Four complete19589-key radial traversals <=78356 components() cache misses; each at most3 potential and6 density accesses,235068/470136 caps. Complete potential construction has4 physical spinor component contractions per miss, <=940272 band component upper. No screening or mutable-owner recovery.')
contract.pop('representation_only_successor')
contract['workflow_lineage'] = dict(frozen_FD_driver_sha256=sha(OLD/'run.py'), frozen_native3g_driver_sha256=sha(NATIVE/'run.py'),
    reused='Ordinary fullCI/owner/tracker/retraction, public mixed finite API and complete exact Lambda/Gram workflow; native normalized signed-table/direct/delta/publicg and raw-h ambient correction.',
    changes='Newliteral13 baseline and exact actual transported direction, fresh3g controls and corresponding budgets; no physics kernel/default/source change. Only standalone consumer exception class names are adapted.',
    no_prior_FD_gate_transfer=True, old_failed_attempts_preserved=True)
contract['direction'] = dict(definition='Exact StageA normalized tangent13(direction12), all3689 coordinates.',
    base='Literal accepted13 original P@x. No coordinate recovery or new retraction at baseline.',
    trial='Original f043 polar retraction of Y13 +/- h*eta in all7 complete same-kappa groups.',
    minimum_norm_for_admission=1e-14, old_alpha1_path_derivative_or_Wolfe_test=False,
    no_direction_or_NIST_selection=True, native_dtype='float64 eta/x/radials; ambient corrected longdouble preserved in exact Fraction algebra')
contract['gradient_control'] = dict(public_retained=1, independent_full_minus_retained=1, independent_direct_normalized_full=1,
    targets_unfiltered=5, actual_table_scope='Exact norm and .2 weight per target, exact nonzero signed support, original-order full keys then retained-only delta keys.',
    original_raw_h_quotient_ambient_correction=True, no_energy_or_gradient_callback=True,
    saved_rawg_gate=1e-9, saved_pg_gate=1e-10, full_signed_linearity_gate=1e-9, projected_linearity_gate=1e-10,
    baseline_only_no_uniform_global_bound=True, HF_partial_not_finite_CI_response=True)
resource = contract['resources']
calls = resource['calls']
calls.update(public_reported_energy_evaluate=6, gradient_from_weights=3, gradient_pair_potential=2118,
    gradient_components=8472, retained_RDM=1, unfiltered_target_RDM=25,
    tracker_resolve=19, tracking_SVD=95, tracker_overlap=20,
    finite_component_cache_miss=78356, finite_potential_cache_miss=235068, finite_density_cache_miss=470136)
resource.update(maximum_total_input_decoded_bytes=192*2**20, input_archive_loads_max=5,
    baseline_gradient_sources={'retained':706,'full_minus_retained':706,'full_normalized':706},
    baseline_gradient_terms={'retained':17233,'full_minus_retained':18074,'full_normalized':19589},
    primitive_census='Three706-source gradients =2118 streamed pair potentials x4 PP/QQ/PQ/QP components =8472; five target tables67655 entries. Four finite kernels x19589 complete angular keys <=78356 components, <=235068 potential producers and <=470136 density producers. All5 full1595 denseCI, no projected-only eigenpair subset.',
    memory_estimate='Prior same-model FD peak1581.664MiB plus baseline3g and literal13 saved arrays; only baseline+current trial H/C retained, source potentials streamed,5 wholeH/C outputs total203522000 uncompressed bytes. Input selected decode <=192MiB, exact bookkeeping/cache caps bound work;3.95GiB hard AS stop. No measured new peak or timing claim.',
    time_estimate='Previous same-model two-h FD104.221s plus fresh3g baseline.300s is prospective whole-run stop, not a completion guarantee.')
resource['signed_term_work_cap'] = dict(complete_angular_term_capacity_per_trial=19589,
    all_four_radial_components=78356, potential_requests=235068, density_requests=470136,
    meaning='Complete structural table capacity, reject whole experiment if exceeded; no small-term/CSF pruning. Four trials, three potential/six density accesses per radial key maximum.')

def spec(shape, dtype='<f8'):
    return dict(shape=shape, dtype=dtype)

fields = {k:spec([3689]) for k in ('x','raw_coordinate_gradient','tangent_gradient','preconditioned_gradient',
    'full_normalized_raw_coordinate','full_normalized_tangent','full_normalized_pg')}
fields.update({k:spec([3689],'<f16') for k in ('raw_h_corrected_raw_coordinate','raw_h_corrected_tangent','raw_h_corrected_pg')})
fields.update(keys=spec([31,2],'<i8'), H=spec([1595,1595]), energies=spec([1595]), mixing=spec([1595,1595]),
    P=spec([31,123]),Q=spec([31,123]),retained_radial=spec([31,246]),inverse_diagonal=spec([3689]))
for i, row in enumerate(json.loads((ROOT/contract['input_roles']['new_layout']['path']).read_text())):
    fields[f'projector_{i}']=spec([246,119])
    fields[f'inverse_blocks_{i}']=spec([row['shape'][1],119,119])
access = {'literal13_arrays':dict(fields=fields,fields_max=len(fields),compressed_bytes_max=64*2**20),
    'stageA_direction':dict(fields={k:spec([3689]) for k in ('literal_x13','direction12','transported_direction13','normalized_transported_direction13')},fields_max=4,compressed_bytes_max=2**20),
    'geometry_workspace':dict(fields={'orbital_keys':spec([24,2],'<i8'),'P':spec([24,123]),'Q':spec([24,123])},basis_prefix=True,fields_max=64,compressed_bytes_max=64*2**20),
    'tracking_ci':dict(fields={'energies_Ha':spec([531]),'mixing':spec([531,531])},fields_max=2,compressed_bytes_max=8*2**20),
    'screened_geometry':dict(fields={f'{side}_{k}':spec([246,246]) for side in ('h','s') for k in contract['fixed']['kappas']},fields_max=14,compressed_bytes_max=16*2**20)}
contract['array_access'] = access
contract['preparation'] = dict(numerical_imports=0, NPZ_reads_or_hashes=0, science_calls=0,
    source_generator='build_source.py copies frozen FD workflow and native3g consumer text; no physical kernel copied/reimplemented or shared source edited.')
contract['measurement_scope'] = 'Single new local retraction direction at literal13; StageA original transport is not the exact derivative of the previous alpha1 polar path and cannot certify Wolfe curvature.'
contract['stop'] = 'First identity/resource/CI/g gate failure preserved; complete all4 declared finite points before ordinary/centered domain ledger gates. Failed ordinary remains failed. No revised h/direction/weights/model, no optimizer or automatic continuation.'
contract['primitive_census'] = dict(whole_dense_H_C_bytes_per_pair=2*1595*1595*8,
    five_H_C_archives_uncompressed_bytes=5*2*1595*1595*8,
    full_signed_target_table_entries=67655, baseline_signed_terms=[17233,18074,19589],
    gradient_source_pairs_each=706, gradient_source_pairs_total=3*706, gradient_components_total=3*706*4,
    full_angular_table_capacity=19589, finite_component_upper=4*19589,
    finite_potential_upper=4*19589*3, finite_density_upper=4*19589*6,
    finite_potential_component_band_upper=4*19589*3*4,
    complete_CI_matrix_entries_total=5*1595*1595,
    complete_CI_sector_cubic_work_sum=sum(n**3 for n in (219,223,367,370,416))*5,
    complete_CI_sector_eigh_upper=5*5, Dirac_retry_eigh_upper=7*3,
    complete_SciPy_eigh_upper=5*5+7*3, retractions=4*7,
    tracker_resolve=5*2+1+4*2, tracking_SVD=(5*2+1+4*2)*5,
    tracker_overlap=1+5*2+1+4*2, initial_SPD_eigvalsh=7+31,
    exact_signed_Gram_Lambda_product_cap=16000000)
contract['root_command'] = f'/data/code/atomx/.venv/bin/python {HERE.relative_to(ROOT)}/run.py --execution-binding {HERE.relative_to(ROOT)}/execution_binding.json'
(HERE/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
print(json.dumps(dict(driver_sha256=sha(HERE/'run.py'),contract_sha256=sha(HERE/'contract.json'),NPZ_reads_or_hashes=0)))
