"""Bounded independent saved-array audit; never imports Atomx or its oracle."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
from decimal import Decimal, localcontext

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze(value):
    return tuple(freeze(x) for x in value) if isinstance(value, list) else value


def decimal_response(h0, h1, c0, c1, e0, e1, phase):
    """Independent finite Rayleigh and residual decomposition on saved matrices."""
    with localcontext() as ctx:
        ctx.prec = 80
        d = Decimal.from_float
        zero = Decimal(0)
        a, b = [d(float(x)) for x in c0], [d(float(x)) for x in c1]
        h, k = [[d(float(x)) for x in row] for row in h0], [[d(float(x)) for x in row] for row in h1]
        e, f = d(float(e0)), d(float(e1))

        def dot(x, y):
            return sum((u*v for u, v in zip(x, y, strict=True)), zero)

        def action(matrix, vector):
            return [dot(row, vector) for row in matrix]

        n0, n1 = dot(a, a), dot(b, b)
        q0, q1 = dot(a, action(h, a))/n0, dot(b, action(k, b))/n1
        old = [Decimal(phase)*x for x in a]
        delta = [x-y for x, y in zip(b, old, strict=True)]
        ac = [[x-(e if i == j else zero) for j, x in enumerate(row)] for i, row in enumerate(h)]
        aq = [[x-(q0 if i == j else zero) for j, x in enumerate(row)] for i, row in enumerate(h)]
        bc = [[x-(f if i == j else zero) for j, x in enumerate(row)] for i, row in enumerate(k)]
        r0, r1, rq = action(ac, old), action(bc, b), action(aq, old)
        rt = [dot(delta, action(ac, delta))/n1, 2*dot(delta, r0)/n1,
              dot(old, r0)/n1, -dot(b, r1)/n1]
        qt = [dot(delta, action(aq, delta))/n1, 2*dot(delta, rq)/n1, dot(old, rq)/n1]
        radial = (dot(b, action(k, b))-dot(b, action(h, b)))/n1
        return dict(norm0=n0, norm1=n1, reported_terms=rt, rayleigh_terms=qt,
                    reported_response=sum(rt, zero), rayleigh_response=sum(qt, zero),
                    quotient_offsets=[q0-e, q1-f], radial=radial,
                    reported_delta=f-e, quotient_delta=q1-q0,
                    residual_norms=[dot(r0, r0).sqrt(), dot(r1, r1).sqrt()])


def main():
    start = time.monotonic()
    contract = json.loads((HERE/'contract.json').read_text())
    cap = contract['resources']
    os.sched_setaffinity(0, cap['actual_cpu_affinity'])
    resource.setrlimit(resource.RLIMIT_AS, (cap['address_space_bytes'], cap['address_space_bytes']))
    signal.alarm(cap['wall_time_seconds'])
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')):
        raise ValueError('single-thread environment required before NumPy import')
    import numpy as np

    checks = []

    def gate(name, condition, observed=None):
        row = dict(name=name, passed=bool(condition))
        if observed is not None:
            row['observed'] = observed
        checks.append(row)
        if not row['passed']:
            write(HERE/'partial_verification.json', dict(checks=checks, numerical_qualification_passed=False))
            raise ValueError(name)

    for name, digest in contract['inputs_sha256'].items():
        gate('input_sha256:'+name, sha(REPO/name) == digest)
    identity = json.loads((BASE/'candidate_source_identity.json').read_text())
    source_map = identity['all_atomx_python_sha256']
    gate('source_count', len(source_map) == 356)
    for name, digest in source_map.items():
        # Preserve the execution's frozen epoch even if a later shared source changes.
        gate('frozen_source:'+name, sha(BASE/'source'/name.removeprefix('src/')) == digest)
    seal = json.loads((BASE/'execution_contract.json').read_text())
    for name, digest in seal['root_reviewed_files_sha256'].items():
        if name.startswith('src/'):
            gate('sealed_source:'+name, source_map[name] == digest)
        else:
            gate('sealed_non_source:'+name, sha(REPO/name) == digest)
    actual = BASE/'actual'
    result = json.loads((actual/'result.json').read_text())
    root = json.loads((BASE/'root_execution.json').read_text())
    log = (BASE/'root_execution.log').read_text()
    gate('root_closed_exit1', root['exit_code'] == 1 and result['pytest_exit_code'] == 1)
    gate('qualification_failed_preserved', result['status'] == 'failed' and result['numerical_qualification_passed'] is False)
    gate('root_log_sha', sha(BASE/'root_execution.log') == root['log_sha256'])
    gate('original_26_pass_1_fail', '1 failed, 26 passed' in log)
    gate('original_stop_x', 'stopping after 1 failures' in log)
    gate('execution_binding', result['execution_sha256'] == sha(BASE/'execution_contract.json') == root['execution_contract_sha256'])
    gate('source_map_binding', result['source_map_sha256'] == contract['fixed']['source_map_sha256'])
    syntax = ast.parse((REPO/'tests/test_mixed_ci_finite_increment.py').read_text())
    unsupported = next(n for n in syntax.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'test_unsupported_scope_stale_density_or_owner_rejects_without_calls')
    changes = ast.literal_eval(unsupported.decorator_list[0].args[1])
    gate('unexecuted_16_unsupported_controls', len(changes) == 16)
    tests_status = dict(passed=26, failed=1, not_executed_due_to_x=17,
                        skipped_unsupported_cases=list(changes), skipped_legacy_controls=1,
                        second_direction_in_failed_test='not reached in live test; saved-array calculation below only')
    calls = json.loads((actual/'candidate_calls.json').read_text())
    oldcalls = json.loads((actual/'frozen_calls.json').read_text())
    for scope, counters in (('candidate', calls), ('old_comparator', oldcalls)):
        gate(scope+':attempted_completed', counters['started'] == counters['completed'])
        for key, limit in counters['limits'].items():
            gate(scope+':call_limit:'+key, counters['completed'][key] <= limit, counters['completed'][key])
    gate('candidate_exact_HCIg', [calls['completed'][k] for k in ('H_build','CI_solve','gradient')] == [13,13,2])
    gate('old_exact_HCIg', [oldcalls['completed'][k] for k in ('H_build','CI_solve','gradient')] == [5,0,1])
    gate('no_original_optimizer', calls['completed']['optimizer'] == 0)
    geometry = dict(np.load(actual/'geometry_inputs.npz', allow_pickle=False))
    assets = dict(np.load(actual/'oracle_assets.npz', allow_pickle=False))
    tables = json.loads((actual/'oracle_signed_tables.json').read_text())
    generated = json.loads((actual/'generated_input_identity.json').read_text())
    diagnostics = json.loads((actual/'finite_diagnostics.json').read_text())
    for named, file in (('generated_arrays_sha256','geometry_inputs.npz'),):
        gate(named, generated[named] == sha(actual/file))
    gate('oracle_assets_binding', tables['oracle_assets_sha256'] == sha(actual/'oracle_assets.npz'))
    gate('oracle_geometry_binding', tables['geometry_inputs_sha256'] == sha(actual/'geometry_inputs.npz'))
    gate('declared_unfiltered_tables', tables['filtering'].startswith('none;'))
    gate('actual_target_relative_weights', generated['targets'] == [[0,1,0,1.0],[0,1,1,3.0]])
    gate('full_dimension29', generated['csf_count'] == 29)
    gate('all13_diagnostics', len(diagnostics) == 13)
    keys = [tuple(map(int, key)) for key in geometry['orbital_keys']]
    gate('ordered_radial_keys', [list(x) for x in keys] == generated['radial_keys'])
    row_groups = [list(map(int, block['columns'])) for block in tables['blocks']]
    gate('complete_captured_blocks', sorted(sum(row_groups, [])) == list(range(29)) and len(row_groups) == 4)
    gate('captured_sizes', list(map(len, row_groups)) == [12,9,7,1])
    gate('target_captured_J0_block_contains_1s2', 0 in row_groups[0])
    kappas = sorted(set(k[1] for k in keys))
    for k in kappas:
        p, s, x = geometry[f'P_{k}'], geometry[f'S_{k}'], geometry[f'x0_{k}']
        gate(f'positive_P_metric:{k}', float(np.max(np.abs(p.T@s@p-np.eye(p.shape[1])))) <= 1e-10)
        for direction in range(2):
            eta = geometry[f'eta{direction}_{k}']
            gate(f'original_coordinate_tangent:{direction}:{k}', float(np.max(np.abs(x.T@eta+eta.T@x))) <= 1e-14)
    spectra = []
    block_columns = []
    ci_controls, radial_controls = [], []
    for index in range(13):
        ci = dict(np.load(actual/f'CI_{index:02}.npz', allow_pickle=False))
        h, c, e = ci['H'], ci['mixing'], ci['energies']
        for label, array, shape in (('H',h,(29,29)),('C',c,(29,29)),('E',e,(29,))):
            gate(f'complete_{label}:{index}', array.shape == shape and array.dtype == np.float64 and bool(np.isfinite(array).all()))
        gate(f'H_exact_symmetric:{index}', np.array_equal(h, h.T))
        resid = float(np.max(np.linalg.norm(h@c-c*e, axis=0)))
        gram = float(np.max(np.abs(c.T@c-np.eye(29))))
        gate(f'full_column_residual:{index}', resid <= 1e-8, resid)
        gate(f'full_CI_Gram:{index}', gram <= 1e-10, gram)
        assigned = []
        point_columns = []
        for block, rows in enumerate(row_groups):
            population = np.sum(c[rows]**2, axis=0)
            columns = np.flatnonzero(population > 1-1e-9)
            columns = columns[np.argsort(e[columns], kind='stable')]
            point_columns.append(columns)
            gate(f'block_complete:{index}:{block}', len(columns) == len(rows))
            outside = sorted(set(range(29))-set(rows))
            gate(f'no_C_block_leakage:{index}:{block}', np.count_nonzero(c[np.ix_(outside,columns)]) == 0)
            gate(f'no_H_block_leakage:{index}:{block}', np.count_nonzero(h[np.ix_(outside,rows)]) == 0)
            assigned += columns.tolist()
            gate(f'block_Gram:{index}:{block}', float(np.max(np.abs(c[np.ix_(rows,columns)].T@c[np.ix_(rows,columns)]-np.eye(len(rows))))) <= 1e-10)
        gate(f'full_spectrum_assigned_once:{index}', sorted(assigned) == list(range(29)))
        mh = hashlib.sha256(repr((h.shape,h.dtype.str)).encode()+h.tobytes(order='C')).hexdigest()
        gate(f'owner_H_bytes:{index}', mh == diagnostics[index]['diagnostic']['endpoint_H_sha256'][1])
        waves = geometry[f'waves_{index}']
        inc = geometry[f'increments_{index}']
        gate(f'full_radial_arrays:{index}', waves.shape == (7,2,35) and inc.shape == waves.shape and waves.dtype == inc.dtype == np.float64)
        radial_gram = 0.
        for k in kappas:
            positions = [i for i,key in enumerate(keys) if key[1] == k]
            values = np.column_stack([waves[i].reshape(70) for i in positions])
            radial_gram = max(radial_gram,float(np.max(np.abs(values.T@geometry[f'S_{k}']@values-np.eye(len(positions))))))
            if index == 0:
                x = geometry[f'x0_{k}']
                dx = np.zeros_like(x)
            else:
                direction = (index-1)//6
                step = (1e-4,-1e-4,1e-6,-1e-6,1e-8,-1e-8)[(index-1)%6]
                x = geometry[f'x_{direction}_{step}_{k}']
                dx = x-geometry[f'x0_{k}']
            gate(f'direct_P_x_point:{index}:{k}', np.array_equal(values,geometry[f'P_{k}']@x))
            increments = np.column_stack([inc[i].reshape(70) for i in positions])
            gate(f'direct_P_dx_increment:{index}:{k}', np.array_equal(increments,geometry[f'P_{k}']@dx))
        gate(f'actual_radial_Gram:{index}', radial_gram <= 1e-10, radial_gram)
        reconstruction = float(np.max(np.abs(waves.astype(np.longdouble)-geometry['waves_0'].astype(np.longdouble)-inc.astype(np.longdouble))))
        gate(f'rounded_endpoint_reconstruction:{index}', reconstruction <= 32*np.finfo(float).eps*max(1.,float(np.max(np.abs(geometry['waves_0'])))), reconstruction)
        gate(f'saved_radial_reconstruction:{index}', reconstruction == diagnostics[index]['diagnostic']['radial']['reconstruction_wave_max'])
        ci_controls.append(dict(index=index,residual=resid,Gram=gram,columns=[x.tolist() for x in point_columns],owner_H_sha256=mh))
        radial_controls.append(dict(index=index,Gram=radial_gram,reconstruction_wave_max=reconstruction))
        spectra.append(ci)
        block_columns.append(point_columns)
    candidate_g = dict(np.load(actual/'candidate_gradients.npz', allow_pickle=False))
    old = dict(np.load(actual/'frozen_comparators.npz', allow_pickle=False))
    for index in (0,1,2,7,8):
        gate(f'frozen_default_H_bitwise:{index}', np.array_equal(old[f'H_{index}'],spectra[index]['H']))
    gate('frozen_complete_signed_gradient_bitwise', np.array_equal(old['baseline_gradient'],candidate_g['baseline']))
    gate('saved_full_gradient_shape', candidate_g['baseline'].shape == candidate_g['endpoint'].shape == (7,70))
    signed_ledgers, finite_controls = [], []
    for index, record in enumerate(diagnostics):
        diag = record['diagnostic']
        gate(f'diagnostic_scope:{index}', diag['diagnostic_only'] and not diag['certified_error_bound'] and not diag['optimizer_adapter'])
        signed = {}
        report_sum, quotient_sum = [], []
        targets = []
        for target_index, target in enumerate(diag['targets']):
            expected_uid, weight = [0,1,target_index], (.25,.75)[target_index]
            gate(f'target_UID_weight:{index}:{target_index}', target['reference_level'] == target['selected_reference'] == target['selected_trial'] == expected_uid and target['weight'] == weight)
            column0, column1 = int(block_columns[0][0][target_index]),int(block_columns[index][0][target_index])
            c0, c1 = spectra[0]['mixing'][:,column0],spectra[index]['mixing'][:,column1]
            phase = 1 if np.dot(c0,c1) >= 0 else -1
            gate(f'whole_column_phase:{index}:{target_index}', target['phase'] == phase)
            response = decimal_response(spectra[0]['H'],spectra[index]['H'],c0,c1,
                                        spectra[0]['energies'][column0],spectra[index]['energies'][column1],phase)
            for field in ('norm0','norm1'):
                gate(f'{field}:{index}:{target_index}', abs(float(response[field])-target[field]) <= 2e-15)
            for field in ('reported_terms','rayleigh_terms','residual_norms'):
                error = max(abs(float(a)-b) for a,b in zip(response[field],target[field],strict=True))
                gate(f'Decimal_{field}:{index}:{target_index}', error <= 1e-12, error)
            offset_error = max(abs(float(a)-b) for a,b in zip(response['quotient_offsets'],target['quotient_minus_reported'],strict=True))
            gate(f'Decimal_quotient_offsets:{index}:{target_index}', offset_error <= 1e-12,offset_error)
            with localcontext() as ctx:
                ctx.prec = 80
                gate(f'finite_reported_identity:{index}:{target_index}', abs(float(response['radial']+response['reported_response']-response['reported_delta'])) <= 1e-30)
                gate(f'finite_quotient_identity:{index}:{target_index}', abs(float(response['radial']+response['rayleigh_response']-response['quotient_delta'])) <= 1e-30)
            report_sum.append(weight*float(response['reported_response']))
            quotient_sum.append(weight*float(response['rayleigh_response']))
            target_density = {}
            for bi, block in enumerate(tables['blocks']):
                projection = assets[f'projection_{bi}']
                pairs = assets[f'pairs_{bi}']
                amplitudes = projection @ c1[block['columns']]
                products = amplitudes[pairs[:,0]]*amplitudes[pairs[:,1]]
                values = np.zeros(len(block['keys']))
                # Dense saved sparse-row contraction, preserving all signed terms.
                for prod,row in zip(products,block['rows'],strict=True):
                    for keypos, coefficient in row:
                        values[keypos] += float(coefficient)*prod
                for key, value in zip(block['keys'],values,strict=True):
                    if value != 0:
                        key = freeze(key)
                        target_density[key] = target_density.get(key,0.)+float(value)
            for key,value in target_density.items():
                signed[key] = math.fsum((signed.get(key,0.),weight*value/target['norm1']))
            targets.append(dict(UID=expected_uid,weight=weight,column0=column0,column1=column1,phase=phase,
                                reported_response=float(response['reported_response']),rayleigh_response=float(response['rayleigh_response'])))
        negative = sum(x < 0 for x in signed.values())
        gate(f'full_unfiltered_signed_count:{index}', len(signed) == diag['signed_density_terms'],len(signed))
        gate(f'negative_signed_terms:{index}', negative == diag['negative_signed_density_terms'] and negative > 0,negative)
        signed_ledgers.append(dict(index=index,label=record['label'],terms=[dict(key=key,coefficient=value) for key,value in signed.items()]))
        radial_errors = [abs(diag['explicit_radial']-float(Decimal(record['oracle_explicit']))),
                         abs(diag['reconstruction_radial']-float(Decimal(record['oracle_reconstruction']))),
                         abs(diag['explicit_radial']+diag['reconstruction_radial']-float(Decimal(record['oracle_complete_radial'])))]
        for ei,error in enumerate(radial_errors):
            gate(f'stored_Decimal_radial:{index}:{ei}', error <= 1e-13,error)
        with localcontext() as ctx:
            ctx.prec = 80
            gate(f'Decimal_stored_domain_additivity:{index}', abs(Decimal(record['oracle_explicit'])+Decimal(record['oracle_reconstruction'])-Decimal(record['oracle_complete_radial'])) <= Decimal('1e-70'))
        oracle_reported = float(Decimal(record['oracle_complete_radial']))+sum(report_sum)
        oracle_quotient = float(Decimal(record['oracle_complete_radial']))+sum(quotient_sum)
        error_reported,error_quotient = abs(diag['reported_change']-oracle_reported),abs(diag['rayleigh_change']-oracle_quotient)
        gate(f'finite_reported_total:{index}',error_reported <= 1e-12,error_reported)
        gate(f'finite_quotient_total:{index}',error_quotient <= 1e-12,error_quotient)
        if abs(oracle_quotient) > 1e-13:
            gate(f'finite_quotient_relative:{index}',error_quotient <= .1*abs(oracle_quotient),error_quotient/abs(oracle_quotient))
        if index == 0:
            gate('base_zero_radial',diag['explicit_radial'] == diag['reconstruction_radial'] == 0.)
        finite_controls.append(dict(index=index,label=record['label'],radial_errors=radial_errors,
            reported_error=error_reported,quotient_error=error_quotient,
            native_rayleigh_change=diag['rayleigh_change'],stored_Decimal_radial=float(Decimal(record['oracle_complete_radial'])),
            independently_recomputed_CI_response=sum(quotient_sum),targets=targets))
    derivatives = []
    for direction in range(2):
        derivative = 0.
        for k in kappas:
            positions = [i for i,key in enumerate(keys) if key[1] == k]
            tangent = geometry[f'P_{k}']@geometry[f'eta{direction}_{k}']
            derivative += sum(float(candidate_g['baseline'][i]@tangent[:,j]) for j,i in enumerate(positions))
        plus,minus = diagnostics[1+direction*6],diagnostics[2+direction*6]
        measured = (plus['diagnostic']['rayleigh_change']-minus['diagnostic']['rayleigh_change'])/(2e-4)
        error = abs(measured-derivative)
        threshold = max(1e-8,1e-5*abs(derivative))
        derivatives.append(dict(direction=direction,step=1e-4,analytic=derivative,central_difference=measured,
                                absolute_error=error,relative_error=error/abs(derivative),original_threshold=threshold,
                                original_gate_passed=error <= threshold,
                                live_test_reached=(direction == 0),source='saved full signed gradient, original P and tangent, native saved finite rayleigh increments'))
    gate('original_first_derivative_failure_reproduced', derivatives[0]['original_gate_passed'] is False)
    gate('original_failure_values_match_log', '-0.008720635851431197' in log and '-0.00872046743791691' in log)
    for name,digest in contract['inputs_sha256'].items():
        gate('unchanged_after:'+name, sha(REPO/name) == digest)
    write(HERE/'signed_density_replay.json', signed_ledgers)
    evidence = dict(CI=ci_controls,radial=radial_controls,finite_increment=finite_controls,
                    derivative=derivatives,test_execution=tests_status,
                    actual_completed_calls=dict(H_build=18,CI_solve=13,gradient=3,optimizer=0),
                    saved_matrix_decimal_rebuilt=False,
                    saved_radial_Decimal_interpretation='13 stored independent Decimal operator strings checked; full25 domain matrix oracle not repeated',
                    target_block_interpretation='Captured12-CSF block includes original1s2 row and supplies J0/+1 target local ranks from frozen test; all four complete blocks checked without re-enumeration')
    write(HERE/'replay_details.json',evidence)
    report = dict(all_passed=all(x['passed'] for x in checks),passed_count=len(checks),checks=checks,
        numerical_qualification_passed=False,failed_original_scientific_gate='central signed directional derivative at h=1e-4',
        original_test_execution=tests_status,contract_sha256=sha(HERE/'contract.json'),
        inputs_sha256=contract['inputs_sha256'],outputs_sha256={name:sha(HERE/name) for name in ('replay_details.json','signed_density_replay.json')},
        source_map_sha256=contract['fixed']['source_map_sha256'],
        actual_audit_resources=dict(affinity=sorted(os.sched_getaffinity(0)),elapsed_seconds=time.monotonic()-start,
                                   max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,address_space_bytes=cap['address_space_bytes']),
        physical_calls=dict(Atomx_imports=0,H_build=0,CI_solve=0,gradient=0,Dirac=0,optimizer=0),
        scope='Independent saved-array consistency audit. Original live numerical qualification failed; unsupported and legacy guards not executed. No optimizer/Armijo/physical accuracy qualification.')
    write(HERE/'verification.json',report)
    print(json.dumps(dict(audit_consistency_passed=report['all_passed'],controls=len(checks),qualification_passed=False,
                         derivatives=derivatives,elapsed_seconds=report['actual_audit_resources']['elapsed_seconds'])))


if __name__ == '__main__':
    main()
