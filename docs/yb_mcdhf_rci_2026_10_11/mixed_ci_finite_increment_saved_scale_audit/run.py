"""Saved-array causal scale audit, with zero Atomx imports or physical calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
from decimal import Decimal, localcontext

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
PRIOR = BASE.parent/'mixed_ci_finite_increment_implementation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((BASE/'contract.json').read_text())
    execution = json.loads((BASE/'execution_contract.json').read_text())
    assert execution['contract_sha256'] == sha(BASE/'contract.json')
    assert execution['runner_sha256'] == sha(Path(__file__))
    assert len(os.sched_getaffinity(0)) == 1
    assert os.environ['OPENBLAS_NUM_THREADS'] == os.environ['OMP_NUM_THREADS'] == '1'
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    def stop(signum, frame):
        raise TimeoutError('saved-array audit60s cap')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(60)
    import numpy as np
    checks = []
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
    for name, expected in contract['inputs_sha256'].items():
        check('entry_'+name, sha(REPO/name) == expected)
    if not all(c['passed'] for c in checks):
        raise ValueError('saved audit input binding failed')
    geometry = np.load(PRIOR/'actual/geometry_inputs.npz', allow_pickle=False)
    gradient = np.load(PRIOR/'actual/candidate_gradients.npz', allow_pickle=False)['baseline']
    keys = tuple(tuple(key) for key in geometry['orbital_keys'].tolist())
    diagnostics = json.loads((PRIOR/'actual/finite_diagnostics.json').read_text())
    by_label = {row['label']: row for row in diagnostics}
    check('exact13savedpoints', len(diagnostics) == 13)
    base_targets = by_label['base']['diagnostic']['targets']
    base_target_identity = [(row['reference_level'], row['selected_reference'], row['weight'])
                            for row in base_targets]
    point_controls = []
    for index, row in enumerate(diagnostics):
        saved = np.load(PRIOR/'actual'/f'CI_{index:02}.npz', allow_pickle=False)
        H, C, energies = saved['H'], saved['mixing'], saved['energies']
        check(f'point{index}_fullsquare', H.shape == C.shape == (len(energies), len(energies)))
        residual = float(np.max(np.linalg.norm(H@C-C*energies, axis=0)))
        gram = float(np.max(np.abs(C.T@C-np.eye(len(energies)))))
        check(f'point{index}_complete_column_residual', residual <= 1e-8)
        check(f'point{index}_complete_Gram', gram <= 1e-10)
        check(f'point{index}_target_weights_id', [(t['reference_level'], t['selected_reference'], t['weight'])
              for t in row['diagnostic']['targets']] == base_target_identity)
        point_controls.append(dict(index=index, label=row['label'], eigenpair_column_residual=residual,
                                   Gram=gram, targets=row['diagnostic']['targets']))
    directions = []
    for direction in range(2):
        derivative, wide_derivative, tangent_norm = 0., np.longdouble(0), 0.
        for k in (-2, -1, 1):
            P, x, eta = geometry[f'P_{k}'], geometry[f'x0_{k}'], geometry[f'eta{direction}_{k}']
            active_keys = [key for key in keys if key[1] == k]
            tangent = P@eta
            tangent_wide = P.astype(np.longdouble)@eta.astype(np.longdouble)
            check(f'd{direction}_k{k}_exact_tangent', np.max(np.abs(x.T@eta+eta.T@x)) <= 1e-14)
            tangent_norm += float(eta.ravel()@eta.ravel())
            for column, key in enumerate(active_keys):
                g = gradient[keys.index(key)]
                derivative += float(g@tangent[:, column])
                wide_derivative += g.astype(np.longdouble)@tangent_wide[:, column]
        check(f'd{direction}_normalized_coordinate_tangent', abs(tangent_norm-1) <= 1e-14)
        scale_rows = []
        for h in (1e-4, 1e-6, 1e-8):
            plus, minus = by_label[f'd{direction}_{h}'], by_label[f'd{direction}_{-h}']
            fields = ('explicit_radial', 'reconstruction_radial', 'reported_response', 'rayleigh_response',
                      'reported_change', 'rayleigh_change', 'direct_reported_root_difference',
                      'direct_retained_quotient_difference', 'floating_EO_difference')
            central = {name: (plus['diagnostic'][name]-minus['diagnostic'][name])/(2*h) for name in fields}
            with localcontext() as ctx:
                ctx.prec = 80
                central['saved_Decimal_radial'] = float((Decimal(plus['oracle_complete_radial'])
                                                       -Decimal(minus['oracle_complete_radial']))
                                                      /(2*Decimal.from_float(h)))
                central['saved_Decimal_explicit'] = float((Decimal(plus['oracle_explicit'])
                                                       -Decimal(minus['oracle_explicit']))
                                                      /(2*Decimal.from_float(h)))
                central['saved_Decimal_reconstruction'] = float((Decimal(plus['oracle_reconstruction'])
                                                       -Decimal(minus['oracle_reconstruction']))
                                                      /(2*Decimal.from_float(h)))
            error = central['rayleigh_change']-derivative
            gate = max(1e-8, 1e-5*abs(derivative))
            scale_rows.append(dict(h=h, central=central, error_vs_gradient=error,
                original_derivative_gate=gate, original_derivative_gate_passed=abs(error) <= gate,
                signed_plus={name: plus['diagnostic'][name] for name in fields},
                signed_minus={name: minus['diagnostic'][name] for name in fields},
                numerator=plus['diagnostic']['rayleigh_change']-minus['diagnostic']['rayleigh_change'],
                response_asymmetry=plus['diagnostic']['rayleigh_response']-minus['diagnostic']['rayleigh_response'],
                reconstruction_asymmetry=plus['diagnostic']['reconstruction_radial']
                                             -minus['diagnostic']['reconstruction_radial']))
        e0, e1, e2 = (row['error_vs_gradient'] for row in scale_rows)
        coarse, medium, fine = (row['central']['rayleigh_change'] for row in scale_rows)
        extrapolated = (1e4*medium-coarse)/(1e4-1)
        estimated_order = (math.log(abs(e0/e1))/math.log(100) if e0 != 0 and e1 != 0 else None)
        difference_order = (math.log(abs((coarse-medium)/(medium-fine)))/math.log(100)
                            if coarse != medium and medium != fine else None)
        reduction = abs(e0/e1) if e1 else None
        supported = (scale_rows[1]['original_derivative_gate_passed'] and
                     abs(e0) >= 20*abs(e1) and abs(extrapolated-derivative)
                     <= scale_rows[1]['original_derivative_gate'])
        directions.append(dict(direction=direction, archived_gradient_derivative=derivative,
            lifted_saved_gradient_derivative=float(wide_derivative), wide_minus_float=float(wide_derivative-derivative),
            coordinate_tangent_norm=tangent_norm, scales=scale_rows,
            h2_extrapolation=extrapolated, extrapolation_error=extrapolated-derivative,
            measured_error_order_coarse_to_medium=estimated_order,
            measured_difference_order_all_three=difference_order,
            error_reduction_coarse_to_medium=reduction,
            coarse_h2_coefficient=(coarse-medium)/(1e-8-1e-12),
            classification='coarse central-truncation supported; no qualification upgrade' if supported else
                           'saved scales do not resolve cause; preserve failed derivative gate'))
    for name, expected in contract['inputs_sha256'].items():
        check('exit_'+name, sha(REPO/name) == expected)
    result = dict(scope='saved13points/two directions only; no new physical or finite-method evaluation',
         contract_sha256=sha(BASE/'contract.json'), execution_sha256=sha(BASE/'execution_contract.json'),
         runner_sha256=sha(Path(__file__)), all_saved_array_checks_passed=all(c['passed'] for c in checks),
         checks_count=len(checks), original_qualification_status='failed_unchanged', directions=directions,
         point_controls=point_controls, actual_H_CI_gradient_optimizer_calls=0,
         original_gradient_derivative_source='saved baseline Euclidean gradient dotted with saved P@eta',
         limitations=['no new step sizes or half-step order test',
                      'finite residual/bank/rounding noise can dominate smallest step',
                      'two-scale h2 extrapolation is diagnostic, not a new derivative acceptance gate',
                      'unsupported consumer guards were not reached in original -x test panel'])
    (BASE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (BASE/'verification.json').write_text(json.dumps(dict(all_passed=all(c['passed'] for c in checks),
        checks=checks, checks_count=len(checks), result_sha256=sha(BASE/'result.json'),
        actual_H_CI_gradient_optimizer_calls=0),indent=2)+'\n')
    signal.alarm(0)
    print(json.dumps(dict(checks_passed=result['all_saved_array_checks_passed'],
        directions=[{k: d[k] for k in ('direction','archived_gradient_derivative',
                    'measured_error_order_coarse_to_medium','error_reduction_coarse_to_medium',
                    'extrapolation_error','classification')} for d in directions],
        derivatives=[[dict(h=r['h'],D=r['central']['rayleigh_change'],error=r['error_vs_gradient'],
                           gate_pass=r['original_derivative_gate_passed']) for r in d['scales']]
                     for d in directions]),indent=2))


if __name__ == '__main__':
    main()
