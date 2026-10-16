"""Once-only saved-array algebra review; never imports Atomx or rebuilds H/CI."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
CASES = ('J1_even', 'J3_even', 'J5_even', 'J1_odd', 'J3_odd')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return plain(value.tolist()) if hasattr(value, 'tolist') else value


def main():
    contract = read(HERE / 'execution_contract.json')
    source = read(HERE / 'current_source_identity.json')
    for name, expected in source['all_atomx_python_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('changed source: ' + name)
    rows, arrays, bindings, checks = {}, {}, {}, []
    original = {}

    def check(name, observed, limit=None):
        ok = bool(observed) if limit is None else bool(observed <= limit)
        checks.append(dict(name=name, observed=observed, limit=limit, PASS=ok))
        if not ok:
            raise ValueError('saved gate: ' + name)

    for case in ('matched',) + CASES:
        folder = HERE / ('actual_' + case)
        result = read(folder / 'result.json')
        check(case + ' live PASS', result['status'].startswith('PASS')
              and all(row['PASS'] for row in result['checks']))
        check(case + ' same source', result['source_map_sha256'] == contract['source_map_sha256'])
        check(case + ' saved gradient bytes', sha(folder / 'gradients.npz') == result['gradient_arrays_sha256'])
        with np.load(folder / 'gradients.npz', allow_pickle=False) as z:
            gradient, direction, keys, vectors = (z[k] for k in
                                                  ('gradients', 'radial_direction', 'keys', 'signed_vectors'))
        check(case + ' complete gradient shape', gradient.shape == (10 if case == 'matched' else 2, 31, 246))
        check(case + ' finite complete arrays', all(np.isfinite(a).all() for a in (gradient, direction, vectors)))
        if arrays:
            first = next(iter(arrays.values()))
            check(case + ' same radial keys', np.array_equal(keys, first['keys']))
            check(case + ' same literal direction', np.array_equal(direction, first['direction']))
        arrays[case] = dict(gradient=gradient, direction=direction, keys=keys, vectors=vectors)
        bindings[case] = dict(result_sha256=sha(folder / 'result.json'),
                              gradient_sha256=sha(folder / 'gradients.npz'))
        for i, state in enumerate(result['states'].values()):
            action = float(np.sum(gradient[i].astype(np.longdouble)*direction.astype(np.longdouble)))
            check(case + '/action/' + str(i), abs(action-state['directional_action_Ha']), 1e-12)
            norm = math.fsum(float(x)*float(x) for x in vectors[:, i])
            check(case + '/norm/' + str(i), abs(norm-state['actual_CI_norm']), 1e-12)
            if case != 'matched':
                key = (state['arm'], tuple(state['UID']))
                check(case + '/unique state/' + str(i), key not in rows)
                rows[key] = dict(**state, gradient=gradient[i])
            else:
                original[state['arm'], tuple(state['UID'])] = state
    check('all ten complete model states', len(rows) == 10)
    base = contract['base_contract']
    with np.load(ROOT / base['inputs']['checkpoint_arrays']['path'], allow_pickle=False) as z:
        cp = {key: z[key] for key in z.files}
    layout_rows = read(ROOT / base['inputs']['layout']['path'])
    check('checkpoint bytes still bound', sha(ROOT / base['inputs']['checkpoint_arrays']['path']) ==
          base['inputs']['checkpoint_arrays']['sha256'])
    keys = tuple(map(tuple, arrays[CASES[0]]['keys'].tolist()))
    direction = arrays[CASES[0]]['direction']
    summaries = {}
    for arm in ('A', 'B'):
        selected = [rows[arm, tuple(uid)] for _, uid in base['UIDs']]
        gradient = np.array([math.fsum(.2*state['gradient'][i, j] for state in selected)
                             for i in range(31) for j in range(246)]).reshape(31, 246)
        raw = np.zeros(3689)
        for index, row in enumerate(layout_rows):
            group, shape, sl = tuple(map(tuple, row['keys'])), tuple(row['shape']), slice(*row['slice'])
            raw[sl] = (cp['projector_'+str(index)].T @ gradient[[keys.index(k) for k in group]].T).ravel()

        def tangent(value):
            projected = value.copy()
            for row in layout_rows:
                shape, sl = tuple(row['shape']), slice(*row['slice'])
                y, a = cp['x'][sl].reshape(shape), value[sl].reshape(shape)
                cross = y.T@a
                projected[sl] = (a-.5*y@(cross+cross.T)).ravel()
            return projected

        tg = tangent(raw)
        pre = np.zeros_like(tg)
        for index, row in enumerate(layout_rows):
            shape, sl = tuple(row['shape']), slice(*row['slice'])
            block = tg[sl].reshape(shape)
            pre[sl] = np.column_stack([b@block[:, j] for j, b in enumerate(cp['inverse_blocks_'+str(index)])]).ravel()
        pg = tangent(pre)
        eol = math.fsum(.2*state['directional_action_Ha'] for state in selected)
        mapped = float(np.sum(gradient.astype(np.longdouble)*direction.astype(np.longdouble)))
        check(arm + ' full EOL action closure', abs(mapped-eol), 1e-12)
        ground = rows[arm, (1, 1, 0)]
        lines = []
        for state in selected:
            if state is ground:
                continue
            delta = state['directional_action_Ha']-ground['directional_action_Ha']
            delta_forward = state['independent_forward_Ha']-ground['independent_forward_Ha']
            baseline = (original[arm, tuple(state['UID'])]['directional_action_Ha']
                        - original[arm, (1, 1, 0)]['directional_action_Ha'])
            check(arm + ' independent line derivative ' + str(state['UID']), abs(delta-delta_forward), 2e-9)
            lines.append(dict(UID=state['UID'], line_derivative_Ha=delta,
                              forward_line_derivative_Ha=delta_forward,
                              difference_Ha=delta-delta_forward,
                              original_model_line_derivative_Ha=baseline,
                              complete_class_response_change_Ha=delta-baseline,
                              same_direction_response_ratio=(delta/baseline if baseline else None)))
        summaries[arm] = dict(EOL_directional_derivative_Ha=eol, lines=lines,
                              reference_metric_tangent_norm=float(np.linalg.norm(tg)),
                              reference_metric_pg_norm=float(np.linalg.norm(pg)),
                              reference_metric_g_dot_pg_Ha=float(tg@pg),
                              stationarity_qualified=False)
    result = dict(status='PASS_saved_complete_density_gradient_review', checks=checks,
                  source_map_sha256=contract['source_map_sha256'], bindings=bindings,
                  summaries=summaries, ordinary_FD005_superseded=False,
                  scope='Readback/algebra only. Independent live radial forward checks already recorded; no new H residual/CI-response/finite-step derivative verification or optimizer admission.')
    target = HERE / 'aggregate001'
    target.mkdir()
    (target / 'result.json').write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), summaries=summaries)))


if __name__ == '__main__':
    main()
