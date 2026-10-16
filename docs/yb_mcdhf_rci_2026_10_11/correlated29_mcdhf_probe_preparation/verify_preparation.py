"""Read-only metadata/AST and pure UID tests; no physical/numerical imports."""
import ast
import copy
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    begin = time.monotonic()
    contract = json.loads((HERE / 'contract.json').read_text())
    source = (HERE / 'run.py').read_text()
    tree = ast.parse(source)
    controls = []

    def check(name, passed):
        controls.append({'name': name, 'passed': bool(passed)})

    inputs = {}
    for role, item in {**contract['input_roles'], **contract['pending_prerequisites']}.items():
        inputs[item['path']] = item['sha256']
        check('input_' + role, sha(REPO / item['path']) == item['sha256'])
    identity = json.loads((REPO / contract['input_roles']['candidate_source_identity']['path']).read_text())
    check('356source_map', identity['source_map_sha256'] == contract['source']['candidate356_map_sha256'])
    check('complete356source_count', len(identity['all_atomx_python_sha256']) == 356)
    archive = REPO / contract['source']['candidate_archive']
    for name, expected in identity['all_atomx_python_sha256'].items():
        check('frozen_source_' + name, sha(archive / name.removeprefix('src/')) == expected)
    for name, expected in contract['source']['critical_sources_sha256'].items():
        check('critical_source_' + name, identity['all_atomx_python_sha256'][name] == expected)
    check('original353_epoch', contract['source']['original353_map_sha256'] ==
          '10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f')
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    probe = functions['probe']
    calls = [n for n in ast.walk(probe) if isinstance(n, ast.Call)]
    attributes = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
    check('one_public_state_import', 'OrbitalOptimizationState' in source)
    check('one_public_from_arrays_callsite', attributes.count('from_arrays') == 1)
    check('one_evaluate_callsite_shared6points', attributes.count('evaluate') == 1)
    check('evaluate_gradient_true_once', source.count('waves, gradient=True)') == 1)
    check('zero_optimizer_call', 'optimize_orbitals' not in source)
    check('no_arbitrary_owner_restore', '__new__' not in source)
    check('admit_before_numerical_imports', isinstance(probe.body[1], ast.Assign)
          and isinstance(probe.body[1].value, ast.Call) and probe.body[1].value.func.id == 'admit'
          and isinstance(probe.body[2], ast.Import))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    check('top_level_only_stdlib', not any('numpy' in ast.unparse(n) or 'atomx' in ast.unparse(n) or 'scipy' in ast.unparse(n) for n in imports))
    check('immutable_external_root_seal', contract['status'] == 'prepared_requires_exact_root_execution'
          and contract['root_authorized'] is False and 'preparation_sha256' in source)
    check('no_rootseal_selfhash_prerequisite', not any('execution_contract' in name for name in contract['pending_prerequisites']))
    limits = next(n.value for n in ast.walk(probe) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'limits' for t in n.targets))
    check('all_attempt_caps', ast.literal_eval(limits) == {
        'H':6, 'CI':6, 'gradient':1, 'validation_Dirac':7, 'initial_hS':7,
        'CI_internal_hS':42, 'mixed_hS':28, 'tracking_resolution':21,
        'tracking_SVD':105, 'tracking_overlap':22, 'geometry_eigvalsh':36,
        'scipy_eigh':37, 'polar_retraction':28, 'polar_numpy_eigh':28})
    check('initial_CI_mixed_hS77', sum(ast.literal_eval(limits)[k] for k in
          ('initial_hS','CI_internal_hS','mixed_hS')) == 77)
    check('both_derivative_gates', 'reported_derivative' in source and 'Rayleigh_derivative' in source
          and source.count('ledger["gate"]') == 2)
    check('original_derivative_thresholds', contract['finite_step']['h'] == 1e-6
          and contract['gates']['directional_absolute'] == 1e-8 and contract['gates']['directional_relative'] == 1e-5)
    check('samepoint_gradient_replay_unexecuted', contract['gates']['same_point_raw_g_pg_replay_executed_by_this_probe'] is False
          and '"same_point_whole_gradient_replay_executed": False' in source
          and '"same_point_raw_g_1e_minus9_pg_1e_minus10_replay_gates_passed": None' in source)
    check('original_stationarity_gates', contract['gates']['orbital_tolerance'] == 1e-7 and
          contract['gates']['energy_tolerance_Ha'] == 1e-8)
    check('new5_predeclared_drift_gate', contract['gates']['new5_coordinate_recovery_max_wave_drift'] == 1e-12)
    check('new_epoch_metric_preserving_old_inverse', contract['metric']['coordinate_count'] == 3451
          and contract['metric']['original7_P_bytes_preserved'] and
          contract['metric']['original24_inverse_blocks_and_diagonal_columns_preserved'])
    check('direct_point_not_polar_repaired', source.count('_polar_retraction(candidate') == 1
          and 'ordinary_anchor' in contract['point_policy'])
    check('raw_and_logical_h_distinguished', 'logical = np.tril(h) + np.tril(h, -1).T' in source
          and 'h.tobytes() == reference_h.tobytes()' in source)
    check('full_gradient_archived_before_qualification', source.index('"_raw_radial_gradient.npz"') <
          source.index('raw_identity()', source.index('def complete_point')))
    check('all5_complete_blocks1046', sum(row[2] for row in contract['fixed']['target_blocks']) == 1046)
    check('two_seeded_directions_only', [r['PCG64_seed'] for r in contract['directions']] == [20261011,20261012])
    check('bounded_resources', (contract['resources']['CPU_affinity_count'],contract['resources']['threads'],
          contract['resources']['address_space_GiB'],contract['resources']['wall_seconds']) == (1,1,3.95,300))
    # Execute only this stdlib/pure identity helper; no driver/physical module import.
    namespace = {}
    exec(compile(ast.Module(body=[functions['target_bindings']], type_ignores=[]), '<pure-target-identity-helper>', 'exec'), namespace)
    targets = json.loads((REPO / contract['input_roles']['tracking_binding']['path']).read_text())['targets']
    def uid(t):
        return t['twoJ'], t['parity'], t['models']['DC']['local_root']
    bindings = [{'reference_levels':[uid(t)],'selected_levels':[uid(t)]} for t in targets]
    actual_symmetry_order = [(1,1),(5,1),(3,1),(3,-1),(1,-1)]
    permutation = sorted(bindings, key=lambda b:actual_symmetry_order.index(tuple(b['reference_levels'][0][:2])))
    for name, supplied in [('original',bindings),('reversed',list(reversed(bindings))),('actual_symmetry_order',permutation)]:
        matched = namespace['target_bindings'](targets,supplied)
        check('physical_UID_order_' + name, all(matched[uid(t)]['reference_levels'] == [uid(t)] for t in targets))
    for name, kind in [('duplicate','duplicate'),('missing','missing'),('wrong_reference','reference'),('wrong_selected_sector','sector'),('multi_reference','multi')]:
        supplied = copy.deepcopy(bindings)
        if kind == 'duplicate':
            supplied[-1] = copy.deepcopy(supplied[0])
        if kind == 'missing':
            supplied.pop()
        if kind == 'reference':
            supplied[-1]['reference_levels'] = [(3,-1,9)]
        if kind == 'sector':
            supplied[-1]['selected_levels'] = [(1,1,0)]
        if kind == 'multi':
            supplied[-1]['reference_levels'] *= 2
        try:
            namespace['target_bindings'](targets,supplied)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check('reject_UID_' + name,rejected)
    # The rejection path executes no imports and requires no fabricated root token.
    reject_ns = {'HERE':HERE, 'REPO':REPO}
    exec(compile(ast.Module(body=[functions['admit']],type_ignores=[]), '<unprepared-admission-rejection>', 'exec'),reject_ns)
    try:
        reject_ns['admit']({'status':'unprepared'}, {})
    except ValueError:
        check('unprepared_rejected_before_import',True)
    else:
        check('unprepared_rejected_before_import',False)
    check('no_actual_output', not (HERE / 'actual').exists())
    check('no_numpy_atomx_scipy_import', not any(n == 'numpy' or n.startswith(('numpy.','atomx','scipy')) for n in sys.modules))
    proof = {'all_passed':all(r['passed'] for r in controls),'checks_count':len(controls),'checks':controls,
             'runner_sha256':sha(HERE/'run.py'),'contract_sha256':sha(HERE/'contract.json'),
             'verifier_sha256':sha(Path(__file__)), 'inputs_sha256':inputs,
             'source_map_sha256':identity['source_map_sha256'],
             'scope':'Metadata, AST, pure UID mapping and early refusal only; no numerical qualification.',
             'NumPy_Atomx_SciPy_imports':0,'H_CI_gradient_Dirac_optimizer_calls':0,
             'elapsed_s':time.monotonic()-begin,'RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'accounting_CPUs':sorted(os.sched_getaffinity(0))}
    (HERE/'preparation_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('checks','inputs_sha256','accounting_CPUs')}))
    if not proof['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
