"""One stdlib-only source/input/budget preparation, no numerical imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    c = load(HERE / 'contract.json')
    checks = []

    def gate(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise ValueError(name)

    gate('not_authorized_or_executed', c['root_execution_authorized'] is False and not (HERE/'actual').exists())
    identities = {}
    for variant in ('ordinary', 'candidate'):
        declaration = c[variant]
        identity = load(REPO / declaration['identity'])
        identities[variant] = identity
        manifest = identity['all_atomx_python_sha256']
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        gate(variant+'_map', digest == declaration['map'] == identity['source_map_sha256'])
        gate(variant+'_356_complete_files', len(manifest) == 356)
        for name, digest in manifest.items():
            gate(variant+'_'+name, sha(REPO / declaration['archive'] / name.removeprefix('src/')) == digest)
    old, new = (identities[k]['all_atomx_python_sha256'] for k in ('ordinary','candidate'))
    gate('same_complete_path_set', set(old) == set(new))
    changed = [name for name in old if old[name] != new[name]]
    gate('only_two_inner_orders_private_convol', changed == ['src/atomx/numerics/bspline/convol.py'])
    oldfile = REPO / c['ordinary']['archive'] / 'atomx/numerics/bspline/convol.py'
    newfile = REPO / c['candidate']['archive'] / 'atomx/numerics/bspline/convol.py'
    oldtext, newtext = oldfile.read_text(), newfile.read_text()
    expected = oldtext.replace('    n_sub = ks\n', '    n_sub = ks + (k // 2 if k >= 0 else 0)\n')
    expected = expected.replace('    n_sub = max(ksp, ksq)\n', '    n_sub = max(ksp, ksq) + (k // 2 if k >= 0 else 0)\n')
    gate('exact_two_line_replacement', expected == newtext)
    oldtree, newtree = ast.parse(oldtext), ast.parse(newtext)
    normalized = copy.deepcopy(newtree)
    replaced = []
    for function in normalized.body:
        if isinstance(function, ast.FunctionDef) and function.name in ('_triang_sym','_triang_pqpq'):
            for node in function.body:
                if isinstance(node, ast.Assign) and any(isinstance(x,ast.Name) and x.id=='n_sub' for x in node.targets):
                    expr = 'ks' if function.name == '_triang_sym' else 'max(ksp, ksq)'
                    node.value = ast.parse(expr, mode='eval').body
                    replaced.append(function.name)
    gate('only_two_assignment_AST_deltas', replaced == ['_triang_sym','_triang_pqpq'] and ast.dump(normalized) == ast.dump(oldtree))
    for rank in (-8,-2,-1,0,1):
        for order in (4,5,6,8,9):
            gate(f'unchanged_negative_k0_k1_{rank}_{order}', order+(rank//2 if rank>=0 else 0) == order)
    for rank in range(7):
        for order in (4,5,6,8,9):
            nodes = order+rank//2
            gate(f'positive_source_degree_{rank}_{order}', 2*nodes-1 >= 2*order-2+rank and nodes <= 12)
    gate('private_matrix_route_unchanged', old['src/atomx/physics/csf/matrix.py'] == new['src/atomx/physics/csf/matrix.py'])
    for role, row in c['inputs'].items():
        gate('input_'+role, sha(REPO/row['path']) == row['sha256'])
    gate('same_point_native_reference_bytes', c['inputs']['saved_point']['sha256'] == c['inputs']['normalized_control_point']['sha256'])
    prior = load(REPO/c['inputs']['normalized_control_result']['path'])
    gate('closed_native_fullnormalized_numeric_controls', prior['all_numeric_controls_passed'] is True and prior['samepoint_full_normalized_gradient_control_completed'] is True and prior['source_map_sha256'] == c['ordinary']['map'])
    tables = load(REPO/c['inputs']['signed_tables']['path'])
    def key(value):
        return tuple(key(v) if isinstance(v,list) else v for v in value)
    targets = {tuple(t['UID']):t for t in tables['targets']}
    gate('five_unique_complete_UIDs', len(targets)==5 and len(tables['target_tables'])==5 and set(targets)=={tuple(x) for x in c['fixed']['target_UIDs']})
    gate('five_labels', {t['label'] for t in targets.values()} == set(c['fixed']['target_labels']))
    gate('all_weights_equal_unchanged', all(t['weight']==.2 for t in targets.values()))
    full = [key(x[0]) for x in tables['full_normalized']]
    union = {key(k) for t in tables['target_tables'] for k,_ in t['unfiltered']}
    selected = [k for k in full if k[0]=='r' and k[1]>=2]
    gate('full_union12403_unique', len(full)==len(set(full))==12403 and set(full)==union)
    gate('every_h_or_nonnegative_radial_key', all(k[0]=='h' or (k[0]=='r' and isinstance(k[1],int) and 0<=k[1]<=6) for k in full))
    gate('all7398_positive_affected_keys', len(selected)==7398)
    sources = {(k[1],tuple(sorted((k[4],k[5])))) for k in selected}
    pairs = {tuple(sorted((k[i],k[j]))) for k in selected for i,j in ((2,3),(4,5))}
    gate('333_sourcepotentials_278_pairdensities', len(sources)==333 and len(pairs)==278)
    for rank in range(7):
        gate(f'all_rank{rank}_counts', sum(k[0]=='r' and k[1]==rank for k in full)==c['full_affected_scope']['by_rank_unique'][str(rank)])
    affected = {}
    for table in tables['target_tables']:
        target=targets[tuple(table['UID'])]
        gate('column_'+target['label'], table['column']==target['column'])
        affected[target['label']]=sum(k[0]=='r' and k[1]>=2 for k,_ in table['unfiltered'])
    gate('complete25568_signed_target_contributions', sum(affected.values())==25568 and [affected[label] for label in c['fixed']['target_labels']]==c['full_affected_scope']['five_target_affected_counts'])
    gate('no_coefficient_screening', tables['no_small_coefficient_filter'] is True and c['full_affected_scope']['small_term_filter'] is False)
    counts = {k:c['anchor_calls'].get(k,0)+2*c['radial_calls_per_variant'].get(k,0) for k in c['max_all_calls']}
    gate('aggregate_exact_call_budget', counts==c['max_all_calls'] and counts['hS']==14 and counts['public_radial_evaluate']==14796)
    gate('anchor_3setup_5rankgroups_20components', c['anchor_calls']['basis_setup']==3 and c['anchor_calls']['factor_get']==20 and c['anchor_calls']['bank_get']==40)
    gate('real_dense_only_no_performance_patch', c['radial_calls_per_variant']['direct_semiseparable']==0 and c['radial_calls_per_variant']['direct_full_bank']==333*4)
    gate('no_H_CI_gradient_optimizer_GPU_budget', all(c['resources'][k]==0 for k in ('max_H','max_CI','max_gradient','max_optimizer','GPU_calls')))
    source = (HERE/'run.py').read_text()
    compile(source,str(HERE/'run.py'),'exec')
    tree=ast.parse(source)
    gate('runtime_syntax', True)
    gate('no_top_level_physics_import', all(not (isinstance(n,(ast.Import,ast.ImportFrom)) and any(getattr(n,'module','') and getattr(n,'module','').startswith(x) for x in ('numpy','atomx','numba','scipy'))) for n in tree.body))
    gate('no_CI_g_optimization_calls', not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ('ci_solve','gradient_from_weights','optimize_orbitals','evaluate')) or (isinstance(n.func,ast.Attribute) and n.func.attr in ('solve','gradient_from_weights','optimize_orbitals'))) for n in ast.walk(tree)))
    gate('no_native_eigen_or_retraction_calls', not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('eigh','eigvalsh','svd','qr','cholesky') for n in ast.walk(tree)))
    gate('fresh_source_isolated_cache', 'env["NUMBA_CACHE_DIR"] = str(output / (phase + "_numba_cache"))' in source and 'for phase in ("anchors", "ordinary", "candidate")' in source)
    gate('exact_CI_normalization_and_signed_difference', 'Fraction.from_float(float(x))**2' in source and 'Fraction.from_float(float(coefficient)) / norm' in source and 'Fraction.from_float(float(v1)) - Fraction.from_float(float(v0))' in source)
    gate('strict_lower_completed_h_and_S', 'np.tril(raw_h) + np.tril(raw_h, -1).T' in source and 'geometry[f"s_{k}"]' in source)
    gate('unchanged_actual_outer_no_prepare_call_radial', source.count('basis.prepare_bspline_data(')==1 and 'n_gauss=16' in source)
    gate('root_execution_guard', 'execution_authorized_by_root' in source and 'Missing exact root seal' in source)
    proof=dict(all_passed=True,checks_count=len(checks),checks=checks,driver_sha256=sha(HERE/'run.py'),contract_sha256=sha(HERE/'contract.json'),candidate_source_identity_sha256=sha(HERE/'candidate_source_identity.json'),candidate_source_map_sha256=c['candidate']['map'],ordinary_source_map_sha256=c['ordinary']['map'],preparation_driver_sha256=sha(HERE/'prepare.py'),actual_scientific_calls=0,NumPy_Atomx_imports=0,numeric_qualification=False,inputs_sha256={row['path']:row['sha256'] for row in c['inputs'].values()})
    (HERE/'preparation_verification.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:proof[k] for k in ('all_passed','checks_count','driver_sha256','contract_sha256','candidate_source_identity_sha256')}))


if __name__=='__main__':
    main()
