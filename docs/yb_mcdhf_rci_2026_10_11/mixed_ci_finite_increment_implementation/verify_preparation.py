"""Readonly stdlib source, fixture and scope proof; no Atomx import/numerics."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
OLD = REPO/'docs/yb_mcdhf_rci_2026_10_11/checkpoint_hamiltonian_reuse_implementation/source002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_functions(source):
    return {node.name: node for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    checks = []
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
    identity = json.loads((BASE/'candidate_source_identity.json').read_text())
    contract = json.loads((BASE/'qualification_contract.json').read_text())
    old_identity = json.loads((OLD.parent/'source_identity002.json').read_text())
    allowed = {'src/atomx/methods/csf/variational.py', 'src/atomx/physics/csf/matrix.py',
               'src/atomx/physics/csf/radial_increment.py', 'src/atomx/numerics/ci_energy_increment.py'}
    changed = set()
    for name, expected in identity['all_atomx_python_sha256'].items():
        check('live_'+name, sha(REPO/name) == expected)
        check('archived_'+name, sha(BASE/'source'/name.removeprefix('src/')) == expected)
        if old_identity['all_atomx_python_sha256'].get(name) != expected:
            changed.add(name)
    check('only_authorized_source_delta', changed == allowed)
    check('all_previous_source_files_retained', set(old_identity['all_atomx_python_sha256']).issubset(
                                                identity['all_atomx_python_sha256']))
    check('canonical_map_digest', hashlib.sha256(json.dumps(identity['all_atomx_python_sha256'],
              sort_keys=True, separators=(',', ':')).encode()).hexdigest() == identity['source_map_sha256'])
    for name in ('methods/csf/variational.py', 'methods/ci.py', 'physics/csf/radial_density.py',
                 'numerics/bspline/orbital_optimization.py'):
        old_source = (OLD/'atomx'/name).read_text()
        new_source = (REPO/'src/atomx'/name).read_text()
        old_functions, new_functions = node_functions(old_source), node_functions(new_source)
        for function, old_node in old_functions.items():
            check(f'old_method_bytes_{name}_{function}', function in new_functions and
                  ast.get_source_segment(old_source, old_node) ==
                  ast.get_source_segment(new_source, new_functions[function]))
    old_matrix = (OLD/'atomx/physics/csf/matrix.py').read_text()
    new_matrix = (REPO/'src/atomx/physics/csf/matrix.py').read_text()
    old_functions, new_functions = node_functions(old_matrix), node_functions(new_matrix)
    old_loop = old_functions['pair_potential'].body[2:]
    helper = new_functions['_bspline_pair_potential'].body[1:]
    check('extracted_loop_AST_and_operation_order',
          [ast.dump(v, include_attributes=False) for v in old_loop] ==
          [ast.dump(v, include_attributes=False) for v in helper])
    for function, old_node in old_functions.items():
        if function in ('_radial_evaluator', 'pair_potential'):
            continue
        check(f'old_matrix_function_{function}', function in new_functions and
              ast.get_source_segment(old_matrix, old_node) ==
              ast.get_source_segment(new_matrix, new_functions[function]))
    for name, expected in contract['prepared_files_sha256'].items():
        check('contract_file_'+name, sha(REPO/name) == expected)
    for name, expected in identity['protected_inputs_sha256'].items():
        check('protected_'+name, sha(REPO/name) == expected)
    for path in list((BASE/'source').rglob('*.py'))+[BASE/'run.py', BASE/'decimal_oracle.py',
                 BASE/'frozen_comparator.py', REPO/'tests/test_mixed_ci_finite_increment.py']:
        ast.parse(path.read_text())
        check('AST_'+str(path.relative_to(REPO)), True)
    check('execution_pending', contract['authorization']['finite_qualification_execution'] is False)
    check('full_declared_H_CI_gradient_caps', contract['resources']['maximum_H_builds'] == 18 and
           contract['resources']['maximum_complete_CI_solves'] == 13 and
           contract['resources']['maximum_full_gradients'] == 3)
    proof = dict(all_passed=all(v['passed'] for v in checks), checks_count=len(checks), checks=checks,
                 actual_H_CI_gradient_optimizer_calls=0, source_map_sha256=identity['source_map_sha256'],
                 qualification_contract_sha256=sha(BASE/'qualification_contract.json'),
                 verifier_sha256=sha(Path(__file__)), validation_scope='stdlib metadata/AST only')
    (BASE/'preparation_verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(all_passed=proof['all_passed'], checks_count=len(checks))))
    if not proof['all_passed']:
        print(json.dumps([c for c in checks if not c['passed']]))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
