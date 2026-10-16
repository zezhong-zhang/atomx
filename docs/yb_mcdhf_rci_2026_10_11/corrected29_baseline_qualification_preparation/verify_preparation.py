"""Minimal source/JSON admission and refusal; no numerical imports or science."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / 'contract.json').read_text())
    controls = []

    def check(name, passed):
        controls.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise ValueError(name)

    spec = importlib.util.spec_from_file_location('corrected29_consumer_preparation', HERE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check('stdlib_import_only', not any(n == 'numpy' or n.startswith('numpy.') or n == 'atomx'
                                       or n.startswith('atomx.') for n in sys.modules))
    module.metadata(contract)
    check('bound_inputs_source_and_closed_scale_prerequisite', True)
    check('no_pending_scientific_prerequisites', contract['pending_prerequisites'] == [])
    for name in ('run.py', 'constraint_bookkeeping.py'):
        ast.parse((HERE / name).read_text())
        check('AST_' + name, True)
    check('separate_corrected_source_epoch', contract['source']['map_sha256'] ==
          'f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe')
    check('original_metric_retained', contract['metric_sha256'] ==
          'a96f1aecd3e18316376121d8255c156a5f5b4c01a21fd0ba86ca91131ae66d73')
    check('complete_original_model', contract['fixed']['target_CSFs'] == 1046
          and len(contract['fixed']['keys29']) == 29 and len(contract['fixed']['five_weights']) == 5)
    check('explicit_budget', [contract['calls'][k] for k in ('H', 'CI', 'gradient_from_weights', 'Dirac')] == [5, 5, 3, 0])
    check('no_sourcechange_optimizer', contract['execution'].find('zero optimizer/Dirac') >= 0)
    run_ast = ast.parse((HERE / 'run.py').read_text())
    attributes = [n.func.attr for n in ast.walk(run_ast) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    check('no_optimizer_invocation', 'optimize_orbitals' not in attributes)
    targets = [{'twoJ': 1, 'parity': 1, 'models': {'DC': {'local_root': 0}}},
               {'twoJ': 3, 'parity': 1, 'models': {'DC': {'local_root': 0}}},
               {'twoJ': 5, 'parity': 1, 'models': {'DC': {'local_root': 0}}},
               {'twoJ': 1, 'parity': -1, 'models': {'DC': {'local_root': 0}}},
               {'twoJ': 3, 'parity': -1, 'models': {'DC': {'local_root': 0}}}]
    bindings = [{'reference_levels': [[t['twoJ'], t['parity'], 0]],
                 'selected_levels': [[t['twoJ'], t['parity'], 0]]} for t in reversed(targets)]
    matched = module.target_bindings(targets, bindings)
    check('shuffled_binding_original_UID', set(matched) == {(t['twoJ'], t['parity'], 0) for t in targets})
    try:
        module.target_bindings(targets, bindings[:-1] + [bindings[0]])
    except ValueError:
        check('duplicate_UID_refused', True)
    else:
        check('duplicate_UID_refused', False)
    try:
        module.execute(contract, {})
    except ValueError as error:
        check('absent_exact_root_execution_refused', str(error) == 'Exact root execution contract required')
    else:
        check('absent_exact_root_execution_refused', False)
    check('refusal_did_not_import_numerical_source', not any(n == 'numpy' or n.startswith('numpy.')
          or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    check('no_actual_directory_created', not (HERE / 'actual').exists())
    proof = {'all_passed': all(row['passed'] for row in controls), 'checks': len(controls),
             'controls': controls, 'runner_sha256': sha(HERE / 'run.py'),
             'contract_sha256': sha(HERE / 'contract.json'),
             'consumer_math_helper_sha256': sha(HERE / 'constraint_bookkeeping.py'),
             'verifier_sha256': sha(__file__), 'numerical_imports': 0,
             'scientific_calls': {'H': 0, 'CI': 0, 'gradient': 0, 'Dirac': 0, 'optimizer': 0},
             'scope': 'source/JSON/AST/refusal/UID preparation; no numerical or derivative qualification',
             'execution_authorized': False}
    (HERE / 'preparation_verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'all_passed': proof['all_passed'], 'checks': len(controls), 'scientific_calls': 0}))


if __name__ == '__main__':
    main()
