"""Minimal stdlib source/metadata/refusal checks; no numerical execution."""
import ast
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / 'contract.json').read_text())
    controls = []

    def check(name, passed):
        controls.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise ValueError(name)

    spec = importlib.util.spec_from_file_location('corrected29_optimizer_preparation', HERE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.metadata(contract)
    check('source_input_closed_corrected_baseline_admitted', source.name == 'candidate_source')
    check('source_epoch_f043', contract['source']['candidate356_map_sha256'] ==
          'f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe')
    check('stdlib_only_no_science', not any(n == 'numpy' or n.startswith('numpy.')
          or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    tree = ast.parse((HERE / 'run.py').read_text())
    optimizer_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == 'optimize_orbitals']
    check('single_public_optimizer_call', len(optimizer_calls) == 1)
    call = optimizer_calls[0]
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    check('ordinary_no_energy_callback_original_gates', keywords['energy_difference'] == 'None'
          and keywords['max_iter'] == '16' and keywords['tolerance'] == '1e-07'
          and keywords['energy_tolerance'] == '1e-08')
    oldtree = ast.parse((REPO / contract['prospective_successor_of']['run.py']['path']).read_text())
    oldcall = [node for node in ast.walk(oldtree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == 'optimize_orbitals'][0]
    check('public_optimizer_invocation_AST_unchanged', ast.dump(call) == ast.dump(oldcall))
    with zipfile.ZipFile(REPO / contract['input_roles']['saved_gradient']['path']) as archive:
        fields = set(archive.namelist())
    check('own_corrected_production_gradient_metric_fields', all(k + '.npy' in fields for k in
          ('x', 'inverse_diagonal', 'production_radial', 'production_raw_coordinate', 'production_preconditioned',
           *[f'projector_{i}' for i in range(7)], *[f'inverse_blocks_{i}' for i in range(7)])))
    check('own_corrected_baseline_not_old500', 'corrected29_baseline_qualification_preparation/actual/'
          in contract['input_roles']['saved_waves']['path']
          and contract['input_roles']['saved_waves'] == contract['input_roles']['corrected_baseline_arrays'])
    rows = [{'label': label, 'UID': uid, 'nominal_gap_cm1': float(i)} for i, (label, uid) in enumerate(
        {'6s_1/2': (1, 1, 0), '5d_3/2': (3, 1, 0), '5d_5/2': (5, 1, 0),
         '6p_1/2': (1, -1, 0), '6p_3/2': (3, -1, 0)}.items())]
    check('physical_lower_and_four_gap_order_invariant', module.nominal_four_gap_rms(rows) ==
          module.nominal_four_gap_rms(list(reversed(rows))) and
          module.physical_rows_by_label(list(reversed(rows)))['6s_1/2']['UID'] == (1, 1, 0))
    try:
        module.physical_rows_by_label(rows[:-1] + [rows[0]])
    except module.QualificationStop:
        check('duplicate_target_refused', True)
    else:
        check('duplicate_target_refused', False)
    try:
        module.admit(contract, {})
    except ValueError as error:
        check('absent_root_seal_refused_before_import', str(error) == 'Wrong exact root execution binding')
    else:
        check('absent_root_seal_refused_before_import', False)
    check('no_actual_directory', not (HERE / 'actual').exists())
    check('refusal_no_numerical_imports', not any(n == 'numpy' or n.startswith('numpy.')
          or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    check('producer_budget_declared', [contract['resources'][k] for k in
          ('max_H', 'max_CI', 'max_full_gradient', 'max_new_accepted_steps')] == [65, 65, 17, 16])
    result = {'all_source_only_checks_passed': all(c['passed'] for c in controls),
              'controls': controls, 'checks': len(controls), 'runner_sha256': sha(HERE / 'run.py'),
              'contract_sha256': sha(HERE / 'contract.json'), 'verifier_sha256': sha(__file__),
              'numerical_imports': 0, 'scientific_calls': 0,
              'scope': 'source/JSON/ZIP-field/AST/physical UID/refusal only, no optimizer qualification',
              'execution_authorized': False}
    (HERE / 'preparation_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_source_only_checks_passed': result['all_source_only_checks_passed'],
                      'checks': len(controls), 'scientific_calls': 0}))


if __name__ == '__main__':
    main()
