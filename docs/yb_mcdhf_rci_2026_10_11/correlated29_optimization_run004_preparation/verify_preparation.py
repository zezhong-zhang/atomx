"""Minimal stdlib successor/source/UID/budget checks; zero numerical imports."""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import types
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

    spec = importlib.util.spec_from_file_location('native29_continuation_preparation', HERE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.metadata(contract)
    check('same_f043_source_and_closed_terminal16_admitted', source.name == 'candidate_source')
    check('stdlib_only_zero_science', not any(n == 'numpy' or n.startswith('numpy.') or n == 'atomx'
          or n.startswith('atomx.') for n in sys.modules))
    aliases = [contract['input_roles'][k] for k in ('saved_gradient', 'saved_waves', 'continuation_baseline_arrays')]
    check('one_exact_terminal_array_three_roles', aliases[0] == aliases[1] == aliases[2]
          and aliases[0]['path'].endswith('run003_preparation/actual/checkpoints/accepted_0016.npz'))
    with zipfile.ZipFile(REPO / aliases[0]['path']) as archive:
        fields = set(archive.namelist())
    check('terminal_contains_complete_own_H_C_g_metric_P_x', all(k + '.npy' in fields for k in
          ('x', 'H', 'energies', 'mixing', 'keys', 'P', 'Q', 'radial_gradient', 'raw_coordinate_gradient',
           'tangent_gradient', 'preconditioned_gradient', 'inverse_diagonal',
           *[f'projector_{i}' for i in range(7)], *[f'inverse_blocks_{i}' for i in range(7)])))
    tree = ast.parse((HERE / 'run.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'optimize_orbitals']
    check('one_public_optimizer_call', len(calls) == 1)
    keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    check('same_default_reported_no_callback_original_gates', keywords['energy_difference'] == 'None'
          and keywords['max_iter'] == '64' and keywords['tolerance'] == '1e-07'
          and keywords['energy_tolerance'] == '1e-08' and keywords['allow_objective_change'] == 'False')
    oldtree = ast.parse((REPO / contract['prospective_successor_of']['run.py']['path']).read_text())
    oldcall = copy.deepcopy([n for n in ast.walk(oldtree) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute) and n.func.attr == 'optimize_orbitals'][0])
    for key in oldcall.keywords:
        if key.arg == 'max_iter':
            key.value = ast.Constant(value=64)
    check('public_optimizer_call_AST_only_maxiter_changed', ast.dump(oldcall) == ast.dump(calls[0]))
    check('same_complete_model_P_metric_UID_weights_and_tracker', contract['fixed'] ==
          json.loads((REPO / contract['prospective_successor_of']['contract.json']['path']).read_text())['fixed'])
    rows = [{'label': label, 'UID': uid, 'nominal_gap_cm1': float(i)} for i, (label, uid) in enumerate(
            {'6s_1/2': [1, 1, 0], '5d_3/2': [3, 1, 0], '5d_5/2': [5, 1, 0],
             '6p_1/2': [1, -1, 0], '6p_3/2': [3, -1, 0]}.items())]
    check('physical_labels_independent_of_list_order', module.nominal_four_gap_rms(rows) ==
          module.nominal_four_gap_rms(list(reversed(rows))))
    try:
        module.admit(contract, {})
    except ValueError as error:
        check('missing_exact_root_authority_refused', str(error) == 'Wrong exact root execution binding')
    else:
        check('missing_exact_root_authority_refused', False)
    # Extract only the declared resource subclass. A stdlib mock replaces the
    # numerical helper base: no NumPy/helper import and no expectation kernels.
    budget_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                       and n.name == 'ContractExactGuardBudget')

    class BudgetExceeded(RuntimeError):
        pass

    class MockBudget:
        def __init__(self):
            self.points = self.quotients = self.examined_pairs = self.nonzero_H_pairs = 0

    namespace = {'guard': types.SimpleNamespace(ExactGuardBudget=MockBudget, ExactGuardBudgetExceeded=BudgetExceeded)}
    exec(compile(ast.Module(body=[budget_node], type_ignores=[]), '<declared-reserve-only>', 'exec'), namespace)
    Budget = namespace['ContractExactGuardBudget']
    budget = Budget()
    budget.points, budget.quotients, budget.examined_pairs = 64, 320, 64 * 115129
    budget.reserve(5, 115129)
    valid = (budget.points, budget.quotients, budget.examined_pairs) == (65, 325, 7483385)
    before = vars(budget).copy()
    try:
        budget.reserve(5, 115129)
    except BudgetExceeded:
        check('point_boundary65_then_refusal_no_counter_mutation', valid and vars(budget) == before)
    else:
        check('point_boundary65_then_refusal_no_counter_mutation', False)
    for name, field, value, roots, pairs in (
            ('quotient_boundary325_no_mutation', 'quotients', 325, 1, 0),
            ('examined_pair_boundary8m_no_mutation', 'examined_pairs', 8000000, 0, 1)):
        budget = Budget()
        setattr(budget, field, value)
        before = vars(budget).copy()
        try:
            budget.reserve(roots, pairs)
        except BudgetExceeded:
            check(name, vars(budget) == before)
        else:
            check(name, False)
    budget = Budget()
    before = vars(budget).copy()
    try:
        budget.reserve(-1, 0)
    except ValueError:
        check('negative_work_refused_no_mutation', vars(budget) == before)
    else:
        check('negative_work_refused_no_mutation', False)
    check('prospective64_257_65_budget_only', [contract['resources'][k] for k in
          ('max_H', 'max_CI', 'max_full_gradient', 'max_new_accepted_steps')] == [257, 257, 65, 64]
          and contract['resources']['wall_seconds'] == 1200)
    check('no_actual_directory_or_numerical_imports', not (HERE / 'actual').exists()
          and not any(n == 'numpy' or n.startswith('numpy.') or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    proof = {'all_source_only_checks_passed': True, 'checks': len(controls), 'controls': controls,
             'runner_sha256': sha(HERE / 'run.py'), 'contract_sha256': sha(HERE / 'contract.json'),
             'verifier_sha256': sha(__file__), 'numerical_imports': 0, 'scientific_calls': 0,
             'scope': 'stdlib source/JSON/ZIPfields/UID/refusal/explicitreserve only; no HCIg replay or derivative inheritance',
             'execution_authorized': False}
    (HERE / 'preparation_verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'all_source_only_checks_passed': True, 'checks': len(controls), 'scientific_calls': 0}))


if __name__ == '__main__':
    main()
