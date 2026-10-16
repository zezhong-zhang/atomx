"""Seal SOURCE/JSON preparation without importing numerics or reading NPZ."""
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'AGENTS.md').is_file())
PARENT = HERE.parent


def sha(path):
    path = Path(path)
    if path.suffix == '.npz':
        raise ValueError('NPZ hashing is forbidden during preparation')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / 'preparation.json').exists() or (HERE / 'handoff.json').exists():
        raise FileExistsError('Frozen preparation is single-write')
    contract = json.loads((HERE / 'contract.json').read_text())
    previous = json.loads((PARENT / 'contract.json').read_text())
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise ValueError(name)

    source = ROOT / contract['source_archive']
    identity = json.loads((ROOT / contract['input_roles']['source_identity']['path']).read_text())
    check('complete365_source_epoch', identity['source_map_sha256'] == contract['source_map_sha256'] and len(identity['all_source_sha256']) == 365)
    check('all365_source_hashes', all(sha(source / k.removeprefix('src/')) == v for k, v in identity['all_source_sha256'].items()))
    roles = contract['input_roles']
    check('all_input_paths_exist', all((ROOT / v['path']).is_file() for v in roles.values()))
    check('all_bound_SHA_schema', all(re.fullmatch('[0-9a-f]{64}', v['sha256']) for v in roles.values()))
    check('all_JSON_source_input_hashes', all(Path(v['path']).suffix == '.npz' or sha(ROOT / v['path']) == v['sha256'] for v in roles.values()))
    check('six_NPZ_identities_deferred', sum(v['path'].endswith('.npz') for v in roles.values()) == 6)
    check('unchanged_physical_calls_and_caps', contract['resources'] == previous['resources'] and contract['expected_exact_completed'] == previous['expected_exact_completed'])
    check('unchanged_operators_and_thresholds', all(contract[k] == previous[k] for k in ('operators', 'gates', 'gate_static_zero_whole_H_Ha', 'additional_original_thresholds', 'bound3689_metric_sha256')))
    check('unchanged_physical_definition', all(contract['fixed'][k] == previous['fixed'][k] for k in previous['fixed'] if k != 'inputs'))
    check('unchanged_original_nonbaseline_inputs', all(roles[k] == v for k, v in previous['input_roles'].items() if k not in ('literal31_DC_H_E_C', 'actual/recovered_owner.json')))
    check('current_saved_DC_explicit_binding', roles['literal31_DC_H_E_C']['sha256'] == '22d8b69d958663ec8cb915ad225eccb2e92032914a7ec342058a2c4e1c46ee9e')
    check('failed_receipt_exact_binding', roles['closed_phaseA_receipt']['sha256'] == 'b8388cf1fcda83d26b53fb99e1443083b22fcdfb67453a3d2f3374533ad7e745')
    check('failure_exact_binding', roles['closed_phaseA_failure']['sha256'] == '6cac50e2b357e4d3b95550196b8628798c42589723cfdf0d045882d6e091e6a7')
    check('offline_result_exact_binding', roles['offline_diagnosis_result']['sha256'] == '7bed5f9bff625e5f13542b26814fdd89537104de0a8d7dddcc1b36ef3991b4d1')
    check('old_failure_preserved', contract['same_epoch_baseline_change']['old_gate_remains_failed'] and not contract['same_epoch_baseline_change']['current_MCDHF_optimizer_qualified'])
    check('no_actual_output', not (HERE / 'actual').exists())
    old_tree, new_tree = [ast.parse(p.read_text()) for p in (PARENT / 'run.py', HERE / 'run.py')]
    for tree in (old_tree, new_tree):
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]]
                check('top_level_stdlib/' + ast.unparse(node), all(n in sys.stdlib_module_names for n in names))
    execute = []
    for tree in (old_tree, new_tree):
        body = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute'))
        for node in ast.walk(body):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
                node.keywords = [k for k in node.keywords if k.arg not in ('current_epoch_DC_baseline_reproduced', 'original_phaseA_failed_byte_gate_passed')]
                for keyword in node.keywords:
                    if keyword.arg == 'status' and isinstance(keyword.value, ast.Constant) and keyword.value.value in ('closed_native31_fixed_RCI_phaseA', 'closed_native31_same_epoch_RCI_phaseA'):
                        keyword.value.value = 'same_execute_body'
        execute.append(ast.dump(body, include_attributes=False))
    check('entire_numerical_execute_AST_unchanged_except_result_labels', execute[0] == execute[1])
    for file in ('run.py', 'verify_preparation.py'):
        ast.parse((HERE / file).read_text())
    check('driver_verifier_AST_parse', True)
    command = "import runpy,sys,json;p=sys.argv[1];sys.argv=[p];runpy.run_path(p,run_name='__main__');print(json.dumps({'numerical_imports':any(k == 'numpy' or k.startswith('numpy.') or k == 'atomx' or k.startswith('atomx.') for k in sys.modules)}))"
    default = subprocess.run(['/usr/bin/python3', '-I', '-c', command, str(HERE/'run.py')], capture_output=True, text=True, timeout=30)
    check('default_SOURCE_only_exit0', default.returncode == 0)
    lines = default.stdout.splitlines()
    check('default_no_numerical_imports', not json.loads(lines[-1])['numerical_imports'])
    help_result = subprocess.run(['/usr/bin/python3', '-I', str(HERE/'run.py'), '--help'], capture_output=True, text=True, timeout=30)
    check('help_exit0', help_result.returncode == 0)
    ruff = subprocess.run([str(ROOT/'.venv/bin/ruff'), 'check', str(HERE/'run.py'), str(HERE/'verify_preparation.py')], capture_output=True, text=True, timeout=30)
    check('Ruff', ruff.returncode == 0)
    links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', (HERE/'README.md').read_text())
    prospective = {'preparation.json', 'handoff.json'}
    check('local_links_exist_or_single_write_outputs', all(p in prospective or (HERE / p).is_file() for p in links))
    preparation = dict(status='SOURCE_JSON_only_prepared_NO_numerical_execution', all_SOURCE_JSON_checks_passed=True,
        checks=checks, driver_sha256=sha(HERE/'run.py'), contract_sha256=sha(HERE/'contract.json'),
        verifier_sha256=sha(__file__), README_sha256=sha(HERE/'README.md'), documentation_impact_sha256=sha(HERE/'documentation_impact.json'),
        source_map_sha256=contract['source_map_sha256'], source_identity_sha256=roles['source_identity']['sha256'],
        source_files=365, deferred_NPZ_file_hashes=6, NPZ_reads_hashes_loads=0, NumPy_Atomx_imports=0,
        physical_calls=0, API_cards_registry_calls=0, sharedsrc_mutations=0, root_binding_required=True,
        original_failed_byte_gate_passed=False, current_MCDHF_optimizer_qualified=False,
        default_mode_summary=json.loads(lines[0]), Ruff=ruff.stdout.strip())
    (HERE/'preparation.json').write_text(json.dumps(preparation, indent=2, allow_nan=False)+'\n')
    handoff = dict(status='READY_FOR_ROOT_REVIEW_ONLY', driver_sha256=preparation['driver_sha256'],
        contract_sha256=preparation['contract_sha256'], preparation_sha256=sha(HERE/'preparation.json'),
        verifier_sha256=preparation['verifier_sha256'], source_map_sha256=contract['source_map_sha256'],
        resources=contract['resources'], science_executed=False, root_execution_binding_created=False,
        numerical_execute_AST_unchanged_except_result_labels=True, all_SOURCE_JSON_checks_passed=True,
        checks_passed=len(checks), original_phaseA_failure_immutable=True, NPZ_reads_hashes_loads=0,
        NumPy_Atomx_imports=0, physical_calls=0, no_cards_or_timer=True,
        root_next='Review driver diff and prospective contract; bind exact driver/preparation/input/source/resources separately, then one guarded foreground run only. Any failure stops; no old-gate reclassification.')
    (HERE/'handoff.json').write_text(json.dumps(handoff, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=handoff['status'], checks_passed=len(checks), run_sha256=preparation['driver_sha256'],
        contract_sha256=preparation['contract_sha256'], preparation_sha256=handoff['preparation_sha256'], handoff_sha256=sha(HERE/'handoff.json'))))


if __name__ == '__main__':
    main()
