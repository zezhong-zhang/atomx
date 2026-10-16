"""Independent stdlib-only final guard-consumer source review; no fixture import."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1]
REPO = TOP.parents[1]
BASE = TOP / 'mixed_ci_finite_increment_guard_qualification002'
PRIOR = TOP / 'mixed_ci_finite_increment_implementation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    contract = json.loads((HERE/'contract.json').read_text())
    os.sched_setaffinity(0,contract['resources']['affinity'])
    cap = contract['resources']['AS_bytes']
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    signal.alarm(contract['resources']['wall_seconds'])
    checks = []

    def gate(name, condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:
            raise ValueError(name)

    for name,digest in contract['inputs_sha256'].items():
        gate('input:'+name,sha(REPO/name) == digest)
    identity = json.loads((PRIOR/'candidate_source_identity.json').read_text())
    source = identity['all_atomx_python_sha256']
    gate('unchanged_candidate_epoch',identity['source_map_sha256'] == contract['source_map_sha256'])
    gate('complete_source356',len(source) == 356)
    for name,digest in source.items():
        gate('current_source:'+name,sha(REPO/name) == digest)
        gate('frozen_source:'+name,sha(PRIOR/'source'/name.removeprefix('src/')) == digest)
    c = json.loads((BASE/'contract.json').read_text())
    prep = json.loads((BASE/'preparation_verification.json').read_text())
    gate('reported_preparation_only',prep['all_passed'] is True)
    for name in ('run.py','test_saved_guards.py','verify_preparation.py'):
        ast.parse((BASE/name).read_text())
        gate('AST_parse:'+name,True)
    test = ast.parse((BASE/'test_saved_guards.py').read_text())
    old_test = ast.parse((REPO/'tests/test_mixed_ci_finite_increment.py').read_text())
    functions = {n.name:n for n in test.body if isinstance(n,ast.FunctionDef)}
    old = next(n for n in old_test.body if isinstance(n,ast.FunctionDef)
               and n.name == 'test_unsupported_scope_stale_density_or_owner_rejects_without_calls')
    new = functions['test_unsupported_scope_rejects_without_physical_calls']
    changes = ast.literal_eval(new.decorator_list[0].args[1])
    gate('original16negativecategories',changes == ast.literal_eval(old.decorator_list[0].args[1]) and len(changes) == 16)
    gate('complete17negativecontrols',list(changes)+['legacy_fixed_and_constrained_mixed'] == c['guard_cases'])
    direction = functions['test_saved_medium_step_original_derivative_gate']
    gate('two_saved_directions',ast.literal_eval(direction.decorator_list[0].args[1]) == (0,1))
    gate('19prospectivecases',c['pytest_cases'] == 19)
    gate('original_medium_gate',c['derivative_gate'] == dict(selected_h=1e-6,relative=1e-5,absolute=1e-8))
    text = (BASE/'test_saved_guards.py').read_text()
    fixture = functions['saved_problem']
    calls = [n for n in ast.walk(fixture) if isinstance(n,ast.Call)]
    producer = [n for n in calls if isinstance(n.func,ast.Name) and n.func.id == 'build_ci_hamiltonian_snapshot']
    attachments = [n for n in calls if isinstance(n.func,ast.Name) and n.func.id == 'CIResult']
    gate('one_genuine_public_producer',len(producer) == 1)
    gate('two_metadata_CI_objects',len(attachments) == 2)
    validation = next(n for n in ast.walk(fixture) if isinstance(n,ast.For)
                      and isinstance(n.target,ast.Name) and n.target.id == 'index')
    gate('CIattach_after_complete_saved_validation',min(n.lineno for n in attachments) > validation.end_lineno)
    guard_source = ast.get_source_segment(text,new)
    for token in ("genuine_public_producer=True","np.array_equal(owner.matrix_Ha, baseline['H'])",
                  "owner.input_sha256 == header['endpoint_input_sha256'][0]",
                  "owner.matrix_sha256 == header['endpoint_H_sha256'][0]",
                  "np.array_equal(f.metric[k], geometry[f'S_{k}'])",
                  "np.array_equal(f.one_body[k], geometry[f'h_{k}'])",
                  "residual <= 1e-8 and gram <= 1e-10",
                  "consumer.state_average, consumer.level_tracking = p['equal'], p['rank_tracker']",
                  "assert any(len(group) == 2 for group, _ in rank_tracker._groups)",
                  "assert scales[0]['gate_passed'] is False",
                  "pytest.approx(derivative, rel=1e-5, abs=1e-8)",
                  "assert p['calls'].started == before"):
        gate('source_invariant:'+token,token in text)
    shell_nodes = [n for n in ast.walk(test) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                   and n.func.value.id == 'object' and n.func.attr == '__new__']
    gate('one_negative_only_malformed_shell',len(shell_nodes) == 1 and
         new.lineno < shell_nodes[0].lineno < new.end_lineno and
         "if change == 'operator':" in guard_source and
         "with pytest.raises(ValueError, match='complete native DC owners'):" in guard_source)
    gate('genuine_operator_coverage_refusal',"ci.hamiltonian_for(f.csfs, waves" in guard_source and
         "nuclear_potential=nuclear, include_breit=True" in guard_source)
    gate('no_accepted_arbitrary_H_fixture',not any(isinstance(n,ast.Call) and
         isinstance(n.func,ast.Attribute) and n.func.attr in ('__new__','__setattr__')
         for n in ast.walk(fixture)))
    limits = dict(H_build=1,CI_solve=0,gradient=0,one_body_assembly=6,nuclear_Dirac_solve=0,
                  polar_retraction=0,physical_overlap=2,tracking_SVD=0,SciPy_eigh=0,
                  NumPy_eigh=0,optimizer=0,radial_increment=0)
    gate('fixed_exact_budget',c['calls'] == limits)
    for key in ('H_build','CI_solve','gradient','one_body_assembly','nuclear_Dirac_solve',
                'polar_retraction','physical_overlap','tracking_SVD','SciPy_eigh','NumPy_eigh','optimizer','radial_increment'):
        gate('forward_counter:'+key,"self.wrap('"+key+"'" in text)
    run = (BASE/'run.py').read_text()
    runtime = ast.parse(run)
    main_node = next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    pytest_import = next(n for n in ast.walk(main_node) if isinstance(n,ast.Import)
                         and any(alias.name == 'pytest' for alias in n.names))
    for token in ("guard_qualification_execution_authorized","root seal mismatch",
                  "explicit single-thread execution is required","host/CPU admission failed",
                  "resource.setrlimit(resource.RLIMIT_AS", "signal.alarm(contract['resources']['wall_time_seconds'])",
                  "output.mkdir(exist_ok=False)", "calls['started'] != contract['calls']",
                  "calls['completed'] != contract['calls']", "verify_inputs(contract)"):
        gate('runtime_guard:'+token,token in run)
    gate('execution_seal_before_test_import',run.index('guard_qualification_execution_authorized') <
         run.index('import pytest') and main_node.lineno < pytest_import.lineno)
    forbidden_imports = []
    for n in runtime.body:
        if isinstance(n,ast.Import):
            forbidden_imports += [a.name for a in n.names if a.name.startswith(('atomx','numpy','scipy','pytest'))]
        elif isinstance(n,ast.ImportFrom) and n.module and n.module.startswith(('atomx','numpy','scipy','pytest')):
            forbidden_imports.append(n.module)
    gate('runtime_module_top_level_is_metadata_only',not forbidden_imports)
    gate('source_verify_both_before_and_after',run.count('verify_inputs(contract)') == 3)  # declaration plus2calls
    gate('failure_preservation',"guard_qualification_passed=False" in run and "original_coarse_qualification_status='failed; immutable'" in run)
    gate('guard_scope_not_full_old_qualification', 'original qualification passed' in c['not_claimed'])
    for name,digest in contract['inputs_sha256'].items():
        gate('unchanged_after:'+name,sha(REPO/name) == digest)
    report = dict(all_passed=all(r['passed'] for r in checks),checks=checks,passed_count=len(checks),
                  verdict='ready for root-sealed bounded successor; no unresolved material source finding',
                  live_guard_qualification_run_by_reviewer=False,live_guard_qualification_passed=None,
                  original_finite_qualification_passed=False,
                  prospective_calls=limits,prospective_cases=19,
                  inputs_sha256=contract['inputs_sha256'],contract_sha256=sha(HERE/'contract.json'),
                  source_map_sha256=c['source_map_sha256'],
                  reviewer_calls=dict(Atomx_imports=0,NumPy_imports=0,pytest_execution=0,H=0,CI=0,gradient=0,Dirac=0,optimizer=0),
                  actual_resources=dict(affinity=sorted(os.sched_getaffinity(0)),AS_bytes=cap,
                                        elapsed_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  limits='Source/AST/hash review only; runtime physical/numerical and refusal outcomes require separate actual execution.')
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_source_controls_passed=report['all_passed'],controls=len(checks),
                         live_numerical_run=False,elapsed_seconds=report['actual_resources']['elapsed_seconds'])))


if __name__ == '__main__':
    main()
