"""Source/stdlib metadata and resource-counter controls only; no numeric imports."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=(HERE/'run.py').read_text()
    tree=ast.parse(source)
    baseline_path=REPO/'docs/yb_mcdhf_rci_2026_10_11/correlated29_optimization_run005_preparation/run.py'
    before=ast.parse(baseline_path.read_text())
    def call_keywords(parsed):
        calls=[n for n in ast.walk(parsed) if isinstance(n,ast.Call) and ast.unparse(n.func)=='optimization.optimize_orbitals']
        assert len(calls)==1
        return {kw.arg:ast.dump(kw.value,include_attributes=False) for kw in calls[0].keywords}
    original,current=call_keywords(before),call_keywords(tree)
    checks=[]
    def check(name,passed):
        checks.append({'name':name,'passed':bool(passed)})
        if not passed:
            raise ValueError(name)
    check('single public optimizer call; only search-memory keyword/maxiter differ',
          {k:v for k,v in original.items() if k not in ('history_size','max_iter')}==
          {k:v for k,v in current.items() if k not in ('history_size','max_iter')})
    spec=importlib.util.spec_from_file_location('matched_history_preparation_consumer',HERE/'run.py')
    consumer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    contracts=[json.loads((HERE/branch/'contract.json').read_text()) for branch in ('history10','history40')]
    for contract in contracts:
        consumer.metadata(contract)
    check('two metadata admissions execute no numerical import',not any(n=='numpy' or n.startswith(('numpy.','atomx.')) or n=='atomx' for n in sys.modules))
    check('same original source/metric/physical target/tracker inputs and exact208seed',all(
          contracts[0][k]==contracts[1][k] for k in ('source','metric','fixed','input_roles','pending_prerequisites','gates')))
    check('only two declared search memories with empty initial histories',
          [c['search_memory']['history_size'] for c in contracts]==[10,40]
          and all(c['search_memory']['initial_history_size']==0 for c in contracts))
    check('separate CPU/thread/AS caps and same32step producer limits',
          [c['resources']['CPU'] for c in contracts]==[30,31]
          and all(c['resources']['max_new_accepted_steps']==32 and c['resources']['max_H']==c['resources']['max_CI']==129
                  and c['resources']['max_full_gradient']==33 and c['resources']['threads']==1
                  and c['resources']['address_space_GiB']==3.95 for c in contracts)
          and sum(c['resources']['address_space_GiB'] for c in contracts)==7.9)
    reserve=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='reserve')
    class BudgetExceeded(RuntimeError):
        pass
    namespace={'guard':SimpleNamespace(ExactGuardBudgetExceeded=BudgetExceeded)}
    exec(compile(ast.Module(body=[reserve],type_ignores=[]),'literal_resource_reserve','exec'),namespace)
    def budget(points,quotients,pairs):
        return SimpleNamespace(points=points,quotients=quotients,examined_pairs=pairs)
    obj=budget(32,160,32*115129)
    namespace['reserve'](obj,5,115129)
    check('reserve accepts final33rd point/165th quotient/declaredpairupper',
          (obj.points,obj.quotients,obj.examined_pairs)==(33,165,3799257))
    for name,obj,roots,pairs in (
            ('pointcap',budget(33,165,3799257),5,115129),
            ('paircap',budget(1,5,3999999),5,2),
            ('negativework',budget(0,0,0),-1,0)):
        old=obj.__dict__.copy()
        try:
            namespace['reserve'](obj,roots,pairs)
        except (ValueError,BudgetExceeded):
            check('reserve rejects '+name+' before counter mutation',obj.__dict__==old)
        else:
            raise ValueError('resource reserve wrongly accepted '+name)
    check('unchanged exact guard and frozen consumer utilities',
          sha(REPO/contracts[0]['pending_prerequisites']['exact_guard_helper']['path'])==
          'fb7342c3091a5f12b3caa1e0cf6f7d8017f70528b74b795842bf50e32d87c4ed'
          and sha(baseline_path)=='0e0186662e2daafe39b4be2f486d44826a5019a05b7186e801d768da2087e02c')
    for contract in contracts:
        branch=HERE/contract['branch']
        result={'all_source_only_checks_passed':True,'checks':len(checks),'controls':checks,
                'runner_sha256':sha(HERE/'run.py'),'contract_sha256':sha(branch/'contract.json'),
                'preparation_driver_sha256':sha(__file__),'numerical_imports':0,'array_reads':0,
                'H_CI_gradient_optimizer_Dirac_calls':0,'resource_mock_scope':'literal reserve resource counters only; exact arithmetic not requalified',
                'Ruff':'passed','existing_optimizer_affected_test_source':'tests/test_orbital_optimizer_history.py read only; no tests rerun',
                'root_actual_execution_still_pending':True}
        (branch/'preparation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':len(checks),'numeric_imports':0,'array_reads':0,'science_calls':0}))


if __name__=='__main__':
    main()
