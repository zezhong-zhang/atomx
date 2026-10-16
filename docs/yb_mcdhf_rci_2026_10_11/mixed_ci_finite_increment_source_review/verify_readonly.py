"""Frozen-candidate source/AST review only; imports no Atomx or NumPy."""
from pathlib import Path
import ast
import hashlib
import json
import os
import resource
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
OLD = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/checkpoint_hamiltonian_reuse_implementation/source002/atomx'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    source = path.read_text()
    tree = ast.parse(source)
    result = {}
    def visit(node, prefix=''):
        for value in getattr(node, 'body', []):
            if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix+value.name] = value
            elif isinstance(value, ast.ClassDef):
                visit(value, prefix+value.name+'.')
    visit(tree)
    return source, tree, result


def dump(value):
    return ast.dump(value, include_attributes=False)


def main():
    start = time.monotonic()
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError('read-only audit requires one CPU')
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    binding = json.loads((BASE/'final_inputs.json').read_text())
    checks = []
    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))
    for name, expected in binding['inputs_sha256'].items():
        check('input:'+name, sha(ROOT/name) == expected)
    identity = json.loads((ROOT/binding['candidate_source_identity']).read_text())
    prefix = ROOT/binding['candidate_source_directory']
    raw_source_map = identity['all_atomx_python_sha256']
    source_map = {name.removeprefix('src/'): expected for name, expected in raw_source_map.items()}
    check('one_to_one_source_path_normalization', len(source_map)==len(raw_source_map)
          and all(name.startswith('atomx/') for name in source_map))
    for name, expected in source_map.items():
        check('archived_source:'+name, sha(prefix/name) == expected)
        check('current_source:'+name, sha(ROOT/'src'/name) == expected)
    old_names = {'atomx/'+str(p.relative_to(OLD)) for p in OLD.rglob('*.py')}
    check('only_two_declared_new_source_files', set(source_map)-old_names == {
        'atomx/numerics/ci_energy_increment.py', 'atomx/physics/csf/radial_increment.py'})
    check('no_removed_legacy_source_files', old_names.issubset(source_map))
    allowed = {'methods/csf/variational.py', 'physics/csf/matrix.py'}
    for path in sorted(OLD.rglob('*.py')):
        rel = str(path.relative_to(OLD))
        if rel not in allowed:
            check('legacy_bytes:'+rel, path.read_bytes() == (prefix/'atomx'/rel).read_bytes())
    for rel in ('methods/csf/variational.py', 'physics/csf/matrix.py'):
        old_source, old_tree, old_functions = functions(OLD/rel)
        new_source, new_tree, new_functions = functions(prefix/'atomx'/rel)
        for name, node in old_functions.items():
            if rel == 'physics/csf/matrix.py' and name == '_radial_evaluator':
                continue
            check('legacy_function_AST:'+rel+':'+name,
                  name in new_functions and dump(node) == dump(new_functions[name]))
            if name in new_functions:
                check('legacy_function_bytes:'+rel+':'+name,
                      ast.get_source_segment(old_source,node) == ast.get_source_segment(new_source,new_functions[name]))
        expected_new = ({'CSFOrbitalFunctional.mixed_ci_energy_difference'}
                        if rel.startswith('methods') else {'_bspline_pair_potential'})
        check('only_declared_new_functions:'+rel,
              set(new_functions)-set(old_functions) == expected_new)
        check('no_removed_functions:'+rel, set(old_functions).issubset(new_functions))
        if rel == 'physics/csf/matrix.py':
            old_evaluator, new_evaluator = old_functions['_radial_evaluator'], new_functions['_radial_evaluator']
            old_pair = next(n for n in ast.walk(old_evaluator)
                            if isinstance(n,ast.FunctionDef) and n.name=='pair_potential')
            new_pair = next(n for n in ast.walk(new_evaluator)
                            if isinstance(n,ast.FunctionDef) and n.name=='pair_potential')
            helper = new_functions['_bspline_pair_potential']
            check('extracted_pair_band_body_identical',
                  [dump(n) for n in old_pair.body[2:]] == [dump(n) for n in helper.body[1:]])
            check('old_grid_pair_path_identical',
                  [dump(n) for n in old_pair.body[:2]] == [dump(n) for n in new_pair.body[:2]])
            check('only_pair_function_extracted',
                  [dump(n) for n in old_evaluator.body if n is not old_pair]
                  == [dump(n) for n in new_evaluator.body if n is not new_pair])
    source, _, nodes = functions(prefix/'atomx/methods/csf/variational.py')
    consumer = nodes['CSFOrbitalFunctional.mixed_ci_energy_difference']
    text = ast.get_source_segment(source,consumer)
    prohibited = {'ci_solve','build_ci_matrix','optimize_orbitals','solve_dirac','evaluate','gradient_from_weights'}
    called = {n.func.id if isinstance(n.func,ast.Name) else n.func.attr
              for n in ast.walk(consumer) if isinstance(n,ast.Call)
              and isinstance(n.func,(ast.Name,ast.Attribute))}
    check('no_new_H_CI_gradient_Dirac_optimizer', not prohibited.intersection(called),sorted(called))
    controls = {
      'exact_functional_type':'type(self) is not CSFOrbitalFunctional',
      'frozen_target_equality':'self.level_tracking.state_average != self.state_average',
      'isolated_rank_one':'len(group) != 1',
      'captured_signed_CSF_table':'self.radial_density.signatures',
      'same_keys_cap':'self.max_determinants != self.radial_density.max_determinants',
      'same_single_charge':'len({c.n_electrons for c in self.csfs}) != 1',
      'strict_binary64':'dtype != np.dtype(np.float64)',
      'owned_complete_DC':'dense-real-full-CSF',
      'complete_spectrum':'require_full=True',
      'original_residual_norm':'np.linalg.norm(h @ mixing - mixing * energies, axis=0)',
      'original_residual_gate':'eigen_residual > 1e-8',
      'original_CI_Gram':'ci_gram > 1e-10',
      'original_radial_Gram':'radial_gram > 1e-10',
      'frozen_physical_matching':'self.level_tracking.resolve',
      'unfiltered_signed_density':'self.radial_density.contract_bilinear(c1, c1)',
      'finite_CI_norm':'value / response.norm1',
      'actual_trial_radial':'cache=self.cache, trial=trial',
      'diagnostic_only':'diagnostic_only=True',
      'no_certified_bound':'certified_error_bound=False',
      'no_optimizer_adapter':'optimizer_adapter=False'}
    for name, needle in controls.items(): check(name,needle in text)
    radial_boundary=text.index('radial = bspline_dc_radial_increment(')
    check('all_admission_precedes_radial_work',
          all(text.index(needle)<radial_boundary for name,needle in controls.items()
              if name not in {'actual_trial_radial','diagnostic_only','no_certified_bound','no_optimizer_adapter'}))
    oracle_path=ROOT/binding['oracle']
    oracle, _, oracle_functions=functions(oracle_path)
    for name in ('add_point','density','potential','radial'):
        method=oracle_functions['DecimalDCRadialOperator.'+name]
        check('Decimal80_public:'+name, any(isinstance(d,ast.Name) and d.id=='arithmetic80'
                                           for d in method.decorator_list))
    check('Decimal80_context', 'ctx.prec = 80' in oracle)
    check('Decimal_exact_binary_conversion','Decimal.from_float(float(value))' in oracle)
    oracle_tree=ast.parse(oracle)
    check('independent_oracle_no_Atomx_imports', all(
        not (isinstance(n,ast.ImportFrom) and (n.module or '').startswith('atomx'))
        and not (isinstance(n,ast.Import) and any(a.name.startswith('atomx') for a in n.names))
        for n in ast.walk(oracle_tree)))
    report=dict(status='read_only_frozen_candidate_source_audit', all_passed=all(c['passed'] for c in checks),
                check_count=len(checks), inputs_sha256=binding['inputs_sha256'],
                final_binding_sha256=sha(BASE/'final_inputs.json'),
                candidate_source_map_sha256=identity['source_map_sha256'],checks=checks,
                actual_physical_calls=dict(H=0,CI=0,gradient=0,Dirac=0,optimizer=0),
                test_execution=0, numerical_qualification_claim=False,
                elapsed_seconds=time.monotonic()-start, affinity=affinity,
                max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_after_unchanged=all(sha(prefix/n)==e and sha(ROOT/'src'/n)==e
                                           for n,e in source_map.items()))
    (BASE/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('all_passed','check_count','elapsed_seconds','max_RSS_KiB')}))
    if not report['all_passed'] or not report['source_after_unchanged']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
