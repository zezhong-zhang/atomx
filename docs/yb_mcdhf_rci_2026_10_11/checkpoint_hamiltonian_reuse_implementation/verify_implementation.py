"""Read-only source/archive/documentation proof; no Atomx imports or solves."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))

    source = json.loads((HERE / 'source_identity002.json').read_text())
    current = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/atomx').rglob('*.py'))}
    check('complete source map', current == source['all_atomx_python_sha256'])
    check('exact source map hash', digest(current) == source['source_map_sha256'])
    check('source count', len(current) == source['complete_source_files'] == 354)
    check('only authorized changes', set(source['authorized_changes']) == {
        'src/atomx/methods/ci.py', 'src/atomx/methods/csf/variational.py',
        'src/atomx/physics/csf/hamiltonian_snapshot.py'} and not source['removed'])
    for identity_name, archive in (('source_identity.json', 'source'),
                                   ('source_identity002.json', 'source002')):
        identity = json.loads((HERE / identity_name).read_text())
        for name, expected in identity['all_atomx_python_sha256'].items():
            check(f'{archive}:{name}', sha(HERE / archive / Path(name).relative_to('src')) == expected)
    old = json.loads((HERE.parent / 'checkpoint_hamiltonian_reuse_design/source_identity.json').read_text())
    check('predecessor353 reference', source['previous_source_map_sha256'] == old['reference_source_map_sha256'])
    check('predecessor353 actual map', digest(old['all_atomx_python_sha256']) == source['previous_source_map_sha256'])
    for suffix, count in (('', 78), ('002', 85), ('003', 85)):
        contract_path = HERE / f'unit_contract{suffix}.json'
        execution = json.loads((HERE / f'unit_execution{suffix}.json').read_text())
        contract = json.loads(contract_path.read_text())
        check(f'unit{suffix} contract', execution['contract_sha256'] == sha(contract_path))
        check(f'unit{suffix} actual pass', execution['all_passed'] and execution['exit_code'] == 0 and
              execution['source_unchanged'] and execution['tests_unchanged'] and execution['stopping_condition'] is None)
        log = HERE / f'unit_controls{suffix}.log'
        check(f'unit{suffix} log', execution['log_sha256'] == sha(log) and f'{count} passed in ' in log.read_text())
        check(f'unit{suffix} resource scope', contract['resources']['cpu_threads'] == 1 and
              contract['resources']['max_address_space_bytes'] == int(3.95 * 1024**3) and
              contract['resources']['max_wall_seconds'] == 300 and contract['no_performance_claim'])
        for name, expected in contract['tests_sha256'].items():
            path = (HERE / 'unit_revision001' / Path(name).name if
                    suffix == '' and Path(name).name == 'test_ci_hamiltonian_snapshot.py' else
                    HERE / 'tests' / Path(name).name)
            check(f'unit{suffix} test:{name}', sha(path) == expected)
    for name, expected in json.loads((HERE / 'unit_contract003.json').read_text())['tests_sha256'].items():
        check(f'final live test:{name}', sha(ROOT / name) == expected)
    check('final runner frozen', sha(HERE / 'run_unit_controls.py') ==
          json.loads((HERE / 'unit_contract003.json').read_text())['runner_sha256'])
    doc_identity = json.loads((HERE / 'documentation_source_identity.json').read_text())
    check('docstring-only source revision', all(doc_identity['all_nondocstring_AST_identical'].values()) and
          doc_identity['new_source_map_sha256'] == source['source_map_sha256'])
    for name in ('src/atomx/physics/csf/hamiltonian_snapshot.py', 'src/atomx/methods/ci.py',
                 'src/atomx/methods/csf/variational.py'):
        ast.parse((ROOT / name).read_text())
        check(f'parse final:{name}', True)
    readme = (HERE / 'README.md').read_text()
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        check(f'document link:{target}', (HERE / target).exists())
    check('explicit qualification boundary', 'not a controlled speedup measurement' in readme and
          '531-CSF qualification' in readme and 'remains above 1e-7' in readme)
    result = dict(scope='read-only source/archive/unit-log/link proof; zero H/CI/gradient/optimizer calls',
                  source_map_sha256=source['source_map_sha256'], checks=checks,
                  check_count=len(checks), all_passed=all(c['passed'] for c in checks),
                  document_sha256=sha(HERE / 'README.md'), verifier_sha256=sha(__file__))
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, indent=2))
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
