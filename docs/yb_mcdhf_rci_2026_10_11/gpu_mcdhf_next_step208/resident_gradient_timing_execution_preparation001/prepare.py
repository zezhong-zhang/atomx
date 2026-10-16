"""Stdlib-only SOURCE/JSON checks; never imports numerical code or hashes NPZ."""

import ast
import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
QUALIFIED = HERE.parent / 'qualification_representation_repair001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def parsed(path):
    text = path.read_text()
    compile(text, str(path), 'exec')
    return text, ast.parse(text)


def dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def find(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def prepare():
    checks = []
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise AssertionError(name)
    c, q = load(HERE / 'contract.json'), load(QUALIFIED / 'contract.json')
    for field in ('fixed', 'gates', 'runtime', 'actual_native31_directional_prerequisite', 'source_archive'):
        check('qualified_identity_' + field, c[field] == q[field])
    proposal = ROOT / c['proposal']['path']
    check('frozen_proposal', sha(proposal) == c['proposal']['sha256'])
    p = load(proposal)
    check('same_5triplet_schedule', c['schedule'] == p['schedule'])
    check('no_extra_event_scope', c['mechanical_caps']['optional_warmup_event_pair_upper'] == 0)
    check('same_primary_scope', c['timing']['primary'] == p['timing']['primary'])
    deferred = []
    for name, row in c['input_roles'].items():
        path = ROOT / row['path']
        if path.suffix == '.npz':
            deferred.append(name)
            check('deferred_array_metadata_' + name, path.is_file() and len(row['sha256']) == 64)
        else:
            check('source_JSON_input_' + name, sha(path) == row['sha256'])
    for name, row in c['literal_helper_source_roles'].items():
        check('own_source_' + name, sha(ROOT / row['path']) == row['sha256'])
    identity = load(ROOT / c['input_roles']['f043_identity']['path'])
    check('f043356_map', len(identity['all_atomx_python_sha256']) == 356 and identity['source_map_sha256'] == c['fixed']['source_map_sha256'])
    for name, digest in identity['all_atomx_python_sha256'].items():
        check('frozen_source_' + name, sha(ROOT / c['source_archive'] / name.removeprefix('src/')) == digest)
    qualified, qt = parsed(QUALIFIED / 'run.py')
    helper, ht = parsed(HERE / 'native31_helpers.py')
    driver, dt = parsed(HERE / 'run.py')
    launcher, lt = parsed(HERE / 'launch.py')
    check('no_runtime_AST_exec', all(not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('exec', 'eval', 'compile')) for tree in (ht, dt, lt) for n in ast.walk(tree)))
    # Exactly the original point preparation statements; only old phase/progress
    # monitoring lives in the new driver instead of inside the literal helper.
    original_setup = textwrap.dedent(qualified[qualified.index('        def archive(role):'):qualified.index('        phase = "fresh_full1595_H_CI"')])
    original_setup = original_setup.replace('phase = "complete_public_setup"\nprogress()\n', '')
    nodes = ast.parse(original_setup).body
    hsetup = find(ht, 'prepare_native31_point').body
    pos = next(i for i, n in enumerate(hsetup) if isinstance(n, ast.FunctionDef) and n.name == 'archive')
    check('literal_setup_all_statements_order', dump(hsetup[pos:pos + len(nodes)]) == dump(nodes))
    for name in ('coordinate', 'tangent', 'metric_action', 'pack', 'signature', 'represent'):
        check('qualified_formula_AST_' + name, ast.dump(find(qt, name), include_attributes=False) == ast.dump(find(ht, name), include_attributes=False))
    # Preserve every qualified HCI/UID/norm/RDM gate, including complete sector
    # cluster loops; new timing assignments do not replace numerical statements.
    original_reference = ast.parse(textwrap.dedent(qualified[qualified.index('        ci = ci_module.ci_solve'):qualified.index('        def coordinate(radial_gradient):')])).body
    actual_reference = find(ht, 'fresh_native31_reference').body
    expected_dumps = [ast.dump(n, include_attributes=False) for n in original_reference]
    actual_dumps = [ast.dump(n, include_attributes=False) for n in actual_reference]
    cursor = 0
    for index, expected in enumerate(expected_dumps):
        while cursor < len(actual_dumps) and actual_dumps[cursor] != expected:
            cursor += 1
        check('literal_reference_statement_' + str(index), cursor < len(actual_dumps))
        cursor += 1
    qcounters = [ast.dump(n, include_attributes=False) for n in ast.walk(qt) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'counted']
    dcounters = [ast.dump(n, include_attributes=False) for n in ast.walk(dt) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'counted']
    check('same_public_alias_instrumentation', qcounters == dcounters)
    for name in ('blocks', 'factors', 'banks', 'moments'):
        check('same_forward_cache_observer_' + name, ast.dump(find(qt, name), include_attributes=False) == ast.dump(find(dt, name), include_attributes=False))
    check('matrix_hS_producer_alias', 'counted(matrix, "build_dirac_matrices", "CI_hS_producer")' in driver)
    check('private_density_captured_alias', 'counted(convol, "_density_sym", "private_density")' in driver)
    check('no_getter_equals_producer', c['resources']['calls']['hS_getter'] == 500 and c['resources']['calls']['CI_hS_producer'] == 7)
    check('22fullg_scope', c['resources']['calls']['public_full_gradient'] == 1 and c['resources']['calls']['private_full_gradient'] == 21)
    check('29NPZ_count_includes_symmetric_readback', c['resources']['calls']['NPZ_load'] == 6 + 1 + 1 + 21)
    check('physical_counts', all(c['resources']['calls'][k] == v for k, v in dict(H=1, CI=1, validation_Dirac=7, scipy_eigh=12, optimizer=0).items()))
    check('proper_root_nonoverlap_required', 'admit_window(binding, contract)' in driver and len(c['resources']['window_receipts_deferred']) == 2)
    check('CPU0_actual_permitted_check', '{caps["controller_CPU"], *caps["CPU4"]} <= original_affinity' in launcher)
    check('child_before_guardian_pin', launcher.index('child = subprocess.Popen') < launcher.index('os.sched_setaffinity(0, {caps["controller_CPU"]})'))
    check('original_masks_finally', 'nb.set_num_threads(previous_NumBa_mask)' in driver and 'os.sched_setaffinity(0, original_affinity)' in launcher)
    # The thin collector owns references only, never copies/computes/compresses
    # a band while the underlying complete raw-gradient call is timed.
    collector = find(dt, 'collect')
    forbidden = {'copy', 'array_equal', 'save', 'savez_compressed', 'fsync', 'write', 'check'}
    names = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(collector)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    check('thin_packet_reference_observer', not names & forbidden)
    check('profile_only_explicit_warmup', "one_call(name, 'explicit_warmup', 0, profile=True)" in driver and "one_call(name, 'warm', index)" in driver)
    check('resident_no_output_reuse', 'owner.gradient_from_weights(waves, full_weights, ci,' in driver and 'expected_weights_sha256=signed_layout.weights_sha256' in driver)
    check('strict_complete_bytes_before_after', 'helper.validate_reference_bytes' in driver and 'actual.tobytes() != expected.tobytes()' in helper)
    check('all_existing_g_gates', c['gates']['g'] == 1e-9 and c['gates']['pg'] == 1e-10 and c['gates']['CI_residual_Ha'] == 1e-8 and c['gates']['CI_Gram'] == 1e-10)
    # Import only this owned stdlib module; confirm tuple/list canonicalization
    # and the non-numerical wrapper protocol without importing Atomx/NumPy.
    spec = importlib.util.spec_from_file_location('native31_resident_source_only_mock', HERE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check('canonical_JSON_tuple_identity', module.plain({'a': ((1, 2), 3)}) == {'a': [[1, 2], 3]})
    check('no_numerical_imports_during_preparation', not any(n == 'numpy' or n == 'numba' or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    # Standard-library generator mock preserves already-owned packet identity.
    packets, forwarded = [object() for _ in range(6)], []
    def original():
        yield from enumerate(packets)
    def forwarding():
        for item in original():
            forwarded.append(item)
            yield item
    consumed = list(forwarding())
    check('thin_mock_same_identity_order', all(a[1] is b for a, b in zip(consumed, packets, strict=True)) and consumed == forwarded)
    result = dict(status='SOURCE_JSON_ONLY_PREPARED_PENDING_EXACT_ROOT_SEAL', all_passed=True,
        checks_count=len(checks), checks=checks, driver_sha256=sha(HERE / 'run.py'),
        helper_sha256=sha(HERE / 'native31_helpers.py'), launcher_sha256=sha(HERE / 'launch.py'),
        contract_sha256=sha(HERE / 'contract.json'), preparation_driver_sha256=sha(__file__),
        qualified_source_sha256=sha(QUALIFIED / 'run.py'), deferred_array_roles=deferred,
        numerical_imports=0, NPZ_loads=0, NPZ_bytes_hashed=0, H=0, CI=0, gradient=0, GPU=0,
        timed_execution=0, scientific_qualification_claimed=False)
    (HERE / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status','all_passed','checks_count','driver_sha256','helper_sha256','launcher_sha256','contract_sha256')}, indent=2))


if __name__ == '__main__':
    prepare()
