"""Bound offline diagnosis; no Atomx imports or new physical evaluations."""
import hashlib
import json
import os
import resource
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    contract = json.loads((HERE / 'contract.json').read_text())
    binding = json.loads((HERE / 'execution_binding.json').read_text())
    if sha(HERE / 'contract.json') != binding['contract_sha256']:
        raise ValueError('Contract changed')
    if sha(Path(__file__).resolve()) != binding['driver_sha256']:
        raise ValueError('Driver changed')
    for role in contract['inputs'].values():
        if sha(ROOT / role['path']) != role['sha256']:
            raise ValueError('Bound input changed: ' + role['path'])
    os.sched_setaffinity(0, {31})
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    import numpy as np

    roles = contract['inputs']
    closed = json.loads((ROOT / roles['failed_root_receipt']['path']).read_text())
    failed = json.loads((ROOT / roles['failed_result']['path']).read_text())
    if closed['status'] != 'foreground_closed' or closed['exit_code'] != 1:
        raise ValueError('Not the original closed failure')
    if failed['message'] != 'Original gate failed: fresh_DC_exact_whole_saved_H':
        raise ValueError('Different failure')
    old_owner = json.loads((ROOT / roles['old_owner']['path']).read_text())
    new_owner = json.loads((ROOT / roles['new_owner']['path']).read_text())
    checks = []

    def check(name, passed, observed=None):
        checks.append(dict(name=name, passed=bool(passed), observed=observed))
        if not passed:
            raise ValueError('Offline diagnosis prerequisite failed: ' + name)

    check('complete_owner_payload_equal', old_owner == new_owner)
    decoded = 0
    archives = []
    for role in ('old_CI', 'new_CI'):
        with np.load(ROOT / roles[role]['path'], allow_pickle=False) as archive:
            check(role + '/exact_fields', set(archive.files) == {
                'H', 'energies', 'mixing', 'keys', 'P', 'Q'})
            arrays = {key: archive[key] for key in archive.files}
            decoded += sum(array.nbytes for array in arrays.values())
            archives.append(arrays)
    old, new = archives
    check('decoded_bytes_cap', decoded <= 100_000_000, decoded)
    for label, arrays in (('old', old), ('new', new)):
        for key, array in arrays.items():
            check(label + '/' + key + '_finite', np.isfinite(array).all())
        for key in ('H', 'mixing', 'energies', 'P', 'Q'):
            check(label + '/' + key + '_binary64', arrays[key].dtype == np.float64)
        check(label + '/H_shape', arrays['H'].shape == (1595, 1595))
        check(label + '/C_shape', arrays['mixing'].shape == (1595, 1595))
        check(label + '/E_shape', arrays['energies'].shape == (1595,))
        check(label + '/H_symmetry', np.array_equal(arrays['H'], arrays['H'].T))
    for key in ('keys', 'P', 'Q'):
        check('literal_' + key + '_bytes_equal',
              old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
              and old[key].tobytes() == new[key].tobytes())

    difference = new['H'] - old['H']
    changed = old['H'].view(np.uint64) != new['H'].view(np.uint64)
    diagonal = np.eye(1595, dtype=bool)
    numerical_changed = difference != 0.0
    same_sign = np.signbit(old['H']) == np.signbit(new['H'])
    uint_old, uint_new = old['H'].view(np.uint64), new['H'].view(np.uint64)
    # Finite same-sign binary64 values have monotone magnitude bit encodings.
    ulp = np.maximum(uint_old[same_sign], uint_new[same_sign]) - np.minimum(
        uint_old[same_sign], uint_new[same_sign])
    at = tuple(int(i) for i in np.unravel_index(np.argmax(abs(difference)), difference.shape))
    measures = dict(
        bytes_equal=bool(not changed.any()),
        changed_entries=int(changed.sum()),
        numerically_changed_entries=int(numerical_changed.sum()),
        diagonal_changed_entries=int(changed[diagonal].sum()),
        off_diagonal_changed_entries=int(changed[~diagonal].sum()),
        positive_difference_entries=int((difference > 0).sum()),
        negative_difference_entries=int((difference < 0).sum()),
        max_abs_difference_Ha=float(np.max(abs(difference))),
        max_location=list(at), old_at_max_Ha=float(old['H'][at]),
        new_at_max_Ha=float(new['H'][at]), signed_at_max_Ha=float(difference[at]),
        max_same_sign_ULP=int(ulp.max()),
        changed_opposite_sign_entries=int((changed & ~same_sign).sum()),
        E_bytes_equal=bool(old['energies'].tobytes() == new['energies'].tobytes()),
        C_bytes_equal=bool(old['mixing'].tobytes() == new['mixing'].tobytes()),
    )
    controls = {}
    residuals = {}
    for name, H, C, E in (
        ('old_on_old', old['H'], old['mixing'], old['energies']),
        ('new_on_new', new['H'], new['mixing'], new['energies']),
        ('old_on_new', new['H'], old['mixing'], old['energies']),
    ):
        residual = np.linalg.norm(H @ C - C * E, axis=0)
        residuals[name] = residual
        controls[name] = dict(max_residual_Ha=float(residual.max()),
                              gate_Ha=1e-8, passed=bool(residual.max() <= 1e-8))
    for name, arrays in (('old', old), ('new', new)):
        gram = float(np.max(abs(arrays['mixing'].T @ arrays['mixing'] - np.eye(1595))))
        controls[name + '_Gram'] = dict(max_error=gram, gate=1e-10, passed=gram <= 1e-10)
    # No acceptance of the historical byte gate is inferred from residual gates.
    output = HERE / 'actual'
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / 'full_signed_difference.npz', delta_H=difference,
                        **{name + '_residual_Ha': value for name, value in residuals.items()})
    for role in roles.values():
        check('post_input_hash/' + role['path'], sha(ROOT / role['path']) == role['sha256'])
    check('post_driver_hash', sha(Path(__file__).resolve()) == binding['driver_sha256'])
    check('post_contract_hash', sha(HERE / 'contract.json') == binding['contract_sha256'])
    result = dict(status='offline_diagnosis_complete', checks=checks, matrix_difference=measures,
                  complete_eigensystem_controls=controls, decoded_array_bytes=decoded,
                  counts=dict(NPZ_load=2, full_matrix_products=5, H=0, CI=0, gradient=0,
                              Dirac=0, optimizer=0, eigensolve=0, SVD=0),
                  original_byte_gate_passed=bool(not changed.any()),
                  original_failed_run_remains_failed=True, Breit_contribution_measured=False,
                  controlled_timing=False, elapsed_uncontrolled_s=time.monotonic() - started,
                  contract_sha256=binding['contract_sha256'], driver_sha256=binding['driver_sha256'],
                  archive_sha256=sha(output / 'full_signed_difference.npz'))
    write(output / 'result.json', result)
    print(json.dumps(dict(status=result['status'], difference=measures, controls=controls)))


if __name__ == '__main__':
    main()
