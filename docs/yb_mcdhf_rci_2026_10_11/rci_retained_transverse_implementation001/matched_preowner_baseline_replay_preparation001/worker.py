"""One native baseline owner in a fresh, exactly bound source process."""
import argparse
import hashlib
import inspect
import json
import resource
import signal
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', choices=('current', 'preowner'), required=True)
    args = parser.parse_args()
    out = D / 'actual' / args.epoch
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    counters = None
    try:
        contract = json.loads((D / 'executor_contract.json').read_text())
        resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('worker90s cap')))
        signal.alarm(90)
        recipe = json.loads((D / 'source_recipe.json').read_text())
        source = D / args.epoch / 'source'
        map_key = 'current365' if args.epoch == 'current' else 'preowner_matched365'
        for name, expected in recipe[map_key].items():
            assert sha(source / name.removeprefix('src/')) == expected, name
        for item in contract['literal_inputs']:
            assert sha(R / item['path']) == item['sha256'], item['path']
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy import linalg

        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods.ci import CIResult, ci_solve
        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.numerics.bspline.basis import prepare_bspline_data
        from atomx.numerics.bspline.dirac_assembly import (
            build_dirac_matrices,
            solve_dirac,
        )
        from atomx.numerics.bspline.knots import make_knots
        from atomx.numerics.bspline.orbital_optimization import _polar_retraction
        from atomx.physics.csf.hamiltonian_snapshot import (
            CIHamiltonianSnapshot,
            build_ci_hamiltonian_snapshot,
        )
        from atomx.physics.csf.matrix import build_ci_matrix
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.physics.csf.tracking import CIBasisTracking, ci_subspace_overlaps

        def source_guard():
            for name, module in tuple(sys.modules.items()):
                if name == 'atomx' or name.startswith('atomx.'):
                    filename = getattr(module, '__file__', None)
                    if filename:
                        assert source in Path(filename).resolve().parents, name

        source_guard()

        class Calls:
            """Python call-profile observation, without function replacement."""
            def __init__(self):
                self.limits = contract['per_worker_primitive_limits']
                self.entered = dict.fromkeys(self.limits, 0)
                self.completed = dict(self.entered)
                functions = dict(H_build=build_ci_matrix, owner_build=build_ci_hamiltonian_snapshot,
                                 owner_matrix_read=CIHamiltonianSnapshot.matrix_for,
                                 evaluate=CSFOrbitalFunctional.evaluate,
                                 gradient=CSFOrbitalFunctional.gradient_from_weights,
                                 public_hS=build_dirac_matrices, physical_overlap=ci_subspace_overlaps,
                                 SVD=np.linalg.svd._implementation, CI_solve=ci_solve,
                                 Dirac_solve=solve_dirac, polar=_polar_retraction)
                self.codes = {inspect.unwrap(f).__code__: key for key, f in functions.items()}
                for f in (linalg.eigh, np.linalg.eigh._implementation):
                    self.codes[inspect.unwrap(f).__code__] = 'eigh'
                self.active = {}
                self.flush()

            def flush(self):
                write(out / 'calls.json', dict(entered=self.entered, completed=self.completed, limits=self.limits,
                                             observation='sys.setprofile; no physical function monkeypatch'))

            def profile(self, frame, event, value):
                key = self.codes.get(frame.f_code)
                if key is None:
                    return
                token = id(frame)
                if event == 'call':
                    if self.entered[key] >= self.limits[key]:
                        raise RuntimeError(f'{key} prospective cap exceeded')
                    self.entered[key] += 1
                    self.active[token] = key
                    self.flush()
                elif event == 'return' and token in self.active:
                    self.active.pop(token)
                    if value is not None:
                        self.completed[key] += 1
                    self.flush()

        counters = Calls()
        sys.setprofile(counters.profile)
        loads = 0

        def load(item):
            nonlocal loads
            loads += 1
            assert loads <= 3
            with np.load(R / item['path'], allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            assert all(np.isfinite(v).all() for v in arrays.values())
            return arrays

        geometry, saved, saved_gradient = [load(item) for item in contract['literal_inputs']]
        keys = tuple(map(tuple, geometry['orbital_keys'].tolist()))
        waves = {key: tuple(part.copy() for part in geometry['waves_0'][i]) for i, key in enumerate(keys)}
        assert saved['H'].shape == saved['mixing'].shape == (29, 29)
        assert saved['energies'].shape == (29,) and saved_gradient['baseline'].shape == (7, 70)

        def nuclear(radius):
            return -2 / radius

        data = prepare_bspline_data(make_knots(35, 9, .005, 30), 35, 9)
        csfs = [c for cfg in contract['configs'] for c in enumerate_jj_csfs(cfg)]
        target = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3)))
        f = CSFOrbitalFunctional(csfs, data, 2, nuclear, state_average=target, retain_hamiltonian=True)
        assert len(csfs) == 29 and f.keys == keys
        for k, h in f.one_body.items():
            raw = geometry[f'h_{k}']
            assert np.array_equal(h, np.tril(raw)+np.tril(raw, -1).T)
            assert np.array_equal(f.metric[k], geometry[f'S_{k}'])
        owner = build_ci_hamiltonian_snapshot(csfs, waves, keys, data, 2, 35,
                    nuclear_potential=nuclear, convol_cache=f.cache, max_determinants=f.max_determinants)
        H = owner.matrix_for(csfs, waves, keys, data, 2, 35,
                            nuclear_potential=nuclear, max_determinants=f.max_determinants)
        np.savez_compressed(out / 'whole_H.npz', H=H)
        assert np.array_equal(H, saved['H']), 'current numerical kernel/owner baseline H mismatch'
        C, E = saved['mixing'].copy(), saved['energies'].copy()
        residual = np.linalg.norm(H @ C-C*E, axis=0)
        gram = float(np.max(np.abs(C.T @ C-np.eye(29))))
        assert residual.max() <= 1e-8 and gram <= 1e-10
        ci = CIResult(E, C, hamiltonian_snapshot=owner)
        f.level_tracking = CIBasisTracking(target, csfs, ci, waves, f.metric)
        _, gradient, _ = f.evaluate(waves, ci_result=ci)
        g = np.array([gradient[key] for key in keys])
        np.savez_compressed(out / 'whole_gradient.npz', gradient=g)
        assert np.array_equal(g, saved_gradient['baseline']), 'same-epoch baseline signed gradient mismatch'
        sys.setprofile(None)
        source_guard()
        assert loads == 3 and counters.entered == counters.completed == counters.limits
        assert owner.schema_version == 1 and owner.hamiltonian_terms == ('dirac-coulomb',)
        header = owner.identity_payload()
        write(out / 'owner.json', header)
        for name, expected in recipe[map_key].items():
            assert sha(source / name.removeprefix('src/')) == expected
        write(out / 'result.json', dict(status='native_baseline_worker_closed', epoch=args.epoch,
              complete_H_array_equal_saved=True, complete_signed_gradient_array_equal_saved=True,
              H_array_bytes_sha256=hashlib.sha256(H.tobytes()).hexdigest(),
              gradient_array_bytes_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
              H_shape=list(H.shape), gradient_shape=list(g.shape), NPZ_loads=loads,
              complete_CI_residuals=[float(v) for v in residual], complete_CI_Gram=gram,
              owner_header_sha256=hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              H_symmetry_max=float(np.max(np.abs(H-H.T))), primitive_completed=counters.completed,
              CI_solve=0, optimizer=0, elapsed_uncontrolled_s=time.monotonic()-started))
    except Exception as error:
        sys.setprofile(None)
        if counters is not None:
            counters.flush()
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), elapsed_uncontrolled_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
