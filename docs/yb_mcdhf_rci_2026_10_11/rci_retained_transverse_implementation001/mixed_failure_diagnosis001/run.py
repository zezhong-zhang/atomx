"""Consume all frozen mixed001 points; no Atomx import or physical producer."""
import argparse
import hashlib
import json
import resource
import signal
import time
from decimal import Decimal, localcontext
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[3]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    contract = json.loads((D / 'contract.json').read_text())
    for item in contract['inputs']:
        assert hashlib.sha256((R / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    if not args.execute:
        print('Input hashes checked; numeric execution requires --execute and root admission.')
        return
    out = D / 'actual'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('90s cap')))
        signal.alarm(90)
        import numpy as np

        A = D.parent / 'mixed001/numeric_assets'
        loads = 0

        def load(name):
            nonlocal loads
            loads += 1
            assert loads <= 16
            with np.load(A / name, allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            assert all(np.isfinite(v).all() for v in arrays.values())
            return arrays

        geometry, gradients, old = (load(n) for n in ('geometry_inputs.npz', 'candidate_gradients.npz', 'frozen_comparators.npz'))
        points = [load(f'CI_{i:02}.npz') for i in range(13)]
        keys = [tuple(k) for k in geometry['orbital_keys'].tolist()]
        sectors = json.loads((A / 'oracle_signed_tables.json').read_text())['blocks']
        targets = sectors[0]['columns']  # Frozen complete J0-even CSF row set, not a root index.
        records = json.loads((A / 'finite_diagnostics.json').read_text())
        assert len(records) == 13 and points[0]['H'].shape == (29, 29)

        def spectrum(H, p):
            h, c, e = (np.asarray(v, dtype=np.longdouble) for v in (H, p['mixing'], p['energies']))
            residual = np.sqrt(np.sum((h @ c - c * e)**2, axis=0))
            gram = float(np.max(np.abs(c.T @ c - np.eye(29))))
            return dict(symmetry=float(np.max(np.abs(h-h.T))), all_column_residuals=[float(v) for v in residual], max_residual=float(residual.max()), Gram=gram)

        def difference(a, b):
            assert a.dtype == b.dtype == np.dtype('float64') and a.shape == b.shape
            changes = np.argwhere(a != b)
            au, bu = a.view(np.uint64), b.view(np.uint64)
            sign = np.uint64(1 << 63)
            ao = np.where(au & sign, ~au, au | sign)
            bo = np.where(bu & sign, ~bu, bu | sign)
            ulp = np.maximum(ao, bo)-np.minimum(ao, bo)
            delta = a.astype(np.longdouble)-b.astype(np.longdouble)
            support = [dict(index=idx.tolist(), current=float(a[tuple(idx)]), frozen=float(b[tuple(idx)]), delta=float(delta[tuple(idx)]), ULP=int(ulp[tuple(idx)])) for idx in changes]
            return dict(array_equal=bool(np.array_equal(a, b)), bytes_equal=a.tobytes()==b.tobytes(), changed_count=len(support), max_absolute=float(np.max(np.abs(delta))), max_ULP=int(ulp.max()), changed_support=support)

        full = [spectrum(p['H'], p) for p in points]
        comparison = {str(i): dict(delta=difference(points[i]['H'], old[f'H_{i}']), frozen_H_current_complete_CI=spectrum(old[f'H_{i}'], points[i])) for i in (0, 1, 2, 7, 8)}
        signed_g = difference(gradients['baseline'], old['baseline_gradient'])
        assert gradients['baseline'].shape == (7, 70)
        directions, quotient = [], []
        for p in points:
            population = np.sum(p['mixing'][targets]**2, axis=0)
            columns = np.flatnonzero(population > 1-1e-9)
            columns = columns[np.argsort(p['energies'][columns], kind='stable')]
            assert len(columns) == 12
            q = []
            for col in columns[:2]:
                c = p['mixing'][:, col].astype(np.longdouble)
                q.append((c @ p['H'].astype(np.longdouble) @ c)/(c @ c))
            quotient.append((q[0]+3*q[1])/4)
        for direction in range(2):
            tangent = np.zeros_like(gradients['baseline'])
            analytic = frozen_analytic = 0.
            for k in sorted({k[1] for k in keys}):
                indices = [i for i, key in enumerate(keys) if key[1] == k]
                values = geometry[f'P_{k}'] @ geometry[f'eta{direction}_{k}']
                for col, i in enumerate(indices):
                    tangent[i] = values[:, col]
                    analytic += float(gradients['baseline'][i] @ values[:, col])
                    frozen_analytic += float(old['baseline_gradient'][i] @ values[:, col])
            tol = max(1e-8, abs(analytic)*1e-5)
            scales = []
            for j, h in enumerate((1e-4, 1e-6, 1e-8)):
                ip = 1+direction*6+2*j
                plus, minus = records[ip], records[ip+1]
                assert plus['label'] == f'd{direction}_{h}' and minus['label'] == f'd{direction}_{-h}'
                parts = {name: (plus['diagnostic'][name]-minus['diagnostic'][name])/(2*h) for name in ('rayleigh_change', 'reported_change', 'explicit_radial', 'reconstruction_radial', 'rayleigh_response', 'reported_response')}
                with localcontext() as context:
                    context.prec = 80
                    parts.update({name: float((Decimal(plus[name])-Decimal(minus[name]))/Decimal.from_float(2*h)) for name in ('oracle_explicit', 'oracle_reconstruction', 'oracle_complete_radial')})
                odd = (geometry[f'waves_{ip}']-geometry[f'waves_{ip+1}']).reshape(7, 70)/(2*h)
                scales.append(dict(h=h, central=parts, error=parts['rayleigh_change']-analytic, original_tolerance=tol, within_original_tolerance=abs(parts['rayleigh_change']-analytic)<=tol, even_rayleigh_change=(plus['diagnostic']['rayleigh_change']+minus['diagnostic']['rayleigh_change'])/2, physical_odd_g_dot=float(np.sum(gradients['baseline'].astype(np.longdouble)*odd)), stored_H_normalized_quotient_central=float((quotient[ip]-quotient[ip+1])/(2*h)), complete_finite_ledgers=[plus, minus]))
            directions.append(dict(direction=direction, analytic=analytic, frozen_gradient_analytic=frozen_analytic, ideal_g_dot_wide=float(np.sum(gradients['baseline'].astype(np.longdouble)*tangent)), scales=scales))
        assert loads == 16
        result = dict(status='saved_diagnostic_closed_original_two_failures_preserved', NPZ_loads=loads, new_scientific_producers=0, Atomx_imports=0, full_spectra=full, full_H_comparisons=comparison, full_signed_gradient=signed_g, directions=directions, original_suite_passed=False, optimizer_authorized=False, limitations=contract['limits'], elapsed_uncontrolled_s=time.monotonic()-started)
        write(out / 'result.json', result)
        print(json.dumps(dict(status=result['status'], NPZ_loads=loads, new_scientific_producers=0)))
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), elapsed_uncontrolled_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
