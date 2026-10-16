"""Inspect Gram loss at the explicit saved full-coordinate complement seam."""
import hashlib
import json
import os
from pathlib import Path

os.sched_setaffinity(0, {27})
import numpy as np
from scipy import linalg

from atomx.numerics.bspline.dirac_assembly import solve_dirac

HERE = Path(__file__).resolve().parent
old = HERE / 'actual001/blocks_ns180.npz'
out = HERE / 'diagnosis001'
out.mkdir(exist_ok=False)
ns = 180
with np.load(old, allow_pickle=False) as saved:
    h, s, c, keep = [saved[k] for k in
                    ('H_k-3', 'S_k-3', 'occupied_k-3', 'active_coordinates')]
H = np.zeros((2*ns, 2*ns))
S = np.zeros_like(H)
H[np.ix_(keep, keep)] = h
S[np.ix_(keep, keep)] = s
constraints = np.zeros((2*ns, c.shape[1]))
constraints[keep] = c
normals = s @ c
q, _ = linalg.qr(normals, mode='full')
b = q[:, c.shape[1]:]
hp, sp = b.T @ h @ b, b.T @ s @ b
sp = np.tril(sp) + np.tril(sp, -1).T
raw = solve_dirac(H, S, ns, -3, nsp=ns-1, nsq=ns,
                  orthogonal_to=constraints)
v = b.T @ raw.coefficients[keep]
gram = v.T @ sp @ v
lower = linalg.cholesky((gram+gram.T)/2, lower=True)
fixed = linalg.solve_triangular(lower, v.T, lower=True).T
correction = fixed-v
sv = linalg.cholesky(sp, lower=True)
raw_residual = hp @ v-(sp @ v)*raw.eigenvalues
fixed_residual = hp @ fixed-(sp @ fixed)*raw.eigenvalues
result = dict(
    hypothesis='Re-embedding all positive roots through a Euclidean complement loses metric Gram accuracy; one ordered Cholesky QR changes only representation in the same complete span.',
    fixed='Saved ns180 kappa-3 Fock/metric/core constraints only; no SCF, Fock build, roots/band/model/gate change.',
    input_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    columns=v.shape[1], coordinates=v.shape[0],
    raw_original_Gram=float(np.max(abs(raw.coefficients.T@S@raw.coefficients-np.eye(v.shape[1])))),
    complement_Gram=float(np.max(abs(gram-np.eye(v.shape[1])))),
    fixed_Gram=float(np.max(abs(fixed.T@sp@fixed-np.eye(v.shape[1])))),
    maximum_column_change_S_norm=float(np.linalg.norm(linalg.cholesky(sp,lower=True).T@correction,axis=0).max()),
    first_root_raw_dual_Ha=float(np.linalg.norm(linalg.solve_triangular(sv,raw_residual[:,0],lower=True))),
    first_root_fixed_dual_Ha=float(np.linalg.norm(linalg.solve_triangular(sv,fixed_residual[:,0],lower=True))),
    gates=dict(original_Gram=1e-10, fixed_Gram=1e-10, column_change_S_norm=1e-8),
    status='DIAGNOSIS_ONLY')
np.savez(out/'arrays.npz',h=hp,s=sp,complement=b,energies=raw.eigenvalues,
         raw=v,corrected=fixed,raw_gram=gram)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
