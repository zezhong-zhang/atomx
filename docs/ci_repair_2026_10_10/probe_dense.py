"""Compare LAPACK drivers on the unchanged native two-electron CI blocks."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import linalg

from atomx import Atom, Basis, CIStateAverage, Correlation
from atomx.scattering.models import ContinuumModel
from atomx.scattering.multielectron_ci import MultiElectronCIChannels

atom = Atom(10, config=["1s2 2s1", "1s1 2s1 2p1"],
            basis=Basis(ns=60, ks=9), corrections="none",
            correlation=Correlation(state_average=CIStateAverage()))
atom.solve(tol=1e-6, max_iter=100)
assert atom.converged
original = linalg.eigh
blocks = []


def capture(matrix, *args, **kwargs):
    blocks.append(np.array(matrix, copy=True))
    return original(matrix, *args, **kwargs)


with patch("atomx.methods.ci.linalg.eigh", capture):
    provider = MultiElectronCIChannels(atom, continuum_model=ContinuumModel.shooting(
        n_grid=16000, r_max=80, orbital_removal="none"))
np.savez(Path(__file__).with_name("native_blocks.npz"), **{
    f"block_{i}": block for i, block in enumerate(blocks)})
for i, block in enumerate(blocks):
    for driver in ("evr", "evd", "ev"):
        values, vectors = original(block, driver=driver)
        metrics = dict(block=i, size=len(block), driver=driver,
                       orthogonality=float(np.max(np.abs(
                           vectors.T @ vectors - np.eye(len(block))))),
                       residual=float(np.max(np.abs(
                           block @ vectors - vectors * values))))
        print(json.dumps(metrics), flush=True)
labels = list(provider.vectors[2])
masks = sorted({mask for vector in provider.vectors[2].values() for mask in vector})
matrix = np.array([[provider.vectors[2][label].get(mask, 0) for label in labels]
                   for mask in masks])
print(json.dumps(dict(magnetic_closure=float(np.max(np.abs(
    matrix @ matrix.T - np.eye(len(masks))))))), flush=True)
