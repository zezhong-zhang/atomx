"""Controlled cache benchmark: identical model, grid and energy samples."""
import json
import time

import numpy as np

from atomx import Atom
from atomx.numerics.bspline import continuum
from atomx.scattering.radiative_recombination import rr_cross_section

atom = Atom(2, config="1s1", engine="bspline", corrections="none").solve()
data = atom.result._bsp_data
energies = np.geomspace(1., 500., 80)
original_spectrum = continuum._continuum_spectrum
original_eigh = continuum.linalg.eigh
counts = [0]
enabled = [True]


def spectrum(kappa, Z, workspace, *args):
    if not enabled[0]:
        workspace.pop("_continuum_spectra", None)
    return original_spectrum(kappa, Z, workspace, *args)


def eigh(*args, **kwargs):
    counts[0] += 1
    return original_eigh(*args, **kwargs)


continuum._continuum_spectrum = spectrum
continuum.linalg.eigh = eigh
records, outputs = [], []
try:
    rr_cross_section(atom.result, energies[:1], multipoles=["E1"])
    for cached in (False, True, False, True, False, True):
        enabled[0] = cached
        counts[0] = 0
        data.pop("_continuum_spectra", None)
        start = time.perf_counter()
        result = rr_cross_section(atom.result, energies, multipoles=["E1"])
        records.append(dict(cached=cached, seconds=time.perf_counter()-start,
                            diagonalizations=counts[0]))
        outputs.append(result.sigma_total_a0_sq.copy())
finally:
    continuum._continuum_spectrum = original_spectrum
    continuum.linalg.eigh = original_eigh
print(json.dumps(dict(energy_samples=len(energies), spline_count=data["ns"], runs=records,
                     bit_identical=all(np.array_equal(outputs[0], r) for r in outputs[1:])), indent=2))
