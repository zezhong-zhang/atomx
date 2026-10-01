"""Independent FAC photon operators on identical, support-matched spinors.

These tests isolate the radiation operator from orbital-solver errors. The
reference uses an exact photon frequency per line, not FAC's grouped table
interpolation. The raw exported spinors are retained through each ilast.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

from atomx.numerics.radial_workspace import make_grid_workspace
from atomx.properties.transitions.angular import angular_eL, angular_mL
from atomx.properties.transitions.multipole import (
    radial_eL,
    radial_eL_velocity,
    radial_mL,
)
from atomx.properties.transitions.rates import einstein_A_eL


@pytest.fixture(scope='module')
def reference():
    folder = Path(__file__).parent/'data'
    metadata = json.loads((folder/'fac_photon_reference.json').read_text())
    path = folder/'fac_photon_spinors.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['spinors_sha256']
    with np.load(path, allow_pickle=False) as arrays:
        yield metadata, arrays


@pytest.fixture(scope='module', params=[(1, 5), (10, 5), (82, 5), (1, 7)])
def common_spinors(request, reference):
    z, order = request.param
    metadata, arrays = reference
    case = next(c for c in metadata['cases'] if c['z'] == z)
    tables = {(o['n'], o['kappa']): arrays[o['array']] for o in case['orbitals']}
    longest = max(tables.values(), key=len)
    r = np.r_[0., longest[:, 0]]
    x, w = np.polynomial.legendre.leggauss(8)
    half = np.diff(r)[:, None]/2
    nodes = (r[:-1, None]+half*(1+x)).ravel()
    weights = (half*w).ravel()
    waves = {}
    for key, table in tables.items():
        np.testing.assert_array_equal(table[:, 0], longest[:len(table), 0])
        support = nodes <= table[-1, 0]
        values = np.zeros((len(nodes), 2))
        values[support] = make_interp_spline(
            np.r_[0., table[:, 0]], np.vstack((np.zeros(2), table[:, 1:])),
            k=order)(nodes[support])
        waves[key] = values.T
    return case, waves, make_grid_workspace(nodes, weights)


@pytest.mark.parametrize('rank', [1, 2, 3, 4])
def test_fac_exact_frequency_rates(common_spinors, rank):
    case, waves, workspace = common_spinors
    rows = [r for r in case['rows'] if abs(r['multipole']) == rank]
    assert len(rows) == 5  # both electric branches/gauges, one magnetic branch
    for row in rows:
        kappa = row['upper'][1]
        spinors = (*waves[1, -1], *waves[tuple(row['upper'])])
        omega = row['omega_Ha']
        if row['multipole'] < 0:
            kernel = radial_eL_velocity if row['gauge'] == 1 else radial_eL
            radial = kernel(*spinors, workspace, rank,
                            kappa_i=-1, kappa_f=kappa, omega=omega)
            angular = angular_eL(kappa, -1, rank)
        else:
            radial = radial_mL(*spinors, workspace, rank, omega=omega)
            angular = angular_mL(kappa, -1, rank)
        rate = einstein_A_eL(omega, (radial*angular)**2, rank, 2*abs(kappa)-1)
        # H 2s–1s M1 is cancellation sensitive; constants and interpolants
        # are not identical to FAC's native quadrature. No fitted correction.
        tolerance = 5e-8 if case['z'] == 1 else 1e-8
        assert rate == pytest.approx(row['fac_A_per_s'], rel=tolerance), row
