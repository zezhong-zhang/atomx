"""Independent kernel projections and finite-frequency exchange invariants."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from atomx.numerics.helmholtz import _scaled_factors, helmholtz_potential
from atomx.physics.csf.retardation import distance_potential


@pytest.mark.parametrize("order", range(5))
@pytest.mark.parametrize("k", [0.0, 1e-9, .1, 1., 3.])
def test_multipole_against_direct_angular_projection(order, k):
    a, b = .7, 1.8
    j, _, dj, _ = _scaled_factors(order, np.array(k*a))
    _, y, _, dy = _scaled_factors(order, np.array(k*b))
    leading = a**order / b**(order+1)
    full = leading*j*y
    retarded = -2*leading*(a*a*dj*y+b*b*dy)

    def project(retardation):
        def integrand(t):
            R = np.sqrt(a*a+b*b-2*a*b*t)
            kernel = (R*np.sinc(k*R/(2*np.pi))**2 if retardation
                      else np.cos(k*R)/R)
            return kernel*eval_legendre(order, t)
        return (2*order+1)/2*quad(integrand, -1, 1, epsabs=2e-14)[0]

    assert full == pytest.approx(project(False), abs=2e-13)
    assert retarded == pytest.approx(project(True), abs=2e-13)


def _cells():
    x, w = np.polynomial.legendre.leggauss(12)
    edges = np.linspace(.01, 8, 80)
    half = np.diff(edges)[:, None]/2
    return edges[:-1, None]+half*(1+x), half*w


@pytest.mark.parametrize("order", range(5))
def test_static_limit_retains_distance_kernel_without_cancellation(order):
    r, weights = _cells()
    rho = r*r*np.exp(-2*r)
    expected = distance_potential(rho, r, weights, order)
    for k in (0., 1e-12, 1e-8):
        actual = helmholtz_potential(rho, r, weights, order, k, component="retardation")
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)
    assert not helmholtz_potential(rho, r, weights, order, 0., component="difference").any()


@pytest.mark.parametrize("component", ["full", "retardation", "difference"])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_radial_contraction_against_split_quadrature(component, order):
    r, weights = _cells()
    k = 1.7
    actual = helmholtz_potential(r*r*np.exp(-2*r), r, weights, order, k,
                                 component=component)
    for i, j in ((4, 3), (30, 6), (65, 2)):
        target = r[i, j]

        def integrand(s):
            lo, hi = sorted((s, target))
            jl, _, djl, _ = _scaled_factors(order, np.array(k*lo))
            _, yh, _, dyh = _scaled_factors(order, np.array(k*hi))
            leading = lo**order/hi**(order+1)
            if component == "full":
                kernel = leading*jl*yh
            else:
                kernel = leading*(lo*lo*djl*yh+hi*hi*dyh)
                kernel *= k*k if component == "difference" else -2
            return s*s*np.exp(-2*s)*kernel

        reference = (quad(integrand, .01, target, epsabs=1e-13)[0]
                     + quad(integrand, target, 8., epsabs=1e-13)[0])
        assert actual[i, j] == pytest.approx(reference, abs=3e-12)


@pytest.fixture(scope="module")
def neon_orbitals():
    from atomx import Atom
    from atomx.physics.csf.breit_ci import _ci_radial_inputs
    result = Atom(10, engine="bspline", nuclear="point", corrections="none").solve().result
    energies = {(s.n, s.kappa): result.orbital_energies[s.label]
                for s in result.subshells}
    return _ci_radial_inputs(result), energies


def test_full_transverse_static_limit_and_hermiticity(neon_orbitals):
    from atomx.physics.csf.matrix import _breit_radial_evaluator
    from atomx.physics.csf.transverse import radial_evaluator
    (orbitals, ws, Z, ns), energies = neon_orbitals
    static = _breit_radial_evaluator(orbitals, ws, Z, ns)
    zero = radial_evaluator(orbitals, ws, {a: 0. for a in orbitals})
    finite = radial_evaluator(orbitals, ws, energies)
    a, c = (1, -1), (2, -1)
    keys = [("t", 0, a, c, c, a),
            ("g", 0, ("P", *a), ("Q", *c), ("Q", *c), ("P", *a)),
            ("g", 1, ("P", *a), ("Q", *c), ("Q", *c), ("P", *a))]
    for key in keys:
        kind, L, a, c, b, d = key
        assert zero(key) == pytest.approx(static(key), rel=3e-9, abs=2e-13)
        value = finite(key)
        assert finite((kind, L, b, d, a, c)) == pytest.approx(value, abs=1e-14)
        assert finite((kind, L, c, a, d, b)) == pytest.approx(value, abs=1e-14)
        assert abs(value-zero(key)) > 1e-12


def test_ci_frequency_is_explicit_and_recorded(neon_orbitals):
    from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
    from atomx.methods.ci import ci_solve
    (orbitals, ws, Z, ns), energies = neon_orbitals
    csfs = enumerate_jj_csfs_many(["1s2 2s2 2p6"])
    kwargs = dict(csfs=csfs, orbitals_dict=orbitals, orbital_basis=get_shared_orbital_basis(csfs),
                  bsp_data=ws, Z=Z, ns=ns)
    with pytest.raises(ValueError, match="requires include_breit"):
        ci_solve(**kwargs, breit_orbital_energies=energies)
    with pytest.raises(ValueError, match="every orbital"):
        ci_solve(**kwargs, include_breit=True, breit_orbital_energies={})
    static = ci_solve(**kwargs, include_breit=True)
    zero = ci_solve(**kwargs, include_breit=True,
                    breit_orbital_energies={a: 0. for a in orbitals})
    finite = ci_solve(**kwargs, include_breit=True,
                      breit_orbital_energies=energies)
    assert zero.ground_energy == pytest.approx(static.ground_energy, abs=2e-11)
    assert abs(finite.ground_energy-static.ground_energy) > 1e-9
    assert finite.provenance["breit"] == "finite-frequency-transverse-in-blocks"
    assert len(finite.provenance["breit_orbital_energies_Ha"]) == len(orbitals)


@pytest.mark.parametrize("Z", [10, 82])
@pytest.mark.parametrize("common_spinors", [False, True])
def test_live_fac_two_electron_matrix_elements(Z, common_spinors):
    import json
    from pathlib import Path
    from atomx.config.subshell import Subshell
    from atomx.numerics.shooting.local import ShootingSolver
    from atomx.physics.csf.breit_ci import breit_terms
    from atomx.physics.csf.matrix import GridCIWorkspace
    from atomx.physics.csf.transverse import radial_evaluator

    data_dir = Path(__file__).parent/'data'
    filename = 'fac_transverse_common.json' if common_spinors else 'fac_transverse_reference.json'
    fixture = json.loads((data_dir/filename).read_text())
    case = next(c for c in fixture['cases'] if c['z'] == Z)
    keys = {tuple(r[k]) for r in case['pairs'] for k in ('a', 'b')}
    keys |= {tuple(s[:2]) for row in case['off_diagonal'] for s in row['spinors']}
    orbitals, energies = {}, {}
    if common_spinors:
        import hashlib
        spinors = data_dir/'fac_transverse_spinors.npz'
        assert hashlib.sha256(spinors.read_bytes()).hexdigest() == fixture['spinors_sha256']
        with np.load(spinors) as data:
            r = data[f'Z{Z}_r']
            orbitals = {(n,k): tuple(data[f'Z{Z}_n{n}_k{k}']) for n,k in keys}
        energies = {(e['n'],e['kappa']):e['energy_Ha'] for e in case['energies']}
        ws = GridCIWorkspace(r, np.ones_like(r), np.full_like(r, -Z))
        tolerance = 1e-7
    else:
        for n, kappa in sorted(keys):
            result = ShootingSolver(Z, [Subshell(n, kappa, 1)], nuclear="point",
                                     n_grid=2000, r_max=125./Z).solve(tol=1e-11)
            assert result.converged
            orbitals[n, kappa] = result._pa[0], result._qa[0]
            energies[n, kappa] = result.total_energy
        ws = GridCIWorkspace(result._grid, result._difr, np.full_like(result._grid, -Z))
        tolerance = 1e-6
    for reference, spectrum in (("fac_static_Ha", {a: 0. for a in keys}),
                                ("fac_finite_Ha", energies)):
        evaluate = radial_evaluator(orbitals, ws, spectrum)
        for row in case['pairs']:
            a, b = row['a'], row['b']
            value = 0j
            for ma in range(-2*abs(a[1])+1, 2*abs(a[1]), 2):
                for mb in range(-2*abs(b[1])+1, 2*abs(b[1]), 2):
                    labels = [(*a, ma), (*b, mb)]
                    value += sum(c*evaluate(key) for key, c in
                                 breit_terms(0, 1, 1, 0, -1., labels))
            assert value.imag == pytest.approx(0., abs=1e-14)
            assert value.real == pytest.approx(row[reference], rel=tolerance)
        elements = []
        for row in case['off_diagonal']:
            labels = [tuple(s) for s in row['spinors']]
            value = sum(c*evaluate(key) for key, c in breit_terms(0, 1, 2, 3, 1., labels))
            if common_spinors:
                # Exported spinors fix the phase convention as well as the
                # shapes, so CI interference requires the signed element.
                assert value.imag == pytest.approx(0., abs=1e-14)
                assert value.real == pytest.approx(row[reference], rel=tolerance)
            else:
                # Orbital phases in independently solved bases are arbitrary.
                assert abs(value) == pytest.approx(abs(row[reference]), rel=tolerance)
            elements.append(value)
        # The last three rows interchange bra and ket, including distinct
        # vertex frequencies. This checks signed Hermiticity, not magnitudes.
        np.testing.assert_allclose(elements[:3], np.conj(elements[3:]), atol=1e-14, rtol=1e-13)
