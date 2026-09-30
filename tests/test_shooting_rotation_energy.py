"""Ordered Coulomb functionals and lifetime of occupied-rotation contexts."""
import numpy as np
import pytest

from atomx.numerics.shooting.coulomb import _slater_integral_slag6
from atomx.numerics.shooting.energy import RadialRotationEnergy
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.operators import one_body_element
from atomx.numerics.shooting.quadrature import slag6_weights


def complete_energy(p, q, r, jacobian, rv, one_body, terms):
    # No partial-state cache or indexed contraction graph in this oracle.
    energy = sum(weight*one_body_element(p[a], q[a], p[a], q[a], kappa,
                                        r, jacobian, rv)
                 for a, (weight, kappa) in enumerate(one_body))
    for coefficient, rank, a, b, c, d in terms:
        left = p[a]*p[b]+q[a]*q[b]
        right = p[c]*p[d]+q[c]*q[d]
        energy += coefficient*_slater_integral_slag6(left, right, rank,
                                                    r, jacobian, len(r))
    return energy


def problem(strided=False):
    r, jacobian, n = sgrid(20., 1e-7, .1, 1200)
    p = np.empty((4, n*2 if strided else n))
    if strided:
        p = p[:, ::2]
    p[:] = [r*np.exp(-r), r*(1-r)*np.exp(-r/2),
            r*(1-2*r+.5*r*r)*np.exp(-r/3), r*r*np.exp(-r)]
    q = .004*p*r
    norm = np.sqrt(np.sum((p*p+q*q)*slag6_weights(n)*jacobian, axis=1))
    p /= norm[:, None]
    q /= norm[:, None]
    one_body = [(1.2, -1), (.8, -1), (2., -1), (1.3, -2)]
    # Fixed, variable, repeated, reversed and signed interference terms.
    # R(00,01) retains odd rotation harmonics; it cannot be replaced by an
    # occupation-product configuration average.
    terms = [(1.25, 0, 2, 2, 2, 2), (.7, 0, 0, 0, 1, 1),
             (-.34, 0, 0, 1, 0, 1), (.29, 0, 0, 0, 0, 1),
             (.41, 2, 3, 3, 3, 3), (.09, 0, 2, 2, 0, 1),
             (.13, 0, 0, 1, 2, 2), (.23, 0, 2, 2, 0, 0),
             (.35, 0, 0, 0, 1, 1)]
    return p, q, r, jacobian, np.full(n, -4.), one_body, terms


def rotate(p, q, initial, angle, a=0, b=1):
    c, s = np.cos(angle), np.sin(angle)
    for values, saved in zip((p, q), initial, strict=True):
        values[a] = c*saved[a]+s*saved[b]
        values[b] = c*saved[b]-s*saved[a]


@pytest.mark.parametrize('strided', [False, True])
def test_rotation_recomputes_changed_pairs_and_retains_interference(strided):
    p, q, r, jacobian, rv, one_body, terms = problem(strided)
    initial = p.copy(), q.copy()
    energy = RadialRotationEnergy(p, q, r, jacobian, rv, one_body, terms).for_pair(0, 1)
    values = []
    for angle in (.33, 0., -.33, .67, 0.):
        rotate(p, q, initial, angle)
        values.append(energy())
        assert values[-1] == pytest.approx(
            complete_energy(p, q, r, jacobian, rv, one_body, terms), abs=3e-11, rel=0)
    assert abs(values[0]-values[2]) > .01
    assert values[1] == values[-1]


def test_new_pair_and_state_get_independent_fixed_contractions():
    p, q, r, jacobian, rv, one_body, terms = problem()
    sweep = RadialRotationEnergy(p, q, r, jacobian, rv, one_body, terms)
    for a, b in ((0, 1), (1, 2)):
        initial = p.copy(), q.copy()
        first = sweep.for_pair(a, b)
        changed_body = [(weight*.7, kappa) for weight, kappa in one_body]
        changed_terms = [(-coefficient, rank, i, j, k, l)
                         for coefficient, rank, i, j, k, l in terms]
        second = RadialRotationEnergy(p, q, r, jacobian, rv,
                                      changed_body, changed_terms).for_pair(a, b)
        for angle in (-.17, .41):
            rotate(p, q, initial, angle, a, b)
            assert first() == pytest.approx(
                complete_energy(p, q, r, jacobian, rv, one_body, terms), abs=3e-11, rel=0)
            assert second() == pytest.approx(
                complete_energy(p, q, r, jacobian, rv, changed_body, changed_terms), abs=3e-11, rel=0)


def test_hydrogenic_coulomb_integral_tracks_fourth_power_of_amplitude():
    z = 1.4
    r, jacobian, n = sgrid(30., 1e-7, .1, 1600)
    p = np.array([2*z**1.5*r*np.exp(-z*r)])
    q = np.zeros_like(p)
    initial = p.copy()
    energy = RadialRotationEnergy(p, q, r, jacobian, np.full(n, -z),
                                  [(0., -1)], [(1., 0, 0, 0, 0, 0)]).for_pair(0, 0)
    for amplitude in (1., .3, -1.2, 1.):
        p[:] = amplitude*initial
        assert energy() == pytest.approx(amplitude**4*5*z/8, abs=3e-10, rel=0)


def test_empty_coulomb_graph_preserves_one_body_rotations():
    p, q, r, jacobian, rv, one_body, _ = problem()
    initial = p.copy(), q.copy()
    energy = RadialRotationEnergy(p, q, r, jacobian, rv, one_body, []).for_pair(0, 1)
    rotate(p, q, initial, .31)
    assert energy() == pytest.approx(
        complete_energy(p, q, r, jacobian, rv, one_body, []), abs=3e-12, rel=0)


@pytest.mark.parametrize('term', [(1., 0, 0, 0, 4, 4), (1., 0, -1, 0, 0, 0),
                                 (1., -1, 0, 0, 0, 0), (1., .5, 0, 0, 0, 0)])
def test_invalid_coulomb_indices_never_reach_unchecked_compiled_reads(term):
    p, q, r, jacobian, rv, one_body, _ = problem()
    with pytest.raises(ValueError, match='Coulomb'):
        RadialRotationEnergy(p, q, r, jacobian, rv, one_body, [term])
