"""Close-root resolution tested against an independent arbitrary-precision oracle."""
import mpmath as mp
import numpy as np
import pytest
from scipy import linalg

from atomx.numerics.hermitian_clusters import refine_real_symmetric_clusters


def test_close_root_slopes_match_same_matrix_high_precision_oracle():
    # An exactly specified DOUBLE matrix, not a rounded assumed spectrum.
    h = np.array([[1e3, 1e-9, 2e-6], [1e-9, 1e3+2e-8, -3e-6],
                  [2e-6, -3e-6, 1e4]])
    d = np.array([[3., 40., 2.], [40., -7., 1.], [2., 1., 0.]])
    with mp.workdps(60):
        _, exact = mp.eigsy(mp.matrix(h.tolist()))
        md = mp.matrix(d.tolist())
        slopes = [float((exact[:, i].T*md*exact[:, i])[0]) for i in (0, 1)]
    refined_slopes = []
    for driver in ('evr', 'evd', 'ev'):
        w, v = linalg.eigh(h, driver=driver)
        original_h, original_w, original_v = h.copy(), w.copy(), v.copy()
        ew, ev, info = refine_real_symmetric_clusters(h, w, v, gap=1e-5)
        np.testing.assert_array_equal(h, original_h)
        np.testing.assert_array_equal(w, original_w)
        np.testing.assert_array_equal(v, original_v)
        measured = np.diag(ev.T @ d @ ev)[:2]
        np.testing.assert_allclose(measured, slopes, rtol=0, atol=2e-6)
        refined_slopes.append(measured)
        assert info['clusters'][0]['end'] == 2
        assert info['max_original_relative_residual'] < 1e-10
        assert info['gram_error'] < 1e-10
        np.testing.assert_allclose(ew, w, rtol=0, atol=1e-10)
    assert np.max(np.ptp(refined_slopes, axis=0)) < 2e-6


def test_complete_exact_degenerate_subspace_retains_projector():
    h = np.diag([2., 2., 3.])
    a = .7
    c = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
    w, v, _ = refine_real_symmetric_clusters(h, [2., 2., 3.], c, gap=1e-5)
    np.testing.assert_allclose(v[:, :2] @ v[:, :2].T, np.diag([1., 1., 0.]), atol=1e-14)
    np.testing.assert_allclose(w, [2., 2., 3.], atol=1e-14)


@pytest.mark.parametrize('gap', [True, 0., -1., np.inf, 1j])
def test_invalid_gap_rejected(gap):
    with pytest.raises(ValueError, match='gap'):
        refine_real_symmetric_clusters(np.eye(2), np.ones(2), np.eye(2), gap=gap)


def test_rejects_incomplete_complex_nonsymmetric_and_wrong_vectors():
    with pytest.raises(ValueError, match='complete'):
        refine_real_symmetric_clusters(np.eye(3), [1., 1.], np.eye(3)[:, :2], gap=.1)
    with pytest.raises(ValueError, match='real'):
        refine_real_symmetric_clusters(np.eye(2, dtype=complex), [1., 1.], np.eye(2), gap=.1)
    with pytest.raises(ValueError, match='exactly symmetric'):
        refine_real_symmetric_clusters([[1., 1e-20], [0., 2.]], [1., 2.], np.eye(2), gap=.1)
    with pytest.raises(ValueError, match='Gram'):
        refine_real_symmetric_clusters(np.eye(2), [1., 1.], np.zeros((2, 2)), gap=.1)
    with pytest.raises(ValueError, match='residual'):
        refine_real_symmetric_clusters(np.diag([1., 2.]), [1., 2.], np.fliplr(np.eye(2)), gap=.1)


def test_cluster_cap_does_not_split_a_closed_cluster():
    with pytest.raises(ValueError, match='resource cap'):
        refine_real_symmetric_clusters(np.eye(3), [1., 1., 1.], np.eye(3),
                                       gap=.1, max_cluster_size=2)


def test_ci_partial_and_davidson_rejected_before_physical_assembly(monkeypatch):
    import atomx.physics.csf.matrix as matrix
    from atomx.methods.ci import ci_solve
    def forbidden(*args, **kwargs):
        raise AssertionError('physical matrix allocated before validating request')
    monkeypatch.setattr(matrix, 'build_ci_matrix', forbidden)
    for options in ({'solver': 'davidson'}, {'n_states': 1}):
        with pytest.raises(ValueError, match='complete dense'):
            ci_solve([object()], {}, [], {}, 1, 1, dense_cluster_gap=.1, **options)


def test_real_charge_functional_refines_same_target_and_reuses_accepted_trial():
    from atomx import Atom, Basis, CIChargeStateAverage, CIStateAverage, Correlation
    from atomx.methods.csf.charge_variational import ChargeStateOrbitalFunctional
    atom = Atom(6, config=['1s2', '1s1 2p1', '2s2'], basis=Basis(ns=30, ks=9),
                corrections='none', correlation=Correlation(radial_optimization='fac_frozen_local'))
    r = atom.solve(tol=1e-6).result
    target = CIChargeStateAverage(((2, 1., CIStateAverage()), (1, 1., CIStateAverage())))
    args = (r.csfs, r.bsp_data, 6, r._nuclear_potential)
    ordinary = ChargeStateOrbitalFunctional(*args, state_average=target)
    refined = ChargeStateOrbitalFunctional(*args, state_average=target, dense_cluster_gap=2.)
    e0, g0, old = ordinary.evaluate(r.orbitals_dict)
    e1, trial = refined.evaluate(r.orbitals_dict, gradient=False)
    e2, g2, accepted = refined.evaluate(r.orbitals_dict, ci_result=trial)
    assert accepted is trial
    assert e1 == e2
    assert e0 == pytest.approx(e1, abs=1e-11)
    for n in trial.sectors:
        np.testing.assert_allclose(trial.sectors[n].state_energies,
                                   old.sectors[n].state_energies, rtol=0, atol=1e-11)
        assert trial.sectors[n].provenance['dense_cluster_refinement']
        np.testing.assert_array_equal(refined.last_sector_weights[n], ordinary.last_sector_weights[n])
    for key in g0:
        np.testing.assert_allclose(g2[key], g0[key], rtol=1e-9, atol=1e-9)
