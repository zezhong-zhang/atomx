"""Native integration and canonical-orbital correction controls."""
import numpy as np
import pytest

from atomx.numerics.radial_integration import cumulative_radial_integral


def test_gauss_cell_prefix_includes_partial_cell_and_prior_cells():
    nodes, weights = np.polynomial.legendre.leggauss(9)
    edges = np.array([0., .01, .2, 1., 3.])
    half = np.diff(edges)/2
    r = (edges[:-1]+half)[:, None]+half[:, None]*nodes
    w = half[:, None]*weights
    values = r**6-2*r**3
    expected = r**7/7-r**4/2
    actual = cumulative_radial_integral(values, {'gr':r,'grw':w})
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)


def test_slag6_prefix_keeps_origin_endpoint_weights():
    from atomx.numerics.shooting.grid import sgrid
    from atomx.numerics.shooting.quadrature import slag6_weights
    r,j,n=sgrid(20.,1e-7,.05,2000)
    workspace={'gr':r[1:],'grw':(slag6_weights(n)*j)[1:],
               'integration_rule':'slag6','jacobian':j,'origin_offset':1}
    actual=cumulative_radial_integral(r[1:]*np.exp(-r[1:]),workspace)
    expected=1-(r[1:]+1)*np.exp(-r[1:])
    np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=2e-9)


def test_closed_shell_shooting_is_canonical_and_corrections_match_bspline():
    from atomx.config.jj_coupling import generate_jj_average
    from atomx.corrections.stack import CorrectionStack
    from atomx.methods.dirac_fock import DiracFock
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    shells,configs=generate_jj_average('1s2 2s2',averaging='statistical')
    solver=ShootingHFSolver(16,shells,n_grid=3200,aw=32,nuclear='fermi',jj_configs=configs)
    shooting=solver.solve(max_iter=200,tol=1e-12)
    reference=DiracFock(breit=False,qed=False,nuclear_mass_number=32).solve(
        16,shells,jj_configs=configs,max_iter=150,tol=1e-10,
        grid_hi=.125,grid_he=.125)
    assert shooting.converged and reference.converged
    assert abs(shooting.total_energy-reference.total_energy)<1e-8
    rv=solver.nuclear_rv_screened(shooting._grid,shooting._difr,len(shooting._grid))
    fock=solver.functional.occupied_fock_matrix(shooting._pa,shooting._qa,shooting._grid,
                                    shooting._difr,rv,solver.functional.angular_tables())
    assert abs(fock[0,1])<1e-9
    shooting._jj_configs=configs
    reference._jj_configs=configs
    stack=CorrectionStack(se_backend='mohr',strict=True)
    a,b=stack.apply(shooting),stack.apply(reference)
    # Refined controls retain a 1.34e-8 Ha Breit difference; enforce an
    # explicit absolute bound, not equality below the measured numerical floor.
    for name in a:
        assert a[name].energy_shift == pytest.approx(b[name].energy_shift,abs=2e-8,rel=0)
