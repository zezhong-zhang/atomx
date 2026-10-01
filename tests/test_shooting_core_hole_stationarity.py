"""A core-hole configuration is a stationary excited state, not a minimum."""
import numpy as np
import pytest


def test_beryllium_core_hole_keeps_radial_identity_and_matches_bspline():
    from atomx.config.jj_coupling import JJConfig, generate_jj_average
    from atomx.methods.dirac_fock import DiracFock
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    from atomx.numerics.shooting.operators import one_body_element
    shells,ensemble=generate_jj_average('1s1 2s2 2p1',averaging='statistical')
    kept=[i for i,s in enumerate(shells) if s.occupation>0]
    shells=[shells[i] for i in kept]
    ensemble=[JJConfig(subshells=[c.subshells[i] for i in kept],weight=c.weight) for c in ensemble]
    solver=ShootingHFSolver(4,shells,aw=9,nuclear='fermi',jj_configs=ensemble,
        n_grid=3200,r_max=75.)
    result=solver.solve(max_iter=200,tol=1e-12)
    reference=DiracFock(breit=False,qed=False,nuclear_mass_number=9).solve(
        # The B-spline orbital-change floor is about 4e-10 on this mesh;
        # the independent total-energy comparison below remains unchanged.
        4,shells,jj_configs=ensemble,max_iter=150,tol=1e-9,
        r_max=75.,grid_hi=.125,grid_he=.125)
    assert result.converged and reference.converged
    assert result.orbital_energies['1s1/2'] < result.orbital_energies['2s1/2']
    assert result.total_energy == pytest.approx(reference.total_energy,abs=2e-8,rel=0)
    assert result.solver_diagnostics['occupied_rotation_residual_Ha'] < 1e-8
    # The physical branch is a saddle: filling the hole lowers the energy.
    r,j=result._grid,result._difr
    rv=solver.nuclear_rv_screened(r,j,len(r))
    a,b=[i for i,s in enumerate(shells) if s.kappa==-1]
    energies=[]
    for angle in (-1e-3,1e-3):
        p,q=result._pa.copy(),result._qa.copy()
        for values in (p,q):
            va,vb=values[a].copy(),values[b].copy()
            values[a]=np.cos(angle)*va+np.sin(angle)*vb
            values[b]=np.cos(angle)*vb-np.sin(angle)*va
        h=sum(s.occupation*one_body_element(p[i],q[i],p[i],q[i],s.kappa,r,j,rv)
              for i,s in enumerate(shells))
        direct,exchange=solver.functional.two_electron_energy(p,q,r,j,len(r))
        energies.append(h+direct-exchange)
    assert max(energies)<result.total_energy-1e-7
