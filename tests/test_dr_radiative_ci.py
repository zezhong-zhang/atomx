"""Final-state CI, photon frequencies and coherent stabilization amplitudes."""
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.angular.mscheme import one_body_action
from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold


def toy_manifold(mixing=.3):
    """A p_1/2 hole and an s² pair; two bound s spectators can mix."""
    keys = [(2,1),(3,-1),(4,-1)]
    r = np.linspace(.001,10.,100)
    waves = {key:(r*np.exp(-r),np.zeros_like(r)) for key in keys}
    t = {(a,b):float(a==b)*(-10 if a[0]==2 else -2 if a[0]==3 else -1)
         for a in keys for b in keys if a[1]==b[1]}
    t[(3,-1),(4,-1)] = t[(4,-1),(3,-1)] = mixing
    man = ResonanceManifold(SimpleNamespace(Z=10),
        [([(2,1)],1), ([(3,-1)],2), ([(4,-1)],0)],
        r_fine=r, dr_fine=np.gradient(r), orbital_map=waves,
        orbital_energies={key:123. for key in keys}, one_body=t,
        bound_radial=lambda key:0.)
    return man


def test_final_ci_mixing_and_photon_frequencies():
    man = toy_manifold()
    frequencies = []
    def dipole(bra, ket, rank, omega=None):
        frequencies.append(omega)
        return 1. if omega is None else 1.+.001*omega**2
    man.electric_radial = dipole
    dets = man.determinants(1)
    assert len(dets)==1
    initial = float(man.hamiltonian(dets)[0,0])
    final_energies, vectors = np.linalg.eigh(np.array([[-22.,.3],[.3,-21.]]))
    static = man.radiative_channels(dets,np.ones(1),initial,photon_approximation='long_wavelength')
    finite = man.radiative_channels(dets,np.ones(1),initial)
    assert len(static)==len(finite)==2
    prefactors=[]
    for i,(s,f) in enumerate(zip(static,finite,strict=True)):
        omega=initial-final_energies[i]
        assert f.final_energy_au==pytest.approx(final_energies[i],abs=1e-13)
        assert f.omega_au==pytest.approx(omega,abs=1e-13)
        assert f.A_per_s/s.A_per_s==pytest.approx((1+.001*omega**2)**2,rel=2e-14)
        prefactors.append(s.A_per_s/(omega**3*abs(vectors[0,i])**2))
    assert prefactors[0]==pytest.approx(prefactors[1],rel=2e-14)
    assert set(v for v in frequencies if v is not None)=={-f.omega_au for f in finite}
    # Orbital eigenvalue bookkeeping must have no influence on a CI photon.
    man.eps[:]=[-1e6]*len(man.eps)
    assert man.radiative_channels(dets,np.ones(1),initial)==finite


def test_fixed_m_rates_and_common_energy_zero():
    man=toy_manifold()
    man.electric_radial=lambda bra,ket,rank,omega=None:1.
    rates=[]
    for tm in (-1,1):
        dets=man.determinants(tm)
        energy=man.hamiltonian(dets)[0,0]
        rates.append(man.radiative_channels(dets,np.ones(1),energy))
    np.testing.assert_allclose([x.A_per_s for x in rates[0]],
                               [x.A_per_s for x in rates[1]],rtol=1e-14)
    # The same one-body constant shifts both three-electron levels equally.
    man._t += .1*np.eye(man.n_orb)
    man._radiative_final_cache=None
    dets=man.determinants(1)
    shifted=man.radiative_channels(dets,np.ones(1),man.hamiltonian(dets)[0,0])
    np.testing.assert_allclose([x.omega_au for x in shifted],
                               [x.omega_au for x in rates[1]],atol=1e-13,rtol=0)
    np.testing.assert_allclose([x.A_per_s for x in shifted],
                               [x.A_per_s for x in rates[1]],rtol=1e-14)


def test_radiative_ci_rejects_dressed_eigenvalue_hamiltonian():
    man=toy_manifold()
    man.one_body=None
    with pytest.raises(ValueError,match='dressed orbital energies'):
        man.radiative_channels(man.determinants(1),np.ones(1),-14.)


def test_one_body_action_preserves_complex_interference_and_weak_terms():
    # |0,2> -> |1,2> by a†_1 a_0: +1; |0,1> -> |1,2> by a†_2 a_0: -1.
    bra=[0b110]
    ket=[0b101,0b011]
    weights=np.array([1e-18+2e-18j,3e-18-1e-18j])
    actual=one_body_action(bra,ket,weights,{(1,0):2.,(2,0):1j})
    assert actual[0]==pytest.approx(2*weights[0]-1j*weights[1],abs=1e-32)
    np.testing.assert_array_equal(one_body_action(bra,ket,weights,{}),[0.])
