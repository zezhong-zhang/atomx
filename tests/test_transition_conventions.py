"""Physical absorption/emission identities and live FAC column conventions."""
from pathlib import Path

import numpy as np
import pytest

from atomx.constants import ALPHA, AU_RATE_PER_S, HARTREE_EV
from atomx.io.fac_writer import read_tra, write_tra
from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength

FAC = Path(__file__).parent/'data/fac_hlike_ne_multipoles.txt'


@pytest.mark.parametrize('rank', [1,2,3,4])
@pytest.mark.parametrize('omega', [.05,1.,100.])
def test_absorption_emission_relation_for_each_multipole(rank, omega):
    # Einstein detailed balance: g_lower f = g_upper A /(2 alpha^3 omega^2).
    lower_j2, upper_j2 = 1, 5
    rate = einstein_A_eL(omega, .7, rank, upper_j2)
    f, gf = oscillator_strength(omega, .7, rank, lower_j2, upper_j2)
    assert gf == pytest.approx((lower_j2+1)*f, rel=1e-14)
    assert rate*(upper_j2+1)/AU_RATE_PER_S == pytest.approx(
        2*ALPHA**3*omega**2*gf, rel=1e-14)


def test_fac_prints_A_and_reduced_amplitude():
    rows = read_tra(FAC)
    assert len(rows) == 4
    for row in rows:
        omega = row['DELTA_E_eV']/HARTREE_EV
        rank = abs(row['multipole_code'])
        from_rate = (row['A_per_s']*(row['twoj_upper']+1) /
                     (2*ALPHA**3*omega**2*AU_RATE_PER_S))
        # FAC dbase.c::OscillatorStrength has its own radial normalization.
        from_amplitude = (omega/(2*rank+1) * (ALPHA*omega)**(2*rank-2) *
                          row['reduced_matrix_element']**2)
        assert from_rate == pytest.approx(row['gf'], rel=2e-6)
        assert from_amplitude == pytest.approx(row['gf'], rel=2e-6)
        assert 'gA_per_s' not in row and 'gf_alt' not in row
    assert rows[0]['reduced_matrix_element'] < 0  # signed, unlike gf


def test_quadrupole_conversion_matches_live_fac():
    # Independently integrated Ne X 3d3/2 -> 1s1/2 Dirac radial strength.
    strength = 0.00023843183991580636
    row = read_tra(FAC)[2]
    _, gf = oscillator_strength(row['DELTA_E_eV']/HARTREE_EV,
                                strength, 2, 1, 3)
    assert gf == pytest.approx(row['gf'], rel=2e-6)


def test_fac_writer_preserves_rate_and_signed_amplitude(tmp_path):
    original = read_tra(FAC)
    for multipole in ('E1','E2'):
        rows = [dict(r, deltaE_eV=r['DELTA_E_eV']) for r in original
                if r['multipole'] == multipole]
        path = tmp_path/f'{multipole}.txt'
        write_tra(rows, path, multipole=multipole, Z=10)
        reread = read_tra(path)
        for before, after in zip(rows,reread, strict=False):
            for field in ('A_per_s','gf','reduced_matrix_element'):
                assert after[field] == pytest.approx(before[field],rel=1e-8)

    magnetic = next(r for r in read_tra(FAC.with_name('fac_hlike_ne_magnetic.txt'))
                    if r['multipole'] == 'M2' and r['mode'] == 0)
    path = tmp_path/'M2.txt'
    write_tra([dict(magnetic,deltaE_eV=magnetic['DELTA_E_eV'])],path,
              multipole='M2',Z=10)
    restored = read_tra(path)[0]
    assert restored['mode'] == 0
    assert restored['A_per_s'] == pytest.approx(magnetic['A_per_s'],rel=1e-8)


def test_fac_reader_rejects_uta_layout(tmp_path):
    path = tmp_path/'uta.txt'
    text = FAC.read_text().replace('6.268055E+12 -1.487110E-01',
                                  '6.268055E+12 -1.487110E-01 0.0 0.0')
    path.write_text(text)
    with pytest.raises(ValueError,match='unsupported'):
        read_tra(path)


@pytest.mark.parametrize('rank', [2,3,4])
def test_finite_frequency_magnetic_operator_against_live_fac(rank):
    from atomx.config.subshell import Subshell
    from atomx.numerics.shooting.local import ShootingSolver
    from atomx.properties.transitions.angular import angular_mL
    from atomx.properties.transitions.multipole import radial_mL
    row = next(r for r in read_tra(FAC.with_name('fac_hlike_ne_magnetic.txt'))
               if r['multipole'] == f'M{rank}' and r['mode'] == 0)
    lo = ShootingSolver(10,[Subshell(1,-1,1)],n_grid=2000,
                        r_max=12.5,nuclear='point').solve()
    up = ShootingSolver(10,[Subshell(rank,-rank,1)],n_grid=2000,
                        r_max=12.5,nuclear='point').solve()
    omega = up.total_energy-lo.total_energy
    radial = radial_mL(up._pa[0,1:],up._qa[0,1:],lo._pa[0,1:],lo._qa[0,1:],
        lo.radial_state.correction_workspace(),rank,omega=omega)
    strength = (radial*angular_mL(-rank,-1,rank))**2
    rate = einstein_A_eL(omega,strength,rank,row['twoj_upper'])
    assert rate == pytest.approx(row['A_per_s'],rel=1e-4)


@pytest.mark.parametrize('rank', [1,2,3,4])
def test_magnetic_retardation_has_regular_zero_frequency_limit(rank):
    from atomx.numerics.radial_workspace import make_grid_workspace
    from atomx.properties.transitions.multipole import radial_mL
    x, weights = np.polynomial.legendre.leggauss(80)
    r, weights = (x+1)*5, weights*5
    p, q = r*np.exp(-r), -.01*r*np.exp(-r)
    workspace = make_grid_workspace(r,weights)
    static = radial_mL(p,q,p,q,workspace,rank)
    finite = radial_mL(p,q,p,q,workspace,rank,omega=1e-8)
    assert finite == pytest.approx(static,rel=1e-14)
