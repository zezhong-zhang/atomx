"""Independent determinant-level check of the MCDHF E1 contraction.

Reconstructs <1s2 1S0|D_q|1s2p J=1,M=1> from explicit 2-electron
determinants + sympy CG, sharing ONLY the production radial integrals.
"""
import sys
import numpy as np
from sympy import Rational, S
from sympy.physics.wigner import clebsch_gordan as CG, wigner_3j as W3J
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom
from atomx.angular.coefficients import reduced_C
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import einstein_A_eL
from atomx.constants import ALPHA, AU_RATE_PER_S

he = Atom(2, config=["1s2", "1s1 2p1"], corrections="none")
he.solve()
mcdhf = he._csf_result
csfs, mixing = mcdhf.csfs, mcdhf.mixing_coefficients
E = mcdhf.state_energies
bsp = mcdhf.bsp_data
od = mcdhf.orbitals_dict
print("orbitals_dict keys:", list(od.keys()))

# locate 1s and 2p_j orbitals
def get_orb(n, kappa):
    for key, val in od.items():
        if key[0] == n and key[1] == kappa:
            return val
    raise KeyError((n, kappa))

P1s, Q1s = get_orb(1, -1)
P2p12, Q2p12 = get_orb(2, 1)
P2p32, Q2p32 = get_orb(2, -2)

# J=1 states: indices with energy near the two J=1 levels.
# From earlier run: idx 2 (triplet, -2.08081842) and idx 4 (singlet, -2.06066442).
# Ground = idx 0.
results = {}
for iJ1 in (2, 4):
    omega = E[iJ1] - E[0]
    # production radial integrals with the same finite-frequency operator
    R12 = radial_eL(P1s, Q1s, P2p12, Q2p12, bsp, 1, kappa_i=-1, kappa_f=1, omega=omega)
    R32 = radial_eL(P1s, Q1s, P2p32, Q2p32, bsp, 1, kappa_i=-1, kappa_f=-2, omega=omega)

    # single-particle m-resolved dipole <j_a m_a| d_q |j_b m_b>, d_q = r*C^1_q
    # using production reduced_C convention (Grant phase), WE theorem:
    # <a|d_q|b> = (-1)^{j_a-m_a} 3j(j_a 1 j_b; -m_a q m_b) * reduced_C(kappa_a, kappa_b, 1) * R_ab
    def d_me(tja, tma, tkb, tmb, q, R, ka, kb):
        # ka, kb kappas; tja etc doubled
        jb = abs(kb) - 0.5
        red = reduced_C(ka, kb, 1)
        if red == 0.0: return 0.0
        w = float(W3J(Rational(tja,2), S.One, Rational(abs(kb)*2-1,2),
                      Rational(-tma,2), S(q), Rational(tmb,2)))
        phase = (-1)**((tja - tma)//2)
        return phase * w * red * R

    # Build J=1, M=1 CSFs from determinants.
    # determinants (m_s of 1s active electron, m of p electron), sum m = 1
    # p1/2 CSF: only det (1s +1/2)(2p1/2 +1/2)
    # p3/2 CSF: d1=(+1/2, +1/2), d2=(-1/2, +3/2) with CGs <1/2 ms 3/2 mp|1 1>
    csf_p12 = {(1, 1): 1.0}
    c1 = float(CG(Rational(1,2), Rational(3,2), S.One, Rational(1,2), Rational(1,2), S.One))
    c2 = float(CG(Rational(1,2), Rational(3,2), S.One, Rational(-1,2), Rational(3,2), S.One))
    csf_p32 = {(1, 1): c1, (-1, 3): c2}

    # <1s2 J=0| D_q |det(m_s, m_p)>: D_q removes the p electron (m_p) and
    # flips one 1s spin-orbital. Ground |1s+1/2, 1s-1/2> determinant.
    # One-body op: sum over 1s spin m: d_q(m<-m_p) a+_{1s m} a_{p m_p}.
    # Acts on |1s_{m_s}, p_{m_p}> -> d_q * sgn * |1s_{m_s}, 1s_m>.
    # Overlap with ground det nonzero if {m_s, m} = {-1/2, +1/2}.
    # q = m - m_p ; need total <0 0|D_q|1 1> -> q = -M = -1
    def amp(csf, R, ka, kb):
        total = 0.0
        for (tms, tmp), coef in csf.items():
            for tm in (-1, 1):
                q = (tm - tmp) // 2
                # m-resolved one-body matrix element <1s m|d_q|p_j m_p>
                mel = d_me(1, tm, kb, tmp, q, R, -1, kb)
                # determinant algebra: |1s_ms, p_mp> ordering (1s first)
                # a+_{1s m} a_{p mp}:  annihilate p (2nd orbital): sign (-1)^1 = -1
                #   -> -|1s_ms>; then create 1s m: -|1s_ms 1s_m>
                # overlap with ground |1s_+1/2 1s_-1/2>:
                #   if (ms,m)==(+1/2,-1/2): |1s_+ 1s_-> -> +1 ; times -1 => -1
                #   if (ms,m)==(-1/2,+1/2): |1s_- 1s_+> = -|1s_+ 1s_-> ; times -1 => +1
                if (tms, tm) == (1, -1):
                    sgn = -1.0
                elif (tms, tm) == (-1, 1):
                    sgn = 1.0
                else:
                    continue
                # only q=-1 contributes for <00|D|11>... keep all and select later
                total += coef * mel * sgn * (1.0 if q == -1 else 0.0)
        return total

    A_p12 = amp(csf_p12, R12, -1, 1)
    A_p32 = amp(csf_p32, R32, -1, -2)
    # CI contraction over the two J=1 CSFs. Find the mixing coefficients:
    # CSF list order printed earlier: idx1 = 1s 2p3/2 J=1, idx4 = 1s 2p1/2 J=1.
    amp_state = mixing[4, iJ1] * A_p12 + mixing[1, iJ1] * A_p32
    A = (4.0/3.0) * ALPHA**3 * omega**3 * amp_state**2 * AU_RATE_PER_S
    results[iJ1] = (omega, A_p12, A_p32, amp_state, A)
    print(f"state {iJ1}: omega={omega:.8f} A_p12={A_p12:.6e} A_p32={A_p32:.6e} "
          f"amp={amp_state:.6e} A_det={A:.6e} s^-1")

trs = he.transitions(multipoles=["E1"])
for tr in trs:
    print(f"production: {tr.initial_label}->{tr.final_label} A={tr.A_rate_per_s:.6e}")
