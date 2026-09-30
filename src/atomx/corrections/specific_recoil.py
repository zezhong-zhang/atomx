"""Two-electron nuclear recoil in the point-Coulomb Breit approximation.

GRASP RIS4 VINTI, RINT_SMS2/3 and SMSNEW supply the radial convention.
Without Z, return the leading sum_{i<j} p_i.p_j / M. Supplying Z adds the
relativistic SMS2/SMS3 terms. Finite-size and radiative recoil are absent.
The correction stack exposes this operator through the opt-in include_sms flag.
"""
from __future__ import annotations

import numpy as np

from atomx.constants import C
from atomx.corrections.recoil import _radial_values


def vinti_matrix(orbitals, workspace):
    """V(a,c) in GRASP convention, before reduced rank-one angular factors."""
    return recoil_radial_matrices(orbitals, workspace)[0]


def _vinti_from_waves(waves, kappas, r, w):
    values = np.zeros((len(kappas), len(kappas)))
    for a, ka in enumerate(kappas):
        pa, qa, _, _ = waves[a]
        la = ka if ka > 0 else -ka-1
        for c, kc in enumerate(kappas):
            lc = kc if kc > 0 else -kc-1
            if (la+lc) % 2 != 1 or abs(abs(ka)-abs(kc)) > 1:
                continue
            pc, qc, dpc, dqc = waves[c]
            f1 = (ka*(ka+1)-kc*(kc+1))/2
            f2 = (ka*(ka-1)-kc*(kc-1))/2
            values[a, c] = np.sum(w*(pa*dpc + qa*dqc - (f1*pa*pc+f2*qa*qc)/r))
    return values


def reduced_sigma(kappa_bra, kappa_ket):
    """Reduced Pauli spin tensor in the Grant convention (GRASP SIGMA_1/2)."""
    from atomx.angular.wigner import wigner_6j
    la = kappa_bra if kappa_bra > 0 else -kappa_bra-1
    lb = kappa_ket if kappa_ket > 0 else -kappa_ket-1
    if la != lb:
        return 0.
    ja, jb = 2*abs(kappa_bra)-1, 2*abs(kappa_ket)-1
    phase = (-1.)**(la+(ja+1)//2+1)
    return phase*np.sqrt(6*(ja+1)*(jb+1))*wigner_6j(1, ja, 2*la, jb, 1, 2)


def relativistic_vinti_matrix(orbitals, workspace, Z):
    """GRASP VINT2: VINTI + RINT_SMS2/CRE + RINT_SMS3.

    Point-Coulomb Breit recoil, excluding radiative and finite-size recoil.
    Only rank-one allowed angular pairs are evaluated; forbidden pairs are zero.
    """
    return recoil_radial_matrices(orbitals, workspace, Z=Z)[1]


def recoil_radial_matrices(orbitals, workspace, *, Z=None):
    """Return (VINTI, VINT2), evaluating spinors and derivatives only once.

    With no Z, both entries refer to the leading momentum matrix. Supplying
    Z adds point-Coulomb Breit recoil to a separate second matrix.
    """
    r, w = workspace['gr'], workspace['grw']
    kappas = [orb[2] for orb in orbitals]
    waves = [_radial_values(p, q, workspace) for p, q, _ in orbitals]
    leading = _vinti_from_waves(waves, kappas, r, w)
    if Z is None:
        return leading, leading
    return leading, _breit_vinti_from_waves(waves, kappas, r, w, Z, leading.copy())


def _breit_vinti_from_waves(waves, kappas, r, w, Z, values):
    from atomx.angular.coefficients import reduced_C, reduced_cl_j2
    for a, ka in enumerate(kappas):
        pa, qa, _, _ = waves[a]
        for c, kc in enumerate(kappas):
            if abs(reduced_C(ka, kc, 1)) < 1e-14:
                continue
            # SIGMA_1/2 use the standard Wigner-Eckart convention. Divide
            # by GRASP CRE / FAC ReducedCL in that same convention. The
            # legacy reduced_C uses reversed 3j projections and differs in
            # sign for unequal-j rank-one channels; mixing these conventions
            # reverses SMS2 relative to the momentum and SMS3 terms.
            angular = reduced_cl_j2(2*abs(ka)-1, 2, 2*abs(kc)-1)
            pc, qc, _, _ = waves[c]
            sms2 = -Z/C*np.sum(w*(reduced_sigma(-ka, kc)*qa*pc
                                  - reduced_sigma(ka, -kc)*pa*qc)/r)
            sms3 = Z/C*np.sum(w*(qa*pc-pa*qc)/r)
            values[a, c] += sms2/angular + sms3
    return values


def momentum_recoil_matrix(csfs, orbitals_dict, workspace, *, Z=None):
    """CSF matrix K/M; supply Z to include GRASP SMS2/SMS3 Breit terms."""
    from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
    keys = tuple(sorted(orbitals_dict))
    index = {key: i for i, key in enumerate(keys)}
    orbitals = [(*orbitals_dict[k], k[1]) for k in keys]
    vinti, corrected = recoil_radial_matrices(orbitals, workspace, Z=Z)

    def integral(key):
        if key[0] != 'r' or key[1] != 1:
            return 0.
        _, _, a, c, b, d = key
        ac, bd = (index[a], index[c]), (index[b], index[d])
        return -(corrected[ac]*vinti[bd]+corrected[bd]*vinti[ac])/2

    matrix = np.zeros((len(csfs), len(csfs)))
    blocks = {}
    for i, csf in enumerate(csfs):
        blocks.setdefault((csf.total_J2, csf.parity), []).append(i)
    for indices in blocks.values():
        matrix[np.ix_(indices, indices)] = exact_block_matrix(
            [csf_signature(csfs[i]) for i in indices], integral)
    return matrix


def momentum_recoil_parameters(csfs, mixing, orbitals_dict, workspace, *, Z=None):
    """Two-body K expectation; Z selects relativistic Breit recoil."""
    matrix = momentum_recoil_matrix(csfs, orbitals_dict, workspace, Z=Z)
    return np.einsum('ik,ij,jk->k', mixing, matrix, mixing, optimize=True)


def configuration_average_recoil(orbitals, workspace, *, Z=None, jj_configs=None):
    """Two-body K for an independently averaged jj configuration.

    Orbitals are (P,Q,occupation,kappa[,label]). For coupled/CI states use
    the CSF matrix instead. Supply jj_configs for an ensemble (labels are
    then required); its pair occupations are averaged explicitly. The rank-one direct term vanishes
    by parity; only exchange between opposite-parity shells contributes.
    """
    return float(sum(configuration_average_recoil_contributions(
        orbitals, workspace, Z=Z, jj_configs=jj_configs).values()))


def configuration_average_recoil_contributions(orbitals, workspace, *, Z=None, jj_configs=None):
    """Return {(orbital_index_a, orbital_index_b): K_ab} for allowed pairs.

    Occupations are electron counts. Reduced angular matrix elements already
    sum over magnetic substates, so pair weights contain occupation fractions
    q_a/(2j_a+1), not another factor of each full subshell degeneracy.
    This is the same operator used by the correction stack and properties.
    """
    from atomx.angular.coefficients import reduced_C
    moments = None
    if jj_configs is not None:
        from atomx.corrections.dbsr_breit import _pair_moments
        moments = _pair_moments(orbitals, jj_configs)
    waves = [(orb[0], orb[1], orb[3]) for orb in orbitals]
    v, vr = recoil_radial_matrices(waves, workspace, Z=Z)
    contributions = {}
    for a, orbital in enumerate(orbitals):
        _, _, qa, ka = orbital[:4]
        fa = qa/(2*abs(ka))
        for b in range(a+1, len(orbitals)):
            _, _, qb, kb = orbitals[b][:4]
            fb = qb/(2*abs(kb))
            pair_weight = fa*fb if moments is None else moments[a, b]/(4*abs(ka*kb))
            angular = reduced_C(ka, kb, 1)
            if angular != 0. and pair_weight != 0.:
                contributions[a, b] = float(pair_weight*angular**2*
                    (vr[a, b]*v[b, a]+vr[b, a]*v[a, b])/2)
    return contributions


def specific_recoil_provenance(mass):
    return {'operator': 'two-body point-Coulomb Breit recoil (GRASP RIS4)',
            'nuclear_mass_me': float(mass), 'order': 'first order in 1/M',
            'radial_terms': ['VINTI', 'RINT_SMS2', 'RINT_SMS3'],
            'omitted': ['finite-size recoil', 'radiative recoil', 'higher orders in 1/M'],
            'evaluation': 'first-order expectation; no degenerate-subspace diagonalization'}
