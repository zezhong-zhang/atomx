"""CI Hamiltonian matrix construction for MCDHF.

Slater-Condon rules:
- 0-diff (diagonal): full 1e + 2e energy of the configuration
- 1-diff (a→b):   <a|h|b> + Σ_c q_c [R^0(ac,bc) - Σ_k c^k R^k(ac,cb)]
- 2-diff (ab→cd): Σ_k [direct R^k(ac,bd) - exchange R^k(ac,db)]
- 3+ diff:        0  (orthogonality)

For CSFs with explicit J-resolution (``total_J2`` set, intermediate
parentage inferable), a J-coupled angular correction is added on top of
the subshell-occupation kernel so that distinct (J, parity) blocks
acquire physically meaningful J-dependent splittings.  See
``_diagonal_J_correction`` and Grant, *Relativistic Quantum Theory of
Atoms and Molecules* (2007) §8.3.
"""

import numpy as np
from atomx.methods.kohn_sham import compute_slater_fk, compute_slater_gk, compute_slater_rk
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid, compute_yk
from atomx.angular.coefficients import exchange_lambda, exchange_k_range
from atomx.angular.wigner import wigner_6j
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices


def build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns):
    """Build the CI Hamiltonian matrix H_{rs} in the CSF basis."""
    n_csf = len(csfs)
    H = np.zeros((n_csf, n_csf))
    for r in range(n_csf):
        for s in range(r, n_csf):
            h_rs = _ci_matrix_element(csfs[r], csfs[s], orbitals_dict,
                                       orbital_basis, bsp_data, Z, ns,
                                       is_diagonal=(r == s))
            H[r, s] = h_rs
            if r != s:
                H[s, r] = h_rs
    return H


def _ci_matrix_element(csf_r, csf_s, orbitals_dict, orbital_basis,
                        bsp_data, Z, ns, *, is_diagonal=False):
    n_diff = csf_r.n_electron_diff(csf_s)
    if n_diff == 0:
        # Subshell occupations match.  When ``is_diagonal`` we are on
        # the matrix diagonal (r == s) and apply the J-coupled
        # correction (Grant §8.3) so distinct (J, parity) CSFs receive
        # different physically meaningful diagonal energies.
        #
        # Two distinct CSFs with identical subshell lists but different
        # ``total_J2`` (parentage-only difference) reuse the
        # configuration-averaged kernel here; their cross-coupling is
        # then zeroed by the block filter in ``methods/ci.py`` when
        # ``(J, parity)`` differs.  Two distinct CSFs sharing both
        # subshells and ``(J, parity)`` do not appear in any
        # single-configuration Fe XVII basis and are left for the
        # general parentage-aware treatment.
        if is_diagonal:
            return _diagonal_with_J(csf_r, orbitals_dict, bsp_data, Z, ns)
        return _diagonal_element(csf_r, orbitals_dict, bsp_data, Z, ns)
    elif n_diff == 1:
        return _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns)
    elif n_diff == 2:
        return _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data)
    return 0.0


def _diagonal_with_J(csf, orbitals_dict, bsp_data, Z, ns):
    """Diagonal CI element with J-coupled angular correction.

    For CSFs with explicit ``total_J2`` and inferable intermediate
    coupling — specifically those with at most two open subshells where
    each open subshell carries a unique parentage (single electron or
    single hole) — adds a J-dependent correction on top of the
    configuration-averaged kernel so different (J, parity) members of
    the same configuration acquire physically meaningful splittings.

    Closed-shell or non-J-resolved CSFs are returned unchanged
    (correction is zero), preserving backwards compatibility.
    """
    base = _diagonal_element(csf, orbitals_dict, bsp_data, Z, ns)
    return base + _diagonal_J_correction(csf, orbitals_dict, bsp_data, ns)


def _get_orb(key, orbitals_dict, ns):
    """Get orbital (P, Q) or zeros if not found."""
    if key in orbitals_dict:
        return orbitals_dict[key]
    return np.zeros(ns), np.zeros(ns)


def _diagonal_element(csf, orbitals_dict, bsp_data, Z, ns):
    """Diagonal: E = Σ_a q_a <a|h|a> + (1/2) Σ_{a,b} [q_a q_b F^0 - exchange]."""
    E_1e = 0.0
    for sh in csf.subshells:
        P, Q = _get_orb((sh.n, sh.kappa), orbitals_dict, ns)
        H0, S0 = build_dirac_matrices(ns, sh.kappa, Z, bsp_data)
        vec = np.concatenate([P, Q])
        norm_sq = vec @ S0 @ vec
        if norm_sq > 1e-20:
            E_1e += sh.occupation * (vec @ H0 @ vec) / norm_sq

    E_J, E_K = 0.0, 0.0
    for sh_a in csf.subshells:
        Pa, Qa = _get_orb((sh_a.n, sh_a.kappa), orbitals_dict, ns)
        tj_a = 2 * abs(sh_a.kappa) - 1
        for sh_b in csf.subshells:
            Pb, Qb = _get_orb((sh_b.n, sh_b.kappa), orbitals_dict, ns)
            tj_b = 2 * abs(sh_b.kappa) - 1

            f0 = compute_slater_fk(Pa, Qa, Pb, Qb, bsp_data, k=0)
            E_J += 0.5 * sh_a.occupation * sh_b.occupation * f0

            for k in exchange_k_range(sh_a.kappa, sh_b.kappa):
                lam = exchange_lambda(sh_a.kappa, sh_b.kappa, k)
                if lam < 1e-15:
                    continue
                gk = compute_slater_gk(Pa, Qa, Pb, Qb, bsp_data, k)
                w_k = (sh_a.occupation * sh_b.occupation
                       / ((tj_a + 1.0) * (tj_b + 1.0)) * lam)
                E_K += 0.5 * w_k * gk

    return E_1e + E_J - E_K


def _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns):
    """1-electron excitation a→b.

    H_rs = sqrt(q) * [<a|h|b> + Σ_c q_c (R^0(ac,bc) - Σ_k c_k R^k(ac,cb))]

    where c runs over common occupied orbitals, and q is the occupation factor.
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 1 or len(gained) != 1:
        return 0.0

    key_a, qa_r, qa_s = lost[0]
    key_b, qb_r, qb_s = gained[0]
    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    kappa_a, kappa_b = key_a[1], key_b[1]

    # Occupation coupling factor
    sqrt_factor = np.sqrt(abs(qa_r - qa_s) * abs(qb_s - qb_r))

    # One-electron: <a|h|b> (nonzero only for same kappa)
    H_1e = 0.0
    if kappa_a == kappa_b:
        H0, S0 = build_dirac_matrices(ns, kappa_a, Z, bsp_data)
        vec_a = np.concatenate([Pa, Qa])
        vec_b = np.concatenate([Pb, Qb])
        H_1e = vec_a @ H0 @ vec_b

    # Two-electron: Σ_c q_c [direct - exchange] with common orbitals
    H_2e = 0.0
    # Common orbitals: those present in BOTH configurations
    occ_r = {(sh.n, sh.kappa): sh.occupation for sh in csf_r.subshells}
    occ_s = {(sh.n, sh.kappa): sh.occupation for sh in csf_s.subshells}

    for sh_c in csf_r.subshells:
        key_c = (sh_c.n, sh_c.kappa)
        # Use the minimum occupation (common electrons)
        q_common = min(occ_r.get(key_c, 0), occ_s.get(key_c, 0))
        if q_common <= 0:
            continue

        Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
        kappa_c = key_c[1]
        tj_c = 2 * abs(kappa_c) - 1

        # Direct: R^0(ac, bc) — Coulomb between "a→b excitation" and orbital c
        rk_direct = compute_slater_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pc, Qc,
                                       bsp_data, k=0)
        H_2e += q_common * rk_direct

        # Exchange: -Σ_k c_k R^k(ac, cb)
        for k in exchange_k_range(kappa_a, kappa_c):
            lam_ac = exchange_lambda(kappa_a, kappa_c, k)
            if lam_ac < 1e-15:
                continue
            # R^k(ac, cb) — exchange integral
            rk_exch = compute_slater_rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb,
                                         bsp_data, k)
            c_k = q_common / (tj_c + 1.0) * lam_ac
            H_2e -= c_k * rk_exch

    return sqrt_factor * (H_1e + H_2e)


def _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data):
    """2-electron excitation (a,b→c,d).

    H_rs = Σ_k [direct R^k(ac,bd) - exchange R^k(ac,db)]
    weighted by angular coupling coefficients.
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 2 or len(gained) != 2:
        return 0.0

    ns = len(list(orbitals_dict.values())[0][0]) if orbitals_dict else 50

    key_a, _, _ = lost[0]
    key_b, _, _ = lost[1]
    key_c, _, _ = gained[0]
    key_d, _, _ = gained[1]

    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
    Pd, Qd = _get_orb(key_d, orbitals_dict, ns)

    kappa_a, kappa_b = key_a[1], key_b[1]
    kappa_c, kappa_d = key_c[1], key_d[1]

    H_2e = 0.0

    # Direct contribution: R^0(ac, bd) (monopole)
    rk_direct = compute_slater_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd,
                                   bsp_data, k=0)
    H_2e += rk_direct

    # Exchange contribution: -Σ_k c_k R^k(ac, db)
    for k in exchange_k_range(kappa_a, kappa_d):
        lam = exchange_lambda(kappa_a, kappa_d, k)
        if lam < 1e-15:
            continue
        rk_exch = compute_slater_rk(Pa, Qa, Pc, Qc, Pd, Qd, Pb, Qb,
                                     bsp_data, k)
        tj_a = 2 * abs(kappa_a) - 1
        tj_d = 2 * abs(kappa_d) - 1
        c_k = lam / ((tj_a + 1.0) * (tj_d + 1.0))
        H_2e -= c_k * rk_exch

    return H_2e


def _diagonal_J_correction(csf, orbitals_dict, bsp_data, ns):
    """J-dependent diagonal correction for two-open-subshell CSFs.

    The configuration-averaged exchange coefficient between two open
    subshells (a, b) in :func:`_diagonal_element` is
    ``q_a q_b /(g_a g_b) * λ^K``.  The proper Grant §8.3 J-coupled
    coefficient for two electrons (or, by particle-hole symmetry, two
    holes / one of each) with unique parentage ``J_a = j_a`` and
    ``J_b = j_b`` coupled to total ``J`` is

        g^K(J) = (-1)^{j_a + j_b + J + K}
                 * λ^K_{ab} * {j_a j_b J; j_b j_a K}

    where ``λ^K_{ab} = (2j_a+1)(2j_b+1) (j_a K j_b; -1/2 0 1/2)^2``.
    The correction returned here is the difference times the Slater
    integral ``G^K(a,b)``, summed over allowed multipoles ``K``, with
    a sign that matches the existing kernel's ``-E_K`` convention.

    For a single hole on subshell ``a`` (``q_a = g_a - 1``) the
    particle-hole-conjugated decomposition gives
    ``ΔE = G^K * (g^K(J) - λ^K/g_a)`` (and the symmetric form for
    holes on ``b``) instead of the electron-electron expression
    ``G^K * (g^K(J) - λ^K/(g_a g_b))``.

    Returns ``0.0`` for closed-shell-only CSFs (no open subshells),
    for CSFs without explicit ``total_J2`` metadata, and for CSFs with
    more than two open subshells (the general case requires a full
    parentage chain that is not inferable from a bare ``CSF`` object
    today — those continue to use the configuration-averaged kernel
    until parentage labels are threaded through the CSF generator).
    """
    if not _has_explicit_J(csf):
        return 0.0

    open_subs = _open_subshells(csf)
    if len(open_subs) != 2:
        return 0.0

    sh_a, sh_b = open_subs
    qa = int(round(sh_a.occupation))
    qb = int(round(sh_b.occupation))
    cap_a = 2 * abs(sh_a.kappa)   # 2j_a + 1 (subshell capacity)
    cap_b = 2 * abs(sh_b.kappa)
    tj_a = cap_a - 1              # 2 j_a
    tj_b = cap_b - 1              # 2 j_b

    # Each open subshell must be a single electron (q=1) or a single
    # hole (q=2j) so that its parentage J = j_subshell is unique.
    is_elec_a = qa == 1
    is_hole_a = qa == cap_a - 1
    is_elec_b = qb == 1
    is_hole_b = qb == cap_b - 1
    if not ((is_elec_a or is_hole_a) and (is_elec_b or is_hole_b)):
        return 0.0
    # q=1 and q=cap-1 can collide for cap=2 (j=1/2 subshells with one
    # electron).  In that case both flags are true and the
    # electron-electron prefactor below is the correct one.
    if is_elec_a and is_hole_a:
        is_hole_a = False
    if is_elec_b and is_hole_b:
        is_hole_b = False

    Pa, Qa_orb = _get_orb((sh_a.n, sh_a.kappa), orbitals_dict, ns)
    Pb, Qb_orb = _get_orb((sh_b.n, sh_b.kappa), orbitals_dict, ns)

    total_J2 = int(csf.total_J2)
    delta = 0.0
    for k in exchange_k_range(sh_a.kappa, sh_b.kappa):
        lam = exchange_lambda(sh_a.kappa, sh_b.kappa, k)
        if lam < 1e-15:
            continue
        sixj = wigner_6j(tj_a, tj_b, total_J2, tj_b, tj_a, 2 * k)
        if abs(sixj) < 1e-15:
            g_K_J = 0.0
        else:
            exp_int = (tj_a + tj_b + total_J2) // 2 + k
            phase = -1.0 if exp_int % 2 else 1.0
            g_K_J = phase * sixj * lam

        if is_elec_a and is_elec_b:
            avg_coef = lam / (cap_a * cap_b)
        elif is_hole_a and is_elec_b:
            avg_coef = lam / cap_a
        elif is_elec_a and is_hole_b:
            avg_coef = lam / cap_b
        else:  # both holes — closed-minus-two-electrons picture
            avg_coef = lam * (1.0 - 1.0 / cap_a - 1.0 / cap_b
                              + 1.0 / (cap_a * cap_b))

        # Final energy uses (- exchange).  The existing kernel
        # subtracts ``avg_coef * G^K`` from the diagonal; the proper
        # J-coupled subtraction is ``g_K_J * G^K``.  Net correction to
        # ADD to the existing diagonal energy is therefore
        # ``(avg_coef - g_K_J) * G^K``.
        gk_val = compute_slater_gk(Pa, Qa_orb, Pb, Qb_orb, bsp_data, k)
        delta += (avg_coef - g_K_J) * gk_val

    return float(delta)


def _has_explicit_J(csf):
    """Whether a CSF carries explicit J-resolution metadata."""
    if int(getattr(csf, "total_J2", 0)) > 0:
        return True
    return " J=" in str(getattr(csf, "label", ""))


def _open_subshells(csf):
    """Return open subshells (0 < occupation < capacity) of a CSF."""
    open_subs = []
    for sh in csf.subshells:
        cap = 2 * abs(sh.kappa)
        q = int(round(sh.occupation))
        if 0 < q < cap:
            open_subs.append(sh)
    return open_subs
