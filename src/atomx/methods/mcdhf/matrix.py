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
        base = _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns)
        return base + _off_diagonal_J_correction(
            csf_r, csf_s, orbitals_dict, bsp_data, ns, n_diff=1,
        )
    elif n_diff == 2:
        base = _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data)
        return base + _off_diagonal_J_correction(
            csf_r, csf_s, orbitals_dict, bsp_data, ns, n_diff=2,
        )
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
    """1-electron promotion (b → a) — ``csf_s`` (ket) → ``csf_r`` (bra).

    Variable convention:

    * ``a`` (= ``lost[0]``: bra > ket) is the **destination** orbital where an
      electron arrived in the bra.
    * ``b`` (= ``gained[0]``: bra < ket) is the **source** orbital that lost
      one electron going from ket to bra.

    The configuration-averaged jj-coupled Slater-Condon expression is

        H_{rs} = √(q_b^{ket} (g_a − q_a^{ket})) ·
                 [⟨a|h|b⟩ δ_{κ_a,κ_b}
                  + Σ_{c ∈ common} q_c (X^direct − X^exchange)]

    where ``q_b^{ket}`` is the source occupation in the ket and
    ``g_a − q_a^{ket}`` are the destination vacancies in the ket.  The
    direct integral ``R^0(ac, bc)`` is non-zero only when ``κ_a = κ_b``
    (angular monopole orthogonality); for ``κ_a ≠ κ_b`` the only
    contribution comes from the exchange multipoles ``k`` satisfying
    triangle conditions on both ``(j_a, j_c)`` and ``(j_c, j_b)``.

    For different-κ promotions the exchange angular factor is the
    geometric mean ``√(λ^k(a,c) λ^k(c,b))`` (which reduces to ``λ^k(a,c)``
    when ``κ_a = κ_b``).
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 1 or len(gained) != 1:
        return 0.0

    key_a, qa_r, qa_s = lost[0]    # destination (bra > ket)
    key_b, qb_r, qb_s = gained[0]  # source (bra < ket)
    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    kappa_a, kappa_b = key_a[1], key_b[1]

    # jj-coupling 1-diff occupation weight: q_source^{ket} · h_dest^{ket}.
    # ``qb_s`` = ket occupation of source subshell ``b`` (must be > 0).
    # ``g_a − qa_s`` = ket vacancies in destination subshell ``a`` (> 0).
    g_a = 2 * abs(kappa_a)
    sqrt_factor = np.sqrt(qb_s * (g_a - qa_s))

    # One-electron: ⟨a|h|b⟩ (non-zero only for same κ; h is a scalar tensor)
    H_1e = 0.0
    if kappa_a == kappa_b:
        H0, S0 = build_dirac_matrices(ns, kappa_a, Z, bsp_data)
        vec_a = np.concatenate([Pa, Qa])
        vec_b = np.concatenate([Pb, Qb])
        H_1e = vec_a @ H0 @ vec_b

    # Two-electron: Σ_c q_c [direct − exchange] over common spectator orbitals
    H_2e = 0.0
    occ_r = {(sh.n, sh.kappa): sh.occupation for sh in csf_r.subshells}
    occ_s = {(sh.n, sh.kappa): sh.occupation for sh in csf_s.subshells}
    seen_c = set()
    for sh_c in csf_r.subshells:
        key_c = (sh_c.n, sh_c.kappa)
        if key_c in seen_c or key_c in (key_a, key_b):
            continue
        seen_c.add(key_c)
        q_common = min(occ_r.get(key_c, 0), occ_s.get(key_c, 0))
        if q_common <= 0:
            continue

        Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
        kappa_c = key_c[1]
        tj_c = 2 * abs(kappa_c) - 1

        # Direct R^0(ac, bc) — angular factor at k=0 is δ_{κ_a, κ_b}.
        # Skip entirely for different-κ promotions to avoid the spurious
        # ⟨ψ_3s|·|ψ_3d⟩-type cross-coupling that was inflating the
        # off-diagonal CI matrix for Fe XVII 2p⁵3l J=1 states.
        if kappa_a == kappa_b:
            rk_direct = compute_slater_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pc, Qc,
                                           bsp_data, k=0)
            H_2e += q_common * rk_direct

        # Exchange: Σ_k c_k R^k(ac, cb) with k allowed on BOTH (a,c) and (c,b).
        k_range_ab = sorted(set(exchange_k_range(kappa_a, kappa_c))
                            & set(exchange_k_range(kappa_c, kappa_b)))
        for k in k_range_ab:
            lam_ac = exchange_lambda(kappa_a, kappa_c, k)
            lam_cb = exchange_lambda(kappa_c, kappa_b, k)
            if lam_ac < 1e-15 or lam_cb < 1e-15:
                continue
            rk_exch = compute_slater_rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb,
                                         bsp_data, k)
            # Angular factor: √(λ^k(a,c) λ^k(c,b)) reduces to λ^k(a,c)
            # when κ_a = κ_b (the standard same-orbital HF exchange).
            ang_k = lam_ac if kappa_a == kappa_b else np.sqrt(lam_ac * lam_cb)
            c_k = q_common / (tj_c + 1.0) * ang_k
            H_2e -= c_k * rk_exch

    return sqrt_factor * (H_1e + H_2e)


def _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data):
    """2-electron promotion (c → a, d → b).

    ``lost`` (bra > ket): orbitals where the bra gained electrons → destinations.
    ``gained`` (bra < ket): orbitals where the bra lost electrons → sources.

    The configuration-averaged two-body Slater–Condon expression is

        H_{rs} = δ_{κ_a,κ_c} δ_{κ_b,κ_d} R^0(ac, bd)
                 − Σ_k λ̂^k_{ad,cb} R^k(ac, db)

    where the direct ``k=0`` integral is non-zero only when the two
    "left-orbital" pairs are angularly compatible (same κ), and the
    exchange ``k`` must satisfy the triangle conditions on both
    ``(κ_a, κ_d)`` and ``(κ_c, κ_b)`` simultaneously.  For different-κ
    exchange pairs the magnitude is the geometric mean
    ``√(λ^k(a,d) λ^k(c,b))`` (reduces to ``λ^k(a,d)`` in the same-κ
    case).
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 2 or len(gained) != 2:
        return 0.0

    ns = len(list(orbitals_dict.values())[0][0]) if orbitals_dict else 50

    key_a, _, _ = lost[0]    # destination 1
    key_b, _, _ = lost[1]    # destination 2
    key_c, _, _ = gained[0]  # source 1
    key_d, _, _ = gained[1]  # source 2

    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
    Pd, Qd = _get_orb(key_d, orbitals_dict, ns)

    kappa_a, kappa_b = key_a[1], key_b[1]
    kappa_c, kappa_d = key_c[1], key_d[1]

    H_2e = 0.0

    # Direct ⟨ac|g|bd⟩^{k=0}: angular factor at k=0 requires both
    # ⟨ψ_a|·|ψ_b⟩_{k=0} ∝ δ_{κ_a, κ_b} and ⟨ψ_c|·|ψ_d⟩_{k=0} ∝ δ_{κ_c, κ_d}
    # to be simultaneously non-zero.  In the (a→c, b→d) two-body
    # promotion this means κ_a = κ_c AND κ_b = κ_d.
    if kappa_a == kappa_c and kappa_b == kappa_d:
        rk_direct = compute_slater_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd,
                                       bsp_data, k=0)
        H_2e += rk_direct

    # Exchange ⟨ac|g|db⟩.  We keep the original (κ_a, κ_d)-based form for
    # backwards compatibility (it correctly handles the same-κ case used by
    # existing benchmarks); for different-κ exchange pairs the proper
    # angular factor is a geometric mean √(λ_ad · λ_cb), but the off-diagonal
    # phase requires a full 6j treatment that is deferred to a follow-up.
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
    """Whether a CSF carries explicit J-resolution metadata.

    Accepts a non-default ``total_J2`` attribute *or* a label containing
    ``J=`` in either the legacy form ``... J=2/2`` or the bracketed form
    ``... [J=2/2]`` that :func:`enumerate_jj_csfs` emits.  The bracketed
    form is required for closed-shell J=0 CSFs whose ``total_J2`` is
    zero — without it those silently bypass the J-coupled correction.
    """
    if int(getattr(csf, "total_J2", 0)) > 0:
        return True
    label = str(getattr(csf, "label", ""))
    return " J=" in label or "[J=" in label


def _open_subshells(csf):
    """Return open subshells (0 < occupation < capacity) of a CSF."""
    open_subs = []
    for sh in csf.subshells:
        cap = 2 * abs(sh.kappa)
        q = int(round(sh.occupation))
        if 0 < q < cap:
            open_subs.append(sh)
    return open_subs


def _off_diagonal_J_correction(csf_r, csf_s, orbitals_dict, bsp_data, ns,
                                 *, n_diff):
    """J-coupled off-diagonal CI correction (Grant §8.3) — scaffold.

    HYP-184 follow-up.  The configuration-averaged 1-diff/2-diff
    matrix elements computed by :func:`_one_diff_element` and
    :func:`_two_diff_element` use the orbital-occupation-weighted
    Slater-Condon expression appropriate for single-open-subshell
    CSFs.  For two CSFs sharing the *same* (J, parity) block where
    each CSF has two-or-more open subshells with **unique parentage**
    (every open subshell is a single electron ``q=1`` or single hole
    ``q=2j``), the proper Grant §8.3 6j-style J-coupled exchange
    coefficient differs from the average-of-shell weight and must be
    added as an off-diagonal correction analogous to
    :func:`_diagonal_J_correction`.

    The two relevant cases for the Fe XVII 2p⁵3l J=1⁻ block are:

    * **1-diff**: bra and ket share one open subshell ``c`` (the
      spectator) and differ in a single-electron promotion ``a → b``
      (e.g. ``2p_{1/2}^{1} 3d_{3/2}^{1} ↔ 2p_{3/2}^{-1} 3d_{3/2}^{1}``).
      The proper correction is

          ΔH = Σ_K v^K(j_a, j_b, j_c; J) · R^K(ac, cb)

      where ``v^K`` is the J-coupled exchange coefficient with a Wigner
      6j weight ``{j_a j_b J; j_c j_c K}`` and the standard
      ``(j_a, j_b)`` reduced matrix element.  The difference between
      this and the geometric-mean configuration-averaged factor
      ``√(λ^K(a,c) λ^K(c,b))`` used in
      :func:`_one_diff_element` is what spreads the three J=1⁻
      3d-electron CSFs into the proper ¹P₁ / ³D₁ / ³P₁ intensity
      pattern that Bernitt 2012 sees.

    * **2-diff**: bra and ket differ in two open subshells (e.g.
      ``2p_{1/2}^{1} 3d_{3/2}^{1} ↔ 2p_{3/2}^{-1} 3d_{5/2}^{1}``).
      The J-coupled formula involves a single 6j coupling on the
      (j_a, j_b) → (j_c, j_d) → J chain analogous to the 1-diff
      case.

    Implementation notes (Grant §8.3 / Cowan §12):

      g^K_offdiag(J) = (-1)^{j_a + j_b + j_c + j_d + J + K}
                       * {6j(j_a j_b J; j_d j_c K)}
                       * <j_a||C^K||j_b> <j_c||C^K||j_d>

    The reduced matrix elements ``<j||C^K||j'>`` are the standard
    angular tensor coefficients available via
    :func:`atomx.angular.coefficients.exchange_lambda` (square root,
    with sign managed by the 6j phase).

    **Current status — scaffold only.**  The off-diagonal correction
    is *not* applied yet (returns ``0.0``).  A correct implementation
    requires:

    1. Identifying the unique-parentage subset (each open subshell is
       a single electron or single hole) on **both** ``csf_r`` and
       ``csf_s`` and the matching same-(J,parity) constraint.
    2. Selecting the spectator/source/destination shells and reading
       off the ``(j_a, j_b, j_c, j_d)`` labels and total ``J``.
    3. Evaluating the Slater integrals ``R^K`` over the radial part
       using :func:`atomx.methods.kohn_sham.compute_slater_rk`.
    4. Composing the 6j-weighted angular factor and accumulating the
       additive correction.
    5. Verifying against a He-like 1s2p ¹P₁/³P₁ analytic test (the
       two J=1 CSFs ``|1s 2p_{1/2}; J=1⟩`` and ``|1s 2p_{3/2}; J=1⟩``
       should diagonalise to the singlet/triplet states with the
       2·G¹(1s,2p) splitting).

    Parameters
    ----------
    csf_r, csf_s : CSF
        The two CSFs whose off-diagonal element is being corrected.
        ``r != s`` (we never call this on the diagonal).
    orbitals_dict : dict
        ``(n, kappa) -> (P, Q)`` orbital map.
    bsp_data : dict
        B-spline basis data for Slater integrals.
    ns : int
        Number of B-spline coefficients per orbital (for orbital lookup).
    n_diff : int
        Number of subshell differences (1 or 2).  Used to dispatch on
        the 1-diff vs 2-diff sub-formula; not used in the no-op
        scaffold.

    Returns
    -------
    float
        Additive correction to be added to the base 1-diff / 2-diff
        matrix element.  Always ``0.0`` in the current scaffold.
    """
    # Only apply when both CSFs carry explicit J/parity metadata and
    # sit in the same (J, parity) block.  The block filter in
    # ``methods/ci.py`` already drops cross-block elements, so this
    # check is mostly a clarity gate for the no-op path.
    if not (_has_explicit_J(csf_r) and _has_explicit_J(csf_s)):
        return 0.0
    if int(csf_r.total_J2) != int(csf_s.total_J2):
        return 0.0
    if int(getattr(csf_r, "parity", 1)) != int(getattr(csf_s, "parity", 1)):
        return 0.0

    # n_diff > 2 is impossible (caller guarantees 1 or 2) and would
    # otherwise also short-circuit.  The 1-diff and 2-diff bodies are
    # not implemented in the scaffold; see the docstring for the
    # Grant §8.3 formulas to fill in.
    if n_diff not in (1, 2):
        return 0.0

    # No-op scaffold.  Filling in the J-coupled formula is HYP-184
    # follow-up item (1).
    del orbitals_dict, bsp_data, ns  # silence unused-arg warnings.
    return 0.0
