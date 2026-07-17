"""Non-local Dirac-Hartree-Fock on the ODE shooting engine.

Solves the jj-averaged Dirac-Hartree-Fock equations with the classic
Froese-Fischer / Hartree *inhomogeneous-equation* approach, relativistic
(Dirac) version:

For each occupied orbital ``a`` the DHF equation is the
integro-differential eigenproblem

    (h_D + V_loc,a(r) - eps_a) [P_a, Q_a] = chi_a(r)

where the LOCAL part collects the nuclear potential, the direct
(Hartree) potential of all shells in the (N-1)-electron convention, and
the same-shell k>0 multipole structure:

    V_loc,a = V_nuc + sum_b q_b Y^0_bb/r - Y^0_aa/r
              - (q_a - 1) sum_{k>=2} Lambda_k(a,a)/(2j_a(2j_a+1)) Y^k_aa/r

and the NON-LOCAL exchange with all other shells is treated as an
inhomogeneous source built from the previous iteration's orbitals:

    chi_a(r) = sum_{b != a} sum_k  q_b Lambda_k(a,b) /
               ((2j_a+1)(2j_b+1)) * Y^k_ab(r)/r * [P_b, Q_b](r)

The angular factors ``Lambda_k`` are the SAME jj-average coefficients
the B-spline ``DFSolver`` uses (``atomx.angular.coefficients
.exchange_lambda`` — DBSR_HF ``av_energy_coef`` with the doubled
same-shell direct term), so both engines solve the *same physics* with
different radial numerics.

Radial solution of the inhomogeneous boundary-value problem
-----------------------------------------------------------
Variation of parameters (Green's function) with the two homogeneous
solutions of ``(h_D + V_loc,a - eps)``:

* ``y_0``   — regular at the origin, integrated OUTWARD with the
  existing Salvat power-series integrator (:func:`doutw`);
* ``y_inf`` — decaying at infinity, integrated INWARD from the
  practical infinity (:func:`dinw`).

For the first-order Dirac system the "Wronskian"
``W = P_0 Q_inf - Q_0 P_inf`` is exactly r-independent
(the system matrix is traceless), and the unique solution satisfying
both boundary conditions is

    phi(r) = -1/(c W) * [ y_0(r)   * int_r^inf (y_inf . chi) dr'
                        + y_inf(r) * int_0^r   (y_0   . chi) dr' ]

with ``(y . chi) = P y_P chi_P + Q y_Q chi_Q`` and ``c`` the speed of
light.  The eigenvalue ``eps_a`` is fixed by the NORMALISATION
condition ``<phi|phi> = 1`` via Newton iteration: since
``d phi/d eps = (h + V_loc - eps)^{-1} phi``, the derivative
``dN/d eps = 2 <phi| (h+V-eps)^{-1} |phi>`` is one extra Green solve
reusing the same homogeneous pair.  A node-count guard (Sturm) keeps
``eps`` bracketed between the neighbouring homogeneous eigenvalues.

Off-diagonal Lagrange multipliers between same-kappa shells are handled
by Schmidt orthogonalisation each sweep — exact for closed shells
(where the multipliers can be gauged away), a documented approximation
for open shells.

Validation: He, Ne, Fe XVII total energies agree with the B-spline
``DFSolver`` (``Atom(numerics='bspline', exchange='hf')``) to <= 1e-4
relative and orbital energies to <= 5e-4 relative
(``tests/test_shooting_hf.py``).
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.spline import splin0
from atomx.numerics.shooting.quadrature import slag6
from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.dirac_integrator import doutw, dinw
from atomx.numerics.shooting.solver import (
    ShootingSolver,
    ShootingResult,
    _yk_over_r,
    _slater_integral_slag6,
)

#: Power-series ODE tolerance for the homogeneous integrations —
#: matches the internal DBOUND tolerance used by the local-exchange SCF.
_EPS_ODE = 1.0e-11
#: "Disable" sentinels for doutw's node-based early exits.
_BIG = 10 ** 9


def _l_of_kappa(kappa: int) -> int:
    return -kappa - 1 if kappa < 0 else kappa


def _count_nodes(p: np.ndarray, ncut: int) -> int:
    """Sign changes of P on (0, r_cut), counting only O(1) lobes.

    Used as a wrong-root backstop for the inhomogeneous solve: landing
    on the root of a neighbouring pole produces a solution dominated by
    the wrong eigenfunction, i.e. full-amplitude lobes.  The threshold
    must stay LOOSE (1e-2 of the peak): exact HF orbitals are only
    nodeless/nr-noded in their dominant region — the non-local exchange
    source adds ~1e-4-amplitude sign wiggles in the far tail (canonical
    same-kappa HF orbitals are mutually orthogonal), which a tight
    threshold would miscount as extra nodes (seen for Ar 1s).
    """
    amax = np.max(np.abs(p[:ncut]))
    if amax <= 0.0:
        return 0
    thresh = 1.0e-2 * amax
    nodes = 0
    last_sign = 0
    for i in range(ncut):
        v = p[i]
        if abs(v) < thresh:
            continue
        s = 1 if v > 0.0 else -1
        if last_sign != 0 and s != last_sign:
            nodes += 1
        last_sign = s
    return nodes


def _homogeneous_pair(e, kappa, r, ind, np_, va, vb, vc, vd):
    """Regular/decaying homogeneous Dirac solutions of the local problem.

    Returns ``(p_reg, q_reg, p_irr, q_irr, w_const, ncut, iotp, nzero)``
    where the ``*_reg`` arrays (length ``ncut``) are the regular-at-origin
    solution integrated outward, the ``*_irr`` arrays (length ``np_``,
    nonzero on ``[1, ncut-1]``) decay at infinity, ``w_const`` is the
    r-independent Wronskian ``P_reg Q_irr - Q_reg P_irr`` (arbitrary
    overall scale — consistent with the returned arrays), ``ncut`` is one
    past the practical infinity, ``iotp`` the outer classical turning
    point and ``nzero`` the node count of the regular solution (Sturm
    index: the number of homogeneous eigenvalues below ``e``).
    """
    ak = float(kappa)
    l = _l_of_kappa(kappa)
    fl1 = 0.5 * l * (l + 1)

    # Outer classical turning point (scan from outside in, as in dbound).
    iotp = 1
    for jj in range(2, np_ + 1):
        idx = np_ + 1 - jj
        rn = r[idx]
        if rn < 1.0e-30:
            continue
        j = ind[idx]
        rvn = va[j] + rn * (vb[j] + rn * (vc[j] + rn * vd[j]))
        if e - (rvn + fl1 / rn) / rn > 0.0:
            iotp = idx
            break
    iotp += 1
    iotp = max(2, min(iotp, np_ - 2))

    # Decaying solution: inward from practical infinity down to r[1].
    p_irr = np.zeros(np_)
    q_irr = np.zeros(np_)
    _, ilast = dinw(e, _EPS_ODE, ak, 1, 0,
                    r, ind, np_, va, vb, vc, vd, p_irr, q_irr)
    ncut = min(ilast + 1, np_)

    # Regular solution: outward from the origin to the practical
    # infinity.  nr=_BIG disables the node early-exit and iotp_in=_BIG
    # disables the past-turning-point blow-up stop — beyond the turning
    # point the growing solution is the numerically dominant one, so the
    # outward direction stays stable all the way to r_cut.
    p_reg = np.zeros(ncut)
    q_reg = np.zeros(ncut)
    nzero, _, _, _ = doutw(e, _EPS_ODE, ak, _BIG, _BIG, 0,
                           r[:ncut], ind[:ncut], ncut,
                           va, vb, vc, vd, p_reg, q_reg)

    # Wronskian: exactly constant in exact arithmetic; evaluate near the
    # turning point where both solutions are well conditioned (median
    # over a small window for robustness against local roundoff).
    lo = max(1, iotp - 3)
    hi = min(ncut - 1, iotp + 4)
    if hi <= lo:
        lo, hi = max(1, ncut // 2), max(2, ncut // 2 + 1)
    w_win = (p_reg[lo:hi] * q_irr[lo:hi] - q_reg[lo:hi] * p_irr[lo:hi])
    w_const = float(np.median(w_win))

    return p_reg, q_reg, p_irr, q_irr, w_const, ncut, iotp, nzero


def _green_solve(chi_p, chi_q, p_reg, q_reg, p_irr, q_irr,
                 w_const, difr, ncut):
    """Particular solution of ``(h_D + V_loc - eps) phi = chi``.

    Variation-of-parameters assembly on the grid (see module docstring).
    ``chi_p/chi_q`` are full-length arrays; the result is returned on
    ``[0, ncut)`` and is zero at index 0 (P(0) = Q(0) = 0).
    """
    u = (p_reg * chi_p[:ncut] + q_reg * chi_q[:ncut]) * difr[:ncut]
    v = (p_irr[:ncut] * chi_p[:ncut] + q_irr[:ncut] * chi_q[:ncut]) \
        * difr[:ncut]
    cum_u = slag6(1.0, u, ncut)
    cum_v = slag6(1.0, v, ncut)
    v_tail = cum_v[ncut - 1] - cum_v
    fac = -1.0 / (SL * w_const)
    p = fac * (p_reg * v_tail + p_irr[:ncut] * cum_u)
    q = fac * (q_reg * v_tail + q_irr[:ncut] * cum_u)
    return p, q


class _OrbitalSolveError(RuntimeError):
    pass


def _solve_orbital_hf(e0, n, kappa, rv_loc, chi_p, chi_q,
                      r, ind, difr, np_,
                      tol_e=1.0e-10, max_newton=60):
    """Solve one inhomogeneous DHF orbital equation for ``(P, Q, eps)``.

    Newton iteration on the normalisation ``N(eps) = 1`` with a Sturm
    node-count bracket.  Falls back to the plain homogeneous eigenvalue
    solver (:func:`dbound`) when the exchange source vanishes
    (single-shell systems like He, hydrogen-like ions).
    """
    l = _l_of_kappa(kappa)
    nr = n - l - 1  # required radial node count

    va, vb, vc, vd = splin0(r, rv_loc, np_, 0.0, 0.0)

    chi_scale = float(np.max(np.abs(chi_p)) + np.max(np.abs(chi_q)))
    if chi_scale < 1.0e-12:
        # No exchange source: ordinary local eigenproblem.
        e_val, ier, p_sol, q_sol, _ = dbound(
            float(min(e0, -1.0e-4)), 1.0e-11, int(n), int(kappa),
            r, rv_loc, va, vb, vc, vd, np_, r, np_)
        if ier != 0:
            raise _OrbitalSolveError(
                f"dbound failed for n={n} kappa={kappa}: ier={ier}")
        norm = slag6(1.0, (p_sol ** 2 + q_sol ** 2) * difr, np_)[np_ - 1]
        fn = 1.0 / math.sqrt(norm) if norm > 1.0e-30 else 1.0
        return p_sol * fn, q_sol * fn, float(e_val)

    e = float(min(e0, -1.0e-4))
    e_lo = -SL * SL          # deep floor (as in dbound)
    e_hi = -1.0e-9           # bound states only
    p = q = None
    ncut = np_
    converged = False

    for _ in range(max_newton):
        (p_reg, q_reg, p_irr, q_irr, w_const,
         ncut, iotp, nzero) = _homogeneous_pair(
            e, kappa, r, ind, np_, va, vb, vc, vd)

        # Sturm bracket: nzero == number of homogeneous eigenvalues
        # below e; the physical root sits just below pole #nr+1, i.e.
        # inside the window where nzero == nr.
        if nzero > nr:
            e_hi = min(e_hi, e)
            e = 0.5 * (e_lo + e) if e_lo > -SL * SL + 1.0 else 1.5 * e
            e = max(e, -SL * SL)
            continue
        if nzero < nr:
            e_lo = max(e_lo, e)
            e = 0.5 * (e + e_hi)
            continue

        p, q = _green_solve(chi_p, chi_q, p_reg, q_reg, p_irr, q_irr,
                            w_const, difr, ncut)
        nrm = slag6(1.0, (p * p + q * q) * difr[:ncut], ncut)[ncut - 1]

        # dN/de via one extra Green solve with the solution as source.
        pd, qd = _green_solve(
            np.concatenate((p, np.zeros(np_ - ncut))),
            np.concatenate((q, np.zeros(np_ - ncut))),
            p_reg, q_reg, p_irr, q_irr, w_const, difr, ncut)
        dn = 2.0 * slag6(1.0, (p * pd + q * qd) * difr[:ncut],
                         ncut)[ncut - 1]

        if dn > 0.0:
            de = (1.0 - nrm) / dn
        else:
            # Off the physical (lower) branch of the pole — take a
            # bounded step guided by the normalisation defect.
            de = 0.1 * abs(e) * (1.0 if nrm < 1.0 else -1.0)

        # Bounded, bracketed step.
        max_step = 0.5 * abs(e)
        de = max(-max_step, min(max_step, de))
        e_new = e + de
        if e_new >= e_hi:
            e_new = 0.5 * (e + e_hi)
        if e_new <= e_lo:
            e_new = 0.5 * (e + e_lo)

        if abs(e_new - e) <= tol_e * abs(e):
            e = e_new
            converged = True
            break
        e = e_new

    if p is None:
        raise _OrbitalSolveError(
            f"orbital n={n} kappa={kappa}: node bracket never reached the "
            f"target window (nr={nr})")
    if not converged:
        raise _OrbitalSolveError(
            f"orbital n={n} kappa={kappa}: eps Newton did not converge "
            f"(last e={e:.8e})")

    # Wrong-root guard: the accepted solution must carry the target
    # radial node count.
    if _count_nodes(p, ncut) != nr:
        raise _OrbitalSolveError(
            f"orbital n={n} kappa={kappa}: converged onto a root with "
            f"{_count_nodes(p, ncut)} nodes (want {nr}) at e={e:.8e}")

    nrm = slag6(1.0, (p * p + q * q) * difr[:ncut], ncut)[ncut - 1]
    fn = 1.0 / math.sqrt(nrm) if nrm > 1.0e-30 else 1.0
    p_full = np.zeros(np_)
    q_full = np.zeros(np_)
    p_full[:ncut] = p * fn
    q_full[:ncut] = q * fn
    return p_full, q_full, float(e)


class ShootingHFSolver:
    """jj-averaged Dirac-Hartree-Fock SCF on the ODE shooting engine.

    Parameters
    ----------
    Z : float
        Nuclear charge.
    subshells : list of Subshell
        Occupied (n, kappa, occupation) subshells (jj-averaged
        fractional occupations supported).
    n_grid : int
        Tentative number of Salvat SGRID points (default 1000 — the HF
        Green-function quadratures benefit from the denser grid; the
        local-exchange starter uses the same grid).
    r_max : float
        Outer grid radius in Bohr.
    aw : float
        Atomic weight for the Fermi nuclear model (0 = point nucleus).
    nuclear : str or None
        ``'point'`` (default) or ``'fermi'`` — same convention as
        :class:`ShootingSolver`.
    jj_configs : list of JJConfig, optional
        Multi-configuration jj-coupled configurations with c_j^2 weights
        (the structure ``MCDHFSolver._jj_configs`` builds).  When given,
        the pair coefficients become the Guerra Eq (10) weighted averages
        ``WW(a,b) = Σ_j c_j² n_a^j n_b^j`` (``WW(a,a) = Σ_j c_j² n_a^j
        (n_a^j − 1)``) via :func:`atomx.config.jj_coupling.compute_WW`,
        and the per-electron Fock coefficients are ``WW(a,b)/q_a`` —
        exactly the B-spline ``DFSolver`` convention.  Each config's
        ``subshells`` must align positionally with ``subshells`` (same
        ``(n, kappa)`` order), and every ``q_a`` must be positive (the
        MCDHF driver floors occupations at 0.001).  When ``None``, the
        statistical products below are used, bit-identical to the
        historical behaviour.

    Notes
    -----
    Open shells use the fractional statistical average
    ``W(a,a) = q(q-1)``, ``W(a,b) = q_a q_b`` (the ``averaging=
    'fractional'`` convention of the B-spline solver).  Off-diagonal
    Lagrange multipliers between same-kappa shells are approximated by
    Schmidt orthogonalisation — exact for closed shells.
    """

    def __init__(self, Z, subshells: Sequence, n_grid: int = 1000,
                 r_max: float = 75.0, aw: float = 0.0,
                 nuclear: Optional[str] = None,
                 jj_configs: Optional[List] = None):
        self.Z = Z
        self.subshells = list(subshells)
        self.n_grid = n_grid
        self.r_max = r_max
        self.aw = aw
        self.nuclear = nuclear
        if jj_configs is not None:
            for cfg in jj_configs:
                if (len(cfg.subshells) != len(self.subshells)
                        or any((c.n, c.kappa) != (s.n, s.kappa)
                               for c, s in zip(cfg.subshells,
                                               self.subshells))):
                    raise ValueError(
                        "jj_configs subshells must align positionally "
                        "with the solver subshells (same (n, kappa) "
                        "order) — compute_WW indexes by position")
            for s in self.subshells:
                if float(s.occupation) <= 1.0e-15:
                    raise ValueError(
                        "jj_configs mode divides WW(a,b) by q_a; every "
                        f"subshell needs q > 0 (got {s.label} with "
                        f"q={s.occupation})")
        self.jj_configs = jj_configs

    def _get_WW(self, i, j):
        """WW(i,j) — generalized occupation product (DBSR_HF convention).

        Mirrors ``DFSolver._get_WW``: with ``jj_configs`` the
        multi-configuration weighted average, otherwise the statistical
        products ``q_i (q_i − 1)`` / ``q_i q_j``.
        """
        if self.jj_configs is not None:
            from atomx.config.jj_coupling import compute_WW
            return compute_WW(i, j, self.jj_configs)
        qi = float(self.subshells[i].occupation)
        if i == j:
            return qi * (qi - 1.0)
        return qi * float(self.subshells[j].occupation)

    # ------------------------------------------------------------------
    # angular tables (reuses the DFSolver's jj-average coefficients)
    # ------------------------------------------------------------------
    def _angular_tables(self):
        """Per-electron Fock coefficients (DBSR_HF hf_matrix / q_a form).

        Statistical (``jj_configs=None``): intra-shell ``q_a − 1``,
        cross-shell ``q_b`` — the historical products, kept verbatim so
        the default path stays bit-identical.  Guerra: every energy
        coefficient ``WW`` divided by the target occupation ``q_a``,
        mirroring ``DFSolver._build_direct_potential`` (``WW(i,j)/q_i``)
        and ``_build_hf_matrix_dbsr`` (``hfm /= q_i``).  Statistically
        ``WW(a,a)/q_a = q_a − 1`` and ``WW(a,b)/q_a = q_b``, so the two
        forms agree up to float rounding.

        Returns ``(intra, exch, direct0)``; ``direct0`` is ``None`` in
        statistical mode (the SCF loop then uses the equivalent
        ``Σ_b q_b Y⁰_bb − Y⁰_aa`` fast path) and otherwise the per-target
        k=0 Hartree coefficients ``direct0[a][b] = WW(a,b)/q_a``.
        """
        from atomx.angular.coefficients import (
            exchange_lambda, exchange_k_range)

        nsh = len(self.subshells)
        kappas = [sh.kappa for sh in self.subshells]
        occ = [float(sh.occupation) for sh in self.subshells]
        tj = [2 * abs(k) - 1 for k in kappas]
        guerra = self.jj_configs is not None

        # Same-shell k>0 multipoles (local F^k structure):
        # coefficient -(q_a - 1) * Lambda_k / (2j_a (2j_a + 1)),
        # with q_a - 1 -> WW(a,a)/q_a in Guerra mode.
        intra = []  # per shell: list of (k, coeff)
        for a in range(nsh):
            terms = []
            la = _l_of_kappa(kappas[a])
            wq_aa = self._get_WW(a, a) / occ[a] if guerra else occ[a] - 1.0
            for k in range(2, tj[a] + 1):
                if (k + 2 * la) % 2 != 0:
                    continue
                lam = exchange_lambda(kappas[a], kappas[a], k)
                if lam < 1.0e-15:
                    continue
                terms.append(
                    (k, -wq_aa * lam / (tj[a] * (tj[a] + 1.0))))
            intra.append(terms)

        # Cross-shell exchange multipoles (non-local source):
        # coefficient q_b * Lambda_k / ((2j_a+1)(2j_b+1)),
        # with q_b -> WW(a,b)/q_a in Guerra mode.
        exch = {}  # (a, b) -> list of (k, coeff)
        for a in range(nsh):
            for b in range(nsh):
                if a == b:
                    continue
                wq_ab = (self._get_WW(a, b) / occ[a] if guerra
                         else occ[b])
                terms = []
                for k in exchange_k_range(kappas[a], kappas[b]):
                    lam = exchange_lambda(kappas[a], kappas[b], k)
                    if lam < 1.0e-15:
                        continue
                    terms.append(
                        (k, wq_ab * lam / ((tj[a] + 1.0) * (tj[b] + 1.0))))
                exch[(a, b)] = terms

        # k=0 Hartree coefficients: only needed explicitly in Guerra mode
        # (statistically they collapse to the target-independent
        # Σ_b q_b Y⁰_bb − Y⁰_aa form the SCF loop builds directly).
        direct0 = None
        if guerra:
            direct0 = [[self._get_WW(a, b) / occ[a] for b in range(nsh)]
                       for a in range(nsh)]
        return intra, exch, direct0

    # ------------------------------------------------------------------
    # total energy (DBSR_HF av_energy_coef formula on the shooting grid)
    # ------------------------------------------------------------------
    def _two_electron_energy(self, pa, qa, r, difr, np_):
        """(E_J, E_K): direct and exchange two-electron energies."""
        from atomx.angular.coefficients import (
            exchange_lambda, exchange_k_range)

        nsh = len(self.subshells)
        kappas = [sh.kappa for sh in self.subshells]
        tj = [2 * abs(k) - 1 for k in kappas]

        dens = [pa[s] ** 2 + qa[s] ** 2 for s in range(nsh)]
        e_j = 0.0
        e_k = 0.0
        for a in range(nsh):
            la = _l_of_kappa(kappas[a])
            for b in range(a, nsh):
                if a == b:
                    # WW(a,a)/2 — statistically q(q-1)/2, bit-identical.
                    c = self._get_WW(a, a) / 2.0
                    if abs(c) < 1.0e-15:
                        continue
                    e_j += c * _slater_integral_slag6(
                        dens[a], dens[a], 0, r, difr, np_)
                    for k in range(1, tj[a] + 1):
                        if (k + 2 * la) % 2 != 0:
                            continue
                        lam = exchange_lambda(kappas[a], kappas[a], k)
                        if lam < 1.0e-15:
                            continue
                        fk = _slater_integral_slag6(
                            dens[a], dens[a], k, r, difr, np_)
                        e_k += c * lam / (tj[a] * (tj[a] + 1.0)) * fk
                else:
                    # WW(a,b) — statistically q_a q_b, bit-identical.
                    ww = self._get_WW(a, b)
                    if abs(ww) < 1.0e-15:
                        continue
                    e_j += ww * _slater_integral_slag6(
                        dens[a], dens[b], 0, r, difr, np_)
                    f_ab = pa[a] * pa[b] + qa[a] * qa[b]
                    for k in exchange_k_range(kappas[a], kappas[b]):
                        lam = exchange_lambda(kappas[a], kappas[b], k)
                        if lam < 1.0e-15:
                            continue
                        gk = _slater_integral_slag6(
                            f_ab, f_ab, k, r, difr, np_)
                        e_k += ww * lam / ((tj[a] + 1.0) * (tj[b] + 1.0)) * gk
        return e_j, e_k

    # ------------------------------------------------------------------
    # SCF driver
    # ------------------------------------------------------------------
    def solve(self, max_iter: int = 60, tol: float = 1.0e-7,
              damping: float = 0.5) -> ShootingResult:
        """Run the DHF SCF.

        Parameters
        ----------
        max_iter : int
            Maximum HF sweeps (each sweep re-solves every orbital with
            exchange sources rebuilt Gauss-Seidel style).
        tol : float
            Convergence threshold on the RELATIVE total-energy change
            between sweeps (capped at 1e-7 — looser values would leak
            SCF error into the cross-engine validation gates).
        damping : float
            Old-orbital fraction in the update mix (0.5 = directed
            default; 0 = full replacement).
        """
        nsh = len(self.subshells)
        occ = np.array([float(sh.occupation) for sh in self.subshells])
        kappas = [sh.kappa for sh in self.subshells]
        ns = [sh.n for sh in self.subshells]

        # 1. Local-exchange starter (Gu alpha_xc — the closest local
        #    model to HF) on the SAME grid.
        base = ShootingSolver(
            self.Z, self.subshells, n_grid=self.n_grid, r_max=self.r_max,
            exchange='gu', aw=self.aw, nuclear=self.nuclear,
        ).solve(max_iter=300, tol=1.0e-9)
        r = base._grid
        difr = base._difr
        np_ = len(r)
        pa = [base._pa[s].copy() for s in range(nsh)]
        qa = [base._qa[s].copy() for s in range(nsh)]
        ev = np.array([base.orbital_energies[sh.label]
                       for sh in self.subshells])
        ind = np.arange(np_, dtype=np.int64)

        # Nuclear potential in r*V form.
        rvnuc = self._nuclear_rv(r, difr, np_)

        intra, exch, direct0 = self._angular_tables()
        # kappa groups for Schmidt orthogonalisation (ascending n).
        groups: Dict[int, List[int]] = {}
        for s in range(nsh):
            groups.setdefault(kappas[s], []).append(s)
        for g in groups.values():
            g.sort(key=lambda s: ns[s])

        tol_e = min(tol, 1.0e-7)
        e_total_prev = None
        e_total = base.total_energy
        converged = False
        iterations = 0
        first_sweep = True

        for it in range(1, max_iter + 1):
            iterations = it
            for a in range(nsh):
                # --- local potential (direct + same-shell F^k) ---
                dens = [pa[s] ** 2 + qa[s] ** 2 for s in range(nsh)]
                if direct0 is None:
                    vel = np.zeros(np_)
                    for b in range(nsh):
                        vel += occ[b] * _yk_over_r(dens[b], 0, r, difr, np_)
                    v_loc = vel - _yk_over_r(dens[a], 0, r, difr, np_)
                else:
                    # Guerra: per-target k=0 coefficients WW(a,b)/q_a
                    # (statistically q_b, resp. q_a - 1 for b == a).
                    v_loc = np.zeros(np_)
                    for b in range(nsh):
                        v_loc += direct0[a][b] * _yk_over_r(
                            dens[b], 0, r, difr, np_)
                for k, coeff in intra[a]:
                    v_loc += coeff * _yk_over_r(dens[a], k, r, difr, np_)
                rv_loc = rvnuc + r * v_loc

                # --- non-local exchange source ---
                chi_p = np.zeros(np_)
                chi_q = np.zeros(np_)
                for b in range(nsh):
                    if b == a:
                        continue
                    terms = exch[(a, b)]
                    if not terms:
                        continue
                    f_ab = pa[a] * pa[b] + qa[a] * qa[b]
                    for k, coeff in terms:
                        ykab = _yk_over_r(f_ab, k, r, difr, np_)
                        chi_p += coeff * ykab * pa[b]
                        chi_q += coeff * ykab * qa[b]

                # --- seed: first sweep uses the homogeneous eigenvalue
                #     plus the first-order exchange shift ---
                e_seed = ev[a]
                if first_sweep and (np.max(np.abs(chi_p))
                                    + np.max(np.abs(chi_q))) > 1.0e-12:
                    va_, vb_, vc_, vd_ = splin0(r, rv_loc, np_, 0.0, 0.0)
                    e_h, ier, p_h, q_h, _ = dbound(
                        float(min(ev[a], -1.0e-4)), 1.0e-11,
                        int(ns[a]), int(kappas[a]),
                        r, rv_loc, va_, vb_, vc_, vd_, np_, r, np_)
                    if ier == 0:
                        nh = slag6(1.0, (p_h ** 2 + q_h ** 2) * difr,
                                   np_)[np_ - 1]
                        if nh > 1.0e-30:
                            fnh = 1.0 / math.sqrt(nh)
                            shift = slag6(
                                1.0, (p_h * chi_p + q_h * chi_q) * difr,
                                np_)[np_ - 1] * fnh
                            e_seed = e_h - shift

                p_new, q_new, e_new = _solve_orbital_hf(
                    e_seed, ns[a], kappas[a], rv_loc, chi_p, chi_q,
                    r, ind, difr, np_)

                # sign alignment with the previous orbital
                if slag6(1.0, (p_new * pa[a] + q_new * qa[a]) * difr,
                         np_)[np_ - 1] < 0.0:
                    p_new = -p_new
                    q_new = -q_new

                # damped update
                p_mix = damping * pa[a] + (1.0 - damping) * p_new
                q_mix = damping * qa[a] + (1.0 - damping) * q_new

                # Schmidt orthogonalisation against lower same-kappa
                # shells (off-diagonal Lagrange multipliers).
                for b in groups[kappas[a]]:
                    if ns[b] >= ns[a]:
                        break
                    ov = slag6(1.0, (p_mix * pa[b] + q_mix * qa[b]) * difr,
                               np_)[np_ - 1]
                    p_mix -= ov * pa[b]
                    q_mix -= ov * qa[b]

                nrm = slag6(1.0, (p_mix ** 2 + q_mix ** 2) * difr,
                            np_)[np_ - 1]
                fn = 1.0 / math.sqrt(nrm) if nrm > 1.0e-30 else 1.0
                pa[a] = p_mix * fn
                qa[a] = q_mix * fn
                ev[a] = e_new

            first_sweep = False

            # --- total energy: E = sum_a q_a eps_a - E_2e ---
            pa_arr = np.array(pa)
            qa_arr = np.array(qa)
            e_j, e_k = self._two_electron_energy(pa_arr, qa_arr,
                                                 r, difr, np_)
            e_2e = e_j - e_k
            e_total = float(np.dot(occ, ev) - e_2e)

            if e_total_prev is not None:
                if abs(e_total - e_total_prev) <= tol_e * max(
                        1.0, abs(e_total)):
                    converged = True
                    break
            e_total_prev = e_total

        oe = {sh.label: float(ev[i])
              for i, sh in enumerate(self.subshells)}
        pa_arr = np.array(pa)
        qa_arr = np.array(qa)
        e_sum_eps = float(np.dot(occ, ev))
        return ShootingResult(
            converged=converged, iterations=iterations,
            total_energy=e_total, orbital_energies=oe,
            Z=self.Z,
            subshells=list(self.subshells),
            energy_components={
                "E_1e": e_sum_eps - 2.0 * e_2e,
                "E_2e": e_2e,
                "E_J": e_j,
                "E_K": e_k,
            },
            _grid=r,
            _equation="dirac",
            _difr=difr, _pa=pa_arr, _qa=qa_arr,
            _model="hf",
        )

    def _nuclear_rv(self, r, difr, np_):
        """r*V_nuc on the grid: point nucleus or Fermi distribution.

        The Fermi SHAPE parameters (c, a) come from the library-wide
        :func:`atomx.potentials.nuclear.fermi_params` (DBSR_HF
        t = 2.30 fm skin + empirical rms radius) — the SAME nuclear
        model ``DFSolver`` solves via ``make_nuclear_potential``.  The
        Salvat literals (R1 = 1.07 A^{1/3} fm, a = 0.546 fm) that
        ``_dhfs_scf`` keeps for bit-for-bit DHFS reproduction give a
        ~4% smaller rms radius at Z=82; with a ~5.5 Ha finite-size
        shift at Pb that left every HF total 0.35 Ha below the
        B-spline engine on identical Guerra weights (measured
        -20914.1136 vs -20913.7625 on the ci route).  The potential is
        still built by grid-native slag6 integration of the density,
        normalized to Z on this SGRID.
        """
        aw = self.aw
        if self.nuclear is not None and not (aw and aw > 0.0):
            if self.nuclear == "point":
                aw = 0.0
            elif self.nuclear == "fermi":
                from atomx.potentials.nuclear import _get_mass
                aw = float(_get_mass(self.Z))
            else:
                raise ValueError(
                    f"nuclear must be 'point' or 'fermi' "
                    f"(got {self.nuclear!r})")
        z = float(self.Z)
        if aw <= 0.5:
            return np.full(np_, -z)
        from atomx.potentials.nuclear import fermi_params
        a_nuc, c_nuc = fermi_params(self.Z, A=aw)
        arg = np.minimum((c_nuc - r) / a_nuc, 500.0)
        xx = np.where(arg < 500.0, np.exp(arg), 1.0e200)
        denn = xx / (1.0 + xx)
        fwd = slag6(1.0, denn * r ** 2 * difr, np_)
        aux = np.zeros(np_)
        idx = np_ - 1 - np.arange(np_ - 1)
        aux[:np_ - 1] = denn[idx] * r[idx] * difr[idx]
        bwd = slag6(1.0, aux, np_)
        fnorm = z / fwd[np_ - 1] if abs(fwd[np_ - 1]) > 1.0e-30 else z
        return -fnorm * (fwd + bwd[np_ - 1 - np.arange(np_)] * r)
