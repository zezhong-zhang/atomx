"""jj-averaged Hartree–Fock fields and energy on supplied radial arrays.

Occupations and pair moments define the physical functional. Grid creation,
boundary solves, orbital rotations and convergence are owned by numerics.
"""
import numpy as np

from atomx.angular.recoupling import l_of_kappa
from atomx.numerics.shooting.coulomb import _slater_integral_slag6, _yk_over_r
from atomx.numerics.shooting.quadrature import slag6_weights


class RadialHartreeFock:
    """Configuration-average direct/exchange operator and energy contractions."""

    def __init__(self, subshells, jj_configs=None):
        self.subshells = list(subshells)
        if jj_configs is not None:
            for cfg in jj_configs:
                if (len(cfg.subshells) != len(self.subshells)
                        or any((c.n, c.kappa) != (s.n, s.kappa)
                               for c, s in zip(cfg.subshells,
                                               self.subshells, strict=False))):
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

    def get_WW(self, i, j):
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

    def angular_tables(self):
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
        from atomx.angular.coefficients import exchange_k_range, exchange_lambda

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
            la = l_of_kappa(kappas[a])
            wq_aa = self.get_WW(a, a) / occ[a] if guerra else occ[a] - 1.0
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
                wq_ab = (self.get_WW(a, b) / occ[a] if guerra
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
            direct0 = [[self.get_WW(a, b) / occ[a] for b in range(nsh)]
                       for a in range(nsh)]
        return intra, exch, direct0

    def coulomb_energy_terms(self):
        """Yield (exchange, coefficient, rank, a, b, c, d) for R^k(ab,cd).

        The chosen occupation ensemble defines these physical coefficients. Radial
        quadrature and reuse of unchanged density pairs belong to numerics.
        """
        from atomx.angular.coefficients import exchange_k_range, exchange_lambda
        for a, shell in enumerate(self.subshells):
            tj_a = 2*abs(shell.kappa)-1
            for b in range(a, len(self.subshells)):
                other = self.subshells[b]
                if a == b:
                    c = self.get_WW(a, a)/2.
                    if abs(c) < 1e-15:
                        continue
                    yield False, c, 0, a, a, a, a
                    for rank in range(2, tj_a+1, 2):
                        lam = exchange_lambda(shell.kappa, shell.kappa, rank)
                        if lam >= 1e-15:
                            yield True, c*lam/(tj_a*(tj_a+1.)), rank, a, a, a, a
                else:
                    ww = self.get_WW(a, b)
                    if abs(ww) < 1e-15:
                        continue
                    yield False, ww, 0, a, a, b, b
                    for rank in exchange_k_range(shell.kappa, other.kappa):
                        lam = exchange_lambda(shell.kappa, other.kappa, rank)
                        if lam >= 1e-15:
                            c = ww*lam/(4.*abs(shell.kappa)*abs(other.kappa))
                            yield True, c, rank, a, b, a, b

    def two_electron_energy(self, pa, qa, r, difr, np_):
        """(E_J, E_K): direct and exchange two-electron energies."""
        densities = {}
        energies = [0., 0.]
        for exchange, coefficient, rank, a, b, c, d in self.coulomb_energy_terms():
            for pair in ((a,b), (c,d)):
                if pair not in densities:
                    i, j = pair
                    densities[pair] = pa[i]*pa[j]+qa[i]*qa[j]
            energies[exchange] += coefficient*_slater_integral_slag6(
                densities[a,b], densities[c,d], rank, r, difr, np_)
        return tuple(energies)

    def rotation_energy_factory(self, pa, qa, r, difr, rv):
        """One frozen Hamiltonian graph for a complete occupied rotation sweep."""
        from atomx.numerics.shooting.energy import RadialRotationEnergy
        terms = [(coefficient*(-1 if exchange else 1), rank, i, j, k, l)
                 for exchange, coefficient, rank, i, j, k, l
                 in self.coulomb_energy_terms()]
        return RadialRotationEnergy(pa, qa, r, difr, rv,
            [(sh.occupation, sh.kappa) for sh in self.subshells], terms).for_pair

    def orbital_field(self, a, pa, qa, r, difr, rvnuc, angular):
        """One definition of the local field and full nonlocal exchange."""
        intra, exch, direct0 = angular
        n = len(r)
        v_loc = np.zeros(n)
        for b, sh in enumerate(self.subshells):
            coefficient = (direct0[a][b] if direct0 is not None else
                           sh.occupation - (1. if a == b else 0.))
            v_loc += coefficient * _yk_over_r(pa[b]**2+qa[b]**2, 0, r, difr, n)
        for k, coefficient in intra[a]:
            v_loc += coefficient * _yk_over_r(pa[a]**2+qa[a]**2, k, r, difr, n)
        chi = self.exchange_action(a, np.array([pa[a], qa[a]]), pa, qa, r, difr, exch)
        return rvnuc+r*v_loc, chi[0], chi[1], v_loc

    def exchange_action(self, a, trial, pa, qa, r, difr, exch):
        """Linear nonlocal exchange action with all other orbitals frozen."""
        n = len(r)
        chi = np.zeros((2, n))
        for b in range(len(self.subshells)):
            if b == a:
                continue
            density = trial[0]*pa[b]+trial[1]*qa[b]
            for k, coefficient in exch[(a, b)]:
                y = coefficient*_yk_over_r(density, k, r, difr, n)
                chi[0] += y*pa[b]
                chi[1] += y*qa[b]
        return chi

    def legacy_orbital_field(self, a, pa, qa, r, difr, rvnuc, angular):
        """Historical Schmidt-mode field, preserving its summation order."""
        intra, exch, direct0 = angular
        np_ = len(r)
        occ = [float(sh.occupation) for sh in self.subshells]
        # --- local potential (direct + same-shell F^k) ---
        dens = [pa[s] ** 2 + qa[s] ** 2 for s in range(len(self.subshells))]
        if direct0 is None:
            vel = np.zeros(np_)
            for b in range(len(self.subshells)):
                vel += occ[b] * _yk_over_r(dens[b], 0, r, difr, np_)
            v_loc = vel - _yk_over_r(dens[a], 0, r, difr, np_)
        else:
            # Guerra: per-target k=0 coefficients WW(a,b)/q_a
            # (statistically q_b, resp. q_a - 1 for b == a).
            v_loc = np.zeros(np_)
            for b in range(len(self.subshells)):
                v_loc += direct0[a][b] * _yk_over_r(
                    dens[b], 0, r, difr, np_)
        for k, coeff in intra[a]:
            v_loc += coeff * _yk_over_r(dens[a], k, r, difr, np_)
        rv_loc = rvnuc + r * v_loc

        # --- non-local exchange source ---
        chi_p = np.zeros(np_)
        chi_q = np.zeros(np_)
        for b in range(len(self.subshells)):
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

        return rv_loc, chi_p, chi_q

    def occupied_fock_matrix(self, pa, qa, r, difr, rvnuc, angular):
        from atomx.numerics.shooting.operators import one_body_element
        weights = slag6_weights(len(r))*difr
        count = len(self.subshells)
        fock = np.zeros((count, count))
        for a, sh in enumerate(self.subshells):
            _, cp, cq, v = self.orbital_field(a, pa, qa, r, difr, rvnuc, angular)
            for b, other in enumerate(self.subshells):
                if sh.kappa == other.kappa:
                    fock[b, a] = one_body_element(pa[b], qa[b], pa[a], qa[a],
                        sh.kappa, r, difr, rvnuc) + weights @ (
                        v*(pa[b]*pa[a]+qa[b]*qa[a])-pa[b]*cp-qa[b]*cq)
        return fock

    def fock_diagnostics(self, pa, qa, r, difr, rvnuc, angular):
        fock = self.occupied_fock_matrix(pa, qa, r, difr, rvnuc, angular)
        occupations = np.array([sh.occupation for sh in self.subshells])
        multipliers = fock*occupations[None, :]
        residual = float(np.max(np.abs(multipliers-multipliers.T)))
        return np.diag(fock).copy(), residual

    def total_energy(self, pa, qa, r, difr, rv):
        from atomx.numerics.shooting.operators import one_body_element
        h = sum(sh.occupation * one_body_element(
            pa[a], qa[a], pa[a], qa[a], sh.kappa, r, difr, rv)
            for a, sh in enumerate(self.subshells))
        j, k = self.two_electron_energy(pa, qa, r, difr, len(r))
        return h + j - k

    def rotation_pairs(self, groups):
        """Allowed same-kappa rotations and whether to track a core-hole saddle."""
        for group in groups.values():
            for i, a in enumerate(group):
                for b in group[i+1:]:
                    if (self.subshells[a].occupation == 2*abs(self.subshells[a].kappa)
                            and self.subshells[b].occupation == 2*abs(self.subshells[b].kappa)):
                        continue
                    lower = a if self.subshells[a].n < self.subshells[b].n else b
                    core_hole = (self.subshells[a].n != self.subshells[b].n and
                        self.subshells[lower].occupation < 2*abs(self.subshells[lower].kappa)-1e-12)
                    yield a, b, core_hole

    def canonical_groups(self):
        """Only fully occupied same-kappa orbitals have rotation invariance."""
        groups = {}
        for i, shell in enumerate(self.subshells):
            if shell.occupation == 2*abs(shell.kappa):
                groups.setdefault(shell.kappa, []).append(i)
        for indices in groups.values():
            if len(indices) > 1:
                yield sorted(indices, key=lambda i: self.subshells[i].n)
