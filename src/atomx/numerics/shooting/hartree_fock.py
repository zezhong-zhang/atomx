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
by a bordered Green solve, followed by variational same-kappa occupied
rotations. The historical Schmidt update is available only through
``constraint_mode="schmidt_legacy"``.

Validation: He, Ne, Fe XVII total energies agree with the B-spline
``DFSolver`` (``Atom(10, engine='bspline', exchange='hf')``) to <= 1e-4
relative and orbital energies to <= 5e-4 relative
(``tests/test_shooting_hf.py``).
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from atomx.numerics.shooting.local import ShootingSolver
from atomx.numerics.shooting.quadrature import slag6, slag6_weights
from atomx.physics.hartree_fock import RadialHartreeFock
from atomx.results import ShootingResult


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
        Integer jj occupation patterns with explicit ensemble weights.
        These define a configuration-average functional. The pair moments
        ``WW(a,b) = Σ_j c_j² n_a^j n_b^j`` (``WW(a,a) = Σ_j c_j² n_a^j
        (n_a^j − 1)``) via :func:`atomx.config.jj_coupling.compute_WW`,
        give per-electron Fock coefficients ``WW(a,b)/q_a`` —
        exactly the B-spline ``DFSolver`` convention.  Each config's
        ``subshells`` must align positionally with ``subshells`` (same
        ``(n, kappa)`` order), and every ``q_a`` must be positive (the
        caller must omit unoccupied shells). This interface does not include
        off-diagonal CSF interference and is not Hamiltonian DF. When ``None``,
        the fractional products below are used, bit-identical to the
        historical behaviour.

    Notes
    -----
    Open shells use the fractional statistical average
    ``W(a,a) = q(q-1)``, ``W(a,b) = q_a q_b`` (the ``averaging=
    'fractional'`` convention of the B-spline solver).  Off-diagonal
    Lagrange multipliers between same-kappa shells are enforced by
    the coupled variational constraints by default. Explicit legacy
    Schmidt mode is available for regression comparisons.
    """

    def __init__(self, Z, subshells: Sequence, n_grid: int = 1000,
                 r_max: float = 75.0, aw: float = 0.0,
                 nuclear: Optional[str] = None,
                 jj_configs: Optional[List] = None,
                 screening_potential=None, constraint_mode='coupled', r_inner=1e-7):
        self.Z = Z
        self.subshells = list(subshells)
        if not np.isfinite(r_inner) or r_inner < 1e-8 or r_inner >= r_max:
            raise ValueError("r_inner must satisfy 1e-8 <= r_inner < r_max")
        self.r_inner = float(r_inner)
        self.n_grid = n_grid
        self.r_max = r_max
        self.aw = aw
        self.nuclear = nuclear
        if constraint_mode not in ('coupled', 'schmidt_legacy'):
            raise ValueError('constraint_mode must be coupled or schmidt_legacy')
        self.constraint_mode = constraint_mode
        #: delta_V(r) of the surrounding plasma (Hartree), added to the
        #: nuclear potential inside the SCF.  ``None`` = no screening.
        self.screening_potential = screening_potential
        self.functional = RadialHartreeFock(self.subshells, jj_configs)
        self.subshells = self.functional.subshells
        self.jj_configs = self.functional.jj_configs

    def _canonicalize_closed_shells(self, pa, qa, r, difr, rvnuc, angular):
        """Fix occupied rotation freedom before assigning n-dependent shifts.

        Only fully occupied same-kappa orbitals have this invariance.
        Open-shell variational rotations must retain their optimized values.
        """
        fock = self.functional.occupied_fock_matrix(pa, qa, r, difr, rvnuc, angular)
        for indices in self.functional.canonical_groups():
            block = fock[np.ix_(indices, indices)]
            _, vectors = np.linalg.eigh((block+block.T)*.5)
            pp = vectors.T @ np.asarray([pa[i] for i in indices])
            qq = vectors.T @ np.asarray([qa[i] for i in indices])
            for j, i in enumerate(indices):
                sign = 1. if pp[j, np.argmax(abs(pp[j]))] >= 0 else -1.
                pa[i], qa[i] = sign*pp[j], sign*qq[j]

    def _rotate_occupied(self, pa, qa, groups, r, difr, rv):
        """Enforce stationarity of the HF functional along same-kappa rotations.

        A two-orbital rotation has Fourier degree four. Configuration
        averages have only even harmonics (five samples); coherent CSF
        states can have odd harmonics too (nine samples). The functional
        declares its harmonic content. This restores occupied-space
        stationarity lost in individually constrained solves.
        """
        from atomx.numerics.orbital_rotations import rotate_occupied
        pairs = tuple(self.functional.rotation_pairs(groups))
        if not pairs:
            return self.functional.total_energy(pa, qa, r, difr, rv)
        return rotate_occupied(pa, qa, pairs,
            lambda: self.functional.total_energy(pa, qa, r, difr, rv),
            pair_energy=self.functional.rotation_energy_factory(pa, qa, r, difr, rv),
            harmonics=getattr(self.functional, 'rotation_harmonics', (2, 4)))

    def solve(self, max_iter: int = 60, tol: float = 1.0e-7,
              damping: float = 0.5, *, seed: Optional[ShootingResult] = None,
              orbital_solver: str = "adaptive", orbital_tolerance=None) -> ShootingResult:
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
        orbital_tolerance : float, optional
            Independent bound on the undamped radial L2 update. The default
            sqrt(min(tol, 1e-7)) targets variational energy accuracy. A property
            calculation can demand tighter orbitals without requesting energy
            differences below floating-point resolution.
        orbital_solver : str
            'adaptive' retains the nonlocal resolvent after a frozen-source
            failure. 'nonlocal_resolvent' uses it for every unconstrained
            kappa channel. Occupied same-kappa channels retain their coupled
            bordered solver. The latter option requires coupled mode.
        seed : ShootingResult, optional
            Reuse radial orbitals and energies from an earlier solve on
            the same grid and nucleus Z. Shell occupations may change.
            Every requested orbital must be present. Arrays are copied;
            invalid or incompatible seeds are rejected explicitly.
        """
        from atomx.numerics.shooting.scf import validate_scf_controls
        validate_scf_controls(max_iter, tol, damping, orbital_tolerance=orbital_tolerance)
        if orbital_solver not in ('adaptive', 'nonlocal_resolvent'):
            raise ValueError('unknown shooting orbital solver')
        if orbital_solver == 'nonlocal_resolvent' and self.constraint_mode != 'coupled':
            raise ValueError('nonlocal_resolvent requires coupled mode')
        nsh = len(self.subshells)
        occ = np.array([float(sh.occupation) for sh in self.subshells])
        kappas = [sh.kappa for sh in self.subshells]
        ns = [sh.n for sh in self.subshells]

        # 1. Local-exchange starter (Gu alpha_xc — the closest local
        #    model to HF) on the SAME grid.
        # It is a different approximate model, so it need not meet the
        # final SCF tolerance. The nonlocal equations retain independent
        # energy, update and stationarity checks below.
        if seed is None:
            base = ShootingSolver(
                self.Z, self.subshells, n_grid=self.n_grid, r_max=self.r_max,
                exchange='gu', aw=self.aw, nuclear=self.nuclear, r_inner=self.r_inner,
                screening_potential=self.screening_potential,
            ).solve(max_iter=300, tol=1.0e-5)
            seed_rows = list(range(nsh))
        else:
            from atomx.numerics.shooting.grid import sgrid
            grid, jacobian, _ = sgrid(self.r_max, self.r_inner,
                max(2.*self.r_max/self.n_grid, .5), self.n_grid)
            if (seed.Z != self.Z or seed._equation != 'dirac'
                    or seed.subshells is None
                    or not np.array_equal(seed._grid, grid)
                    or not np.array_equal(seed._difr, jacobian)):
                raise ValueError('shooting seed must have the same Z, Dirac equation, and radial grid')
            lookup = {(s.n, s.kappa): i for i, s in enumerate(seed.subshells)}
            if len(lookup) != len(seed.subshells):
                raise ValueError('duplicate orbitals in shooting seed')
            try:
                seed_rows = [lookup[s.n, s.kappa] for s in self.subshells]
            except KeyError as exc:
                raise ValueError(f'shooting seed is missing orbital {exc.args[0]}') from exc
            for values in (seed._pa, seed._qa):
                if (values is None or values.shape != (len(seed.subshells), len(grid))
                        or not np.all(np.isfinite(values))):
                    raise ValueError('invalid shooting seed spinors')
            base = seed
        r = base._grid
        difr = base._difr
        np_ = len(r)
        # Own dense orbital buffers, shared by the field and rotation kernels.
        # Row updates stay in place; no per-energy list-to-array copy is needed.
        pa = np.array([base._pa[s] for s in seed_rows])
        qa = np.array([base._qa[s] for s in seed_rows])
        ev = np.array([base.orbital_energies[sh.label]
                       for sh in self.subshells])
        if seed is not None:
            if not np.all(np.isfinite(ev)) or np.any(ev >= 0):
                raise ValueError('shooting seed needs finite bound orbital energies')
            for a in range(nsh):
                norm = slag6(1., (pa[a]**2+qa[a]**2)*difr, np_)[-1]
                if not np.isfinite(norm) or norm <= 1e-20:
                    raise ValueError('zero-norm shooting seed orbital')
                pa[a] /= math.sqrt(norm)
                qa[a] /= math.sqrt(norm)

        # Nuclear potential in r*V form, plus the plasma screening shift
        # dV(r) = V_screened(r) - (-Z/r) sampled on the same grid.
        rvnuc = self.nuclear_rv_screened(r, difr, np_)

        intra, exch, direct0 = self.functional.angular_tables()
        # kappa groups for Schmidt orthogonalisation (ascending n).
        groups: Dict[int, List[int]] = {}
        for s in range(nsh):
            groups.setdefault(kappas[s], []).append(s)
        for g in groups.values():
            g.sort(key=lambda s: ns[s])

        if self.constraint_mode == 'coupled':
            metric = slag6_weights(np_)*difr
            for group in groups.values():
                block = np.array([[pa[a], qa[a]] for a in group])
                gram = np.einsum('iar,r,jar->ij', block, metric, block)
                values, vectors = np.linalg.eigh(gram)
                if np.min(values) < 1e-10:
                    raise ValueError('linearly dependent same-kappa seed orbitals')
                transform = (vectors / np.sqrt(values)) @ vectors.T
                block = np.einsum('ij,jar->iar', transform, block)
                for a, wave in zip(group, block, strict=True):
                    pa[a], qa[a] = wave

        from atomx.numerics.shooting.scf import iterate_nonlocal_orbitals

        def field(a, pa, qa):
            if self.constraint_mode == 'coupled':
                return self.functional.orbital_field(a, pa, qa, r, difr, rvnuc,
                                           (intra, exch, direct0))[:3]
            return self.functional.legacy_orbital_field(a, pa, qa, r, difr, rvnuc,
                                               (intra, exch, direct0))

        def evaluate(pa, qa, ev):
            variational_energy = None
            if self.constraint_mode == 'coupled':
                variational_energy = self._rotate_occupied(pa, qa, groups, r, difr, rvnuc)
            e_j, e_k = self.functional.two_electron_energy(pa, qa, r, difr, np_)
            e_2e = e_j - e_k
            energy = (float(np.dot(occ, ev) - e_2e) if variational_energy is None
                      else variational_energy)
            return energy, (e_j, e_k, e_2e)

        def stationarity(pa, qa):
            return self.functional.fock_diagnostics(pa, qa, r, difr, rvnuc,
                                          (intra, exch, direct0))[1]

        outcome = iterate_nonlocal_orbitals(pa, qa, ev, ns=ns, kappas=kappas,
            groups=groups, r=r, difr=difr, initial_energy=base.total_energy,
            max_iter=max_iter, tolerance=tol, damping=damping,
            constraint_mode=self.constraint_mode, first_sweep=seed is None,
            field=field, evaluate=evaluate, stationarity=stationarity,
            exchange_action=lambda a, trial, pa, qa: self.functional.exchange_action(
                a, trial, pa, qa, r, difr, exch), orbital_solver=orbital_solver,
            state_update=getattr(self, 'state_update', None), orbital_tolerance=orbital_tolerance)
        converged, iterations, e_total = outcome.converged, outcome.iterations, outcome.energy
        e_j, e_k, e_2e = outcome.components

        rotation_residual = None
        if self.constraint_mode == 'coupled':
            self._canonicalize_closed_shells(pa, qa, r, difr, rvnuc,
                                             (intra, exch, direct0))
            ev, rotation_residual = self.functional.fock_diagnostics(
                pa, qa, r, difr, rvnuc, (intra, exch, direct0))
        oe = {sh.label: float(ev[i])
              for i, sh in enumerate(self.subshells)}
        pa_arr = np.array(pa)
        qa_arr = np.array(qa)
        from copy import deepcopy

        from atomx.calculation_spec import resolved_solver_spec
        spec = resolved_solver_spec(self, method='DHF', engine='shooting',
            controls=dict(max_iter=max_iter, tolerance=tol, damping=damping,
                          orbital_solver=orbital_solver, orbital_tolerance=orbital_tolerance,
                          stable_green_tail=self.constraint_mode == 'coupled'))
        return ShootingResult(
            calculation_spec=spec,
            converged=converged, iterations=iterations,
            total_energy=e_total, orbital_energies=oe,
            Z=self.Z,
            subshells=list(self.subshells),
            _jj_configs=deepcopy(self.jj_configs),
            energy_components={
                "E_1e": e_total - e_2e,
                "E_2e": e_2e,
                "E_J": e_j,
                "E_K": e_k,
            },
            _grid=r,
            _equation="dirac",
            _difr=difr, _pa=pa_arr, _qa=qa_arr, _rv_nuc=rvnuc,
            _model="hf",
            solver_diagnostics={
                "initializer": "local_gu" if seed is None else "supplied_orbitals",
                "initializer_iterations": base.iterations,
                "initializer_converged": base.converged,
                "initializer_tolerance": 1e-5 if seed is None else None,
                "constraint_mode": self.constraint_mode,
                "orbital_solver": orbital_solver,
                "nonlocal_resolvent_scope": "unconstrained_kappa_channels",
                "exchange": "nonlocal_jj_average",
                "green_boundary": ("full_box_scaled" if self.constraint_mode == "coupled"
                                   else "legacy_practical_infinity"),
                "green_source_quadrature": ("dirac_evanescent_fitted_gauss16"
                    if self.constraint_mode == "coupled" else "slag6"),
                "occupied_rotation_residual_Ha": rotation_residual,
                "undamped_orbital_update_norm": outcome.orbital_update_norm,
                "nonlocal_resolvent_recoveries": outcome.nonlocal_recoveries,
                "nonlocal_stagnation_promotions": outcome.stagnation_promotions,
                "nonlocal_resolvent_max_residual": outcome.nonlocal_residual,
                "nonlocal_exchange_continuations": outcome.exchange_continuations,
                "exchange_continuation_min_overlap": outcome.continuation_min_overlap,
                "exchange_continuation_max_residual": outcome.continuation_max_residual,
                "nonlocal_resolvent_residual_form": "shifted_green_identity",
                "orbital_update_tolerance": (math.sqrt(min(tol, 1e-7))
                    if orbital_tolerance is None else orbital_tolerance),
                "closed_shell_canonicalized": self.constraint_mode == 'coupled',
                "core_hole_rotation": "nearest_stationary_point",
                "convergence_criterion": ("energy_change_and_occupied_rotation_and_orbital_update"
                    if self.constraint_mode == "coupled" else
                    "energy_change_and_orbital_update" if orbital_tolerance is not None else
                    "relative_total_energy_change"),
            },
        )

    def nuclear_rv_screened(self, r, difr, np_):
        """Compatibility method for the shared nuclear/environment field."""
        from atomx.potentials.radial_nucleus import screened_nuclear_rv
        return screened_nuclear_rv(self.Z, r, difr, np_, nuclear=self.nuclear,
            atomic_weight=self.aw, screening_potential=self.screening_potential)

    def _nuclear_rv(self, r, difr, np_):
        from atomx.potentials.radial_nucleus import nuclear_rv
        return nuclear_rv(self.Z, r, difr, np_, nuclear=self.nuclear,
                          atomic_weight=self.aw)
