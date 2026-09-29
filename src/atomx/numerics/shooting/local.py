"""Local-potential SCF and frozen-orbital shooting drivers.

Slater field and energy operators are supplied by ``physics.slater``;
radial propagation and iteration are owned by this numerical package.
"""
import math

import numpy as np

from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.local_scf import iterate_local_orbitals
from atomx.numerics.shooting.quadrature import slag6
from atomx.numerics.shooting.scf import validate_scf_controls
from atomx.numerics.shooting.spline import splin0
from atomx.physics.slater import _expval, _local_field
from atomx.potentials.slater_seed import slater_initial_field
from atomx.results import ShootingResult


def _dhfs_scf(z, nshell, shell_n, shell_kappa, shell_occ, shell_l, shell_2j,
              np_, rn, r2, drn, alpha, max_iter, epsc, aw, equation,
              slater_method, vscr):
    """Full DHFS SCF loop — DHFS.f lines 700-888.

    Parameters:
        equation: 0 = Dirac, 1 = Schrödinger
        slater_method: 0 = legacy power-series ODE Slater integral,
            1 = SLAG6 forward+backward (accurate for heavy Z; see
            :func:`_slater_integral_slag6`).  Only affects the
            post-SCF ``_expval`` total-energy assembly — the SCF
            itself does not call ``_slater_integral``.
        vscr: r*delta_V(r) of the plasma screening on the SAME SGRID this
            function builds (empty array = no screening).  It is added to
            the nuclear potential, so the SCF solves the screened
            Hamiltonian rather than an unscreened one.
    """
    r, difr, np_ = sgrid(rn, r2, drn, np_)
    r, difr, np_, vnuc, rden, qelec = slater_initial_field(
        z, shell_occ, r, difr, aw, vscr)
    ev, converged, iteration, ier_max, rv, pa, qa = iterate_local_orbitals(
        r, difr, vnuc, rden, shell_n, shell_kappa, shell_occ, shell_l,
        equation, max_iter, epsc, qelec <= 1.0, _local_field,
        (alpha, qelec, nshell, z))

    # Compute total energy via EXPVAL (DHFS.f:984-1103)
    # Uses Slater integrals + DCJ3 angular coefficients for exact result.
    # `slater_method` picks the Slater integral backend (0 = legacy
    # power-series ODE, 1 = SLAG6 forward+backward).
    e_total, ekin_total = _expval(
        nshell, shell_kappa, shell_occ, shell_2j,
        pa, qa, ev, rv, vnuc, r, difr, np_, equation,
        slater_method)

    return ev, e_total, converged, iteration, ier_max, r, difr, np_, rv, pa, qa


def dfs_frozen_orbital(result, n: int, kappa: int, e_guess: float = -1.0):
    """Bound orbital (n, kappa) in the FROZEN converged DFS potential.

    ``result`` is a ShootingResult from an ``exchange='gu'`` (or other
    local-exchange) solve carrying ``_rv`` — the converged r*V(r).  This
    is exactly FAC's scheme for excited/virtual orbitals: eigenstates of
    the same optimized central potential as the occupied ones
    (OptimizeRadial).  Returns ``(P, Q, eps)`` on ``result._grid``,
    slag6-normalized.
    """
    r = result._grid
    rv = result._rv
    difr = result._difr
    if r is None or rv is None:
        raise ValueError("result carries no converged potential (_rv); "
                         "re-solve with the extended shooting solver")
    np_ = len(r)
    va, vb_arr, vc, vd = splin0(r, rv, np_, 0.0, 0.0)
    e_val, ier, p_sol, q_sol, ilast = dbound(
        float(e_guess), 1.0e-9, int(n), int(kappa),
        r, rv, va, vb_arr, vc, vd, np_, r, np_)
    if ier != 0:
        raise RuntimeError(f"dbound failed for n={n} kappa={kappa}: ier={ier}")
    f1 = (p_sol ** 2 + q_sol ** 2) * difr
    f1_int = slag6(1.0, f1, np_)
    norm = 1.0 / math.sqrt(f1_int[np_ - 1]) if f1_int[np_ - 1] > 1e-30 else 1.0
    return p_sol * norm, q_sol * norm, float(e_val)


class ShootingSolver:
    """DFS SCF using ODE shooting — Salvat/Radial method.

    Parameters:
        Z: nuclear charge
        subshells: list of Subshell objects with n, kappa, occupation
        n_grid: number of radial grid points (default 1000)
        r_max: outer radius in Bohr (default 75.0)
        alpha_x: Slater exchange parameter (default 1.5). Ignored when
            ``exchange='gu'`` — α_xc is then computed by
            :func:`atomx.physics.local_exchange.get_hxs`.
        aw: atomic weight in g/mol (0 = point nucleus)
        equation: 'dirac' or 'schrodinger'
        exchange: 'slater' (default; uses ``alpha_x``) or 'gu' (Gu's
            Z- and N-dependent α_xc; FAC's local-exchange model).
        fac_ihx, fac_hxs: forwarded to ``get_hxs`` when
            ``exchange='gu'``; defaults match FAC's own defaults.
        slater_method: post-SCF Slater integral backend used by
            ``_expval`` when assembling the total binding energy.
            ``'slag6'`` (default) uses direct 6-point Lagrange forward
            and backward Hartree-Y integrals — accurate for heavy-Z
            densities where the original Salvat/DHFS.f power-series
            ODE stalls at its ``MAX_NSTEPS`` cap (and over-binds by
            20–30 % for Z ≥ 74).  ``'power_series'`` keeps the legacy
            backend for regression comparisons.  The SCF loop itself
            is unaffected — it uses direct SLAG6 integrals everywhere.
    """

    def __init__(self, Z, subshells, n_grid=1000, r_max=75.0, alpha_x=1.5,
                 aw=0.0, equation='dirac',
                 exchange='slater', fac_ihx=-1.0, fac_hxs=1.0,
                 slater_method='slag6',
                 exchange_alpha_fn=None, nuclear=None,
                 screening_potential=None, r_inner=1e-7):
        if equation not in ('dirac', 'schrodinger'):
            raise ValueError(f"equation must be 'dirac' or 'schrodinger' (got {equation!r})")
        if slater_method not in ('slag6', 'power_series'):
            raise ValueError(
                f"slater_method must be 'slag6' or 'power_series' "
                f"(got {slater_method!r})"
            )
        if exchange not in ('slater', 'gu'):
            raise ValueError(
                f"exchange must be 'slater' or 'gu' (got {exchange!r})")
        # Decoupled physics choices (review E2/E3):
        # * exchange_alpha_fn(Z, N_total) -> Slater alpha overrides the
        #   built-in 'slater'/'gu' selection, so any Slater-family
        #   functional can be injected without touching this solver.
        #   (The rho^(1/3) FORM is fused into the JIT'd _sfield — a
        #   non-Slater functional form needs a kernel variant, which is
        #   the documented boundary of this hook.)
        # * nuclear='point'|'fermi' resolves the nuclear model through
        #   the shared potentials.nuclear mass table instead of a raw
        #   atomic-weight number; the legacy aw= parameter still wins
        #   when nonzero for backward compatibility.
        self.exchange_alpha_fn = exchange_alpha_fn
        if nuclear is not None and not (aw and aw > 0.0):
            if nuclear == "point":
                aw = 0.0
            elif nuclear == "fermi":
                from atomx.potentials.nuclear import _get_mass
                aw = float(_get_mass(Z))
            else:
                raise ValueError(
                    f"nuclear must be 'point' or 'fermi' (got {nuclear!r})")
        self.Z = Z
        self.subshells = subshells
        if not np.isfinite(r_inner) or r_inner < 1e-8 or r_inner >= r_max:
            raise ValueError("r_inner must satisfy 1e-8 <= r_inner < r_max")
        self.r_inner = float(r_inner)
        self.n_grid = n_grid
        self.r_max = r_max
        self.alpha_x = alpha_x
        self.aw = aw
        self.equation = equation
        self.exchange = exchange
        self.fac_ihx = fac_ihx
        self.fac_hxs = fac_hxs
        self.slater_method = slater_method
        #: delta_V(r) of the surrounding plasma (Hartree), added to the
        #: nuclear potential inside the SCF.  ``None`` = no screening.
        self.screening_potential = screening_potential

    def _effective_alpha(self) -> float:
        from atomx.physics.slater import slater_exchange_alpha
        return slater_exchange_alpha(self.Z, sum(s.occupation for s in self.subshells),
            exchange=self.exchange, alpha_x=self.alpha_x, fac_ihx=self.fac_ihx,
            fac_hxs=self.fac_hxs, exchange_alpha_fn=self.exchange_alpha_fn)

    def solve(self, max_iter=300, tol=1e-9):
        validate_scf_controls(max_iter, tol, 0.0)
        n_shells = len(self.subshells)
        shell_n = np.array([sh.n for sh in self.subshells], dtype=np.int64)
        shell_kappa = np.array([sh.kappa for sh in self.subshells], dtype=np.int64)
        shell_occ = np.array([float(sh.occupation) for sh in self.subshells])

        # Also need l and 2j for Schrödinger
        shell_l = np.zeros(n_shells, dtype=np.int64)
        shell_2j = np.zeros(n_shells, dtype=np.int64)
        for i, sh in enumerate(self.subshells):
            k = sh.kappa
            if k < 0:
                shell_l[i] = -k - 1
                shell_2j[i] = -2 * k - 1
            else:
                shell_l[i] = k
                shell_2j[i] = 2 * k - 1

        eq_flag = 0 if self.equation == 'dirac' else 1

        # Grid parameters matching DHFS defaults
        r2 = self.r_inner
        drn = max(2.0 * self.r_max / self.n_grid, 0.5)

        alpha_eff = self._effective_alpha()
        slater_flag = 1 if self.slater_method == 'slag6' else 0

        # Plasma screening, sampled on the SAME SGRID _dhfs_scf builds
        # below (sgrid is deterministic), in the r*V form the SCF works in.
        vscr = np.zeros(0)
        if self.screening_potential is not None:
            r_scr, _difr_scr, np_scr = sgrid(self.r_max, r2, drn,
                                             self.n_grid)
            vscr = (r_scr * np.asarray(self.screening_potential(r_scr),
                                       dtype=float))
            if vscr.shape[0] != np_scr:
                raise ValueError(
                    "plasma screening potential returned "
                    f"{vscr.shape[0]} values for an {np_scr}-point grid")

        (orb_E, E_total, converged, iters, ier_max, r, difr, np_actual,
         rv_conv, pa_conv, qa_conv) = _dhfs_scf(
            float(self.Z), n_shells, shell_n, shell_kappa, shell_occ,
            shell_l, shell_2j,
            self.n_grid, self.r_max, r2, drn,
            alpha_eff, max_iter, tol, self.aw, eq_flag,
            slater_flag, vscr)

        oe = {}
        for i, sh in enumerate(self.subshells):
            oe[sh.label] = orb_E[i]

        # Bound-state sanity gate.  ``dbound`` brackets the eigenvalue
        # between the potential minimum and the effective potential at the
        # outer grid point, ``esup = V(r_max) + l(l+1)/r_max²``.  When the
        # state does not fit in the box the bracket collapses and the
        # returned eigenvalue is the *ceiling*, not a solved state.
        # DHFS.f discards the status code, so ``Atom(Z=1, config="6s1",
        # numerics="shooting")`` used to return the ceiling with a 4 %
        # error and ``converged=True`` (31 % for 7s, and the whole Na I
        # Rydberg series 9s..13s pinned to one value).  See
        # docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md finding C4.
        #
        # The ceiling must come from the *converged potential*, not from
        # ``-Z/r_max``: for a neutral heavy atom the screened potential at
        # the box edge is ~-1/75 Ha rather than -82/75, so the bare-Z form
        # would flag perfectly well resolved core orbitals (Pb 5d).
        v_edge = float(rv_conv[-1]) / float(r[-1])
        abs_ceiling = abs(min(v_edge, 0.0))
        unbound = []
        if abs_ceiling > 0.0:
            for sh in self.subshells:
                e_orb = oe[sh.label]
                # A pinned state lands within ~1e-10 of the ceiling; a
                # genuinely resolved one sits measurably below it.
                if abs(e_orb) > abs_ceiling * (1.0 + 1e-6):
                    continue
                unbound.append((sh.label, e_orb))

        if ier_max >= 5 or unbound:
            from atomx.numerics.shooting.dirac_orbital import OrbitalSolveError

            detail = ", ".join(
                f"{lab}={e_orb:+.6f}" for lab, e_orb in unbound
            ) or "bracket collapsed (ier=5)"
            raise OrbitalSolveError(
                f"shooting box too small for Z={self.Z}: the effective "
                f"potential at r_max is {v_edge:+.6f} Ha and {detail}. "
                f"Raise r_max (currently {self.r_max}); the default box "
                f"resolves states up to n <~ sqrt(32.6 Z)."
            )
        if ier_max != 0:
            converged = False

        # Preserve the native grid, potential and spinors in the result.
        from atomx.calculation_spec import resolved_solver_spec
        spec = resolved_solver_spec(self, method='DFS', engine='shooting',
                                    controls=dict(max_iter=max_iter, tolerance=tol))
        return ShootingResult(
            calculation_spec=spec,
            converged=converged, iterations=iters,
            total_energy=E_total, orbital_energies=oe,
            Z=self.Z,
            subshells=list(self.subshells),
            _grid=r,
            _equation=self.equation,
            _rv=rv_conv, _difr=difr, _pa=pa_conv, _qa=qa_conv,
        )
