# R-matrix Multi-Channel Inner Region — measured gap + design (2026-10-01)

## Measured state (probes this session, scripts/rmatrix_burke_abinitio.py)

The package (scattering/r_matrix/, ~3000 lines) is real plumbing: channel
table, direct/exchange CC potentials, Bloch B-spline inner region,
outer matching, eigenphase machinery, Breit-Wigner fitter.  23 tests
green.  BUT the "Burke-1971 benchmark" is a REPRESENTATION test of
empirically injected poles (l2_pseudostates.py states this itself):
with include_l2_pseudostates=True the fit returns the tabulated values
to 5 decimals; with it FALSE the computed eigenphase is FLAT across the
pole windows:

| block | pure-CC eigenphase variation over +-8 Gamma | needed |
|---|---|---|
| 1Se(2s2) | 0.106 rad | ~pi rise |
| 1Po(2s2p) | 0.005 rad | ~pi rise |

The inner region as built (projectile in the static+exchange potential
of ONE frozen hydrogenic target orbital) is distorted-wave grade: it
has no dielectronic correlation, cannot bind the (2s2) quasi-state,
and its coupling to the open channel carries no Feshbach structure.
NO ab initio observable has ever validated the CC Hamiltonian.

## External review (kimi, 2026-10-01): REDESIGN — accepted

All 8 issues valid.  The fatal ones: (1) the draft conflated e+He+
(2 electrons — where the Burke poles live; He+ has NO target
correlation, all correlation is chi_j) with e-He (3 electrons — where
"correlated He targets" would apply); (2) the claimed sparse-CI reuse
is blocked by the 128-spinor two-word mask cap (~960 continuum spinors
for l_c<=2) AND by frozen-core-less phase-1b combinatorial explosion
(2e9 quadruples for a fully-active 2-electron system); (4) jj m-scheme
CI vs nonrelativistic LS cc_inner/outer is a full-stack mismatch, and
cc_inner's scipy scalar-P splines are NOT the CI workspace's Dirac
(P,Q) weak-form splines; (5) surface-amplitude extraction (eigenvector
output + antisymmetrised channel projection + boundary evaluation) is
the main new work, not a "generalisation"; (3) Bloch operator and
Buttle correction are both absent (grep buttle = 0); (7) same-kappa
bound/continuum orthogonality does NOT hold in a spline basis
(off-diagonal S is direct evidence) — Lagrange multipliers or explicit
orthogonalisation required.

## Redesigned (v2, decision-first)

**D1 SYSTEM: e + He+ (2 electrons), LS coupling, nonrelativistic.**
The Burke gates live here; He+ is a one-electron target so ALL
dielectronic correlation comes from the chi_j (2l, 2l') L^2
configurations — exactly Burke 1971's own construction.  e-He (3e) and
the relativistic DARC route (jj CI reuse) are LATER phases with their
own projects (see D6).

**D2 REPRESENTATION: channel-radial, not m-scheme determinants.**
For 2 electrons the close-coupling + chi_j expansion is built directly
in LS-coupled channel-radial form on cc_inner's OWN spline workspace:
Psi_k = sum_i A[Phi_i(1) u_i(2)] + sum_j c_j chi_j — a block matrix
problem of order ~(|channels| x n_basis + n_chi), NOT a determinant
CI.  The 128-spinor mask cap and phase-1b explosion never arise.
(Masking fix for the general jj route stays a separate backlog item
with the overflow assert it deserves.)

**D3 HAMILTONIAN: the true 2-electron operator in the box.**
Direct + exchange coupling between channel blocks from the existing
angular_cc kernels; chi_j-continuum and chi_j-chi_j blocks from the
same two-electron integrals; Bloch b=0 symmetric form documented
(cc_inner already builds it — name it); bound-continuum orthogonality
by explicit Gram-Schmidt on the spline grid (Burke & Robb 1975
alternative: Lagrange multipliers — pick one, document).

**D4 SURFACE + BUTTLE.**  Eigenvector-level output (not just spectra);
channel-projected boundary amplitudes w_ik; R_ij(E) = (1/2a) sum_k
w_ik w_jk / (E_k - E) PLUS the Buttle correction per channel (solve
the homogeneous single-channel problem on the same grid; s-wave first
to break).  Pole-count convergence gate: R(E) at n_basis = 40/60/80.

**D5 GATES (numbers fixed by D1).**
  G-A  He+ n=2 threshold (analytic -0.5 Ha) and 1s binding (-2.0) —
       free sanity from the target orbitals
  G-B  1Se(2s2) pole AB INITIO within 0.02 eV of -0.77816 Ha, pi-rise
       in the PURE eigenphase (no injected poles), sampling resolution
       <= Gamma/4 written INTO the gate
  G-C  1Po / 1De widths within 30%
  G-D  unitarity + Buttle-corrected continuity + n_basis convergence

**D6 LATER PHASES (explicitly out of scope now).**  e-He 3e with
correlated CI targets; relativistic DARC (jj) inner region — requires
the mask-representation redesign (>=4-word or occ-array) and the
phase-1b candidate generation fix for core-less systems; each reuse
claim must ship a "what is missing" list.

**Missing-list for D3 reuse (per review rule):**
- angular_cc direct/exchange kernels: REUSABLE as-is
- cc_inner spline workspace: REUSABLE as the basis; its Hamiltonian
  assembly is REPLACED (static+exchange -> true 2e blocks)
- surface.py: formula generalises, but eigenvector plumbing is NEW
- Buttle: ENTIRELY NEW (nothing exists)
- orthogonality: NEW (none exists)


## Diagnosis round 2 (2026-10-01, channel-resolved): correction + root cause

CORRECTION of the "distorted-wave grade" verdict: cc_inner is a TRUE
two-electron CC construction — exact LS direct/exchange angular
coefficients, exact non-local exchange kernel, Bloch b=0 symmetric
form, proper surface amplitudes.  The earlier flat-eigenphase verdict
stands, but the mechanism is NOT missing operator structure.

CHANNEL-RESOLVED MEASUREMENT (1Se, 3-channel, a=25, n_basis=60):
- E=-0.7364: comp (0.97, 0.03, 0.00) — an OPEN-channel 1s-es pseudostate,
  not a resonance.
- Closed-dominated eigenvalues ALL sit at -0.61..-0.47 (n=2 continuum
  pseudostates); NO closed-like eigenpair anywhere near -0.78.

ROOT CAUSE: frozen HYDROGENIC 2s/2p targets.  The (2s2)1S quasi-bound
pole needs the two n=2 electrons to bind ~0.28 Ha below the n=2
threshold — dielectronic relaxation that hydrogenic He+ orbitals cannot
represent; the closed-channel pole lands ~0.18 Ha too high and the
Feshbach structure never forms.  This is EXACTLY Burke 1971's own
history: their contribution over earlier 3-state work was variational
CONTRACTED n=2 orbitals (effective zeta ~ 1.5 vs hydrogenic 1.0).

REVISED WORK PLAN (v3): the inner-region operator is DONE; the work is
the ORBITAL layer:
1. Contracted 2s-bar/2p-bar target orbitals, single exponent zeta
   (Burke-style 2s-bar ~ N r exp(-zeta r)).
2. Tune zeta so the 1Se closed-channel pole lands at -0.77816 (one
   parameter, one anchor symmetry).
3. PREDICT 1Po/1De poles and widths with the SAME zeta (G-B/G-C as
   real prediction gates).
4. Buttle correction + convergence gate unchanged (D4/D5).
The v2 representation decisions (D1/D2) are unaffected.

## v3 progress log (2026-10-02)

1. Contracted targets (zeta sweep, `rmatrix_contracted_zeta.py`): the
   closed 1Se pole moves -0.60 -> **-0.689** at zeta~1.15-1.20 —
   single-STO contraction delivers 78% of the Burke shift.  2s-bar is
   orthogonal to 1s BY CONSTRUCTION (r(1-(zeta+2)r/2)e^{-zeta r});
   post-hoc Schmidt breaks P(0)=0 and log-diverges <1/r>.
2. Orthogonal pair channel EXCLUDED (theorem-grade, 
   `rmatrix_chi_pair.py`): any phi perpendicular to (1s, 2s-bar) has
   E >= E(3s) = -0.222 Ha, so its pair block cannot reach -0.778.
   Measured e_phi = +0.04..+1.5 Ha, phi-composition ~0.01.
3. REMAINING SPEC (precisely): non-orthogonal chi_j blocks — extend
   assemble_cc_inner_region with correlation configurations chi =
   (phi_pair)^2 carrying NON-TRIVIAL cross blocks:
     S[chi, (i,b)]     = 2 <phi|Phi_i> <phi|B_b>          (both spins)
     H[chi, (i,b)]     = <phi|h|Phi_i><phi|B_b> + <phi|Phi_i><phi|h|B_b>
                         + <phiphi|V|Phi_i B_b>_LS-sym
     H[chi, chi']      = 2<phi|h|phi'> + <phiphi|V|phi'phi'>
   All kernels (y_k, direct/exchange coefficients) already exist; the
   work is the block plumbing + a regression gate (chi-free case must
   reproduce the 3-channel numbers digit-exactly).

## v3 progress log addendum (2026-10-02): chi-as-extra-channel excluded

The non-orthogonal pair channel was IMPLEMENTED (cc_inner pair_targets:
non-diagonal S and one-body cross blocks for 1S all-s; regression-clean
23/23 when unused) and FAILS structurally: a smooth-STO phi lies almost
entirely inside (2s-bar x splines), so the extra channel adds ~no span
and S goes indefinite (Cholesky failure).  Combined with the earlier
variational-bound exclusion of ORTHOGONAL phi, the conclusion is sharp:
**the remaining 0.09 Ha lives in richer TARGETS (Burke's multi-STO
2s-bar), not in extra channels.**  Next: 2-term STO target
(c1, z1, c2, z2), single channel per l — structurally immune to both
exclusions.
