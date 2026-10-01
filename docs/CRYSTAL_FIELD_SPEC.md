# Crystal-Field Track Spec (opened 2026-10-01)

## Status

First gated increment GREEN (20/20 `tests/test_crystal_field.py`):
operator module complete, assembler m-resolved channel wired
(python + numba serial + numba parallel), all physics gates passing.

## Convention

V_CF = Σ_{k,q} B^q_k · r^k · C^q_k(r̂),  C^q_k = √(4π/(2k+1)) Y_{kq}

- **B_kq are radial-free Wybourne A_kq expressed in cm⁻¹** (1 au =
  219474.63 cm⁻¹); ⟨r^k⟩ in bohr^k is applied inside the operator on
  the workspace Gauss grid (`rpow_grid`).  NOT Stevens parameters.
- Values may be complex (general sites).  Hermiticity requires
  B_{k,−q} = (−1)^q B*_{kq}.  CI wiring needs REAL elements — complex
  site sets must first go through a tesseral (real) representation
  (FOLLOW-UP, see below).
- Provenance: B_kq are INPUTS (DFT-IAM decomposition, literature,
  `point_charge_bkq` for validation).  The module never invents them.
- Element: ⟨nκ_a m_a|V|nκ_b m_b⟩ = Σ_k B^{q=m_a−m_b}_k · ⟨r^k⟩ ·
  (−1)^{j_a−m_a} (j_a k j_b; −m_a q m_b) ⟨κ_a‖C^k‖κ_b⟩ with
  `reduced_cl_j2` (GRASP/FAC phase — the legacy `reduced_C` differs by
  (−1)^{j_a+j_b+k}, harmless in two-vertex CI products, WRONG for
  one-body operators).

## Assembler channel (`one_body_m_delta`)

The CF q≠0 terms MIX m within a kappa; q=0 terms are m-diagonal but
m-DEPENDENT.  Neither the base one-body value (m-blind, keyed
(n,κ)) nor `one_body_delta` (m-blind dict) can carry them.  The new
channel: callable (ia, ic) over spinor indices, same-kappa partner
groups INCLUDING a == c (diagonal additions), emitted as a separate
pass in `_assemble_reference` and as phase-1a' tables + a second
one-body walk in both numba kernels.  Empty tables → zero cost when
unused.

## Gates (all in tests/test_crystal_field.py)

- **G1** radial k=0 == 1 (pins the P=rR measure convention)
- **G2** Hermiticity cf(ia,ib) == conj(cf(ib,ia)) under Hermitian B sets
- **G3** per-subshell trace of CF diagonal == 0 ∀k>0 (C^k traceless)
- **G4** digit-exact vs INDEPENDENT path: CG-built spinor spherical
  harmonics + vectorised angular quadrature (no 3j of the operator
  route) — 9 (κ_a,κ_b,k,q) combinations over all m.  This gate caught:
  the reduced_C phase (q=0 diagonals), the B/q disconnection (G2 sign
  flips), and its own quadrature conjugation bug (Y_{l,−m}=(−1)^m Y*)
- **G5** python/numba element-identical; H Hermitian; spectrum moves

## Bugs caught by the gates (do not reintroduce)

1. `reduced_C` legacy phase in a one-body context (see above).
2. B-lookup disconnected from the tensor component: the angular
   machinery derives q from m_a − m_b; the B sum MUST be keyed by that
   same q (`if kq != q: continue`).
3. An m_a ≠ m_c-only mixing channel silently drops all q=0 CF shifts.

## Follow-ups

1. **Tesseral real-B conversion** for general low-symmetry sites (needed
   before any real-lattice CI run; axial/cubic sets are already real).
2. **point_charge_bkq validation**: its Y_kq against sympy + a
   G4-style direct-potential quadrature of Σ q_i/|R_i − r| (currently
   validation-grade only; the multipole expansion constants are
   asserted by construction, not yet digit-tested).
3. **Off-diagonal n** CF elements (⟨4f|r^k|5f⟩ etc.) — deferred;
   needs the two-body relaxation story first.
4. **Eu CF anchor protocol**: free-ion anchors calibrated empirically
   (CI_MBPT_SIGMA2_SPEC verdict), CF from DFT-IAM or literature B_kq,
   splitting checked against NIST crystal-field levels.

## External review round (2026-10-01, kimi-for-coding, FIX-FIRST → fixed)

Per the new workflow (proposals get an external-model critique from
physics/numerics/code angles before shipping), the committed module was
reviewed; verdict FIX-FIRST with 10 ranked issues.  Tier-1 fixes
applied same day (24/24 gates green):

1. **point_charge_bkq electron sign** — V_electron = −φ: overall minus
   added (splittings were hole-like/inverted).  Now gated: axial unit
   charge gives exactly A_k^0 = −q/R^{k+1} (G7).
2. **Y_{k,−q} conjugation** — the old branch kept a factorial ratio AND
   skipped the conjugate (two errors cancelling in A^q_k by accident);
   the review-driven "fix" first kept the ratio — magnitude-wrong;
   correct relation is Y_{k,−q} = (−1)^q conj(Y_{k,q}) with NO ratio.
   Y is now module-level `_Y_kq`, digit-validated against sympy
   assoc_legendre (G7c).
3. **r<R guard** — `r_inside` is now a REQUIRED argument; charges inside
   the shell raise instead of silently producing divergent B.
4. **Cross-n same-kappa elements ENABLED** (were silently zeroed —
   wrong Hamiltonian for any CI with 4f+5f; the "two-body relaxation"
   deferral rationale was bogus for a one-body operator).  Gated (G6).
5. **Build-time B validation** — q range hard-checked; Hermiticity
   partner check warns (default) or raises (strict) — non-Hermitian
   input no longer flows silently into H.  Gated (G7b); the
   point-charge output now feeds a strict build as an end-to-end test.
6. **Single-quadrature normalisation** — norm now = rpow(k=0) on the
   same Gauss grid (was: Gauss-grid radial ÷ external _norm_sq = two
   quadrature rules dividing; also removes the private import).
7. **Unit hazard documented** — literature Wybourne B_kq INCLUDE
   ⟨r^k⟩; here B are radial-free A_kq: divide ⟨r^k⟩ out first.

Honesty note: the earlier "python + numba serial + numba parallel wired"
claim overstates — the parallel kernel (n≥2000) m-channel segment has
NO test coverage below the trigger (Be M=0 sectors cap ~1.4k dets; DF
refuses unbound 6s/6d).  It is a structural mirror of the gated serial
path; the first Eu-scale CF run must include a python-vs-numba
spot-check.

Remaining from the review (scheduled, not blocking):
- tesseral (real) B representation for low-symmetry sites — the
  complex capability has no CI consumer until this lands (float() of
  complex raises by design).
- h1m table: per-group CSR instead of global sorted keys + binary
  search (a small-table design tax, not a bottleneck).
- phase-1a' should skip frozen-core pairs using the existing `frozen`
  set; cf() should pre-group bkq by q (O(1) lookup).
- lru_cache closures hold the workspace with no clear() API.
- library-level adapter from build_cf_operator to the assembler channel
  (the only glue today is test-local; KeyError→0.0 adapters must
  fail loud).
- layering: r^k moments belong in the MAT/basis layer, not here.
