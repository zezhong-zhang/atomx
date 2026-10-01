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
