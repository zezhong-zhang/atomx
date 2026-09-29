# ADR-0001 — Unified `ElectronicStructureProvider` Total Design & Dynamic Task Graph

- **Status:** Proposed (awaiting Pojo-organized `claude + codex + gemini` debate — this is the HYP-921 acceptance gate)
- **Deciders:** Tech Lead (claude), with codex + gemini debate, ratified by Pojo
- **Date:** 2026-07-19
- **Epic:** HYP-902 · **This issue:** HYP-921 (总体设计 + 动态任务图)
- **Companion:** HYP-905 carries the *field-level* dataclass contract; this ADR is the *systems-level* boundary + capability + DAG layer. HYP-903 (reuse audit) and HYP-906/HYP-922 (research) feed the capability/tolerance tables here — this ADR does **not** duplicate their inventory work, it consumes their conclusions.

> **Implementation status (2026-09-29):** The native provider, shared payload/capability contracts and optional PySCF adapter now exist. The proposed debate/ratification status above is historical governance metadata, not a claim that these interfaces are absent. ABACUS, DeePTB and the complete proposed task graph are not established by these implementations. See [current capabilities](../ARCHITECTURE_CAPABILITIES.md).

> **Scope discipline.** This document is design-only. No business code is written here. All signatures below that reference existing atomx symbols are anchors for the debate, not new API. Where a signature is *proposed*, it is marked **(NEW)**.

---

## 1. Context & Problem

Atomx today computes Auger/autoionization spectra from its **own** relativistic B-spline / shooting / MCDHF orbitals. HYP-902 wants the same downstream kernels (CSF/CI, two-electron Coulomb matrix elements, continuum coupling, spectrum synthesis) to run on orbitals/Hamiltonians produced by **external** electronic-structure codes:

- **PySCF** — molecular & finite-cluster GTO/quantum-chemistry (SCF, CASSCF/CI, MRPT).
- **ABACUS** — periodic solids in an LCAO basis with k-points.
- **DeePTB** — ML tight-binding surrogate that emits a Hamiltonian/overlap in an atomic-orbital basis.

The three back-ends differ on nearly every axis that the physics kernels care about: basis (GTO vs. LCAO vs. TB), relativistic treatment (non-rel / scalar-rel / SOC), periodicity (Γ-only vs. k-mesh), orthogonality (orthonormal vs. non-orthogonal overlap `S`), and epistemic status (converged ab-initio vs. ML prediction with an OOD risk).

The **key architectural finding** (verified in `src/atomx/scattering/protocols.py`): atomx already decouples its scattering/property layer from the SCF back-end via a *structural* protocol, `AtomicResultLike`, plus a mirror-able capability/adapter pattern in `src/atomx/corrections/protocol.py` (`Correction.supports()/.evaluate()`). **We extend these two existing patterns rather than inventing a parallel stack.**

### System invariants (inherited from HYP-902, non-negotiable)

1. **I1 — Basis & units traceable.** Every orbital/matrix element carries its basis identity and is expressible in Hartree atomic units at the kernel boundary.
2. **I2 — Core hole & final state explicit.** The initial core-hole and the two-hole final state are named, never implied by array position.
3. **I3 — Direct/exchange combined at amplitude level.** Never sum `|direct|² + |exchange|²`; combine the amplitudes, then square (matches `auger_rate_two_electron` and MCDHF matrix conventions).
4. **I4 — Energy & intensity conventions consistent.** One energy zero, one intensity normalization, declared and enforced end-to-end.
5. **I5 — Approximation level explicit.** Every result tags its theory level and every approximation is labelled, never silent.
6. **I6 — ML never bypasses the gate.** No DeePTB/ML output reaches a published spectrum without passing an ab-initio benchmark and an uncertainty gate; on gate failure the pipeline **fails closed**.

---

## 2. Decision (summary)

**D1. Contract, not inheritance.** Introduce `ElectronicStructureProvider` as a `runtime_checkable` `Protocol` (mirror of `Correction`), producing a payload that satisfies a **superset** of today's `AtomicResultLike`. Existing atomx solvers already satisfy the base protocol, so they become the reference "provider" for free and every benchmark keeps running unchanged.

**D2. Two-tier output surface.** The provider returns an `ElectronicStructurePayload` **(NEW)** that:
   - is *structurally* an `AtomicResultLike` for the atomic/molecular path (fields `Z`, `subshells`, `orbital_energies`, `orbitals_raw`), and
   - carries **optional** blocks (`overlap` `S`, `kpoints`, `spin/SOC`, `hamiltonian`, `periodic cell`) that periodic/non-orthogonal/ML back-ends need, each with a documented "absent ⇒ meaning" rule.

**D3. Capability negotiation before compute.** A kernel asks `provider.capabilities()` and `provider.supports(request)`; if a required capability (e.g. `energy_normalized_continuum`, `two_particle_rdm`) is missing, the kernel either selects a documented **degradation path** (§10) or **fails closed** with a typed error — never silently produces a number of unknown validity.

**D4. Provenance is a first-class, mandatory field**, not a free-form dict. Every payload and every `SpectrumResult` carries a `Provenance` record (§6.3) sufficient to reproduce and to audit which invariant each number satisfies.

**D5. Adapters live at the edge.** `pyscf_adapter`, `abacus_adapter`, `deeptb_adapter` are thin translators (external object → `ElectronicStructurePayload`, units → Hartree, provenance stamping). The physics kernels never import a back-end.

**D6. ML is gated, not trusted.** DeePTB enters only behind an OOD detector + uncertainty estimate + fail-closed policy, and is validated against the same ab-initio/experimental assets as the exact path (§9 stage 6, §11).

---

## 3. System boundary & module responsibilities

```
                 ┌─────────────────────────── external world ───────────────────────────┐
                 │   PySCF            ABACUS               DeePTB                          │
                 └──────┬───────────────┬───────────────────┬───────────────────────────┘
        ADAPTER LAYER   │               │                   │      (new: src/atomx/providers/)
                 ┌──────▼──────┐ ┌──────▼──────┐    ┌────────▼────────┐
                 │pyscf_adapter│ │abacus_adapter│    │ deeptb_adapter  │  + native_adapter (wraps SCFResult)
                 └──────┬──────┘ └──────┬──────┘    └────────┬────────┘
                        └───────────────┴──── ElectronicStructureProvider (Protocol) ──┘
                                            │  yields ElectronicStructurePayload (⊇ AtomicResultLike)
   ┌────────────────────────────────────────▼─────────────────────────────────────────┐
   │ CONTRACT / NORMALIZATION LAYER (new thin core: src/atomx/providers/contract.py)     │
   │  · unit coercion → Hartree (atomx.units)   · provenance stamping                     │
   │  · capability negotiation + fail-closed     · optional S/k/SOC block validation      │
   └────────────────────────────────────────┬─────────────────────────────────────────┘
   PHYSICS KERNELS (existing, reused — NOT rewritten):                                    │
   ┌──────────────┬──────────────────┬─────────────────────┬──────────────────────────┐ │
   │ CSF / CI     │ two-electron V_ee │ continuum coupling  │ two-hole correlated final │ │
   │ angular/*    │ scattering/       │ numerics/bspline/    │ scattering/autoionization │ │
   │ methods/ci   │  two_electron/*   │  continuum.py,       │  mscheme_ci, jj_level     │ │
   │ mcdhf/matrix │  bspline/slater   │  shooting/dirac_cont │  (+ Cini–Sawatzky, HYP-911│ │
   └──────────────┴──────────────────┴─────────────────────┴──────────────────────────┘ │
   ┌────────────────────────────────────────▼─────────────────────────────────────────┐ │
   │ SPECTRUM SYNTHESIS (existing): spectrum/*, lineshapes/*  → SpectrumResult          │◄┘
   └────────────────────────────────────────────────────────────────────────────────────┘
```

| Module | New/Reuse | Responsibility | Must NOT do |
|---|---|---|---|
| `providers/contract.py` **(NEW)** | new (thin) | Protocol defs, capability enum, unit coercion, provenance stamping, fail-closed errors | Any physics |
| `providers/*_adapter.py` **(NEW)** | new | Translate one back-end → payload; stamp provenance | Import a physics kernel |
| `scattering/protocols.py` | reuse+extend | Base structural protocol; add optional-block accessors | Break existing duck-typing |
| `angular/*`, `methods/ci`, `mcdhf/matrix` | reuse | CSF construction, CI, angular `W_k`, two-particle couplings | Know about back-ends |
| `scattering/two_electron/*`, `bspline/slater` | reuse | `R^k` Slater integrals, direct/exchange **at amplitude level** | Assume orthonormal basis without checking `S` |
| `numerics/**/continuum` | reuse | Energy-normalized continuum electron | — |
| `scattering/autoionization/*` | reuse | Auger widths, level/manifold enumeration, per-channel accumulation | — |
| `spectrum/*`, `lineshapes/*` | reuse | Line list → broadened `SpectrumResult` | Invent a second intensity unit |

---

## 4. Coupling matrix (module × module)

Legend: **●** strong/data-flow coupling · **○** weak/optional · **—** none. Rows *provide to* columns.

| provides ↓ / consumes → | Adapter | Contract | CSF/CI | V_ee (Rᵏ) | Continuum | Two-hole | Spectrum |
|---|---|---|---|---|---|---|---|
| **External code**       | ● | — | — | — | — | — | — |
| **Adapter**             | — | ● | — | — | — | — | — |
| **Contract**            | — | — | ● | ● | ● | ● | ○ |
| **CSF/CI (angular Wₖ, RDM)** | — | — | — | ● | ○ | ● | — |
| **V_ee / Rᵏ (slater)**  | — | — | ○ | — | ● | ● | — |
| **Continuum**           | — | — | — | ● | — | ● | — |
| **Two-hole final state**| — | — | — | — | — | — | ● |
| **Spectrum**            | — | — | — | — | — | — | — |

**Cut-set / stable interfaces** (where we freeze contracts so a change on one side does not ripple):

- **C-α (Payload):** Adapter → Contract → kernels. Frozen by HYP-905 dataclasses.
- **C-β (Orbital access):** `get_orbital()/list_occupied_orbitals()/grid_data()` in `scattering/protocols.py` — already stable; extend, don't break.
- **C-γ (Auger call):** `auger_rate_two_electron(P_a,Q_a,P_b,Q_b,P_c,Q_c,P_eps,Q_eps, *, angular_W, k, bsp_data, method)` — the amplitude-level contract (I3). Adding a basis with overlap `S≠I` must route through a documented biorthogonalization step *before* this call, not inside it.
- **C-δ (Line list → Spectrum):** `(energy, intensity, width, provenance)` tuples → `SpectrumResult`. Frozen intensity/energy units (I4).

The only **new** couplings introduced by this epic are the top-left block (External→Adapter→Contract). Everything downstream of C-α is existing, tested code.

---

## 5. Provider capability matrix (minimum vs. optional; back-end mapping)

**Minimum (MUST) capabilities — a payload is rejected fail-closed without these:**

| Capability | Meaning | Satisfies |
|---|---|---|
| `orbitals(P,Q)` | radial (or projected-radial) large/small components per orbital; `Q=0/None` ⇒ treated as non-rel small component | I1 |
| `orbital_energies` | ε per labelled orbital, Hartree | I1,I4 |
| `subshells/occupations` | labelled `(label, kappa|l, occupation)` | I2 |
| `core_hole_spec` | explicit initial core-hole subshell(s) | I2 |
| `units_declared` | every quantity tagged, coercible to Hartree | I1,I4 |
| `provenance` | code+version+method+basis+approx-level | I5 |

**Optional (MAY) capabilities — presence unlocks a code path; absence selects a documented degradation (§10):**

| Capability | Unlocks | Absent ⇒ |
|---|---|---|
| `overlap_S` | non-orthogonal basis (LCAO/GTO/TB) | assume orthonormal; assert `max|S−I|<tol` else fail-closed |
| `kpoints` | periodic Auger (ABACUS) | Γ-only / molecular path |
| `spin_soc` | SOC-resolved `kappa`, jj levels | scalar-rel; jj from LS via recoupling, flagged approximate |
| `two_particle_rdm` | correlated two-hole final state from CI | independent-hole / Cini–Sawatzky model (HYP-911) |
| `hamiltonian_matrix` | TB/DeePTB direct-H path | orbital-only path |
| `energy_normalized_continuum` | direct per-unit-energy Auger amplitude | box-normalize then apply `continuum_energy_normalization_factor` |
| `uncertainty` | ML gating, error bars on lines | treated as **0 ± unknown** ⇒ ML gate **fails closed** (I6) |

**Back-end capability mapping (to be confirmed/filled by HYP-922 research):**

| Capability | native (atomx) | PySCF | ABACUS | DeePTB |
|---|---|---|---|---|
| orbitals P,(Q) | ● (Dirac) | ● (GTO→radial proj.) | ● (LCAO) | ● (TB basis) |
| overlap_S | I | ● non-orth GTO | ● non-orth LCAO | ● from TB `S` |
| kpoints | — | — | ● | ● |
| spin_soc | ● | ○ (x2c/BP) | ○ | ○/planned |
| two_particle_rdm | ● (CI/mscheme) | ● (CASSCF/CI) | ○ | — |
| energy_norm continuum | ● | ✗ (GTO cannot represent continuum ⇒ **degradation D-2**) | ✗ | ✗ |
| uncertainty | n/a (exact) | n/a | n/a | ● (**required**) |

> **Sharp edge flagged for debate:** GTO/LCAO bases cannot represent an oscillatory continuum electron. The Auger *final* channel needs a genuine continuum orbital. **Proposed resolution:** the continuum electron is *always* produced by atomx's own `numerics` continuum solver in the residual-ion potential built from the provider's bound density; the provider supplies bound orbitals + potential only. This keeps the continuum contract (`C-γ`) inside atomx for all back-ends. **This is the single most important open question for the debate (Q1).**

---

## 6. Data schema, units & provenance

### 6.1 `ElectronicStructurePayload` (NEW — field list; dataclass detail belongs to HYP-905)

```
ElectronicStructurePayload:
  # --- AtomicResultLike core (frozen contract C-α/C-β) ---
  Z: int
  subshells: Iterable[SubshellLike]              # label, kappa, occupation
  orbital_energies: Dict[str, float]             # Hartree
  orbitals_raw: Dict[int, List[(P, Q, occ)]]     # by kappa
  # --- optional blocks (each: value | None, with "None ⇒ …" documented) ---
  overlap: Optional[OverlapBlock]                # S matrix + basis-label order
  kpoints: Optional[KPointBlock]                 # weights, fractional coords, per-k orbitals
  spin: Optional[SpinBlock]                      # collinear/SOC, jj vs LS flag
  potential: Optional[PotentialProvider]         # V(r) for continuum solve (§5 Q1)
  two_particle: Optional[TwoParticleBlock]       # RDM / CI vectors for correlated final state
  hamiltonian: Optional[HamiltonianBlock]        # TB/DeePTB H,S in AO basis
  uncertainty: Optional[UncertaintyBlock]        # per-quantity σ (required if provider is ML)
  provenance: Provenance                         # MANDATORY
```

### 6.2 Units — one rule
All numbers cross the C-α boundary **either** as a plain float already in Hartree atomic units **or** as a `pint` `Quantity` coerced via `atomx.units.to_hartree`. Adapters are the *only* place a non-atomic unit may appear. Spectrum output declares its axis/intensity unit in `SpectrumResult.provenance` (I4). No kernel guesses a unit.

### 6.3 `Provenance` (NEW — mandatory record)

```
Provenance:
  provider: str            # "pyscf" | "abacus" | "deeptb" | "atomx-native"
  code_version: str        # e.g. "pyscf 2.6.2"
  method: str              # "RHF/CASSCF(8,8)" | "PBE+SOC" | "deeptb-v2-ckpt#..."
  basis: str               # basis identity → satisfies I1
  theory_level: TheoryLevel
  approximations: list[str]# every non-exact step, explicit → I5
  core_hole: str           # initial hole subshell(s) → I2
  final_state: str         # two-hole final config → I2
  energy_zero: str         # declared reference → I4
  is_ml: bool              # gate trigger → I6
  ood_status: Optional[str]# "in-domain" | "ood" | "unknown"
  uncertainty_summary: Optional[dict]
```

Every `SpectrumResult` aggregates the provenance of its contributing lines; a line whose provenance fails an invariant is dropped **and reported**, never silently averaged in.

---

## 7. Uncertainty & failure semantics

- **Typed errors, fail-closed.** Missing MUST capability, `S` too far from `I` without a biorthogonalization path, an ML payload lacking `uncertainty`, or an OOD-flagged ML input ⇒ a typed `ProviderContractError` / `UncertaintyGateError`. The pipeline **stops and reports**; it does not emit a best-effort number (I6).
- **Uncertainty propagation.** Ab-initio path: convergence-based error (basis/k-mesh/CI-space) recorded in `uncertainty_summary`. ML path: model σ + OOD score, propagated to per-line σ and shown as spectrum error band.
- **Degradation is opt-in and logged.** A degradation path (§10) may run only when explicitly permitted by the request; it stamps `approximations += [...]` so the loss of rigor is visible downstream (I5).

---

## 8. Stage gates (phase discipline: molecule → cluster → periodic → two-hole → ML)

Each gate must be green before the next stage's implementation issues leave `backlog`.

| Gate | Deliverable | Green criterion | Owner |
|---|---|---|---|
| **G0 Contract** | this ADR + HYP-905 dataclasses | debate consensus; native solver satisfies protocol with 0 regressions | claude |
| **G1 PySCF molecule** | provider+adapter, molecular Auger MVP | reproduce an atomx-native molecular/He-like Auger width within research tolerance | codex/qa |
| **G2 Cluster/embedding** | local-orbital bridge, chemical shift | core-hole chemical shift vs. reference within tol | codex/qa |
| **G3 ABACUS periodic** | LCAO adapter, k-resolved Auger | periodic line shape vs. experiment/ref; k-convergence shown | codex/qa |
| **G4 Two-hole correlated** | Cini–Sawatzky / cluster-CI final state | correct U→0 and U→∞ limits; spectral weight sum rule | codex/qa |
| **G5 ML acceleration** | DeePTB adapter + gate | ML path reproduces G1–G4 benchmarks within σ; OOD fail-closed demonstrated | codex/qa/security |

**Validation anchors already in-repo (reuse — HYP-530):** `tests/data/fe17_kll_auger_fac.json` (5 dominant Fe XVII KLL Auger widths vs. FAC, 10% tol), `tests/test_auger_kll_vs_fac.py`, `tests/test_fe17_3c_3d_vs_bernitt2012.py` (3C/3D vs. Bernitt 2012). These are the KLL/line-ratio gates every back-end must eventually clear.

---

## 9. Acceptance matrix (invariant × how verified × asset)

| Invariant | Verified by | Asset / gate |
|---|---|---|
| I1 basis/units traceable | provenance schema test; unit round-trip via `atomx.units` | HYP-905 tests |
| I2 core-hole/final-state explicit | payload requires `core_hole`+`final_state`; missing ⇒ error | contract tests |
| I3 direct/exchange at amplitude | numeric test: amplitude-sum ≠ rate-sum on a case where interference matters | `two_electron` + auger tests |
| I4 energy/intensity consistent | end-to-end line→spectrum unit assertion | spectrum tests |
| I5 approximation explicit | every degradation stamps `approximations` | contract tests |
| I6 ML gated | OOD/uncertainty-missing input ⇒ fail-closed; ML vs. ab-initio within σ | HYP-919 QA |
| Physics correctness | Fe XVII KLL widths, 3C/3D ratio | `fe17_kll_auger_fac.json`, Bernitt 2012 |

---

## 10. Risk & degradation strategy

| # | Risk | Likelihood/Impact | Degradation / mitigation |
|---|---|---|---|
| R1 | **Continuum electron in GTO/LCAO basis** (§5 Q1) | High/High | **D-1:** continuum always solved by atomx `numerics` in residual-ion potential from provider density; provider supplies bound + potential only |
| R2 | Non-orthogonal basis breaks Slater `Rᵏ` assumptions | High/High | **D-2:** biorthogonalize / Löwdin-orthogonalize before C-γ; assert `S` residual; else fail-closed |
| R3 | GTO radial projection loses cusp / small component | Med/Med | **D-3:** flag `approximations+=["nonrel-projection"]`; restrict claims to valence Auger; heavy-Z ⇒ prefer native/Dirac |
| R4 | ABACUS k-mesh under-convergence | Med/High | **D-4:** k-convergence gate in QA; report band vs. mesh |
| R5 | DeePTB OOD silent extrapolation | Med/High | **D-5:** OOD detector + uncertainty gate, fail-closed (I6) |
| R6 | Two-hole correlation `U` mis-scaled | Med/High | **D-6:** validate limits (U→0 band-like, U→∞ atomic multiplet) + sum rule (HYP-916) |
| R7 | Provider API drift (PySCF/ABACUS/DeePTB versions) | Med/Med | pin versions in provenance; adapter contract tests per version |
| R8 | Scope creep: kernels grow back-end knowledge | Med/High | enforce "kernels never import a back-end" in review; adapters are the only edge |

---

## 11. Dynamic task graph (DAG) & PR切分

The epic HYP-902 **already** has a stage-ordered child DAG. HYP-921's job is to *annotate it with contracts and gates*, not recreate it. Mapping of the existing children onto this design (⇐ = consumes contract from):

```
STAGE 1  (design/audit/research — G0)
  HYP-903 audit ─┐
  HYP-922 research(data semantics/capability) ─┼─► HYP-905 interface ADR ⇐ THIS (HYP-921)
  HYP-906 research(benchmarks/tolerances) ─────┘        │  (C-α/C-β/C-γ/C-δ frozen here)
STAGE 2  (G1 PySCF molecule)   ⇐ C-α, C-γ
  HYP-904 provider base + PySCF adapter → HYP-910 molecular core/two-hole MVP → HYP-907 QA
STAGE 3  (G2 cluster)          ⇐ C-α
  HYP-909 cluster/embedding + local-orbital bridge → HYP-923 milestone ; HYP-908 QA
STAGE 4  (G3 ABACUS periodic)  ⇐ C-α (+overlap_S,+kpoints)
  HYP-913 LCAO periodic interface/core-hole contract → HYP-912 ABACUS MVP → HYP-914 QA
STAGE 5  (G4 two-hole correlated) ⇐ C-γ (+two_particle)
  HYP-911 research ADR (Cini–Sawatzky/cluster-CI) → HYP-918 impl → HYP-916 QA ; HYP-924 milestone
STAGE 6  (G5 ML)               ⇐ full contract + uncertainty
  HYP-915 DeePTB contract/OOD/active-learning gate → HYP-917 adapter+kernel → HYP-919 QA ; HYP-925/926 milestone
STAGE 7-8 (release gate / system QA)
  HYP-920 integration gate → HYP-927 acceptance
```

**PR切分 principle (half-day → one-day granularity, deliver-through-pr):**

- Each adapter is its own PR; the thin `providers/contract.py` is a *separate, first* PR (unblocks all adapters).
- One PR = one capability or one back-end stage; never mix "add contract" with "add ABACUS".
- Every physics-touching PR ships with the relevant reuse-benchmark green (KLL / 3C-3D / limit tests).
- No PR merges to `main` without independent QA + Pojo integration (per HYP-902 rule).

**Immediate next actions after debate consensus (for Pojo to dispatch):**
1. Freeze HYP-905 dataclasses from §6 field lists → unblocks Stage 2.
2. HYP-922 to fill the §5 back-end capability table with verified PySCF/ABACUS/DeePTB facts.
3. HYP-903 audit to confirm the reuse cut-sets (C-β/C-γ) hold with no hidden back-end assumptions.

---

## 12. Open questions for the debate (claude ⇄ codex ⇄ gemini)

- **Q1 (blocking):** Continuum electron for GTO/LCAO back-ends — accept D-1 (atomx always owns the continuum solve in a provider-supplied residual potential)? Alternatives: B-spline re-expansion of provider density, or a hybrid Kohn/close-coupling. *Recommend D-1.*
- **Q2:** Non-orthogonal basis — Löwdin vs. biorthogonal (dual-basis) before `Rᵏ`? Impact on I3 amplitude combination.
- **Q3:** Is `two_particle_rdm` a MUST for the correlated final state (Stage 5), or do we ship independent-hole first and add correlation behind a capability flag? *Recommend flag-gated, ship independent-hole at G1–G3.*
- **Q4:** Provider granularity — one `solve(request)` call, or split `bound_states()` / `potential()` / `two_particle()` so the continuum-solve (Q1) composes cleanly? *Recommend split accessors behind one Protocol.*
- **Q5:** Uncertainty representation for the ML gate — single scalar σ per line vs. covariance; what OOD score triggers fail-closed (I6)?

---

## 13. Consequences

- **Positive:** kernels stay back-end-agnostic and fully tested; adding a 4th code = one adapter; invariants are enforced structurally (typed errors) rather than by convention; ML is quarantined behind a gate.
- **Negative / cost:** an extra thin contract layer and per-back-end adapter maintenance; continuum-in-provider-potential (D-1) adds an atomx-owned solve step to every external-back-end run.
- **Neutral:** existing native path is unchanged and becomes the reference provider, so every current benchmark keeps guarding regressions during the whole epic.
