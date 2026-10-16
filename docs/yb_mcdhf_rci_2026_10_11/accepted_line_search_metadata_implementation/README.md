# Accepted numerical comparison and exact stored-H guard

The public optimizer now exposes the actual preceding accepted line-search comparison in checkpoint diagnostics. This is a numerical metadata addition. The reported EOL objective, signed public gradient, polar retraction, inverse metric, Armijo expression, tolerances, CI path and cache behavior are unchanged. No public Rayleigh objective or energy callback was added.

The complete356-file candidate archive is [source/](source/). Its map is `500ac4c3c372e0eb51e6d9910ef3830bf8eb323b211b52ba3ec08ccbf7b1b794`; the optimizer hash is `525085b5fc56fcc12d9423d677f067186793d7e83cc6ed12487a03e529de76e1`. [source_identity.json](source_identity.json) binds each dirty source file. [before/](before/) preserves the exact previous356 epoch `0e5496…`, including its optimizer `b3814054…`. Only that optimizer file differs. The current-source diff is [source_change.diff](source_change.diff), independent of Git HEAD and unrelated earlier work.

The728 [source checks](source_verification.json) pass: removing exactly the approved insertions recovers every prior optimizer byte; all355 other source files and all356 archive files match their identity. Syntax compilation did not execute the module. Ruff passes. This does not establish runtime metadata lineage: a future authorized public solve must validate the actual comparison and samepoint H/gradient. No additional optimizer test or physical run was executed here.

At checkpoint zero, `diagnostics['accepted_line_search']` is `None`. Later checkpoints receive a fresh shallow copy with these primitive fields:

| Field | Meaning |
| --- | --- |
| `parent_energy_ha`, `trial_energy_ha` | Actual energies in the preceding comparison |
| `step`, `directional_derivative` | Actual pre-retraction step and `g·direction` |
| `energy_change_ha`, `armijo_rhs_ha` | Actual compared change and unchanged `1e-4*step*derivative` |
| `attempt`, `backtrack` | Zero-based main/fallback attempt and backtrack, including earlier failed evaluations |
| `stable_callback_used` | True only when an existing callback actually replaced the change |

Progress callbacks and final returned diagnostics retain their previous fields. The numerical comparison remains the previous one; the metadata does not infer slope from the finite polar displacement.

## Pure-array consumer

[exact_reported_guard.py](exact_reported_guard.py) is an owned reusable consumer, initially outside the public physical API. It performs no Atomx import, physical builder, eigensolve, radial increment, CI solve or gradient. The caller first validates the complete retained owner at the actual point, whole eigenpairs and Gram matrices, physical targets and sectors, source/objective identity and original radial manifold. Input mutation during the call is unsupported.

```python
budget = ExactGuardBudget()
point = exact_reported_point(
    retained_H, selected_whole_signed_columns, selected_reported_roots,
    actual_target_weights, actual_reported_EO,
    target_ids=canonical_unique_UID_strings, budget=budget,
)
point_record = point.ledger()
decision = reported_acceptance_decision(previous_point, point,
                                        diagnostics['accepted_line_search'])
```

At zero, construct and save the point without a comparison. Each later point uses its own validated H; cache only the previous immutable arithmetic certificate. The budget is17 points including zero,85 target quotients, at most5 targets per point and3 million **examined** upper-triangle signed pairs. Nonzero H pairs are counted separately. There is no small-amplitude cutoff: every exact nonzero CI component and all its signed pairs are examined; only exact H zeros are omitted from accumulation. A budget exit preserves the previous verified checkpoint.

For each supplied binary64 column, integer dyadic accumulation forms the exact finite quotient `q=cᵀHc/(cᵀc)`. The off-diagonal factor2 and actual per-state norm are retained. Each actual binary64 weight is used without renormalization. Let `D=Σwq−EO_float`, which includes both `q−λ` and the actual averaging offset. For the actual recorded change Δ and RHS R, the conservative exact margin is

`m = |D0| + |D1| + |exact(EO1−EO0)−Δ| + |exact(1e−4*step*slope)−R|`.

The guard requires `Fraction(Δ)+m <= Fraction(R)`. The inequality is evaluated in exact rational arithmetic; the displayed float margin rounds upward and never decides the gate. A passing guard implies the normalized expectations of the supplied stored matrices satisfy the exact product RHS of the original comparison. It does not certify radial assembly error, physical-H accuracy, true eigenvalue precision, the finite-CI `c′` gradient term, orbital stationarity or the experimental spectrum. Quotients remain diagnostic and never replace reported energies.

The optimizer has already numerically accepted a candidate and evaluated its gradient when the checkpoint arrives. The driver must save the candidate and guard result before calling it a **verified** point. On an unsafe candidate, save forensic arrays and the previous verified state, then stop before another update; do not publish the candidate as qualified or continue silently. The original residual `1e-7`, energy indicator `1e-8`, complete eigenpair and identity gates remain separate.

## Qualification and retained failure

The first [array invocation](array_execution.json) failed in the saved-data selector: the targets list also contains D/P multiplet summary rows without a CI column. Its contract, helper, driver, traceback and failure remain intact. [array_failure_disposition.json](array_failure_disposition.json) identifies the unexecuted saved-Yb quotient gates. Its analytical counters are explicitly reconstructed from the closed driver and traceback, rather than presented as a serialized live ledger.

Root reviewed [array_qualification002/driver_change.diff](array_qualification002/driver_change.diff), [contract](array_qualification002/contract.json) and [seal](array_qualification002/root_review_seal.json), then authorized one successor. The only scientific-selection change filters the seven metadata rows to the five labeled states; their UID and0.2 weights match the frozen bare physical reference. Output paths/import wiring changed for independent archival. Helper, source, arithmetic and gates stayed unchanged.

The successor executed once and passed19 arithmetic checks: independent full-`i,j` signed Fraction references, relative and whole-column phase, actual norm scaling, unequal-weight ensemble, a resolved acceptance, refusal of a merely rounded reported decrease, and all five saved-Yb quotients against the independent Decimal80 ledger. Its [result](array_qualification002/result.json) records8 points,13 quotients,115159 examined pairs and54690 nonzero H pairs. CPU31/BLAS1,1GiB/30s was respected; no scientific producers ran. The0.164s launcher wall time is observational and is not a production performance comparison.

This19-check arithmetic qualification and the728-check source proof are different evidence. The Yb correlated29 rotation derivative qualification failed its unchanged gate; actual optimization is **HOLD**. No16-step run, MCDHF convergence, improved experimental gap, or Rayleigh production qualification follows from these results.
