# Ar I MBPT(2): the −1.058 Ha anomaly was a virtual-sum double count

**Status: fixed 2026-09-23** (uncommitted → committed same day; see
`src/atomx/methods/mbpt.py`).  The test
`tests/test_nist_levels.py::test_ar_mbpt2_correlation_energy_is_physical`
stays `xfail` — the gate `|E2| < 0.35` Ha is still violated at −0.682 Ha —
but the reason now records the resolution instead of an open anomaly.

## Symptom

Ar I `mbpt2_energy` on the closed-shell DHF reference returned
E2 = **−1.057787 Ha**, ~1.5× argon's total correlation energy
(~−0.7 Ha).  The original diagnosis (preserved in the xfail reason)
varied the virtual-space controls (`e_virt_max`, `kappa_max`,
`n_virt_max`) and watched the sum swing by orders of magnitude, and
concluded "system-specific anomaly, candidate near-degeneracy in the
3p/3d virtual manifold, needs its own diagnosis".

## Root cause: cross-κ virtual pairs were summed twice

The virtual double sum in `_mbpt2_pair_contribution` iterated

```python
for kappa_r in kappa_list:
    for kappa_s in kappa_list:        # full Cartesian product
```

The antisymmetrised two-electron amplitude obeys

    A_J(ab; sr) = −(−1)^{j_r+j_s−J} A_J(ab; rs)
    ⇒ |A_J(ab;sr)|² = |A_J(ab;rs)|²,   Δε(sr) = Δε(rs)

so the ordered channels (κ_r, κ_s) and (κ_s, κ_r) contribute
**identically**.  Every cross-κ virtual pair was therefore counted
twice.  (Within one κ block the `orb_r_idx ≤ orb_s_idx` guard already
kept orbital pairs unordered — only the κ loops were wrong.)

## Fix

`for kappa_s in kappa_list[ka_r:]` — unordered κ channels only, with
the antisymmetry argument in a comment at the loop.  Same-κ blocks keep
the orbital-index guard unchanged.  Side benefit: the virtual-channel
loop work is halved (3.2M of the 4.0M pre-fix terms were the
duplicates).

A `term_recorder` diagnostic hook was added to `mbpt2_energy` so any
second-order sum can be attributed to its dominant channels without
re-running under a debugger.

## Verification (kept-side reconstruction, no pre-fix code re-run)

Recording every kept term on Ar I and splitting by κ equality:

| quantity | value |
|---|---|
| terms kept | 4,032,400 (same-κ 828,040; cross-κ 3,204,360) |
| same-κ contribution | −0.306761 Ha |
| cross-κ contribution | −0.375513 Ha |
| **E2 kept** | **−0.682274 Ha** |
| same + 2·cross (predicts pre-fix) | **−1.057787 Ha** |
| pre-fix measured value | −1.057787 Ha |

The reconstruction matches the recorded pre-fix number to all six
printed digits: the removed contribution was *exactly* the cross-κ
part counted a second time — no other term moved.

Physical reading: −0.682 Ha ≈ 97 % of Ar's ~−0.7 Ha total correlation
energy.  An MP2 rung recovering ~100 % of the correlation is
conventional (MP2 typically overshoots slightly), so the value is now
plausible; the remaining `|E2| < 0.35` gate failure is the
series-convergence heuristic itself, not a defect.  The panel gate in
`tests/test_mbpt2.py` (He literature MP2/CBS value) passes unchanged —
He is 1s² so its double count lived in the small p½/p₃½ cross channel
and the literature gate has margin.

## Open questions

- Whether to replace the `|E2| < 0.35` heuristic with a
  physics-grounded bound (e.g. |E2| ≤ 1.2 × estimated total
  correlation) is left to a later pass; the xfail documents the
  current state honestly in the meantime.
- The historical virtual-space variation numbers in the xfail reason
  (e_virt_max etc.) were measured with the double-counting code and
  are superseded for predictive use; they are kept as diagnostic
  history.
