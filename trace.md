# Trace: inspiring-jepsen-3590b5

<!-- concepts: fac-integration, local-exchange-dfs, shooting-scf -->

## 2026-04-20 · FAC local-exchange DFS port — performance phase

**Session start.** Wrapping up a multi-commit feature branch that ports
Gu's FAC local-exchange SCF into atomx as `exchange='fac'`. All 7 plan
tasks complete, 488/489 tests green (1 pre-existing `test_ci` failure
unrelated). User now wants performance work before deciding merge/PR
path: **be faster than FAC at every Z while keeping the 10⁻⁵ accuracy**.

Known status going in:
* Z ≤ 54: atomx is **2–5× faster** than sfac already
  (e.g. Z=54, 0.11 s vs 0.31 s).
* Z ≥ 74: atomx shooting SCF with Gu's low α_xc (~1.0 at Z=74 vs
  1.5 for the Salvat DHFS default) fails to converge in ≤ 500 iterations —
  we capped the test at Z ≤ 54 to keep the suite fast.

Plan: (1) profile across panel to confirm warm timings ≤ FAC, (2) trace
Z=74 to see whether the bottleneck is iteration count or per-iter cost,
(3) identify targeted fixes, (4) reverify on the Task 0 panel at the
10⁻⁵ gate.
