# Task 0 — Physics calibration for the FAC/Gu local-exchange port

**Purpose.** Before writing any runtime code, verify that the closed-form
α_xc(Z, N) formula we intend to port reproduces FAC's converged
α_xc to the precision required by the plan's gate (≤ 10⁻³ relative
L∞, see `2026-04-20-fac-calculator.md` §Task 0).

## What "α_xc" means here

FAC's local exchange is `V_x(r) = -α_xc · (3ρ(r)/π)^{1/3}`, where
`α_xc = hxs · GetHXS(Z, N, ihx)` (Gu 2008 eq.). The converged α_xc is
printed to the `out.pot` header as the `AHX` field
(`radial.c:2804 "fprintf(f, \"#    AHX = %15.8E\\n\", potential->ahx);"`).
That field is the *single-number* output of the full GetHXS chain FAC
evaluated at its default `hxs=1, ihx=-1`. We reproduce it in pure
Python from the same tables and compare.

## Scripts in this directory

1. `extract_AHX.py` — drives `sfac` across a Z × config panel, one
   run per row, and collects `AHX` from each `out.pot` header into
   `calibration.csv`.
2. `verify_formula.py` — loads `calibration.csv`, computes the same
   α_xc with a pure-Python implementation of Gu's formula (to-be-in
   `atomx.physics.local_exchange.get_hxs`), and emits `FINDINGS.md`
   with the residuals.
3. `FINDINGS.md` — final verdict. Must pass the 10⁻³ gate before
   Task 1 starts.

## Gate

| Panel row check               | Threshold                              |
|-------------------------------|----------------------------------------|
| Max |α_atomx − α_FAC| / α_FAC | ≤ 10⁻³                                 |
| Tables hit the documented `k` branch for every (Z, N) | "yes" |
| No surprises in sfac `HXS` / `IHX` header fields     | sfac ran defaults |
