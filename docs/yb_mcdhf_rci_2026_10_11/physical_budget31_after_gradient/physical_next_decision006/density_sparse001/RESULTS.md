# Actual Yb angular density storage qualification

**PASS: 199/199 checks.** The current native `CIRadialDensity` uses the
existing exact sparse CSF projection. The retained
[`source_before.py`](source_before.py) and current implementation were run
against the same actual angular dependencies and the same saved full CI
amplitudes. The complete A1595/B2574 CSF spaces and all 31 orbital identities
were preserved. No Hamiltonian, CI, radial solve or orbital gradient was run.

| Complete model | Dense projection bytes | CSR projection bytes | Projection nnz | Maximum radial-key difference |
|---|---:|---:|---:|---:|
| A, 91 NR / 1595 CSFs | 19,988,256 | 219,864 | 15,671 | 7.11e−15 |
| B, 100 NR / 2574 CSFs | 48,996,952 | 475,304 | 35,559 | 7.11e−15 |
| Total | 68,985,208 | 695,168 | 51,230 | 7.11e−15 |

These are actual retained projection array payloads: dense FP64 versus CSR
data, column indices and row pointers. They exclude angular tables, Python
objects, caches and solver workspaces. The payload reductions are 90.91× and
103.09× for A/B. They are storage ratios, not runtime speedups or total-memory
reductions.

For each model, 12 full-key comparisons covered the five-state equal-weight
`.contract`, five unfiltered diagonal `.contract_bilinear` results, five
signed transitions to the second root of the same full symmetry sector, and
one fixed random signed pair. Every before/after union key was compared at
the original 1e−12 absolute threshold. **There were no keyset differences.**
All original signed projection coefficients agreed exactly, with unchanged
column order and shared angular structures. The complete C Gram passed
1e−10, and all five UID/column/energy bindings matched the saved results.

Different floating-point accumulation orders produced sign changes in 11/6
unfiltered cancellation-level values for A/B. Their largest magnitudes were
7.78e−21 / 1.09e−21. These values and every full signed weight remain in the
saved output; none was removed to pass a gate. The implementation's existing
`.contract` threshold was unchanged.

The actual run used four density constructors, 20 projection calls, ten
angular coefficient tables and all 43 predeclared highest-weight
nullspaces. It bound 358 actual dirty source files, verified all 77 imported
Atomx module hashes, and checked eight whole input files. All 356 frozen
f043 archive source files remained unchanged. Before and after used the
same current angular module, whose bytes match frozen f043.

One authorized run used CPU25, one thread, a 6 GiB address-space cap, and
300/315 s inner/outer limits. It finished in 85.82 s at 2091.63 MiB peak RSS;
these costs are uncontrolled diagnostics. Ruff and the relevant diff check
passed. No additional solve or retry was performed.

The [complete result](actual/result.json),
[compact closure summary](closure_summary.json),
[predeclaration](contract.json), [driver](compare.py), and
[receipt](receipt.json) retain the evidence. `actual/partial.json` is a
historical intermediate `RUNNING` checkpoint; the final result and closure
summary are authoritative. Completed scientific outputs were not rewritten.

This qualifies the representation change at these existing spaces. It does
not resolve full 4f12 angular-table construction, certify selected-state
tracking/ownership, improve the experimental spectrum, or change the three
previous D3/2 derivative failures and failed stationarity gates. Root owns
the shared-source increment, live tests and current guide updates. No new
general insight is promoted from this narrow storage measurement.
