# Closed31 DC identity failure: saved-array diagnosis

The [original phaseA failure](../actual/failure.json) remains failed:
65 controls passed, then the exact whole-H comparison with the frozen f043
recovered31 baseline failed. It built one fresh DC H/CI and five complete
sector eigensystems. Static Breit and zero-frequency interactions were never
entered, so no transverse contribution has been measured here.

The [prospective offline contract](contract.json) held both already-closed
1595-dimensional eigensystems, all31 radial P/Q arrays and entire owner
payload fixed. The [foreground receipt](root_execution.json) closed with exit0;
the [result](actual/result.json) passed45 prerequisite/hash controls.
It decoded81,557,328 bytes from two saved archives and performed five full
matrix products. It made zero new H, CI, gradient, Dirac, eigensolver or
optimizer calls. Input and consumer hashes remained unchanged.

| Whole-matrix observation | Value |
| --- | ---: |
| Differing binary64 entries | 177457 |
| Differing diagonal / off-diagonal entries | 699 / 176758 |
| Maximum absolute difference | 7.275957614183426e−12 Ha |
| Old eigensystem residual on old H, all1595 roots | 2.631101109489068e−11 Ha |
| Fresh eigensystem residual on fresh H, all1595 roots | 1.0960341614908495e−11 Ha |
| Old eigensystem residual on fresh H, all1595 roots | 2.9018657436316248e−11 Ha |
| Old / fresh full Gram error | 3.1086244689504383e−15 / 3.552713678800501e−15 |

Entire owner payloads and ordered keys/P/Q bytes match. Both stored H are
symmetric; all three residual controls satisfy the original1e−8 Ha gate,
and both full Gram matrices satisfy1e−10. The complete signed difference
is retained in [the output archive](actual/full_signed_difference.npz).
Large ULP distances near zero and sign changes of tiny off-diagonal entries
are retained as diagnostics; they are not an absolute physical error scale.

For this same real symmetric matrix representation,
`||deltaH||2 <= 1595 * max(abs(deltaH))` bounds any ordered eigenvalue shift.
Twice this bound gives approximately **0.0051 cm⁻¹** for an excitation-energy
difference (using1 Ha≈219474.63 cm⁻¹). This is a conservative mathematical
bound between these two finite matrices, not a bound on missing physics,
state matching outside this space, or total experimental error. It excludes
this particular source-epoch matrix discrepancy as a thousand-cm⁻¹ mechanism.

The exact byte gate has not been loosened or passed. A separate prospective
same-current-source baseline contract would be required before continuing
the static/zero-frequency contribution measurement. No correction can be
spliced onto the later optimized terminal16 spectrum.
