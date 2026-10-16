# Native31 FD representation-only successor — prepared, not run

The original [failed execution](../execution_preparation001/RESULTS.md) is
unchanged. Public `gradient_from_weights` produces one contiguous246-entry P+Q
vector per orbital; the qualified writer saves31 such vectors as31x246.
The consumer now checks `(31,246)` and feeds `corrected[i]` directly to exact
represented-number arithmetic. The [two-line diff](representation_diff.patch)
preserves the saved dtype and `as_integer_ratio` without casts or reshaping.

The [source census](representation_review.json) checks every other archive
representation from its writer/API source: P/Q31x123, coordinate vectors3689,
seven original246x119 projectors, group inverse119x119 blocks, full1595 H/C,
and two119x2 antisymmetric direction blocks. No numerical archive was opened
or rehashed. The physical f043 source, own recovered31 point, UID/.2 weights,
reported production objective, finite-CI response, exact fullsigned Lambda/Gram
algebra, both h/gates,5H5CI0g/70hS and CPU31/3.95GiB300+315s remain identical.

24 SOURCE/JSON/AST/refusal controls and Ruff pass. This is concrete prepared
code, not a passing FD experiment. The original run's two FD domains remain
unavailable; it establishes no FD prerequisite or optimization authorization.
A distinct exact root binding is required before any new numerical imports.
Both predeclared ordinary and independent centered domains are saved separately,
and an ordinary FAIL cannot be upgraded by a centered PASS.
