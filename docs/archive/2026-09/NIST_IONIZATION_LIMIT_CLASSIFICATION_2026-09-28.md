# Ionization-limit rows are not bound levels

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The full standard level inventory contains 6,128 rows whose term is `Limit`.
Some carry the preceding configuration label through NIST table context. A
bound-state energy calculated from that label must not be compared with the
threshold energy in the limit row.

The level preparation and comparison now classify these rows as
`ionization_limit`, retain their original source energies and uncertainties,
and leave bound-level predictions/scores empty. The isotope-level comparison
does the same. Configuration and ground-configuration centroid builders exclude
limit rows from their weighted manifolds. The separate ionization comparison
continues using its NIST ionization references.

The refreshed inventory retains all 119,394 standard rows:

- 109,030 classified bound-level rows with numeric energy and zero reference;
- 6,128 ionization-limit rows;
- 1,983 unresolved configurations;
- 844 rows lacking an evaluated zero reference;
- 1,409 other rows with missing numeric energy.

No existing calculation result was overwritten. The preparation script remains
idempotent; rerunning it added no configurations. Tests exercise a limit row
with a plausible orbital label, preserve it without a fabricated prediction,
and verify it cannot make an otherwise complete J manifold appear incomplete.

An additional notation audit found 130 rows using orbital letters beyond `o`,
including 16 limit rows. Supporting the remaining 114 bound-level rows still
requires extending the parser and verifying the numerical/QED path. Examples
from the archive include `13q`, term `2Q`, J=25/2, and `20y`, term `2Y*`, J=39/2.
NIST's notation and n/l/j rules are described at
<https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy-10>.
