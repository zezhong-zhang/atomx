# Uranium density CLI: performance and model correction

Measured on 2026-09-24 using the project `.venv/bin/python`.

`scripts/density_gs.py` documented a B-spline statistical configuration
average (`term=AV`), but inherited `Atom`'s shooting/CI defaults. Uranium
therefore entered a 386-CSF state-specific calculation while its exported
density still used statistical occupations. Repeated stack samples found
it constructing exact CI/Gaunt angular tables.

The script defaults to B-spline CI-averaged MCDHF. Statistical DHF is
optional via `--averaging statistical`. In that mode, the Gaunt
correction is evaluated as the trace over the same jj-configuration
ensemble: distinct magnetic spinor pairs have probabilities
`<q_i(q_i-1)>/[g_i(g_i-1)]` within a subshell and
`<q_i q_j>/(g_i g_j)` between subshells. Contracting these with direct minus
exchange Gaunt integrals avoids constructing CI matrices. `--correlation s/sd`
expands the default CI configuration space and requires CI averaging.
Both modes export the solved occupations, including occupied correlation
orbitals, rather than imposing statistical occupations on a CI result.

The CLI also defaults BLAS/OpenMP to one thread before importing numerical
libraries. Explicit environment settings take precedence. An uncontrolled
threading trial remained inside small LAPACK eigenproblems after two minutes.
No grid reduction, tolerance relaxation, or correction disabling was used.
In statistical mode, Gaunt and QED remain post-SCF energy corrections.

| Run | Measurement |
| --- | --- |
| Original CLI, cProfile plus stack sampling | Still constructing CI/Gaunt tables after 330 s; interrupted |
| Statistical uranium generation, without cProfile | 70.6 s, 22 SCF iterations, 29 subshells |
| Statistical uranium with cProfile and WF_TXT comparison | 78.0 s total; 75.7 s generation |

These are local timings with other validation work running, not an isolated
microbenchmark. These completed uranium timings apply to the optional
statistical model, not to the CI default. They do not establish a CI speedup
or equivalence to a resolved CI ground-state density.

Validation: 100 focused physics/numerics tests passed, plus the separate
open-shell CLI export test. The new ensemble contraction agrees with
degeneracy-weighted resolved-CSF traces for closed, one-electron, p, d, f,
and mixed-shell configurations, for both statistical and equal jj weights.
The existing test that rejects an unspecified open-shell Gaunt state
continues to pass.

For statistical uranium, the exported native-grid trapezoidal electron count is
92.01312 (relative quadrature error 1.43e-4); no normalization was forced.
Against `dataset/WF_TXT/92.txt`, the existing comparison reports
`L1 = 0.2771 electrons` and relative `L2 = 0.001997`.

Run the default CI calculation, including comparison when the reference exists:

```sh
.venv/bin/python scripts/density_gs.py 92 --force -v
```

Reproduce the statistical benchmark with `--averaging statistical` added.
Existing HDF5 files are rebuilt when the averaging, correlation expansion,
or engine differs (including older files missing model metadata), or when
`--force` is supplied. New files record
the engine, averaging, correlation, corrections, tolerance, and iteration count.
