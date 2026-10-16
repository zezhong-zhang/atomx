# Same-epoch transverse31 execution: DC complete, static timed out

The new same-source baseline reproduced the complete DC H exactly and passed
all86 completed controls. The declared1200s inner wall stop then interrupted
the static H construction. **No complete static or independent zero-frequency
RCI spectrum was produced.** This execution supplies no new Breit transition
shift and does not qualify a current-source MCDHF optimizer.

The [closed failure](actual/failure.json) records `TimeoutError: Declared1200s cap`.
The [root receipt](root_execution.json) has foreground exit1, elapsed
1200.481891691s, source/input identities unchanged and no controlled timing
claim. Its outer `resource_stop` is null because the inner driver caught and
saved its own timeout; this is not absence of a resource stop.

| Actual operation | Entered | Returned complete |
| --- | ---: | ---: |
| Full H / native CI | 2 / 2 | 1 / 1 |
| Complete target-sector eigensolves | 5 | 5 |
| Transverse factory | 1 | 1 |
| Static scalar requests | 261889 | 261889 |
| Independent zero-frequency Helmholtz kernels | 0 | 0 |
| Gradient / Dirac / optimizer / polar | 0 | 0 |

The completed model is DC only: all31 recovered radial orbitals, the complete
1595-CSF five target sectors, five original physical UIDs and their0.2 weights
were retained. The source epoch is the archived current365 implementation.
The DC baseline comes from the already closed current-source DC attempt,
not the historical f043 bytes whose exact H gate failed. The original failure
and the [saved-array difference diagnosis](../saved_DC_failure_diagnosis001/RESULTS.md)
remain unchanged. This is the originally recovered31 point, not terminal16;
no correction may be transferred between them.

The same-epoch completed DC archive SHA is
`22d8b69d958663ec8cb915ad225eccb2e92032914a7ec342058a2c4e1c46ee9e`.
Its complete spectrum controls, signed cluster comparisons and the completed
raw h₀/S₀ blocks at failure are retained in `actual/`. H₀/S₀ had7 misses;
the same-orbital static stage reused all seven blocks. Preserving those
one-body blocks does not imply completion of the transverse matrix.

The failure SHA is
`949931f460f805fb4a63be7ec8366089ec05f943e1741a4ecd818fe3063ffe20`,
and the root receipt SHA is
`b31d681c8dcbfba2f9cf4d026e7fa6070c9f814eb686784654ba81292de9a052`.
No retry, shortened model, changed physical gate or correction inference is
part of this closed experiment.

The [source-only reuse investigation](../../source_batch_reuse_note001/README.md)
identifies static Gaunt pair waves/potentials as an exact-reuse candidate.
That is a code observation, not a measured runtime bottleneck or an implemented
speedup. A separately bound candidate and complete fixed-key equivalence check
must precede adoption or another full31 measurement. The old operator source,
static/zero replacement convention and original frequency-related limits
remain in force.
