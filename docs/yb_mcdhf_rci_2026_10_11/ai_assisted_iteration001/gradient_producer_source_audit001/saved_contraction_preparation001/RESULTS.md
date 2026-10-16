# Complete saved-H contraction replay, closed

The [frozen preparation](README.md) was executed in the foreground under its
original CPU28/thread1, 1.5 GiB AS, inner120s/outer135s bounds. The
[root receipt](root_execution.json) records exit0, no wall stop and all389
source/input guard hashes unchanged. [The result](actual/result.json) passes
96/96 controls. Elapsed0.732s is uncontrolled, not a performance benchmark.

Only two already closed1595-state archives were loaded: StageB baseline and
h0-plus. All complete eigenvectors, residuals, Gram matrices, five physical
UIDs, signed phases and actual norms were checked. Four full H-times-C
products,20 matrix-vector products and five Delta-H contractions were used.
No H/CI/gradient, optimizer, radial key or new physical experiment was run.

| Five-state weighted quantity | Ha |
| --- | ---: |
| Complete-matrix radial contraction | 3.8321583218072087e−7 |
| Complete signed finite CI response | 9.82059091139472e−12 |
| Direct stored-H quotient change | 3.83225652771617e−7 |
| Matrix radial + CI response − direct quotient | +1.5234702329271956e−20 |
| Saved radial polynomial − matrix radial | +2.8954207639972986e−11 |
| Direct quotient − saved exported quotient | −9.27182612707626e−20 |

At this specific pair, finite CI response and saved quotient export reproduce
the complete-matrix arithmetic. The radial-polynomial discrepancy is already
present before adding the CI response. This narrows the investigation to the
radial-increment versus stored-H producer routes, including contraction/order
and projection/symmetrization arithmetic. The replay has not isolated a single
operation, proved exact mathematical zero, repaired a gradient or qualified
optimizer continuation. The source audit is [one directory above](../README.md).

The original StageB ordinary finite-difference failures remain failed, as do
stationarity and experimental gates. This fixed saved-pair arithmetic closure
does not measure missing physics or justify transfer to another point/source.
