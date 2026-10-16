# Saved central-difference scale diagnosis

The closed finite qualification remains failed. Saved results support coarse
central-difference truncation as the cause of its derivative failure; they do
not establish a new numerical qualification or warrant a source/gate change.
This audit imports NumPy for saved-array algebra only, with no Atomx imports,
H build, CI solve, gradient call, orbital update or finite diagnostic evaluation.

| Direction | Saved analytic derivative | D(1e−4) error | D(1e−6) error | D(1e−8) error |
|---|---:|---:|---:|---:|
| Seed413711 | −0.008720467437916912 | −1.6841351428535656e−7 | −2.5927678407033916e−11 | −7.872247097251428e−9 |
| Seed413712 | +0.012938447545994816 | +2.1141501325670775e−6 | +2.2519264751552637e−10 | +3.993151462267086e−9 |

Both saved1e−6 and1e−8 comparisons meet the original rel1e−5/abs1e−8 derivative
criterion. Both1e−4 comparisons fail it; original pytest stopped on direction0,
so direction1's original derivative test and unsupported guards were not reached.
The h=1e−4→1e−6 error reductions are6495.51 and9388.18; observed orders are1.9063
and1.9863. Two-scale h² extrapolation errors are−9.0872e−12 and+1.3779e−11.
These independent saved-step trends support a coarse central O(h²) truncation
explanation. There are no additional half-steps to establish a clean asymptotic
order experiment, and the extrapolation is an explanation, not an acceptance
substitute. The1e−8 error increase indicates a smallest-step noise regime; this
record does not isolate bank, endpoint, eigenpair or summation rounding as its
single source. No persistent analytic-gradient mismatch is demonstrated by
these saved scales.

[Full result](result.json) includes every signed ±energy/radial/reconstruction/
CI-response component, central components at all three scales, original gate
values, full saved13-point residual/Gram controls, tangent-normalization controls,
and complete target identities. [Verification](verification.json) binds the
result and original immutable inputs. [Prospective contract](contract.json)
binds the closed failure,13CI arrays, geometry, gradient, signed diagnostics,
source identity, test and original execution contract. [Readonly driver](run.py)
performs no physical evaluation.

The prospective contract was written before the one saved-array execution. The
owner interpreted the delegated instruction to prepare a contract before
execution as authorization to run that saved-only audit. Root subsequently
clarified that its driver review and explicit approval were expected first;
that message arrived after this one calculation. All output is retained, root
has been informed, no additional calculation is performed and these files do
not claim a root execution seal. Original scientific evidence and candidate
source/test/contract/gates remain unchanged.

The next action is root review of this explanation and the preserved failure.
A numerical qualification successor, if chosen, requires its own concrete
contract and authorization; unsupported guards still need actual execution.
