# Failed finite mixed-CI qualification: independent saved-array audit

The live qualification failed its original constrained central-derivative gate.
It ran 26 passing cases, then stopped at the first derivative direction under
`pytest -x`. The 16 unsupported-input cases and one legacy-guard case were not
executed. The second direction inside the failed test was also not reached.
The [original result](../actual/result.json), [root execution](../root_execution.json)
and [failure log](../root_execution.log) are retained without modification.

This separate [prospective audit contract](contract.json) authorized one
CPU thread, 1 GiB address space and 30 seconds, with no Atomx imports, H builds,
CI solves, gradients, Dirac solves or optimizer calls. The independent
[saved-array verifier](verify_saved.py) completed in 0.1513 seconds with
58.47 MiB peak RSS. Its [verification](verification.json) records 1,341 passing
**audit consistency** controls. `numerical_qualification_passed` remains false.

The audited finite model has 29 CSFs, all four captured symmetry blocks
(12, 9, 7 and 1 CSFs), seven radial orbitals, `ns=35`, and complete real
Dirac–Coulomb H/CI arrays at thirteen fixed points. The two inherited physical
target UIDs are `(2J=0, parity=+1, root=0/1)` with normalized weights 0.25/0.75.
Frozen producer/table identities establish these labels; this audit checks the
complete captured block partition and target columns without re-enumerating CSFs.

The complete spectra satisfy the original gates at every saved point:

| Saved-array control | Maximum observed | Original gate |
| --- | ---: | ---: |
| Whole-column CI residual | 3.3411e−15 Ha | 1e−8 Ha |
| Full CI Gram error | 2.7756e−15 | 1e−10 |
| Actual radial S-Gram error | 1.6653e−15 | 1e−10 |
| Radial/native versus stored Decimal increment discrepancy | 7.6572e−19 Ha | 1e−13 Ha |
| Total finite increment versus independent saved-matrix response | 7.6572e−19 Ha | 1e−12 Ha |

All complete blocks retain exact off-block H/CI zeros and complete root counts.
The original P-times-coordinate radial points and P-times-increment arrays
match bitwise. The five frozen ordinary H comparators and complete baseline
signed gradient also match bitwise. These comparisons establish unchanged
default results; they do not establish the physical accuracy of that gradient.

Independent contraction of the complete saved signed angular tables reproduces
233 retained terms and 67 negative terms at every point. Full signed coefficients
are saved in [signed_density_replay.json](signed_density_replay.json).
An independently written 80-digit Decimal calculation recomputes the CI finite
Rayleigh/residual response from each saved H/C/E pair, preserving phase,
normalization and finite residual terms. The thirteen producer Decimal radial
strings are hash-bound and checked for native agreement and domain additivity.
The original 25 full radial Decimal matrix domains are **not regenerated** here;
their original thirteen live-test passes are reported execution evidence.

The derivative discrepancy is independently reproduced from the complete saved
baseline gradient and the original P-projected tangent. Both values below use
the saved `±1e−4` points and the unchanged `max(1e−8, 1e−5 |derivative|)` gate:

| Direction | Gradient dot tangent | Central difference | Absolute discrepancy | Relative discrepancy | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | −0.00872046743791691 | −0.008720635851431197 | 1.68414e−7 | 1.93124e−5 | FAIL |
| 1 | +0.012938447545994816 | +0.012940561696127383 | 2.11415e−6 | 1.63401e−4 | FAIL |

Direction 0 exactly reproduces the failure log. Direction 1 is an independent
saved-array observation; the original live test did not reach it. This fixed-step
discrepancy does not distinguish finite-step error from an analytic-gradient
error. No step sweep, new physical invocation, tolerance change or source edit
was performed. The finite increment agreement does not resolve the derivative
gate, authorize optimizer/Armijo use, establish a general error bound, or improve
the previously failing Yb orbital-stationarity gate.

The original completed physical calls remain 18 H builds, 13 complete CI solves
and three full gradients, including the old-source comparator. This saved-array
audit adds zero such calls. The complete numerical evidence, target columns,
response errors, resource record and unexecuted-case list are in
[replay_details.json](replay_details.json). Source map
`0e5496eb7eab63df2e44d08c4d986574821b33b806a7d3c760e243bda5cc2193`
and all 38 input-file hashes are bound in the audit proof; frozen 356 source
files are checked directly without importing them.
