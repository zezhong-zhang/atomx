# High-n s-state self-energy repair

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The explicit deuterium audit exposed a negative 8s self-energy. The production
Mohr backend used FAC's `Klamaq` expression whenever n>5. For s states that
expression gives `F=-(4/3)ln(k0)+1/2`, omitting the leading logarithm and using
the non-s magnetic formula at l=0.

CODATA 2018, equations (14)-(15), gives for s states:

`F = (4/3) ln[(Z alpha)^(-2)] + 10/9 - (4/3) ln(k0) + O(Z alpha)`.

Source: <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=932441>, page 10.
The local FAC `faclib/coulomb.c` implementation was checked directly; copying
its fallback was not sufficient to establish physical correctness for s states.

The backend now restores these leading terms using NIST Bethe logs. Tabulated
n<=5 results and non-s fallback results are unchanged. Higher-order one-loop,
radiative-recoil, finite-size and higher-loop terms remain outside this repair.
The result is still a model QED calculation, not a precision QED prediction.

## Ledger repair

Nine verified live production workers were terminated before changing the
backend. All interrupted rows and affected completed rows are retained in
`high_ns_qed_history` in each v3 ledger. The migration requeued 130 main-ledger
results, 32 isotope results and nine interrupted jobs. Its live-process gate
refuses to reclaim an active owner. Workers restarted after the pilot passed.

Ledger metadata now records `mohr_high_ns_policy=codata_a41_a40_v1`. New Mohr
results carry `_mohr_high_ns_policy` in their correction components. The
corrected-ledger audit requires this marker for affected configurations. The
v3 Breit/recoil/nuclear model identifier remains unchanged; this policy marker
and preserved history distinguish the self-energy repair within that model.

## Deuterium control

The 8s self-energy changed from -7.586829755e-10 to +2.558791829e-9 Ha.
The 1s–8s excitation residual against the archived NIST level changed from
-0.000861928449 to -0.000133826892 cm^-1. This is an improvement by about 6.4,
but remains much larger than the quoted NIST level uncertainty of 2.3e-7 cm^-1.
The repaired deuterium ledger has all 39 configurations completed and passed
the stored-energy consistency audit.

Validation: 27 QED/table/correction-stack/ledger-audit tests and one migration
test passed. The migration test checks history retention, selective requeue,
refusal to overwrite an already migrated queue, and refusal to reclaim a live
worker. The formula test uses independently written CODATA coefficients.
