# Native31 bounded orbital relaxation

The sole native Atomx f043 optimizer returned after 16 new steps. All 1,582
live controls and all 16 exact finite stored-H acceptance checks passed;
the [scalar closure](completion.json) passed 77 controls. This is a
nonstationary MCDHF iterate in the declared 31-orbital, complete1595-CSF,
69-electron DC model. Neither convergence nor experimental accuracy passed.

| Quantity | Initial | Literal terminal16 |
|---|---:|---:|
| Reported equal-five-state EOL, Ha | −14067.42402931245 | −14067.496296074172 |
| Original projected-metric residual | 0.005162180923400686 | 0.68254770188387 |
| Original energy indicator, Ha | 0.00009089562518897 | 0.6018594158445788 |
| Four-line nominal NIST RMS, cm−1 | 1915.9833377599443 | 1283.7699121117464 |

The original residual1e−7 and indicator1e−8 gates both failed at every
checkpoint. The indicator is not a rigorous energy or spectral error bound.
The finite stored-H EOL descent is −0.07226676172285806Ha. A lower EOL and
smaller nominal RMS do not certify a correct stationary spectrum.

| Physical target | Energy change, Ha | Transition shift, cm−1 | Terminal line, cm−1 | Nominal gap, cm−1 |
|---|---:|---:|---:|---:|
| 6s1/2, lower | −0.06517668547894573 | 0 | 0 | 0 |
| 5d3/2 | −0.08443256705504609 | −4226.17751048693 | 21464.301557320276 | −1496.498442679724 |
| 5d5/2 | −0.08211102844870766 | −3716.658680665500 | 22852.53050680512 | −1480.159493194879 |
| 6p1/2 | −0.06634448254771996 | −256.301831176181 | 25725.279608163462 | −1336.540391836537 |
| 6p3/2 | −0.06326904508387088 | +418.678672482489 | 29779.416149205746 | −612.813850794253 |

The 5d centroid changed from26217.705139605252 to22297.23892701118cm−1,
and its splitting from878.7101196634139 to1388.2289494848446cm−1.
The 6p centroid changed from28234.352130928717 to28428.03730219165cm−1,
and its splitting from3379.1560373836146 to4054.136541042284cm−1.
Both lower and upper states belong to each same-point DC calculation; no
patches from another orbital point or Hamiltonian were added. NIST was
reporting only and selected neither a point nor weights.

The abrupt increase starts at12→13: tangent norm0.1974986→2.1989705,
residual0.00843694→0.1539784, while EO decreases0.00634220Ha.
Terminal tangent norm is5.0626554. All five original physical UIDs remained
in their respective symmetry's local root0. Full normalized and raw-h/
ket-normalization-corrected residuals agree with retained residuals to
approximately1e−13 along the trajectory. The metric, inverse blocks,
seven original nuclear projectors, shared objective and .2 weights remained
unchanged. These facts restrict explanations but do not establish a full
derivative certificate or identify why the search entered this region.

Actual work was23H/23CI/51weighted gradients,85 complete unfiltered target
tables, one public optimizer and17 checkpoints. Cumulative exact guard:
85 quotients,4,623,405 examined pairs,2,001,021 nonzero H pairs. There were
no line-search fallbacks and no energy callback. Root foreground exit0,
265.988991s uncontrolled wall time; this is not a speed claim. No candidate
failed qualification; literal accepted16 is the last qualified checkpoint,
not a best-gap selection. No continuation is authorized by this closure.

The [receipt](../root_execution.json), [actual result](../actual/result.json)
and [completion](completion.json) are bound by
[closure_docimpact.json](closure_docimpact.json). Closure consumed JSON and
saved scalar/Fraction ledgers only: zero NPZ reads/rehashes, zero numerical
replay or scientific calls. It verifies archived ledger consistency and
does not constitute an independent second H/C/gradient solve. Current-source
regression failures and the earlier failed31 FD representation attempt remain
immutable and separate from this historical f043 run.
