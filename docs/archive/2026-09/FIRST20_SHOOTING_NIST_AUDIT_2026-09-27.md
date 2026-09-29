# H–Ca shooting benchmark and NIST source audit — 27 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## What was calculated

`scripts/benchmark_first20_shooting.py` ran nonlocal Dirac–Hartree–Fock
shooting for each neutral element H–Ca, projected its spinors onto the same
75 Bohr B-spline physics grid used by the existing compact-ground jobs, and
completed the B-spline solve with Breit, QED, and the selected isotope's
leading one-body normal mass shift. Shooting was an initial guess; its raw
energy was never substituted for a corrected B-spline result. The benchmark
uses each first ion's previously validated ground energy to form an
ionization difference. Full row data and failures are checkpointed in
`output/nist_spectra/first20_shooting_benchmark.jsonl` and displayed on the
interactive theory comparison page.

All **20 of 20** seeded neutral calculations converged and passed the
orbital-order, radial-node, and projection gates. The largest difference
from the direct B-spline ground energy was **5.44×10⁻⁹ Ha**, versus the
benchmark's **10⁻⁷ Ha** branch threshold. The largest projection L2 error
was **2.30×10⁻⁵**. The 20 calculation times totaled **24.6 s** with one
BLAS thread on this host. This is strong evidence that the first-20 neutral
ground discrepancies with NIST are not a wrong shooting/B-spline root.

## Which NIST values are comparable

NIST's [Ground States and Ionization Energies help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html)
defines the listed ionization as removal from the *specific ground level*.
The H I entry is parenthesized, which NIST identifies as theoretical rather
than evaluated experimental data. Of the other 19 neutral entries, seven
(He, Li, Be, Na, Mg, K, Ca) have unique ground levels on both sides of
ionization, so their comparison with a single-configuration DHF energy is
direct; their RMS relative ionization difference is **10.13%**. The other
twelve neutral entries contain open-shell multiplets. Comparing their DHF
configuration averages with NIST ground terms gives **9.41%** RMS but mixes
different observables. Where complete evaluated J manifolds are available,
the repository's `2J+1` centroid reference gives **7.62%** RMS for those
twelve. The shooting seed reproduces the old energies, so it does not by
itself repair correlation, configuration mixing, or term-resolved energies.
The website's default H–Ca ionization score combines these **19 evaluated
like-for-like references** (seven unique-level comparisons and twelve
complete evaluated open-shell configuration centroids); its RMS relative
error is **8.63%**. H I's theoretical ionization and the mixed
configuration-average-versus-ground-term values remain visible outside
that score.

The hydrogenic Ritz map is a different benchmark: it uses one-electron ions
and often level-derived rather than independently observed wavelengths.
On the same observed H I 1s–2p (J=3/2) row, the wavelength is 121.56699 nm;
point-nucleus Dirac predicts 121.50025146 nm, while corrected mass-aware
DHF+Breit+QED predicts 121.56704187 nm. The respective photon-energy
errors are +0.05493% and −0.000043%. The DHF value is a configuration
average rather than a resolved 2p fine-structure component. Thus the large neutral-atom residuals
do not show that the mass-aware DHF radial solution is worse for hydrogen.
They show that the neutral multielectron and one-electron Ritz panels
measure different transitions and include different reference types.

## First-20 ionization error budget

`scripts/audit_first20_dhf_error_budget.py` independently recomputed both
ground stages for every H–Ca neutral ionization on the production grid. It
checked each against the stored energy within 10⁻⁷ Ha and verified that
Dirac–Coulomb + Breit + QED + leading normal mass shift reconstructs the
reported total. All **20/20** pairs passed. The checkpointed component data
are in `output/nist_spectra/first20_dhf_error_budget.jsonl` and are displayed
beside the residuals in the interactive comparison. Selected neutral results
are:

| Ion | NIST ionization (eV) | DHF+B+QED+mass (eV) | Breit+QED+mass change (eV) | Remaining theory−NIST (eV) |
| --- | ---: | ---: | ---: | ---: |
| He I | 24.5874 | 23.4430 | −0.00526 | −1.14441 |
| Be I | 9.3227 | 8.0447 | −0.00110 | −1.27802 |
| N I | 14.5341 | 12.2015 | −0.00155 | −2.33262 |
| O I | 13.6181 | 14.5685 | −0.00277 | +0.95047 |
| Mg I | 7.6462 | 6.6144 | −0.00327 | −1.03182 |
| Ca I | 6.1132 | 5.1344 | −0.00628 | −0.97876 |

For each evaluated H–Ca neutral, the absolute correction change is at most
**0.0072 eV** (Ne I); it is below **1.6%** of the magnitude of that ion's
remaining error. The corrections cannot close the approximately 1 eV gaps.
The gap is not solely a term-averaging artifact: He, Be, Mg, and Ca each
have unique ground levels on both sides. Their single-configuration DHF
ionization energies need differential correlation and possibly smaller
state-specific physics contributions. Open-shell rows additionally mix
configuration averages with NIST ground terms. For example, N I changes
from **−2.333 eV** against the NIST ground-term ionization to **−0.978 eV**
against its complete evaluated configuration centroid. O I changes from
**+0.950 eV** to **−1.276 eV**, because its NIST centroid ionization is
15.845 eV rather than the ground-term 13.618 eV. The interactive H–Ca
panel now offers this like-for-like centroid view separately. Resolved
ground-term ionizations still require state-resolved theory. The
[NIST ionization help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html)
defines the reference as a ground-level ionization, including its evaluated,
semiempirical, and theoretical notation.

The gap has an independent mean-field check. NIST's CCCBDB reports
nonrelativistic Hartree–Fock ionizations near **23.44 eV** for
[He](https://cccbdb.nist.gov/ie2x.asp?casno=7440597), **8.04 eV** for
[Be](https://cccbdb.nist.gov/ie2x.asp?casno=7440417), and **6.61 eV** for
[Mg](https://cccbdb.nist.gov/ie2x.asp?casno=7439954), versus NIST's
evaluated **24.587**, **9.323**, and **7.646 eV**. Their close agreement
with atomx's same-configuration DHF magnitudes is evidence that the large
neutral errors primarily reflect omitted correlation. CCCBDB uses different
Hamiltonians and finite basis sets, so this is a magnitude check rather than
an exact code-to-code validation. For open-shell N and O, CCCBDB's
term-specific HF ionizations cannot be treated as configuration-average
references; the evaluated NIST centroids above are the appropriate
like-for-like comparison.

## One-electron Breit physics correction

The scalar Breit correction previously assigned a self-pair to one-electron
ions. Breit is a two-electron interaction. The
[`DBSR_HF` average-energy coefficients](/home/zzhang/code/dbsr_hf/DBSR_HF/hf_MOD_energy.f90)
explicitly use `q(q−1)` for same-subshell pair counts, which vanishes for
one electron; GRASP builds Breit as two-electron integrals over CSFs.
`~/code/radial/DHFS` uses local Slater exchange and is useful for checking
the radial Dirac solver, but is not an exact-exchange DHF energy oracle.
`compute_breit_energy` now enforces the exact zero for total occupancy one,
and its closed-shell DBSR_HF comparison still passes.

GRASP's RCI input also offers separate normal and specific mass shifts in
[`getcid.f90`](/home/zzhang/code/grasp/src/appl/rci90_mpi/getcid.f90).
Atomx currently applies the leading one-body normal mass term but not the
general two-electron specific mass shift or relativistic recoil. GRASP's
[`rci90mpi.f90`](/home/zzhang/code/grasp/src/appl/rci90_mpi/rci90mpi.f90)
builds transverse two-electron integrals for CSFs; atomx's single-configuration
scalar Breit is a configuration average, not its term-resolved replacement.
The remaining partial-shell Breit pair weighting in atomx is approximate;
its small H–Ca *ionization difference* cannot justify high-Z open-shell
accuracy without a separate CSF-resolved benchmark.

All **237/237** completed one-electron ledger configurations were recomputed
with this correction, including every H-like ground stage through Og; all
dependent line estimates were refreshed transactionally. The spurious
ground-state Breit grew from **0.00002663 Ha** at He II to **0.02681 Ha**
at Ca XX, **3.06136 Ha** at U XCII, and **7.59665 Ha** at Og CXVIII.
These are total-energy shifts. The much smaller corrections to H–Ca neutral
first ionizations above remain far below the missing multielectron energy.
The repair provenance is in
`output/nist_spectra/one_electron_breit_repair.jsonl`.
Against NIST's **theoretical** one-electron ionization entries, the repaired
H I, He II, and Ca XX ground thresholds differ by −0.000031%, −0.000003%,
and −0.000387%, respectively. These values validate the scale of the
one-electron correction but are not experimental accuracy scores.

The complete first-20 ledger audit confirms **3,470/3,470** configuration
jobs done, **210/210** ion-stage grounds done, and an exact key/wavelength/
medium match for all **35,802** observed NIST line rows. Only **10** line
rows pass the strict resolved-line gate; **32,410** are labelled as
configuration-average estimates. These counts describe calculation and
classification scope, not an accuracy score over 35,802 line positions.

## Original NIST holdings and remaining source gap

The local ASD v5.12 line/level archive has **296,998** standard-ion lines,
**3,835** separate isotope lines, and **119,394** standard-ion levels plus
**912** isotope levels. These sums match the corresponding official NIST
[line holdings grid](https://physics.nist.gov/cgi-bin/ASD/lines_pt.pl)
and [level holdings](https://physics.nist.gov/cgi-bin/ASD/levels_pt.pl)
after the repository's level-order reconciliation; no indexed standard-ion
or isotope row is currently missing. All 1,079 direct line-detail exports
were row-matched before optional fields were imported. The original NIST
[lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)
also distinguishes observed wavelengths from level-derived Ritz values;
generated possible Ritz transitions are not stored as measured lines.

The one identified NIST feature still absent from the database is the
**plasma-diagnostics line-pair metadata and ratio plots**. NIST's
[version history](https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml)
lists this feature for selected C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, and Ni
ions, and its lines help says the output should link to line-pair ratio
details. Seven direct probes retained in
`output/nist_spectra/nist_diagnostics_audit.json` returned HTTP 500 or
NIST's own “Query error” for diagnostic requests, while the ordinary C III
control query succeeded. Thus the archive labels diagnostics as a
**source-access gap**, not zero diagnostic records. It must be rechecked
against a working NIST endpoint before claiming every NIST field is present.

## All-element calculation state

The all-element ledger has ground energies for all **7,021** ion stages.
At the start of this audit, **14,382** configurations were done and
**2,203** were pending. Two old Mo VI `10p`/`11p` failures were returned to
the queue after confirming that the current QED label parser reads the full
principal number. A single-threaded resumable worker was started for the
remaining pending configurations; its process identity and log are in
`output/nist_spectra/full_dhf_worker_2026-09-27.json` and `.log`. The
website can be refreshed from the ledger while this worker runs.

One provenance correction was made before restarting production:
`basis_for_job` had recorded 50 Bohr for compact states while `Atom`'s
unspecified-box default actually used 75 Bohr. Newly claimed jobs now pass
and record 75 Bohr explicitly. Older `r_max_bohr=50` rows retain ambiguous
legacy metadata because some independent repair jobs deliberately used an
explicit 50 Bohr box; they must not be mass-rewritten without per-job
provenance.

## DBSR_HF Breit implementation and method provenance

`src/atomx/corrections/dbsr_breit.py` now ports the average-configuration
coefficient construction from `DBSR_HF/hf_MOD_energy.f90` and the eight
Grant–Pyper terms from `DBSR_HF/hf_energy_BR.f90`. It uses explicit
jj-configuration weights for `WW(i,j)`, including `q_i(q_i−1)` for a
same-subshell pair. It evaluates the ordered radial triangle `r₁ < r₂`,
as `DBSR_HF/dbsr_lib_dbs.f90::msk_ppqq` does. Applying the old full
`Yᵏ` radial convolution to each of the eight terms double counted the
He-like case and missed the Ne/Ar reference energies.

On shooting DHF orbitals the new implementation gives He I
`0.0000637756970 Ha`, Ne I `0.01664313377 Ha`, and Ar I
`0.13236038913 Ha`, against recorded DBSR_HF energies
`0.000063777432919`, `0.016643630840490`, and `0.132364619357042 Ha`.
These agree to at most `3.2 × 10⁻⁵` relative on different radial
representations. A B-spline DHF check also gives He `0.0000637771090 Ha`
and Ne `0.01664240795 Ha` on a modest 55-spline grid. The current
DBSR_HF executable was unavailable, so the comparisons use the recorded
output anchors and the local Fortran source.

New DHF corrections use this DBSR_HF static Breit **configuration
average** by default. It is not an exact term-resolved Breit operator;
MCDHF retains its Gaunt CI operator, which can split levels but omits
retardation. The existing all-element ledger and first-20 benchmark were
computed before this change, with the historical scalar Breit. Their
calculation script now specifies `breit_mode="scalar_legacy"` so the
running worker and resumed jobs use one consistent model. The website
labels those historical rows; they should be regenerated with the new
default before attributing a change in NIST residual to the Breit fix.

The separate, completed H–Ca rerun in
`output/nist_spectra/first20_dbsr_breit.jsonl` recalculated both ground
stages for all 20 neutral ionizations. It validated that each total-energy
change equals the new-minus-old Breit energy to `10⁻⁷ Ha`, with the same
shooting seed, B-spline grid, QED backend, isotope mass, and leading
normal-mass shift. The largest ionization change is **0.003204 eV** (Ne I).
For the 19 evaluated like-for-like references, RMS relative error changes
from **8.62845%** to **8.62654%**. The RMS residual in absolute energy
changes from **0.98054 eV** to **0.97980 eV**. This confirms that the
DBSR_HF Breit correction is physically important for the method definition
but does not explain the roughly 1 eV mean-field gaps. The comparison site
now plots the historical and new residuals separately for these 20 stages.

## Correlation diagnosis and full-table hold

The all-element worker was stopped with no running configuration left in the
ledger. Its interrupted Ag II configuration was restored to `pending`, and
`full_table_run_status=paused_for_first20_physics_audit` was recorded. The
14,535 finished rows were retained. The run should not resume until the
first-20 model and state matching have a validated correlated benchmark.
`generate_full_dhf_nist.py` now refuses worker claims while that hold is
recorded, so a scheduled worker cannot silently restart the calculation.

`scripts/audit_first20_correlation_gap.py` evaluates the closed-shell He I
Dirac–Coulomb MBPT(2) energy in atomx's established reference basis, and
uses the zero electron-correlation energy of one-electron He II to isolate
the differential term. It gives −0.03696818725 Ha for He I, or a
+1.005956 eV ionization shift. Adding this term as a **diagnostic** to the
DBSR_HF-Breit DHF+QED+mass result moves He I from 23.442977 to
24.448933 eV, compared with the evaluated NIST 24.587389 eV. The
residual shrinks from −1.144412 to −0.138456 eV: 87.9% of the DHF gap
is removed by second-order correlation. The additive result is not a
single self-consistent calculation with all corrections and does not close
the remaining higher-order and basis/model error. The MBPT(2) code accepts
closed shells only, so it cannot supply the open-shell ion stages for the
remaining first-20 comparisons. These still require validated J-resolved
correlated theory and matched experimental terms.

A fresh helium `correlation="sd"` MCDHF probe with the default virtual
space converged to −2.89780844 Ha (17 CSFs), compared with the same
Dirac–Coulomb DHF energy −2.86181332 Ha. It recovers 0.03599512 Ha of
correlation, slightly less than the second-order result because the default
CI active space is small. Its convergence alone is not enough to validate
excited spectroscopic labels or its transfer to open-shell first-20 ions.

`scripts/benchmark_he_active_space.py` extends the same He I SD calculation
through the 3s, 3p, and 3d virtual shells. The n≤3 space has 98 CSFs and
gives −2.90197463 Ha, or −0.04016131 Ha relative to the DHF reference.
Adding each Dirac–Coulomb correlation gain diagnostically to the production
DHF+Breit+QED+mass ionization moves He I from 23.442977 to 24.422454 eV
(n≤2) and 24.535822 eV (n≤3). Against NIST 24.587389 eV, the residual
falls from −0.164935 to −0.051567 eV. The 0.113368 eV active-space shift
is much larger than a precision target. The n≤3 result is not an active-space
extrapolation or a fully combined correction calculation.

The reproducible `scripts/benchmark_be_correlation_probe.py` reran Be I
and Be II with the current default SD-MCDHF active space. Their dominant
lowest CSFs have the expected `J=0+` and `J=1/2+`, respectively. The
separate Dirac–Coulomb DHF energies agree with the production values to
better than 10⁻⁶ Ha. The 399/271-CSF SD energies are −14.65404172 and
−14.32071861 Ha, giving **9.070184 eV** for Be I ionization instead of
the Dirac–Coulomb DHF **8.045780 eV**. This removes **80.2%** of the gap
to evaluated NIST **9.322699 eV**; **−0.252515 eV** remains. The probe
does not yet include Breit, QED, recoil, active-space convergence, or an
independent correlated radial grid. Its orbital residual is about 10⁻⁵,
which is not enough to support high-precision spectroscopic claims.

`scripts/benchmark_mg_correlation_probe.py` tests Mg I and Mg II with
the default frozen-core SD-MCDHF space. Their optimized CI orbitals carry
a **rotated spectroscopic label**: the radial orbital called `4s1/2` has
normalized overlap **0.94064** (Mg I) and **0.99092** (Mg II) with each
stage's independent DHF `3s1/2`; the lowest CSF occupies that mapped
orbital and has the expected J/parity. The neutral mapped orbital has two
large-component nodes, the physical 3s count, despite its 4s label.
Consequently a dominant-CSF text match would reject the physical state,
while a label-only comparison could also misassign excited lines. The
matched Dirac–Coulomb SD ionization is **7.532950 eV** versus DHF
**6.617690 eV** and evaluated NIST **7.646236 eV**: the default SD space
removes **89.0%** of this gap but leaves **−0.113286 eV**. Its 1s/2s/2p
excitations were frozen; Breit, QED, recoil, active-space extrapolation,
and independent-grid validation are still absent. The website labels this
as a diagnostic and displays the measured orbital overlaps.

## Stratified check beyond calcium

`scripts/audit_periodic_dhf_sample.py` audits 19 already completed neutral
and first-ion ground pairs in shell families spread from Ne/Na/Mg through
Fr/Ra/Rn. Every NIST reference in the score is evaluated; all 15 selected
unique-ground comparisons use the listed IE and all four selected noble-gas
comparisons use a complete evaluated 2J+1 configuration centroid. The
historical-scalar-Breit DHF ledger lies below the matched NIST ionization
in **19/19** pairs, with mean signed residual **−0.93869 eV** and RMS
absolute residual **1.05644 eV**. Residuals differ by electronic family:
Na/Rb/Fr are −0.184/−0.396/−0.270 eV, Zn/Cd/Hg are
−1.625/−1.696/−1.484 eV, and Ne/Kr/Xe/Rn centroids are
−1.779/−0.987/−0.899/−0.920 eV. U I is excluded because its evaluated
ground-configuration centroid is incomplete; Og I has no evaluated NIST
ionization reference. The website plots every selected pair with a family
filter. This is a selected model diagnostic, not a random sample or a
high-accuracy claim for heavy atoms: their DHF radial branches, nuclear
model, recoil, correlation, Breit and QED contributions need separate
validation.

The archived Zn I 9.394197 eV and Rn I 10.74850 eV listed ground-term
energies were spot-checked against the current
[NIST Zn output](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=1&biblio=1&e_out=0&el_name_out=1&ion_charge_out=1&spectra=zinc&unc_out=1&units=1)
and [NIST Rn output](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=1&biblio=1&e_out=0&el_name_out=1&ion_charge_out=1&spectra=radon&unc_out=1&units=1).
Rn's scored 12.025335 eV is instead the derived complete configuration
centroid ionization, as the report labels it.
