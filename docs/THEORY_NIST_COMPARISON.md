# Atomx theory compared with NIST spectra

Run the spectrum atlas build first, then:

```bash
.venv/bin/python scripts/import_nist_isotopes.py
.venv/bin/python scripts/generate_full_dhf_nist.py --max-z 20
.venv/bin/python scripts/generate_multielectron_cache.py \
  --system 'He I' --system 'Li I' --system 'Na I' --system 'C V' \
  --system 'Ne VIII' --system 'Ne IX' --system 'Ar VIII'
.venv/bin/python scripts/compare_theory_nist.py
```

The isotope importer saves NIST isotope masses and chooses the most abundant isotope when NIST reports an abundance; otherwise it records a provisional first-tabulated choice. The initial full ledger had **15,659 configuration calculations** in `output/nist_spectra/full_dhf.sqlite`; the in-place ground-selection repair described below added 187 NIST-backed configurations, bringing the ledger to **15,846** jobs. A later repair of unlisted ionic grounds added 711 source-traced candidate configurations, then 30 nearby ionic alternatives were queued, so the current ledger has **16,587** jobs. The H–Ca scope contains **3,470 configurations** across **all 210 ion stages**; 55 ground configurations were added beyond the observed-line endpoint queue so stages without observed lines still received a calculation. The full all-element run includes a ground configuration for each of the 7,021 ion stages. It runs with `breit=True`, `qed=True`, and a finite nuclear mass for the selected isotope. One worker periodically rebuilds the interactive page. `--max-calculations N` runs a bounded batch; `--prepare-only` creates the job ledger without calculating. `--worker-only --max-z 20` claims only H–Ca pending jobs. Failed configurations explicitly marked `retry_pending` can be retried with `--worker-only --retry-only --max-z 20`; the retry tries a GSZ SCF start for core-hole configurations and stores the successful starting potential. The separate benchmark command checkpoints selected DHF/MCDHF calculations in `benchmarks/_nist_work/cache.json`; its `--dhf-only`, `--ci-only`, `--system`, and `--all-panel` options remain available.

Workers can also select ion charges with `--min-charge Q --max-charge Q`. This lets a bounded worker finish Mg II and Mg III while the much larger Mg I Rydberg bases remain in progress, without reclaiming another worker's active calculation. For example, `--worker-only --min-z 12 --max-z 12 --min-charge 1 --max-charge 2 --max-calculations 24 --refresh-site-every 0` processes only those two Mg ion stages. The queue still records each NIST observed line, including rows that cannot be assigned to a unique configuration or term.

For the wider Z=21–118 queue, `.venv/bin/python scripts/maintain_full_dhf_workers.py --max-running 5` keeps a bounded pool of pending-job workers alive. It counts PIDs already recorded in the ledger, starts fresh workers only when memory and swap exceed the configured reserves, and writes one log per worker to `output/nist_spectra/worker_logs/`. `--once` reports capacity without starting jobs. Within a bounded 512-job lookahead, workers now claim smaller Coulomb radial boxes first (after ground jobs), so several hours-long neutral Rydberg configurations do not occupy every worker while compact ionic states wait. Completed jobs retain their original source identities and claim records. Failed-job repair remains with the separate supervisors; the pool never silently treats a failed calculation as complete.

### H–Ca batch and numerical repair (26 September 2026)

The initial **3,470-job H–Ca pass** ended with 2,546 converged, 913 failed, and 11 numerically untrusted energies. The high-`n` and failed excited-state jobs are now being recalculated with larger radial boxes and an alternate SCF start where appropriate; the live site reports their changing status. Every one of the **35,802 observed H–Ca NIST line rows** has an explicit comparison status, including rows awaiting revised endpoint energies. All **210 H–Ca NIST ionization-energy entries** have a DHF+B+QED+normal-mass prediction. NIST labels **68 unbracketed evaluated**, **94 bracketed semiempirical**, and **48 parenthesized theoretical** entries. The ground-state results give **5.72% RMS relative error against the 68 evaluated entries** and **3.52% when all reference types are pooled**. The dashboard and CSV keep the reference type visible.

Start `.venv/bin/python scripts/view_nist_spectra.py --port 8765` and open <http://127.0.0.1:8765/comparison>. The comparison is part of the atlas site and links directly to the selected ion's full NIST lines and levels. The generated `index.html` can also be opened directly. The same directory contains `matched_lines.csv`, `multielectron_matched_lines.csv`, `multielectron_matched_levels.csv`, `comparison.json`, `record_audit.sqlite`, and periodic-table PNGs for observed hydrogenic, Ritz, and multielectron comparisons.

## Hydrogen-like Dirac comparison

The line comparison uses atomx's `hydrogenic_dirac_energy` for a point-nucleus, one-electron Dirac-Coulomb model. For each hydrogen-like ion it predicts vacuum wavelengths for classified **1s₁/₂ → np₁/₂,₃/₂** transitions with `n = 2…6`. NIST lines are paired by the lower and upper configurations and J values. No nearest-wavelength matching is used. Only NIST rows explicitly marked `vacuum` enter the comparison, so air and vacuum values are never mixed.

For an observed NIST wavelength, the signed photon-energy error is

\[
100\left(\frac{E_\mathrm{calc}}{E_\mathrm{NIST}}-1\right)
=100\left(\frac{\lambda_\mathrm{NIST}}{\lambda_\mathrm{calc}}-1\right)\%.
\]

The element color is the RMS of those signed errors in percent. The dashboard also offers the median absolute error and displays each paired wavelength. Every element from H to Og is shown. Gray means **no matched reference**, not zero error. In hydrogenic mode, color for an element assesses its *hydrogen-like ion*, not its neutral atom or every ionization stage.

The hydrogenic observed-line view uses **all classified transitions** that the Dirac model can calculate, with separate point-nucleus and leading isotope-mass options. The clean 1s → np, n = 2…6, subset can be selected separately for the point-nucleus mode: **70 matched transitions in 10 ions**. The Ritz Lyman view has **640 transitions in 110 ions**, but a Ritz wavelength is computed from NIST energy levels, and some levels are theoretical. Ritz agreement is therefore a reference consistency check, **not experimental validation**. NIST describes this distinction in its [spectral-lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html).

The point-Dirac observed-line audit currently compares **290 of 236,495 observed line rows** across 10 hydrogen-like ions after same-⁠n intervals are excluded. The complete audit adds the multielectron comparisons described below and gives every other row an explicit status. It also covers **all 119,394 standard NIST level records** after source reconciliation: 3,411 hydrogen-like levels and 110 ionization limits have a Dirac value; multielectron results are added only for states passing the benchmark pairing check. Eight source-backed line-endpoint placeholders have no reported level energy and cannot enter an energy score. NIST level values and Ritz wavelengths can be theoretical, so these level checks are separated from observed-line validation.

Of the observed line rows, **185,015** have both upper/lower configurations and J values in the line export; **50,764** lack a complete configuration pair there. An exact join on NIST level IDs recovers endpoint configurations for **25,878** of those rows while preserving the original line fields. **24,640** recovered rows have two distinct supported configurations, **1,164** stay unresolved within one configuration, and **74** use unsupported notation. The remaining **24,886** need further identification. The interface never pairs lines merely because wavelengths are close.

`record_audit.sqlite` has **one row per NIST line and level**, including every record that has no atomx prediction. Its `status` says whether the record was compared, lacks a calculation or validated state pair, has an unsupported classification, or is degenerate in the point-nucleus model. Separate `line_model_audit` and `level_model_audit` tables keep the available DHF/MCDHF rungs, including the mass-aware DHF rung. The atlas's complete tables join the preferred calculated value and error onto every row. The headline score counts unique observed NIST rows across models; unassessed rows are never treated as zero-error successes.

The model omits finite nuclear size, recoil, QED, and other corrections. These omissions are especially visible in H I and heavy ions. Observed lines can also be rounded or unresolved multiplets, so the displayed error is a descriptive benchmark, not a fit weighted by measurement uncertainty. Stick heights are uniform and compare **line positions only**; they are not emitted intensities. The atlas viewer shows the NIST A-value locator separately.

The site's **one-electron control** compares H I point Dirac, DHF, DHF+B+QED, and mass-aware DHF+B+QED on the same two observed 1s–2p rows. For the J=3/2 row at 121.56699 nm, point Dirac gives 121.50025146 nm (0.05493% photon-energy error); DHF+B+QED without nuclear motion gives 121.50183374 nm (0.05363%); the first-order mass-aware DHF run gives 121.56804253 nm (−0.00087%). DHF assigns one energy to the 2p configuration and cannot resolve its two fine-structure components; point Dirac can. Scaling the point Dirac transition energy by the leading reduced-mass factor, using the [NIST CODATA 2022 proton/electron mass ratio](https://physics.nist.gov/cuu/pdf/wall_2022.pdf), gives 121.56642 nm for illustration. Neither mass treatment is a complete recoil/QED calculation. The hydrogenic and multielectron periodic-table views represent **different ions and different matched-line sets**, so their color scores do not rank methods directly.

To reduce the remaining difference, compare identical observed NIST rows and the same isotope/medium; use a common nuclear model and mass convention for both solvers; check DHF radial-basis convergence against the analytic one-electron Dirac limit; then add recoil, finite-nucleus and radiative corrections as separate documented terms. For multielectron atoms, DHF configuration averages need correlation and state-resolved level treatment before fine-structure lines can be predicted reliably. Breit/QED alone cannot supply missing correlation or fix a wrong orbital branch.

## Full observed-line DHF run and nuclear motion

The finite-mass Hamiltonian adds the nuclear kinetic energy `(Σ p_i)²/(2M)`. It contains a one-electron normal mass term and a two-electron specific mass term. The current DHF job uses atomx's **leading normal mass shift**, estimated from the converged configuration's virial energy and the selected NIST isotope mass. General open-shell specific mass shift, relativistic recoil, isotope-specific charge radii, and nuclear binding-mass refinements are **not yet calculated**. The database labels this as `one-body normal mass shift` rather than full recoil. For isotope-unspecified ASD lines, the most abundant isotope is an explicit modeling assumption, not an NIST assignment. NIST's [isotope table](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?all=all&ascii=ascii2) is saved with a SHA-256 hash; H-1 uses the CODATA proton mass directly.

All **236,495 observed line rows** are entered in `full_dhf.sqlite` before calculation. Initially **183,179** rows have two expandable configurations and an air/vacuum medium, **50,764** lack a configuration pair, **1,805** have unsupported or incomplete configurations, **717** have both endpoints in one configuration, and **30** are hydrogenic transitions within the same principal-⁠n manifold. The last two groups cannot be resolved by configuration-average DHF and are explicitly excluded from accuracy scoring. The NIST shorthand is expanded only by adding fully occupied inner subshells; ambiguous configurations are not guessed. The level-ID backfill adds 23 endpoint configuration jobs to the live ledger and records NIST level-ID provenance for every recovered row. Each successful configuration is checkpointed with its isotope mass, energy, normal-mass term, convergence and elapsed time. Lines are updated as their endpoint energies arrive. Failed, inverted or numerically untrustworthy configurations keep explicit statuses. For NIST air wavelengths the calculated vacuum wavelength is converted with the standard-air Peck–Reeder index described in [NIST's lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html), and photon-energy error is evaluated in vacuum. The site's coverage and periodic table refresh while the run proceeds.

H I's original mass-aware RMS was dominated by invalid same-⁠n predictions: one 6p→6s record has a NIST wavelength of 7.591×10⁹ nm while the configuration-average calculation gave 46,379 nm. The numerical high-`n` orbitals also failed the analytic one-electron Dirac energy check in the old 50-Bohr box. The repaired calculation expands the box for H `n≥4` and rechecks every completed one-electron energy. The strict DHF line score now requires both configurations to have **one physically possible term/J level**, as well as one matching NIST level. NIST listing only one member of a multiplet is insufficient. Lines involving unresolved 2p fine-structure components remain `configuration_average_only`. Same-⁠n point-Dirac comparisons are excluded from its periodic-table score because a Dirac-Coulomb baseline omits the Lamb shift that controls these near-degenerate intervals. These are method-validity gates, not fitted wavelength cutoffs.

### Debugging the large errors before full-table production

The first many-electron pass found a real solver branch error. A converged Be I `1s1 2s2 2p1` calculation assigned the **deep** orbital to the `2s` label and the **shallow** orbital to `1s`, giving about **474 nm** for a NIST line near **10.738 nm**. The same-kappa energy minimizer had rotated the occupation-inverted core-hole pair and changed the requested excited state into an Aufbau-like state. The solver now protects that pair while retaining the rotations needed for ordinary ground-state optimization. A fresh isotope-aware Be I calculation has the expected orbital ordering and gives **10.863 nm** for that configuration-average transition. All many-electron configuration jobs made with the old solver were invalidated in the resumable ledger; they must be recomputed before scoring.

A later H–Ca retry audit found that the same minimizer also exchanged equal-occupation core-hole orbitals, including He I `1s1 4s1` and Li II `1s1 3s1`. Its guard now protects any same-kappa pair whose lower-n orbital is not filled. Both examples converge with correctly ordered orbitals under DHF+Breit+QED+normal mass shift, while the Li ground-state rotation regression still passes. Previously failed H–Ca jobs have a separate retry queue; recovery does not turn a configuration average into a resolved term/J prediction.

The remaining He I 2058.692 nm example is a different issue: NIST identifies a **singlet 2s→2p** line, while DHF gives one average energy for all singlet and triplet levels of each configuration. Its average prediction near **1023.6 nm** is therefore not an exact singlet-line prediction. The NIST levels themselves show 1s2s singlet and triplet energies separated by more than 6,400 cm⁻¹. The ledger retains a calculated wavelength for such rows with `configuration_average_only`, but the periodic accuracy score excludes them. Cached DHF lines follow the same rule. A Na I MCDHF branch with a >0.5 Ha corrected-versus-plain excitation jump was also removed from line scoring pending state-tracking repair. None of these exclusions deletes NIST records.

The Li I ground-to-2p pilot checks that the narrower solver fix does not damage ordinary states. The corrected isotope-aware DHF result is **673.411 nm** versus an observed **670.961 nm** component, a **−0.364% photon-energy error**. The 2p configuration has two NIST fine-structure levels, so this remains an illustrative average, outside the strict line score. Its residual gap is consistent with missing state-specific correlation and fine-structure resolution; a mass correction alone does not close it.

The H–Ca batch exposed a separate high-`n` QED data bug: the self-energy adapter read only the first character of an orbital label, so `10p` was treated as `1p`; FAC's built-in Bethe-logarithm table also ends at `n=8`. The adapter now parses the full principal number and uses the published [NIST Technical Note 1467 Bethe logarithms](https://www.nist.gov/publications/bethe-logarithms-rydberg-states-numerical-values-n-le-200-0) through `n=200` (the [authors' numerical table](https://qedtheory.org/wp-content/uploads/2021/01/values.pdf) is recorded with a SHA-256 hash in the data module). Earlier high-`n` jobs were invalidated and recalculated. This supplies the hydrogenic Bethe-log input to atomx's approximate screened self-energy model; it does not make the multielectron QED correction exact.

## H–Ca ionization energies

The bounded run also calculates the NIST ground configuration for **every one of the 210 ion stages** from H through Ca, including stages without observed spectral lines. For stage `q`, the theoretical ionization energy is `E_ground(Z,q+1) − E_ground(Z,q)`; a fully stripped ion has energy zero. The [interactive site](http://127.0.0.1:8765/comparison) shows charge-stage rows and two selectable periodic ionization maps. The unique-level map uses only unbracketed evaluated NIST values for which **both configurations have a physically unique ground level**. That leaves **29 of 68** evaluated H–Ca comparisons, with **5.62% RMS**. A separate configuration-centroid map compares the atomx average against `2J+1`-weighted NIST level centroids when both ground-configuration manifolds have their full expected degeneracy and all contributing levels are evaluated. It admits **67 of 68** evaluated H–Ca comparisons, with **5.09% RMS**. The archive preserves NIST's raw `g`; centroid weights come from J because [the source audit](NIST_SOURCE_AUDIT_2026-09-26.md) found raw `g` disagreements. These scores measure different observables and are labelled separately. The table and [complete CSV](/data/code/atomx/output/nist_spectra/theory_comparison/ionization_dhf_nist.csv) retain both residuals, all reference types, uncertainties, and explicit calculation statuses. They also identify whether each ground configuration came from NIST levels, NIST ground shells, or an isoelectronic estimate; a bare nucleus is explicit. The interactive table displays the same provenance and NIST reference codes, linking resolvable `L…` codes to the official bibliography. The comparison uses NIST's stored cm⁻¹ energies when available, because [NIST notes](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html) that its eV conversion loses some precision. These DHF values include Breit, QED and the leading normal mass shift in both adjacent ground-configuration calculations. Correlation, specific mass shift and isotope-specific nuclear effects also limit accuracy.

O I → O II demonstrates the need for both views. Atomx's configuration-average ionization is **14.5685 eV** versus NIST's ground-term **13.6181 eV**. Independent [NIST CCCBDB Hartree–Fock calculations for O](https://cccbdb.nist.gov/ie2x.asp?casno=17778802) are around **12.0 eV** across many basis sets, so interpreting atomx's positive ground-term residual as an ordinary HF correlation error would be misleading. From the archived NIST levels, weighting each member of `2s² 2p⁴` and `2s² 2p³` by `2J+1` puts their centroids **0.941 eV** and **3.168 eV** above the respective ground terms. The resulting centroid ionization is **15.8449 eV**; atomx is **8.06% low** against that reference. The oxygen row remains outside the unique-level score and enters the centroid score instead.

## Excited configuration centroid accuracy

The [interactive comparison](http://127.0.0.1:8765/comparison) also has a
separate periodic map and NIST-versus-atomx plot for excited configuration
centroids. For each DHF configuration, the importer groups NIST levels with the
same expanded configuration, derives weights from `2J+1`, and requires their
sum to equal the full microstate degeneracy. It scores the excitation relative
to the ion's ground configuration only when both complete manifolds have
evaluated NIST energies and both DHF calculations converged. This compares
the closest available configuration-average reference; it does not represent
resolved term/J line accuracy. Strong configuration mixing can make a NIST
principal-configuration assignment approximate even when its J-weight sum is
complete. The complete
[configuration CSV](/data/code/atomx/output/nist_spectra/theory_comparison/configuration_centroids_dhf_nist.csv)
keeps all **16,587** live jobs, including incomplete, theoretical, and pending
reference statuses. The map updates as calculations finish. An independent
crosscheck found the centroid builder agrees exactly with the ionization
builder for all **210 H–Ca ground manifolds**.

## Multielectron line positions

The default interactive map uses **mass-aware DHF with `breit=True, qed=True`**. Plain DHF, DHF+B+QED without nuclear motion, MCDHF, and MCDHF+B+QED are selectable. The audit's main row prefers mass-aware DHF when it has a paired line; the model tables retain every available rung. Its vacuum transition wavelength is `10⁷ / (E_upper − E_lower)` with energies in cm⁻¹. The full DHF run matches source configurations, while the smaller benchmark panel also checks term/J/level identity. DHF assigns one energy to a configuration, so it cannot distinguish terms or fine structure within that configuration. MCDHF state matching follows the benchmark's energy-ordered `(2J, parity)` blocks. A block is excluded if its calculated and NIST dimensions differ (except for the benchmark’s validated low-state tail rule) or if the ordered dominant configurations disagree. No nearest-wavelength pairing is used. The site compares methods on the **same shared NIST lines**, avoiding a score difference caused only by differing line coverage.

The MCDHF benchmark needs a stronger state-identity audit before its excited-state scores are treated as current accuracy results. The correlated variational functional in `variational.py` minimizes the **ground-state** CI energy, although the user-facing method describes an average-level calculation. A current-code He I six-configuration control has 17 CI levels, but its uncorrected `1s3s` singlet is at −0.465765 Ha and its triplet at −1.174236 Ha, both above the approximately −2 Ha He II continuum limit. Positional and dominant-configuration matching admits the singlet and produces a spurious roughly 345,000 cm⁻¹ residual. In a reduced `1s² + 1s3s` control, the orbital called `3s` finishes with one significant radial node instead of two; the initial Guerra/DHF seed has two. A frozen-orbital CI control retains two nodes. This identifies a ground-state optimization and spectroscopic-label mismatch, not evidence that a valid MCDHF excited-state prediction is that inaccurate. The benchmark now rejects a light He-like CI panel if any purported bound state lies more than 0.1 Ha above its analytic one-electron ionization threshold. Cached panel rows predate the current solver code, and the site marks them historical. The DHF+B+QED+mass ledger and its source-complete line statuses are independent of this MCDHF benchmark issue.

The initial selected-system pass provisionally paired **67 observed multielectron lines**, but many were configuration-average DHF values assigned to individual term/J components. An earlier audit admitted **13 distinct additional observed rows** across the MCDHF pilot systems and one H I mass-aware control; the new continuum gate can remove some of those He-like rows, so the page and ledger report the current counts. Configuration-average wavelengths appear in a separate unscored table on the interactive page and in `configuration_average_lines.csv`; the full NIST archive remains available throughout.

The H–Ca ledger marks a physically unique-level line `radial_box_unresolved` when the estimated Coulomb outer turning radius, `2n²/(q+1)` Bohr, exceeds its recorded radial box. Diffuse states are being rerun with an adaptive box and outer grid. A flagged wavelength remains in the site's unscored table and `unscored_dhf_lines.csv` but does not affect the periodic accuracy color. The criterion is a conservative domain check, not a convergence proof for rows below the threshold. Configuration-average wavelengths remain visible for all completed multiplet endpoints; they are not exact observed component predictions or part of the strict score.

The first six H–Ca workers each inherited a 63-thread numerical-library
pool on a 32-core host, slowing the high-Rydberg B calculations. Repeating
B I `1s²2s²20d` with one numerical thread gave the same energy within
`4×10⁻¹⁴` Ha in **41 s** rather than the old worker's **655 s**. Their six
unfinished jobs were safely returned to the queue and six single-thread
workers restarted. The generator now defaults fresh CLI workers to one
OpenBLAS, OMP, and MKL thread unless the caller explicitly sets those
variables.

The B I high-Rydberg `2s² nd` branch offers a direct convergence check.
The calculated binding energies below the B II `2s²` core for `n=35,36,37`
are **89.72754, 84.80988, and 80.28645 cm⁻¹**. Subtracting the archived
NIST evaluated level energies from its **66,928.04 cm⁻¹** B I ionization
limit gives **89.68, 84.72, and 80.21 cm⁻¹**, respectively; the listed
level uncertainty is **0.22 cm⁻¹**. These nearby differences support the
selected Rydberg orbital branch, while the absolute transition wavelengths
still inherit the DHF ionization-threshold error: atomx gives **7.928756 eV**
for B I versus [NIST's evaluated 8.298019 eV](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=on&biblio=on&e_out=0&el_name_out=on&format=0&ion_charge_out=on&ion_conf_out=on&level_out=on&order=0&seq_out=on&shells_out=on&sp_name_out=on&spectra=B&submit=Retrieve+Data&unc_out=on&units=1). At the 27 September 2026 audit, the `38d` and `39d`
calculations were in convergence repair; no energy from their failed attempts
was accepted. A subsequent hydrogenic-start calculation with 0.35
orbital damping converged for B I `39d` in **49 iterations** with residual
`6.05×10⁻⁶`, ordered orbitals, and the same B-11 mass and radial grid as the
production job. Its binding below B II is **72.2645 cm⁻¹**, versus **72.44
cm⁻¹** from the archived NIST level and ionization limit; the NIST level
uncertainty is **0.22 cm⁻¹**. The validated result is stored in the ledger
with its damping provenance, and light-atom retry jobs now try that branch
first. B I `38d` resisted the standard `10⁻⁵` orbital-change tolerance from
four starts, although the total energy stabilized across long independent
runs. An explicitly labelled `10⁻⁴` SCF tolerance converged on the production
grid in 27 iterations with residual `8.44×10⁻⁵` and ordered orbitals. A second
grid converged independently and changed the energy by only **0.00455 cm⁻¹**.
The calculated binding below B II is **76.1165 cm⁻¹**, versus **76.01 cm⁻¹**
from the archived NIST level and ionization limit (combined listed uncertainty
**0.221 cm⁻¹**). The production-grid result and both-grid validation are
recorded in the ledger metadata and comparison site. This validates its
numerical energy at a documented looser orbital tolerance; it does not fix
the B I absolute ionization-threshold error or resolve its multiplet lines.

Each newly claimed job records the worker PID, Linux process start tick, and
its original queue. The H–Ca and heavy-ground supervisors check those exact
process identities and return abandoned jobs to their original queues. This
was added after one Mg I `3s 50p` retry worker exited while its job remained
marked running; that job was returned to the retry queue and reclaimed by a
replacement worker. Four extra tracked H–Ca workers accelerated Na I; after
Na I completed, their four Mg I claims were returned to `pending` and the
regular pool reduced to six workers to leave memory for the large `n=50–59`
Mg grids. Jobs claimed by older workers before this tracking was
added remain untracked and require a manual liveness check if those workers
exit. A separate `--refresh-only` supervisor rebuilds the comparison
site every five minutes while long SCF repair jobs occupy the calculation
supervisor. The comparison builder serializes concurrent rebuild requests and
publishes its JSON and HTML only after each file is complete.

The observed Mg I series reaches `3s 59p`. Its production radial grid needs 668 B splines, so the earlier 512-spline cap would exclude 16 `3s np` endpoints (`n=44–59`). The cap is now 700 while the radial spacing and box rule remain unchanged. A trial that doubled the outer spacing shifted Li I `27p` by about 1.3 cm⁻¹, so these jobs retain the finer grid. Mg I `3s 44p` converged with 517 splines after 2,160 seconds. Its calculated binding below the Mg II core is **59.508 cm⁻¹**; the NIST evaluated ionization limit **61,671.05 cm⁻¹** minus its listed `3s 44p` level **61,611.67 cm⁻¹** gives **59.38 cm⁻¹**. The newly completed `3s 42p` and `3s 58p` bindings are **65.44** and **34.17 cm⁻¹**, compared with **65.31** and **33.78 cm⁻¹** from the corresponding archived NIST levels. These checks identify the selected Rydberg branch, not strict wavelength agreement or grid convergence for every higher `n`. The remaining members continue to run; the live ledger and dashboard show their current status.

On 27 September the H–Ca ledger reached **3,467/3,470** converged
configuration jobs, with all 210 ground stages complete and three high-`n`
Mg I jobs (`3s 52p`, `3s 56p`, and `3s 57p`) still running. `3s 37p`
completed after 14,807.74 seconds.
The first `3s 56p` and `3s 57p` attempts each exhausted three undamped SCF
starts without converging; their isolated damped-start repairs are now
active. `3s 52p` remains active on its repair worker.
Its binding below the calculated Mg II ground is **84.872 cm⁻¹**; the
archived NIST `3s 37p` level at 61,586.33 cm⁻¹ and evaluated limit at
61,671.05 cm⁻¹ imply **84.72 cm⁻¹**. This checks the Rydberg branch but
does not remove the Mg I absolute ionization-threshold error.
`scripts/audit_first20_dhf.py --max-z 20` confirms exact identity for all
35,802 observed H–Ca line rows, no failed completed energies, and no
nonpositive completed adjacent-stage ionization energy. It reports
`all_configurations_complete: false` until the three Mg jobs finish.

### Isotope-specific comparisons

The comparison page now audits all **3,835** archived NIST isotope line rows
and all **912** isotope level rows independently of the standard-ion tables.
Of the isotope lines, **315** have an observed wavelength and **3,520** are
Ritz-only. The line audit currently has **10** observed unique-level DHF
comparisons and **106** unscored DHF configuration-average estimates; the
remaining observed rows retain explicit missing-classification or missing-pair
statuses. The isotope level audit includes the level-only K isotopes and
retains NIST's original configuration, term, J, uncertainty, source ID,
reference, and bracket/parenthesis notation. Each level excitation uses the
lowest numeric level of its own isotope spectrum as its zero. It has **15**
evaluated unique-level DHF comparisons beyond the 17 ground reference rows;
configuration averages and NIST semiempirical/theoretical/offset levels stay
separate. Download the complete
[line](/data/code/atomx/output/nist_spectra/theory_comparison/isotope_lines_dhf_nist.csv)
and [level](/data/code/atomx/output/nist_spectra/theory_comparison/isotope_levels_dhf_nist.csv)
audits from the page.
[NIST's levels help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
defines square brackets as semiempirical, parentheses as values not based on
evaluated experimental data, and `+x`-style suffixes as uncertain offsets
between disconnected level systems. [NIST's lines
help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html) defines
Ritz wavelengths as level-derived values, so they are separate from observed
line measurements here.

The mass transfer uses each named isotope's nuclear mass and rescales the
calculated leading one-body normal mass correction as `1/M`. H-like
inter-`n` rows also show a separate reduced-mass point-Dirac baseline.
These isotope estimates omit the specific mass shift, field shift,
relativistic recoil, and isotope-dependent radiative terms. In particular,
they should not be read as a full isotope-shift theory or as multiplet-resolved
line predictions.

The same example shows why a good Rydberg binding does not imply a good absolute line: atomx predicts Mg I `3s² → 3s 44p` at **187.65 nm** versus NIST's **162.3069 nm**. Its DHF Mg I ionization limit is **6.6144 eV**, while [NIST ASD's evaluated value](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=on&biblio=on&e_out=0&el_name_out=on&format=1&level_out=on&seq_out=on&shells_out=on&spectra=Mg&unc_out=on&units=1) is **7.646236 eV**, a **1.032 eV threshold error** that dominates this excitation. Independently, the [NIST CCCBDB Mg calculation table](https://cccbdb.nist.gov/ie2x.asp?casno=7439954) lists nonrelativistic Hartree–Fock ionization energies of **6.605–6.624 eV** across many basis sets, including **6.614 eV** with 6-31G. Its close agreement with atomx's relativistic DHF+B+QED+mass result strongly suggests that the threshold deficit is primarily a single-configuration Hartree–Fock correlation limitation, rather than a gross error in atomx's Mg ground-state branch. This is an independent magnitude check, not an exact method or basis match. The listed `44p` member is part of a multiplet, so the wavelength remains an unscored configuration-average estimate. State-specific correlation is needed to improve the threshold and absolute lines; changing the high-`n` radial grid cannot correct that gap.

These calculations use a restricted configuration model. Some accepted blocks still have large energy residuals, especially for He-like ions. The site displays those errors; it does not select the better rung per ion or treat a missing line as a zero-error success. Because the validated blocks cover only a fraction of each ion's states, the plotted spectrum is a subset of its NIST lines. NIST observed wavelengths are the experimental reference; the level panel below is a separate check on NIST level values, some of which are theoretical. In Na I, an older cached MCDHF+B+QED calculation placed the 4s state incorrectly and caused an apparent ~4,154% same-line RMS error. Recomputing that rung with convergence recorded yielded ~4.23% on the same five lines, close to corrected DHF's ~4.33%. A fresh **plain MCDHF** run still places 4s about 3.8 Hartree above ground despite reporting convergence. Its huge error is an orbital/state-branch failure, and convergence alone cannot certify physical state identity. The corrected result also needs a controlled orbital-branch check because switching Breit/QED changed this state by several Hartree, far larger than a credible Na I radiative or Breit correction. The site labels cache status and large level residuals rather than interpreting this as an intrinsic method ranking.

The archived values for all six matched multielectron ions were checked against fresh direct NIST ASD line queries, along with H I, Fe XVII and U II. All nine live queries matched the archived rows and compared fields exactly; see `output/nist_spectra/live_sample_verification.json`.

## Level benchmark panel

`--panel PATH` accepts the CSV output from `benchmarks/verify_nist_levels.py` with columns such as `system`, `rung`, `n_levels_compared`, and `rms_cm1`. Cached DHF/MCDHF results override corresponding CSV rows, so the chart and line overlay use the same state energies. Its bars compare NIST **excitation energies**; the observed line overlay is scored separately. A chosen rung is shown consistently for all available systems; no per-element best rung is selected. Read `benchmarks/NIST_VERIFICATION.md` for the block matching rules and known method limits.

This is an honest coverage overview of theory accuracy, not a claim that all 7,021 possible ionization stages have been calculated and validated. The NIST atlas inventories every stage and its available reference data; full atomx calculations across all elements and charge states require a much larger controlled computation campaign, isotope choices, state matching, and intensity modeling.

The broader ground-state queue retains explicit solver failures and retries them separately. Cu I `3d¹⁰4s` did not converge from ordinary TFDA or GSZ starts, even after 200 iterations. A genuinely nuclear-only start converged in 124 iterations; TFDA with 0.35 orbital damping converged in 85, and the two total energies agreed within `1×10⁻⁷` Hartree. The production repair reran Cu I with the damped branch and stored **−1652.250679949 Ha** with the selected start recorded. Cu II converged with 200 TFDA iterations at **−1652.010060926 Ha**. Their atomx Cu I ionization energy is **6.54758 eV** versus NIST ASD's **7.72638 eV**; the independent [NIST CCCBDB Cu HF table](https://cccbdb.nist.gov/ie2x.asp?casno=7440508) includes several basis-dependent values near **6.5–6.7 eV**, which is a useful magnitude check, not a same-method benchmark. The guarded fallback is now available to future heavy-ion repair jobs; no unconverged Cu value was used. The long-running heavy ground-state supervisor uses `scripts/supervise_first20_dhf.py --min-z 21 --max-z 118 --ground-only --max-running-ground 2`; it stages each newly failed ground job for one repair attempt and leaves the active main workers' jobs untouched. During the simultaneous Mg I `n=50–59` work, the eight all-element main workers were stopped, their eight ground claims returned to `pending`, and two tracked main workers restarted. The two active heavy ground repairs continued. This reduced memory pressure while retaining progress in both queues.

Two later excited configurations, V I `3d⁵` and Cr I `3d⁶`, still stalled under the ordinary damped retries. Holding 0.70 orbital damping until the residual fell below `10⁻⁶` converged them in 99 and 102 iterations, respectively, with correct orbital order and radial nodes. Their same-grid, same-isotope DHF+Breit+QED+normal-mass energies are **−947.453317326 Ha** and **−1048.824468889 Ha**. The [V probe](/data/code/atomx/output/nist_spectra/v_i_3d5_release_1e6_probe.json) and [Cr probe](/data/code/atomx/output/nist_spectra/cr_i_3d6_release_1e6_probe.json) retain residuals, grid, correction flags, orbital energies, and mass corrections. `scripts/apply_dhf_damping_probe.py` checks these fields and the current ledger mass before accepting a probe and refreshing its line comparisons. A guarded late-release fallback is now available to future explicit repair jobs.

Lr I `7s²7p` exhausted five direct DHF starts and initially remained a failed
ground job. A converged local-exchange Slater calculation supplied occupied
orbitals for a fresh **nonlocal DHF+B+QED+normal-mass** solve; that final solve
converged in 31 iterations with ordered orbitals. The production ground energy
is **−37570.855197825 Ha**. An independent 50/1 Bohr grid converged to
**−37570.855197192 Ha**, a difference of **0.139 cm⁻¹** from the production
245/3.5 grid. The [grid report](/data/code/atomx/output/nist_spectra/lr_i_ground_grid_crosscheck.json)
is also included in the site's numerical validations. The adjacent Lr II
ground gives **3.6716 eV** versus NIST ASD's **4.96 eV**. The site labels this
as a configuration-average comparison: NIST lists only part of the lower
`7p` manifold, so there is no complete evaluated centroid reference for a
strict level comparison. The Slater energy itself is not used in the score.

### Highly charged ion ground-shell repair (27 September 2026)

A check of adjacent completed ground energies found **27 negative ionization
energies**, all from the same 69-electron sequence. The cause was ground-state
selection, not a converged-energy sign error. NIST's level file abbreviates
Os VIII and similar grounds as `5d¹`; `expand_nist_config` cannot restore the
missing ionic core from that token alone. The earlier fallback inserted the
neutral thulium `4f¹³6s²` configuration. NIST's separate
[ground-state table](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html)
specifies `[Xe]4f¹⁴5d` for Os VIII. The two source tables provided **188**
unambiguous abbreviated-level cases whose selected DHF ground differed from
the complete ground-shell field. `scripts/repair_dhf_ground_selection.py`
changed their ground selection transactionally, added **187** missing
configuration jobs, and retained old calculated configurations as ordinary
jobs for provenance. The comparison now reads the ledger's selected ground
jobs directly, so it marks these ionization pairs as waiting until the correct
configurations finish instead of displaying the old negative values. A
post-repair sweep found **zero** nonpositive ionizations among completed
selected ground pairs. In Mt VII and Ds VIII, NIST's `ground shells` field
reads `[Rn]5f¹⁴6d³`, while its separate `ground configuration` field and
level table both specify `6d²7s`. The level configuration remains selected;
all source fields are preserved in
`output/nist_spectra/dhf_ground_selection_repair.json`. NIST defines
[ground shells and ground configuration](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html)
as separate fields and cautions that the configuration is the largest
component of an approximate eigenvector.

### Ground candidates beyond direct NIST coverage

The NIST ground-state export stops at Z=110. The former fallback filled each
unlisted ion like a *neutral* atom with the same electron count, which can
select a very wrong shell for a highly charged ion. For Fl LXXXVII
(Z=114, charge 86), it chose `3d⁸4s²`. The NIST ionic 28-electron sequence
through Ds instead lists `3d¹⁰`; a fresh atomx calculation places that
configuration **313.327439 Ha lower** than the old one with the same nuclear
mass, Breit, QED, and radial grid. The source export itself does not assign a
Fl LXXXVII ground state. The correction traces the nearest lower-Z NIST ionic
member and marks the selection as extrapolated.

The source-informed fallback now covers 937 stages without a direct NIST
assignment; 65 stages retain the neutral estimate. It added **711** distinct
configurations to the live queue; 30 second-nearest ionic candidates then raised it to **16,587** jobs. Old calculated
configurations remain available for an energy check. When both candidates
converge, the lower DHF energy determines the selected theoretical ground;
the periodic recheck records every switch. Among the first **70** converged
candidate pairs, the ionic analogue was lower in **67** and the old neutral
candidate in **3**. These counts will change as workers finish. All **7,021**
stages now appear in the ionization CSV, including stages without a NIST
ionization energy; such rows show the theory value and an explicit missing
reference status, never an experimental error score. The [selection
report](/data/code/atomx/output/nist_spectra/dhf_isoelectronic_ground_repair.json)
and [live candidate audit](/data/code/atomx/output/nist_spectra/dhf_ground_selection_watch.json)
record the source and energy checks.

The ionic sequence itself changes near the 88- and 89-electron shells.
As a separate control, Rf XVI (Z=104, charge 15) was calculated in the
second-nearest NIST analogue's `5f⁷6s²6p²` configuration on the same 50-Bohr,
1-Bohr outer grid and with the same isotope mass and corrections. It converged
to **−38479.746743025 Ha**, which is **0.545684287 Ha above** the selected
`5f⁸6s²6p` candidate. The [control record](/data/code/atomx/output/nist_spectra/rf_xvi_ground_third_candidate_check.json)
supports this one assignment; it does not establish that every unlisted ion's
lowest possible configuration has been found.
The [nearby-analogue queue](/data/code/atomx/output/nist_spectra/dhf_nearby_ionic_candidates.json)
adds the **30** missing second-nearest configurations for the 88- and
89-electron sequences across Z=104–118. They use a separate
`candidate_pending` queue and the same isotope mass as the selected ground.
The automatic selector compares converged energies from these nearby NIST
ionic analogues, the nearest analogue, and the old neutral filling. A source
analogue remains provisional until its competing DHF calculations finish.

Pd I `[Kr]4d¹⁰` also resisted ordinary starts and 0.35 orbital damping. A
0.70-damped TFDA start converged in **71 iterations** with residual
`1.89×10⁻⁶` and ordered orbitals; an independent production rerun reproduced
**−5038.350897545 Ha** and recorded the **+0.026136195 Ha** leading normal
mass term. A 0.90-damped trial did not converge after 220 iterations and its
energy was excluded. Atomx's Pd I configuration-average ionization is
**6.44191 eV** against [NIST ASD's evaluated 8.336839 eV](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=on&biblio=on&e_out=0&el_name_out=on&level_out=on&shells_out=on&spectra=Pd&unc_out=on&units=1).
Because Pd II `4d⁹` has a multiplet, this ground-term difference is
indicative; the complete evaluated-configuration centroid reference is
**8.51236 eV**, or **24.32%** above atomx. The numerical convergence repair
does not remove the physical single-configuration error.

The Ti I `3d⁴` excited configuration converged with 0.70-damped TFDA on the
production 50-bohr grid after 215 iterations. An independent 50-bohr grid
with 0.9-bohr outer spacing converged after 204 iterations to an energy within
**0.00030 cm⁻¹**. Both have correctly ordered orbitals and residual below
`1×10⁻⁵`, so the production energy **−852.153718821 Ha** is now accepted in
the ledger. Two 60-bohr trials did not converge, and a finer 50-bohr trial
either lost bound 3d roots or remained unconverged. Those branches are retained
as numerical cautions in `output/nist_spectra/ti_i_3d4_validation.json`.
Ti I line positions remain configuration-average estimates, not resolved
term/J comparisons.

### Diagnosing inverted configuration orders

The comparison page now distinguishes a DHF configuration-order reversal
from a disagreement with an evaluated NIST configuration centroid. The
[complete inversion audit](/data/code/atomx/output/nist_spectra/theory_comparison/inverted_configuration_audit.csv)
retains each affected ion, configuration pair, NIST row IDs, and both centroid
gaps. Within H–Ca, the **249** affected observed line rows split into **95**
where evaluated NIST centroids have the same configuration ordering as DHF,
**two** with an actual centroid-order disagreement, and **152** without a
complete evaluated NIST centroid pair. An individual NIST line connects
specific term/J levels; its positive photon energy does not imply the same
ordering for whole configuration centroids. All such DHF line estimates remain
unscored.

The broader audit exposed three especially clear false SCF branches. In
Sc IV `3p⁵4p`, Sc V `3p⁴4p`, and Ti V `3p⁵4p`, the converged GSZ result
had labelled core `3p` radial orbitals with **two full-amplitude nodes**;
the physically expected count is one. The existing orbital-energy order
guard accepted all three. Independent TFDA starts on the same grids, with
the same isotope masses, Breit and QED settings, converged with one core
`3p` node and lower total energies. The production solver now checks the
radial node count for core-hole `p` orbitals before accepting a result.
Its tested range is Z=21–74 for n≤3 core `p` orbitals; previously
completed configurations outside the audited repairs have not been
retroactively certified by it.

The [Sc IV](/data/code/atomx/output/nist_spectra/sc_iv_4p_branch_repair.json),
[Sc V](/data/code/atomx/output/nist_spectra/sc_v_4p_branch_repair.json), and
[Ti V](/data/code/atomx/output/nist_spectra/ti_v_4p_branch_repair.json)
branch reports retain every tested start, convergence residual, node/order
checks, isotope mass, and selected energy. The selected TFDA+0.70-damped
branches lowered the original ledger energies by **0.549104**, **0.535504**,
and **0.799715 Hartree**, respectively. After the transactional ledger
updates and line refresh, full-table NIST-centroid-order conflict rows fell
from **333 to 209**. Sc V's `4p→4d` configuration ordering is now correct;
Sc IV and Ti V still have NIST-centroid disagreements of physical model
size. Missing configuration interaction is a plausible contributor for these
open-core excited states: [NIST's levels help](https://physics.nist.gov/PhysRefData/ASD/Html/levelshelp.html)
describes a listed configuration as the largest component of a calculated
level eigenvector, so it need not be a pure configuration. Lower SCF energy
and correct radial nodes do not establish
term-resolved line accuracy.

The same diagnosis also repaired the `2p⁵3p` branches of Sc XII, Ti XIII,
V XIV, Kr XXVII, and five additional ions at Z=29, 31, 36, and 38. Their
stored GSZ branches all reproduced the wrong `2p` node count; the selected
TFDA branches passed the node and orbital-order checks. The
[branch repair manifest](/data/code/atomx/output/nist_spectra/dhf_core_p_branch_repairs_summary.json)
cross-checks every retained report against the current ledger. Across these
first 12 repairs, selected total energies fell by **6.916627 Ha** in aggregate;
this aggregate is a bookkeeping sum across different ions, not a physical
observable. After rebuilding the comparison, NIST-centroid-order conflict
rows fell further to **193**. Sc XII's large evaluated-centroid excitation
residual decreased from about **441,826** to **263,490 cm⁻¹**, leaving a
substantial method error after the numerical branch was corrected.

The same simple node rule rejects all tested Ne I `2p⁵3p` starts, including
their lower-energy TFDA branch. The original Ne I job was left unchanged;
its 2p radial lobes require a separate state analysis. This is why the
production guard starts at Z=21. The other previously completed
core-hole jobs are still subject to targeted review as the full calculation
continues.

A resumable [cross-element root audit](/data/code/atomx/output/nist_spectra/dhf_core_p_root_audit.jsonl)
then checked 20 short completed GSZ jobs across Z=21–38 by recomputing the
stored branch on the same radial grid. Nineteen reproduced their energy and
core-`p` node count. V XX `1s²2p3p` reproduced the stored energy exactly but
its labelled `2p` orbitals had one full-amplitude node instead of zero. A
converged, node-correct TFDA branch was **0.082104 Ha higher** than the
incorrectly labelled branch. The repair gate now requires the stored GSZ
branch to reproduce its energy and fail the node check, then accepts the
validated replacement regardless of the direction of the total-energy
change. V XX was updated transactionally and is the **13th** verified repair.
This example demonstrates why simply choosing the lowest SCF energy among
different orbital identities can give the wrong named configuration.
The later Sc X `3p4p` case reproduced the same failure with two significant
nodes on its labelled `3p` orbitals; its node-correct TFDA energy is
**0.049785 Ha higher**. A direct solver regression checks both branches.

The resumable audit has checked **at least 792** completed GSZ core-hole jobs
in Z=21–40. **765** reproduced both their stored energy and expected core-`p`
node count; **27** reproduced a wrong-node branch, and all 27 have been
repaired. A further 163 attempts interrupted by a concurrent solver-source
reset are kept in a separate error log and are not counted as orbital audits.
The [live audit summary](/data/code/atomx/output/nist_spectra/dhf_core_p_root_audit_summary.json)
reports how many current GSZ jobs in this scope still await the same check;
that count changes as the all-element workers finish new configurations.
The [repair manifest](/data/code/atomx/output/nist_spectra/dhf_core_p_branch_repairs_summary.json)
verifies **48** corrected branches in total: 12 earlier examples, 27 found
by the Z=21–40 audit, and nine heavier examples described below. These are
checks of orbital identity, not claims
that the resulting DHF excitation energies equal NIST term energies.

An [exploratory heavy-ion probe](/data/code/atomx/output/nist_spectra/dhf_heavy_core_p_root_probe_summary.json)
reran **102** stored GSZ core-hole configurations across 16 tested elements
from Z=42 to 118. **93** reproduced the expected low-`p` node profile. Xe
XLV, Ba XLVII, and W LXV `2p⁵3p` each reproduced a stored branch whose
labelled `2p` orbitals had one significant node instead of zero. Same-grid
TFDA starts passed the node and orbital-order checks, lowering their total
energies by **0.451656**, **0.469142**, and **0.769742 Ha**, respectively.
Six additional W configurations with an open `3p` shell reproduced stored
branches whose labelled `3p` orbitals had two significant nodes instead of
one. Valid alternate branches changed their energies by **3.002–14.759 Ha**.
All nine ledger rows and dependent line estimates were repaired; the
[raw probe results](/data/code/atomx/output/nist_spectra/dhf_heavy_core_p_root_probe.jsonl)
and per-branch reports retain the evidence. This extends the production
guard through Z=74, with Xe and W regressions. At this checkpoint, all **80**
remaining completed GSZ jobs with low-`p` holes at Z=41–74 had been checked;
new jobs can add to that count. The probe does not certify all heavy-ion
core-hole states, and Z>74 has not been validated for the guard.

The larger Sc I `3d²4s` versus `3d4s4p` configuration-order conflict affects
61 observed lines. A [four-start same-grid probe](/data/code/atomx/output/nist_spectra/sc_i_inverted_pair_branch_probe.json)
for both configurations reproduced their stored energies from GSZ, TFDA,
and two damped TFDA starts within **1×10⁻⁸ Ha** per configuration. The
calculated order remains opposite the complete evaluated NIST configuration
centroids. Changing the SCF start alone therefore does not resolve this
discrepancy; state-specific correlation and configuration mixing remain
possible contributors, and these lines stay flagged rather than scored.

All **30** second-nearest ionic ground candidates in the 88- and 89-electron
sequences have now converged. None is lower than the current selected
ground among those 30 comparisons; the larger 711-pair neutral-versus-ionic
candidate audit remains in progress.

On 27 September, all **7,021 selected ion-stage ground jobs** were complete.
The full-table `scripts/audit_first20_dhf.py --max-z 118` check passed for
all **236,495** observed standard line identities, ground-stage coverage,
finite completed energies, positive adjacent-stage ground ionizations, and
completed Rydberg-series ordering. Excited configurations and alternate
ground candidates remain in the calculation queue; the audit does not mark
the entire 16,587-job ledger complete.
The updated all-element ionization comparison has **311** unbracketed
evaluated NIST values. **81** have a physically unique ground level on both
sides, giving **7.47% RMS** relative error; **234** have complete evaluated
configuration centroids, giving **6.73% RMS** against those centroids. These
are distinct reference observables. The other evaluated rows remain visible
with their calculation and manifold statuses, outside those strict scores.

For additional atomx calculations, use the atlas page's calculated-line CSV upload. It compares any selected ion whose calculated transitions have configurations, J values and vacuum wavelengths; unresolved or ambiguous classifications are left unmatched. See [atlas instructions](NIST_SPECTRUM_ATLAS.md) for the CSV columns.
