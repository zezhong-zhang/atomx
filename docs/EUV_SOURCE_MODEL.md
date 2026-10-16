# Laser-to-EUV source models and experimental alignment

The native tin line table now feeds an absolute, energy-conserving reduced
source model. A second spatial, multi-charge LTE interface now supplies
charge-neutral EOS response, explicit continua, layered transfer, conduction
and Drude wave deposition. Both remain conditional models, **not validated
industrial conversion-efficiency predictions**. A separate bounded multi-charge
non-LTE kinetic closure now resolves physical resonance lifetimes and photon
energy moments; it is qualified on finite analytic controls. A separate native
Sn12+/13+/14+ complete4d pilot now supplies partial state-resolved PI/RR, while
explicit signed CI Auger widths now use the same final-ion continuum and
complete bound-state expansions. A same-orbital full-n4 Sn12+/13+ resonance
campaign is active; complete real Sn kinetic coverage remains missing. The original atomic-structure
task and its unmet core-correlation gates remain in [the CI guide](TIN_EUV_BASELINE.md).

## Current atomic gates (2026-10-16)

No complete experimental alignment or absolute CE result is accepted. The
completed **single declared native V38 n35/l6 Zn30-hole-only
CI+second-order model** gives:

| Sn12+ target | Calculated(cm^-1) | Minus experiment(cm^-1) | Fixed acceptance |
|---|---:|---:|---|
|3F3|9792.557610|+6.557610|Conditional PASS30|
|3F4|18662.388746|+98.388746|FAIL50|
|1D2|24852.326800|+14.326800|FAIL identity; inside25 energy interval|
|3P2|38872.919949|-171.080051|FAIL30|
|1G4|40991.785106|+1273.785106|FAIL44|

This preserves complete paper-fold6 n4 CI and all0..3 core-hole correction
ranks, including positive box pseudostates. Full source/input identities,
independent Q anchors, failures and limitations are recorded in the
[active partition account](#complete-positive-box-basis-and-exhaustive-q-partition-2026-10-16).
Sn13+ full-n7 CCSD still misses13179(8) by-35.184775cm^-1; a different
method's one successful interval cannot fill that failed gate. The fixed n35/l6 radial check fails stability; the independently qualified
nonperturbative4s-hole discriminator also fails overall experiment. Current API capabilities remain distinct from these private
source-bound research drivers; no new contest release is produced while
these scientific gates fail.

The later original-source audit finds a different core partition: it includes
4p-hole/outside paths omitted from this native Zn30-hole-only model. Its
source-matched finite controls and corrected interpretation are retained
[below](#independent-original-ambit-full-particle-hole-control-2026-10-16);
conditional intervals from these different models must not be combined.

## Implemented chain and ownership

| Component | Implementation | Contract |
|---|---|---|
| Population generator | `plasma.crm.CollisionalRadiativeModel.rate_matrix(sparse=True)` | Same physical rates as the existing dense CRM; source columns, conservative CSR; explicit optional escape factors multiply A only |
| Sparse propagation | `plasma.population_network` | Constant-generator exponential propagation; explicit collision-strength tables reuse the CRM Maxwell average; no clipping or normalization repairs |
| Absolute line radiation | `plasma.euv_source.LTELinePlasma` | Common level energies/g and Einstein A; all supplied levels in the LTE partition, including levels without emitted lines |
| EUV transport | `spectrum.slab_transfer` | Exact homogeneous hemisphere integral; conservative layered angular formal solution, supplied boundary illumination and separate directional `trace_ray` |
| Laser coupling | `plasma.laser_absorption` | Explicit classical collisions/log; original single-pass slab and stable coherent multilayer Drude waves through critical/overdense cells |
| Material dynamics | `plasma.euv_source.evolve_lte_slab` | One-temperature, fixed-charge ideal EOS including excitation heat capacity; symmetric homologous slab expansion |
| Non-LTE alternative | `plasma.euv_source.evolve_cr_slab` | Same energy ledger and CRM physical rates; fixed charge, supplied Omega for every transition, local escape-probability closure; dense joint Jacobian capped explicitly at 256 levels by default |
| Multi-charge LTE | `plasma.charge_mixture`, `plasma.mixture_radiation` | Supplied contiguous positive charge partitions/IPs; self-consistent neutrality, excitation/ionization EOS derivatives; matched line tables and explicit continuum mode |
| LTE continua | `spectrum.lte_continuum`, `spectrum.continuum` | Shared free-free coefficient and stimulated opacity; sourced Gaunt provider and supplied PI cross sections for detailed-balance free-bound radiation |
| Spatial source | `plasma.lagrangian_source` | Smooth-flow 1D Lagrangian ideal plasma, pressure work/kinetic energy, flux-limited conduction, signed radiation heating/cooling and full laser ledger |
| Finite multi-charge kinetics | `plasma.nonlte_source` | Explicit physical resonance/EII/TBR/RR/Omega data, self-consistent electron count and excitation/ionization energy; optically thin fixed-volume source, default256-state cap |
| Explicit native CI PI/RR | `scattering.photoionization.ExplicitCIPhotoionization` | Supplied complete adjacent CI spectra, signed Dyson removal, final-level projected Hartree continuum and Milne photon moments; no inferred residual Fock cascade |
| Explicit native CI Auger | `scattering.autoionization.ci.ExplicitCIAutoionization` | Full signed Dirac one-body plus Coulomb amplitudes, separate interference ledger, actual CI thresholds and unbranched inverse capture; full magnetic sums or independently checked exact coupled scalar reduction |
| Absolute experiment | `reference_data.euv_experiment` | Counts-to-energy gain, wavelength grid, shots and angular calibration are explicit; no absolute CE inferred from relative spectra |
| Native input/CLI | `workflows.tin_source`, `tools/run_tin_source.py` | Stream every saved native line, bind identities/hashes/counts, reject incompatible data or caps; paired laser-on/off calculations |

Sparse CRM storage supports bound-free and DR processes already supplied by
the CRM. The original **fixed-charge non-LTE source** rejects those channels.
The separate multi-charge spatial source supplies an LTE Saha EOS and explicit
continua. The separate `nonlte_source` solves finite multi-charge kinetics in
a fixed-volume optically thin closure; it is not yet coupled to that spatial
LTE driver. Complete matched Sn collisional/PI/Auger inputs remain absent;
the independent12-level complete4d PI/RR pilot below cannot be spliced into
the115715-level paper-space power table.
The original fixed-charge CR slab rejects missing Omega tables; the new
finite source requires them in its explicitly selected tabulated-excitation
mode. Neither consumer invokes a silent gf-based fallback. Current native Sn data have no matched collision-strength
table; consequently the large Sn calculation uses **fixed-charge LTE**, while
the connected non-LTE mode is qualified on independently soluble small systems.
The sparse generator alone does not establish a scalable joint non-LTE solve.

## Equations, units and energy bookkeeping

Source geometry, density, time and power use SI; temperature and atomic energies
are in eV. Line powers are `n_upper A_ul E_photon`, including photon energy.
They are not the earlier paper figure's relative `g A exp(-E_upper/T)` weights.
For a normalized frequency profile phi:

```text
epsilon_nu = n_upper A h nu_0 phi_nu                 [W m^-3 Hz^-1, 4pi]
alpha_nu   = A c²/(8 pi nu_0²) (g_u/g_l n_l - n_u) phi_nu [m^-1]
F_both     = epsilon_bin L beta(alpha_bin L)
beta(tau)  = [1/2 - E_3(tau)]/tau
```

The thin limit is `F_both=epsilon_bin L`; each thick LTE face tends to
`pi B_nu`. A normal-ray `exp(-tau)` is not the angular escape fraction.
Stimulated emission is included. Population inversions reject the passive
model. Emission and absorption are independently assembled from Einstein A,
so the Planck limit is a physical check, not a fitted source normalization.
Einstein factors use actual line centers. Finite bins/profiles introduce wing
Kirchhoff errors; they do not supply exact continuum detailed balance.

Each line first deposits its **actual integrated power** in its containing
energy bin. A declared Gaussian response is column-normalized on the finite
positive-energy grid, with a seven-sigma numerical kernel. All saved lines,
including weak lines, are retained. Grid refinement is needed for opacity and
band boundaries. The default 0.2 eV FWHM is a **phenomenological model input**,
neither a calculated density broadening law nor a paper/experimental parameter.

The laser dielectric is `eps=1-wp²/[w(w+i nu_ei)]`, with attenuation
`alpha_laser=2 w Im(sqrt(eps))/c`. Entrance reflection and single-pass
absorption/transmission sum to one. The default homogeneous-corona admission
requires `n_e/n_critical <= 0.8`; an overdense target requires a spatial model,
not an automatic density change. Classical weak-coupling and one-temperature
validity are additional physical assumptions, not certified by Drude arithmetic.

With ion column N, areal mass M and edge speed v, the reduced symmetric
expansion uses `Ldot=2v`, `K/A=M v²/6`, `vdot=6P/M`. Thus kinetic gain equals
`P Ldot` exactly. This is a reduced pressure-work closure, **not an Euler PDE
solution for ablation, shocks or density gradients**. The LTE per-ion energy is

```text
U/eV = 3/2 (1+q) T + <E_excitation>
dU/dT = 3/2 (1+q) + Var(E_excitation)/T²
```

Only escaped radiation cools matter; trapped energy is returned locally in
the stationary-transfer closure. Non-LTE also subtracts the actual excitation
energy derivative from the thermal equation. In a radiative-decay-only test,
excitation energy supplies escaping photons while electron temperature stays
constant: a second subtraction of photon power would be incorrect. Non-LTE
eliminates one population coordinate so numerical Jacobian variations remain
in the particle-conserving tangent space. It never renormalizes trial states.
Line-averaged escape factors are the adjoint of the same binned emission map.
Trapped photons are assumed to cycle in their originating transition; full
overlapping-line conversion and radiation pumping are not implemented.

The original fixed-charge modes check, throughout the trajectory:

```text
initial stored energy + incident laser
  = final internal + kinetic + reflected laser + transmitted laser + escaped EUV
```

Fixed-charge preparation/ionization energy is constant and excluded on both
sides. Radiation energy storage, pressure and material conduction are excluded
by the reduced closure. A small ledger residual certifies this conditional
model, not the adequacy of these omissions.

## Efficiency convention and reproducible native calculation

The band is **13.365–13.635 nm** (13.5 nm ±1%). One slab face is one hemisphere.
Reported numerator/denominator are time integrals of escaped band energy and
**incident**, not absorbed, laser energy. Since the initial corona is preheated,
both quantities are retained:

```text
gross ratio = E_band,on,one-face / E_laser,incident
added-pulse ratio = (E_band,on - E_band,off),one-face / E_laser,incident
```

The subtraction is a conditional added-pulse response, not a complete prepulse
plus main-pulse CE. It can be negative in other parameter regimes; no clipping
is performed. The small baseline is not comparable to a published multi-charge
droplet CE or to a source's electrical/wall-plug efficiency.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python tools/run_tin_source.py \
  --atomic-run /path/to/native/fold6_complete \
  --output output/sn12-source \
  --rho-g-cm3 0.0001 --thickness-um 30 --temperature-eV 20 \
  --laser-wavelength-um 10.6 --peak-intensity-W-cm2 1e8 --duration-ns 5 \
  --bin-width-eV 0.05 --fwhm-eV 0.2 --coulomb-log 5
```

The pulse is sin² over 5 ns: incident fluence is 0.25 J/cm². The density is
an explicitly prepared corona, not solid tin; at 0.01 g/cm³ a homogeneous
CO2-illuminated model fails the critical-density admission. Planning caps cover
line-table construction, not a claimed whole-process RSS hard limit. No giant
CI archive or dense square over 115,715 levels is loaded. Gaussian/sparse
matrix contractions replace Python line loops during time evolution.

Outputs include input/source identities, atomic file hashes, all populations
at both endpoints, temperature/size/velocity and energy trajectories, absolute
emission/opacity/escaped spectra, and independently integrated time spectra.
The latter must agree with the ODE radiation/band ledgers within 5e-4 without
rescaling. Output paths must be new. Temperature-floor or source-change events
reject certification; failures remain in `failure.json`.

## Verification and physical frontier

The completed native baseline uses115715 partition states and1628730 saved
E1 lines. In the declared20eV,1e-4g/cm³,30µm,5ns sin² case above,
laser absorption is36.0722%; the one-face gross band ratio is0.751100%,
and its laser-off-subtracted added-pulse ratio is0.256984%. The latter is
**not full-cycle source CE**. The largest trajectory ledger defect is2.343e-8;
independent sampled band-spectrum replay differs by at most2.086e-5.
At fixed width and ODE tolerance, refining0.05→0.025eV bins changes the
added-pulse ratio by0.4460%; tightening ODE rtol2e-7→1e-9 changes it by
3.15e-9 relatively. Neither is an experimental error estimate. The baseline
wall time is15.37s and measured peak RSS366.7MiB, including input/table
construction, both pulses, independent spectra and serialization; concurrent
load prevents a controlled hardware-scaling claim.

The affected shared CRM/plasma/spectrum/architecture panel passes141 tests.
An earlier shared Numba-cache failure and the initial diagnostic mistakes
remain in the local source report's `failures.md`; no physics tolerance or
rate was fitted to obtain a desired CE. Current artifacts and lossless full
power-consumer input are in `atomx-tin-source-model.zip` under the task root;
the earlier atomic/challenge ZIPs remain sealed.

`tests/test_euv_source_model.py` and `tests/test_tin_source_loader.py` check
analytic two-level CR and its collisional LTE limit, excitation underflow,
conservative sparse storage, independent angular quadrature, thin/blackbody
limits, classical/Drude laser limits, Einstein/Planck consistency, excitation
heat capacity, the adiabatic `T L^(2/3)` invariant, coupled non-LTE energy
exchange, absolute powers, native state identity and reject-without-trimming
resource gates. Existing CRM/scattering/spectrum/architecture tests cover the
changed shared generator. No NIST or experimental CE acceptance is claimed.

The next physically necessary development is matched Sn multi-charge EII,
excitation, RR and DR data, including resonances, with density-dependent EOS
and a consistent non-LTE energy closure. LTE continua, spatial transport,
wave deposition and conduction are implemented below; shock/ablation, neutral
solid-target preparation and separate electron/ion temperatures remain absent. Population uncertainties, high-level
radiative omissions, bare-CI spectral shifts and line widths must enter the
CE error budget before optimizing a target or wavelength. A full pulse-cycle
comparison must match target preparation, geometry, laser pulse, collection
hemisphere, spectral band and detector response. NIST constrains atomic
identifications/energies; published matched spectra and absolute laser/EUV
measurements constrain source efficiency.

## Multi-charge spatial development

`SahaMixture` requires every declared partition and every adjacent ground-state
ionization potential. Energies share the cumulative ionization zero, so changes
of charge cost energy rather than only changing pressure. Neutrality determines
`n_e = n_i <q>` in log space. Heat capacity and the density derivative of internal
energy include charge response; both are required for compression/expansion.
Partitions are owned immutable snapshots and source hashes bind actual energies,
weights, thresholds and state IDs. Charge-boundary truncation remains explicit;
normalization of a finite charge set does not certify that its end stages suffice.
Neutral q=0, continuum lowering, degeneracy and nonideal/solid EOS are absent.

States above an included charge's next ground-continuum threshold are rejected.
They require explicit resonance, autoionization and dielectronic-recombination
coverage, rather than admission as stable levels or silent deletion. The highest
supplied charge has no supplied next threshold; its continuum coverage is an
additional caller responsibility. A frozen single-ion LTE demonstration remains
a historical conditional calculation, not a qualified Saha input.

`MixtureRadiation` binds each charge's energies, weights, line source, mass and
common energy grid. It never substitutes an absent ion. Continuum mode is
mandatory: `none`, `free-free`, or `free-free+bound-free`. The latter requires
actual supplied PI edges and their coverage. The Gaunt provider is explicit;
unit/Born/hydrogenic approximations are not automatically certified for high-Z,
low-temperature tin. LTE Milne reciprocity is independently checked against
the existing RR integrator. This is not non-LTE free-bound kinetics.

`layered_transfer` uses exact constant-cell formal solutions and angular
quadrature. It returns both face fluxes and cell heating/cooling. Foreground
absorption can make a cell's net radiative loss negative; this is retained.
The local emission-minus-absorption check is independent of face telescoping.
`trace_ray` consumes path lengths along a specified line of sight and returns
radiance per solid angle. It does not substitute hemisphere-integrated flux
for a spectrometer measurement.

`drude_layer_stack` uses a decaying-wave reflection recursion and Poynting-flux
cell losses. It admits critical/overdense layers without truncating density.
The exactly lossless zero-index singularity is rejected. This normal-incidence,
local, coherent planar dielectric treatment omits oblique rays, nonlocal or
strong-field physics. It does not validate the classical collision closure.
The original single-pass corona cap still applies only to that transport mode.

`evolve_lagrangian_slab` advances fixed-mass cells and material nodes with vacuum
mechanical boundaries. It includes pressure work and kinetic energy, EOS density
response, closed-end flux-limited Spitzer conduction, radiation exchange between
cells and incident/reflected/transmitted/absorbed laser energy. Positive geometry,
passive optics, RHS/cell resource caps and the energy ledger are admission gates.
It rejects crossed cells instead of pretending to solve shocks. There is no
solid ablation mass feed, viscosity/shock solver, two-temperature relaxation,
spherical expansion or time-dependent radiation energy/momentum. Qualification
on manufactured two-charge systems is numerical evidence only, not a Sn source
prediction. Public functions are explicit module APIs; the large native CLI still
uses its declared original fixed-charge preparation.

## Finite physical resonances and non-LTE energy

`CRMDRSatelliteSource` remains the existing **effective** satellite-feed model:
its capture input already includes radiative stabilisation branching. Its virtual
satellite population cannot be reinterpreted as a physical autoionizing population.
The new `CRMAutoionizationChannel(resonance, continuum, partial_rate_per_s,
source_id)` instead declares every physical outgoing Auger channel. The shared
CRM generator adds the partial loss and the unbranched inverse capture, using
`scattering.dielectronic_recombination.dielectronic_capture_rate_cm3_s`:

```text
alpha_DC,c(T) = lambda_e^3 g_r/(2g_c) Gamma_A,c exp[-(E_r-E_c)/T]
Gamma_total  = sum_c Gamma_A,c + sum_l A_rl + supplied collisional losses
```

These are isolated narrow-resonance Maxwellian impact rates, without resonant
interference or an implicit density-suppression model. Parent energies and
statistical weights remain separate for different continua. Radiative branching
emerges from the solved lifetime; multiplying alpha_DC by a branching factor
before solving the resonance would apply that branching twice. A static Schur
elimination recovers the independent existing isolated DR coefficient exactly.
The original fixed-charge slab rejects the new charge-changing channels.

`NonLTEKineticPlasma` binds unique state IDs, positive contiguous charges and
common global energies to owned CRM arrays. Reported total Auger widths must
equal the explicit partial channels. Above-next-continuum states require an
explicit resonance model. Bound-free thresholds and statistical weights must
match those same states. An effective satellite, missing Omega in tabulated
mode, absent native PI provider for RR, live/fallback EII or mismatched threshold
rejects; omitted processes are explicit, not inferred. Provider provenance is
caller-supplied and scalar rates/arrays/endpoints enter the source identity.
The top charge boundary still needs physical coverage assessment.

The shared Milne integral now supports `photon_energy_moment=1`, returning
**eV cm³/s** rather than the default **cm³/s**. Both moments use the same actual
PI cross section. For sigma(E) proportional to E^-2, the independently soluble
mean RR photon energy is I+T; inserting I+3T/2 instead would give the wrong
thermal response. Missing providers reject in the moment consumer. Negative
explicit EII coefficients now reject rather than being silently replaced by zero.
Extreme Saha underflow rejects the coupled inverse-EII consumer until a reduced
rate representation is supplied; it does not set a finite inverse process to zero.

`evolve_nonlte_isochoric` conserves nuclei by eliminating one population
coordinate. With p-dot=R(n_e,T)p, n_e=n_i <q>, and all atomic energies on the
same zero, its per-nucleus energy equation is:

```text
U = 3/2 (1+<q>) T + sum_s p_s E_s
3/2 (1+<q>) Tdot = Q/ni - P_lines/ni - P_RR/ni
                    - sum_s E_s pdot_s - 3/2 T sum_s q_s pdot_s
```

All terms in this expression use eV per nucleus per second; public deposited
power uses W/m³ with an explicit joule conversion. Thus EII consumes ionization
and new-electron thermal energy, Auger releases resonance energy and adds an
electron, and RR photons remove the cross-section-weighted captured-electron
energy. Stored excitation supplying spontaneous photons is counted once.
A common energy-zero shift does not change the temperature equation or its
residual conditioning. Signed tangent probes for the ODE Jacobian are never
clipped; actual returned populations/temperature and the full matter/photon
ledger are independently gated.

This source is prepared, ideal, one-temperature and fixed-volume. Radiation is
optically thin supplied bound-bound plus spontaneous RR; no free-free, pumping,
transport, work/expansion, nonideal EOS or solid target is implicitly connected.
The separate spatial LTE source still supplies its own conduction/transport.
The default256-state and joint-Jacobian budget reject rather than reducing
physical states. CSR rate storage alone does not remove the dense Radau Jacobian
in this finite driver. Applying its current10-buffer planning rule to115715
levels would be about1.07TB; this is an **algorithm-specific plan**, not a lower
bound on sparse kinetic methods. No such allocation or large native kinetic solve
was attempted. A scalable joint solve and complete real atomic rates remain
independent requirements.

The source-bound qualification in local evidence
`~/playground/tasks/atomx-rccsdt-euv/build/euv_nonlte_20261015` reports:

- **18 new independent controls;191 affected checks pass**, covering the original
  source/CRM/transport panels and additional DR reference/density checks. Unbranched
  phase volume, multiple continuum parents, one emergent DR branch, the RR first
  moment, electron/ionization thermal terms, finite source ownership/admission,
  energy gauge and the original fixed-charge rejection are exercised.
- A manufactured three-state10ns heating/resonance/EII/TBR/RR run changes mean
  charge11.45→11.849619 and temperature30→35.462886eV. Deposited heat is187.245272,
  escaped line energy8.000050 and RR energy0.044547eV/nucleus; maximum ledger
  defect1.26e-12. These are numerical controls, not Sn CE. Observed wall1.30s
  includes2300 RHS calls under shared load, not a controlled performance claim.
- A collision-only finite model independently relaxes to the Saha energy root:
  kinetic37.3198419879eV versus root37.3198419865eV; maximum energy defect3.17e-11.
- The actual saved native Sn12+ power table has5274 states with saved outgoing
  radiative lines. At the auxiliary theoretical258eV threshold,512 candidate
  resonance states have saved outgoing lines while110421 lack them; the latter
  count means **missing saved outgoing data**, not zero physical decay widths.
  The conditional32eV, optically thin, exact line-center13.365–13.635nm power
  fraction from above-threshold upper states is **11.7802%**, versus5.0007%
  conditional population. At20/24eV that power fraction is2.5487%/5.5671%.
  These are the existing bare-CI fixed-q model diagnostics, without broadening
  or transport; they are not an inferred autoionization rate or CE correction.
- A source-bound OPEN-ADAS public Sn unresolved ADF11 ACD/SCD/PLT/PRB query returned
  no results. Private-manual filenames do not supply public data. The available
  native power NPZ supplies no Omega, adjacent-ion identity, EII, PI or partial
  Auger tables, so no missing process is fabricated and no real Sn kinetic or
  new experimental-spectrum acceptance is claimed.

## Native adjacent-CI partial PI and RR

`scattering.photoionization.ExplicitCIPhotoionization` consumes explicit
`(csfs, BlockedCISpectrum)` charge sectors with one common B-spline orbital,
nuclear and bare-DC binding. Every block/root and magnetic partner is retained;
energy differences use the absolute CI totals. It checks H residuals, Gram,
symmetry, common orbital norm, magnetic norm and final-ion particle count.
No threshold or wavefunction is imported from a different charge optimization.
The default retains every root and magnetic partner. An explicit
`state_selection` must cover every supplied electron sector and keep complete
degenerate root clusters; it retains the full original CSF coefficients and
original root IDs. Selected roots are checked against the complete native H.
`projection_engine='streamed'` accumulates signed root vectors without retaining
the whole CSF projection bank. `magnetic_mode='highest-parent'` and
`'highest-all'` are explicit scalar-operator representations; unpolarized PI/RR
rejects an incomplete initial multiplet. The default remains `'complete'`.
The bounded operator rejects excess levels/magnetic entries/waves before the
corresponding work. Missing daughter configurations remain missing channels.

The shared `angular.dyson.dyson_annihilation` contracts signed real Fock vectors
without magnitude screening. Both this explicit-sector operator and the
existing two-/many-electron continuum path use it. The new operator retains
coherence between all radial components before squaring, averages the initial
multiplet and sums final magnetic states. It uses finite-frequency length E1
and the existing energy-normalized shooting kernel. The continuum field is the
final level's spherical direct Hartree density plus the supplied nuclear field;
Pauli projection removes all supplied same-kappa bound orbitals. Nonlocal
continuum exchange, interchannel/Fano coupling, EII and Auger are excluded.
The off-grid nuclear callable remains caller-owned and must remain immutable;
its values at native quadrature nodes are checked before cache use.

`recombination_moments` integrates spontaneous nonrelativistic Maxwellian Milne
rates directly at Gauss-Laguerre electron energies `T*x`, returning cm3/s and
eV cm3/s photon moments. Exact kinetic nodes and final-level fields are reused
across all initial levels; no fitted/interpolated PI tail is introduced.
Refine the quadrature order and radial/continuum representations separately.

The actual pilot command is `tools/qualify_tin_photoionization.py`. Its fixed
model is120Sn Fermi, statistical Sn12+ VN orbitals, bare DC and the complete
`[Kr]4d2 / [Kr]4d / [Kr]` spaces:9+2+1 fine levels and20 candidate partial
bound-free pairs. This is a separate small physical model, not a truncation
silently applied to the sealed full n=4 baseline. At ns90/ks9/R60bohr, the
native ground IPs are258.56548822 and284.24359600eV. NIST ASD5.12 lists
parenthesized theoretical values `(258)+/-3` and `(282)+/-3`eV, respectively:
differences+0.565488 and+2.243596eV. This is consistency with auxiliary theory,
not an experimental IP or cross-section validation. No NIST threshold shift
was applied. [NIST source](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=on&biblio=on&e_out=0&el_name_out=on&format=1&level_out=on&seq_out=on&shells_out=on&spectra=Sn&unc_out=on&units=1).

For a recombining ground parent at32eV, partial sums are9.24223e-13cm3/s
(Sn13+ to all retained Sn12+4d2 levels) and1.28527e-12cm3/s
(Sn14+ to both retained Sn13+4d levels), at ns90/8000 shooting nodes/order32.
They exclude capture into every other configuration and all DR. They are not
complete ionic recombination coefficients or an EUV peak/CE correction.
Across all nonzero partial rates and photon moments, order16->32 changes at
most1.201e-4 relative, shooting4000->8000 at most2.169e-6, and ns90->110 at
most1.474e-4. These pass the predeclared0.5%/0.5%/1% numerical targets but
leave correlation, continuum-exchange and experimental errors unbounded.

Ten new independent checks cover sorted-list Fermi algebra, tiny signed
amplitudes, coherent bright/dark radial interference anchored by spin spherical
harmonics, complete36-electron spectator-core d2->d1 removal strength2,
existing native two-electron normalization, analytic Milne moments, identity,
ownership and admission guards. The73-test affected continuum/CI/CRM/source
panel passes without skips. Three actual native runs observed12.1--15.6s and
438--501MiB peak RSS on shared hardware; these are not controlled speedup
measurements. The32-order three-temperature calculation reuses1152 unique RR
waves, followed by a separate PI diagnostic grid.

Local replayable evidence: `/home/zzhang/playground/tasks/atomx-rccsdt-euv/build/euv_native_rates_20261015/`.
Protocol, full common SCF snapshots, signed CI H/C arrays, per-pair PI/RR,
refinement comparisons, source hashes and the development ZIP accompany it.
That sealed PI/RR pilot does not contain the later full-n4 resonance data below.
Excitation/EII, population/transport coupling and absolute experimental CE remain
outstanding.

## Explicit signed CI Auger and balanced Sn thresholds

`ExplicitCIAutoionization(snapshot)` consumes the same checked adjacent-sector
snapshot as PI. The default `engine='magnetic-pairs'` retains every signed
one-/two-body Fock amplitude, including cancellation between h_D and Coulomb.
It reports partial widths in s^-1, separate h-only/V-only/interference terms,
actual CI electron energies and channel provenance. Missing residual levels
remain missing channels. `inverse_capture_cm3_s` uses the shared isolated,
narrow-resonance Maxwellian inverse with the true g_resonance/g_ion; it does
not multiply a radiative stabilization branch.

`engine='compiled-reduced'` evaluates one nonzero highest-weight magnetic
component per allowed continuum kappa and divides by its Clebsch-Gordan
coefficient. Orthogonality of coupled final-ion/electron states gives exactly
the complete scalar width; it is not a magnetic-state cutoff. With
`magnetic_mode='highest-all'`, only these exact representatives are assembled.
The final-ion spherical scalar density uses signed off-diagonal radial
coherences and is independent of M for a pure-J state. Shared
`numerics.continuum_actions.ContinuumFockAction` performs the complete one-/two-body
action with no magnitude screening; common occupied spinors are contracted
with their fermion phase and full direct/exchange contribution. Its declared
work bound is N_det [N + C(N,2) n_spinor]. h, core-Coulomb and active-Coulomb
inputs remain separate until their signed sum. Spinor and work limits reject
the whole request rather than removing orbitals or amplitudes.

Eight Auger checks include independent native two-electron h/V widths, inverse
phase volume, signed phases, root admission, full-M versus coupled widths,
highest-M scalar density and a four-electron closed-1s2 core contraction.
Five numerical Fock checks use independent sorted-list creation/annihilation;
the PI panel adds an independent scalar-density oracle. The affected
CI/continuum/CRM/source panel initially passed102 tests in15.43s. After
the optional Breit interface, the expanded operator/architecture panel passes
**160 tests in26.29s**, with3 optional external EXP-T checks skipped,
after adding complete-block orbital/RDM/gradient controls.
These qualify the operator within its declared direct-Hartree continuum;
nonlocal continuum exchange and interchannel/Fano coupling remain absent.

The active local evidence is
`/home/zzhang/playground/tasks/atomx-rccsdt-euv/build/euv_resonance_alignment_20261015/`.
It retains120Sn Fermi, ns90/ks9/R60, the original statistical Sn12+ VN
spinors and bare DC. A fresh SCF differed from the archived coefficients by
3.499e-10 and failed the unchanged1e-12 identity gate. The successful replay
uses the exact archived spinors; no orbital-identity tolerance was relaxed.
The initial four-root calculation with an uncorrelated daughter remains a
diagnostic and is superseded for rates by balanced full-n4 fold6 Sn13+ CI.

Both Sn13+ even J3/2 and J5/2 blocks retain every CSF/root:3733+4810 levels.
Their eigen-equation defects are at most5.457e-12Ha and their independent
first-three-CSF streamed-H control differs by at most8.186e-12Ha. The actual
Sn12+ ground-to-Sn13+ threshold is **258.79915055eV**,0.79915eV above NIST's
auxiliary theoretical258(3)eV, not an experimental IP validation. The same
bound model gives **13709.229696cm^-1** for the Sn13+ ground fine splitting.
Windberger et al. measure **13179(8)cm^-1**: the native result is high by
530.229696cm^-1, **4.02329%**, and fails experimental alignment.
[Primary experiment and FSCC comparison](https://arxiv.org/abs/1605.04236).
The paper's Breit/Lamb entries are comparison data, not corrections added to
Atomx energies. Native Breit/QED and outside-n4 correlation remain required.

The lower-level `SpinorCIHamiltonian` now forwards explicit static or
orbital-difference transverse options to the shared signed integral exporter;
`solve_ci_blocks` keeps its DC-only contract. Six same-space signed-H/phase
controls pass for static, zero and finite frequencies. The combined CI and
transverse panel has16 passes and3 optional external EXP-T skips. Actual
full3733/4810 DC matrices replay **bitwise identically** after this optional
interface change. Native static Breit then gives **13283.327709cm^-1**, a
splitting change of **-425.901987cm^-1**. The independent3-CSF DCB matrix
controls differ by at most1.092e-11Ha; full-H ground residuals are below
1.819e-12Ha. Both lower and upper energies are recomputed, with no paper
shift added. The remaining measured difference is **+104.327709cm^-1**;
the predeclared8cm^-1 experimental gate still **fails**. This is a separate
bound-state diagnostic, not a DCB correction spliced into DC Auger rates.

476 saved radiating Sn12+ parents lie above the balanced threshold. Full
signed partial widths to both correlated ground fine partners are complete
for this window, with separate continuum refinement. The first4000-node run completed476 parents/952 candidate channels
(874 open), observing885.33s and2422MiB RSS on shared hardware. It exposed
**94 parents with incomplete saved E1 endpoints** and **3 parents failing**
the predeclared operational isolation check
`hbar*(Gamma_A+A_saved)/min(open kinetic energy, nearest same-J spacing)<0.1`.
The worst ratio is0.30910 for a state only0.00505082eV above threshold.
These failures are retained; the direct branch fields and inverse-capture
coefficients are diagnostics, not admitted complete DR/source data.

The4250 possible missing E1 pairs lead into577 previously unobserved even
resonances, all above threshold. The declared observable therefore expands
to **all1471 resonances** between the actual threshold and292.93105eV, with
all6288 strictly lower candidate levels and every allowed J/parity E1 block. Both
4000- and8000-node Auger calculations for all995 new even parents are complete;
full signed CI expansions remain unchanged. No weak amplitude, configuration
or rate is discarded. The completed radiative graph and explicit resonance admission are
required before using this inventory for kinetics.

The complete retained Sn13+ all-J/both-parity omitted-threshold
audit now passes: all27 blocks are checked, the lowest omitted threshold is
**324.10094562eV**, above the maximum saved radiating parent energy
**292.93105336eV**. The audit used the second complete saved root for the two
included ground-partner blocks and the lowest full-CSF eigenvalue of every
other block; no higher eigenvalue is needed to establish closure. The first
attempt's NumPy-bool JSON serialization failure is retained, with no physical
gate changed. For the original476 parents,4000->8000 shooting refinement now
passes: the maximum nonzero partial-width change is5.960e-5 relative. This
does not override the3 isolation failures; the94 radiative gaps are measured
by the completed E1 graph below. Complete window E1
now retains **1727701 nonzero lines** from1471 upper resonances to **6288
strictly lower candidate levels**, without a weak-line cutoff. Every
overlapping original line passes the predeclared1e-8 relative/1e-5s^-1
absolute replay gate. Observed wall/RSS are934.05s/4595MiB on shared hardware,
not a controlled speedup. All995 even-parent rates now pass4000->8000-node refinement with a maximum
nonzero partial-width change2.906e-5 relative. There are2942 candidate and
2864 open Auger channels across all1471 parents. Even after retained
channel closure, configurations beyond n4 fold6, narrow-resonance validity
and multi-charge collision/population/transport
must be checked independently. No new EUV peak or absolute CE acceptance is
inferred from these rates.

## Exact highest-component Dyson reconstruction for PI and RR

`ExplicitCIPhotoionization(..., dipole_mode='reduced-dyson')` reconstructs
all magnetic removal amplitudes from each pure-J highest component:
`<Jf Mf|a_(n,k,m)|Ji Mi> = D_(n,k) CG(Jf Mf,j_k m|Ji Mi)`.
The measured signed highest-component matrix element determines D, including
fermion and radial phases. The subsequent q=0 continuum E1 trace retains
every Mi/Mf/electron magnetic partner and divides by the true initial g.
There is no additional statistical or1/3 factor, radial incoherent sum or
weak-amplitude deletion. Exact angular selection rules can give zero channels.

The default complete-M route and source identity remain unchanged. The
explicit reduced mode permits `magnetic_mode='highest-all'`; the complete
mode now rejects incomplete initial **or final** multiplets for unpolarized
PI/RR. Highest-only snapshots remain separately usable for scalar Auger.
The source hash distinguishes the explicit reduced dipole mode. Inputs must
still be pure-J signed native CI states in a common orbital/nuclear binding;
a highest determinant vector alone cannot certify J.

Four new controls compare every signed E1 tensor element to independent
complete-M determinant annihilation in mixed radial/configuration states and
closed-core d2->d1, including coherent darkness, final-state phase reversal,
forbidden triangles and invalid magnetic support. Native continuum PI and
Milne rate/photon moments agree within1e-13 relative in the independent
helium control. The initial33-test affected PI/Auger/acyclic/architecture panel passes
with an isolated Numba cache. An earlier unrelated shooting cache-unpickle
failure and the first test fixture's list/string configuration error are
retained; no physics tolerance was loosened.

The optional `dyson_engine='compiled'` now applies the identical ordered
signed finite bitset action and exact integer magnetic-support forms for at
most63 magnetic spinors. It retains every coefficient and fermion phase;
Python remains default and supports arbitrary integer masks. Explicit numerical
payload/work caps reject rather than remove a coordinate. The physical source
identity is unchanged by this algorithm choice; measurement source hashes and
`plan['dyson_engine']` distinguish the implementation. Six sorted-list/high-bit/
tiny-amplitude/integer-form controls pass, and the affected45-test panel passes.
A same-eight-channel native pilot gives **bitwise identical** RR and photon
moments to Python, with observed wall32.83->22.26s on shared hardware.
The slow incomplete full-inventory run and its completed two symmetry blocks
are preserved with an interruption record; the same complete4818-root inventory
restarts with the qualified compiled engine. No timing or scientific gate is
relaxed. The later combined operator/density/gradient/resume/cache/architecture
panel passes **233 tests in11.91s**.

The actual full-n4 RR pilot retains four complete Sn12+ bound J=0 even roots
and both full3733/4810-CSF correlated Sn13+ ground partners, in the **same**
archived orbitals and bareDC energies as the Auger/cascade inventory.
Construction and eight candidate partial channels complete in32.83s with
1238MiB observed RSS. This qualifies that pilot's native entry path, not
complete RR coverage or performance scaling. The complete4818-bound-root inventory has completed with9636 candidate
channels, including1606 exact angular zeros, using the compiled removal
engine,16-node Laguerre integration and4000-node shooting. Observed full-run
wall/RSS are1101.71s/4959MiB. All2484 rate/photon moment values from the414
channels in the earlier two completed Python symmetry blocks replay **bitwise**.
At32eV the lower/upper initial Sn13+ partners have partial RR coefficients
2.77711924e-12/2.77489962e-12cm3/s and direct RR photon moments
6.64689934e-10/6.64517747e-10eV cm3/s on the first grid. Both complete
refinements have now finished. Comparing every one of57816 rate/photon
moments, including48180 nonzero values without weak screening, gives maximum
relative changes1.9297298e-4 for16->32 Laguerre nodes and2.3782598e-6 for
4000->8000 shooting nodes; both unchanged1% gates pass. The32/8000 lower/upper
partial coefficients at32eV are2.77684930e-12/2.77462936e-12cm3/s and direct
photon moments6.64615061e-10/6.64442761e-10eV cm3/s. Numerical convergence
in this retained channel inventory does not certify full ionic RR or an
experimental rate. See `bound_rr_complete_refinement_audit.json`.
These data must never be replaced by the separate earlier complete4d model.
Excited-ion channels and exchange/interchannel, collision and experimental
gates remain independent.

## Explicit coherent CI direct electron-impact ionization

`scattering.electron_ionization.ci.ExplicitCIElectronIonization` consumes the
same explicit adjacent-CI snapshot as PI/Auger. It uses the actual two-state
CI energy difference, complete signed reduced Dyson radial spinors and all
supplied bound-state Pauli projectors. The factored shared DW driver accepts
immutable `EIIDWBoundChannel` inputs directly; it creates no fictitious SCF
result and does not substitute a Koopmans binding energy. The existing SCF
wrapper keeps its selection, field and default approximations.

For an unpolarized pure-J initial state with a complete final-ion magnetic
trace, the removal density is diagonal in kappa/m. Each same-kappa radial
block is D_n D_n'/(2j+1). The native radial input is the **unnormalized signed**
Dyson sum, with occupation one in the existing single-electron DW prefactor;
there is no further initial/final g or spectroscopic multiplier. Independent
full signed Fock/magnetic sums with both d_j channels and multiple interfering
radials check this identity. The interaction is instantaneous Coulomb with
complete direct-minus-exchange amplitudes. Incoming and both outgoing waves
use one selected final-ion direct Hartree field. Nonlocal distortion exchange,
interchannel coupling, postcollision effects, transverse exchange and indirect
excitation-autoionization remain absent.

Three independent/native CI EII checks pass. An actual native He replay gives
bitwise old/factored SCF cross sections, all orbital contributions and sharing
integrands; no numerical gate is relaxed. The combined affected EII/CI/operator
panel passes124 checks with2 retained strict expected failures: high-energy
FAC EII and the excitation DW/Born limit. These known gaps are not repaired
by the new CI interface.

The first real Sn pilot keeps the original full n4fold6 ground Sn12+ J=2 and
both balanced full Sn13+ fine partners, with the original bareDC orbitals and
thresholds. At incident260/300/500eV, order6->12, shooting4000->8000 and
box80->160 refinements change partial cross sections by at most2.35626e-4,
1.62326e-6 and5.08524e-6 relative; their1% gates pass. The L2->3 partial-wave
change reaches0.60632 and **fails**. Further L4/6/8 values still do not converge. The laterL14->16 maximum
change3.39985e-3 andL16->18 change7.78535e-4 pass the original1% gate on
all retained nonzero pilot cross sections. At L16 the additional order12->24,
combined shooting8000/box160->16000/320 and radial subdivision1->2 changes
are2.74564e-5,9.52680e-7 and1.74997e-4; all1% gates pass. Thus this three-
energy, one-ground-parent/two-ground-daughter diagnostic passes its declared
numerical controls; it is not full ionic EII or an experimental rate.

The separate `plasma.rates.maxwellian_rate_direct` evaluates a callable on
all threshold-shifted Gauss-Laguerre nodes with physical CODATA units, without
an interpolated table or fitted tail. `ExplicitCIElectronIonization.rate_coefficient`
records every sampled energy/cross section and the actual CI threshold.
Independent analytic constant-flux, energy-dependent adaptive quadrature and
native CI entry controls pass11 focused checks; the affected thermal/EII
panel passes43 with one retained high-energy expected failure. Actual32eV
Sn order8/16 andL18/24 partial-integral controls are complete. The32eV
L24/order16 rates to the two ground-ion fine partners are
1.68613664001e-13 and1.10256109254e-14cm3/s. Order8->16 changes are
1.253e-5/9.967e-6 relative; L18->24 changes are6.709e-6/6.775e-6.
Both thermal1% gates pass, while individual high-energy nodes still change
by up to5.737%/5.750%. Thermal-observable convergence therefore does not
certify every sampled cross section at1%, a complete ionic source rate or CE.
No geometric tail, weak-channel cutoff or normalization fit is used.
The full L24 thermal job reaches its declared1800s limit after saving the
complete lower-ion16-node result; its missing upper-ion result is not filled
in. The first J5/2 recovery actually uses40 threads despite requested
OMP/BLAS settings; it is stopped and its evidence retained. Explicit
NUMBA_NUM_THREADS=1 repeats complete both16-node channels with closed source
guards. The lower-channel energies, cross sections and rate replay bitwise.
Observed runtimes960.36s/941.98s on shared hardware are not controlled timing.
The final audit is `Sn12_CI_EII_thermal_integral_final_audit.json`; the original
budget failure and thread-policy correction remain separate evidence.

A separate original Sn12+ J=6 odd root0 metastable has partial direct
thresholds190.171502/191.871230eV to the two ground-ion partners. In this
retained model, the lower-ion removal channel is exactly forbidden; the
upper-ion channel has only coherent f7/2 removal. At200/260/300eV its L18
cross sections are1.71804e-22/8.50918e-22/1.02410e-21cm2. L16->18,
sharing12->24 and combined shooting8000/box160->16000/320 maximum changes
are4.45074e-4,7.94155e-5 and4.52759e-6 relative; each passes1%.
A lower threshold alone therefore does not establish a strong allowed
ground-ion process. Excited residual ions, EA and populations remain absent.
Evidence: `explicit_CI_EII_development`, `Sn12_CI_EII_*` and
`Sn12_CI_EII_initial_refinement_audit.json` under the active evidence root.

## Conditional capture and complete-window radiative cascades

The completed window adds1727701 nonzero E1 lines without weak screening.
All191269 overlapping original lines replay within5.738e-10 relative;
the predeclared1e-8/1e-5s^-1 gate passes. Previously missing endpoints change
the original476 parents' total E1 widths by at most1.286e-10 relative.
Thus those omitted links are numerically negligible for those total widths;
they cannot explain the optical or EUV experimental discrepancy.

Combining all lower bound-state E1 lines produces **3165162 radiative lines**
over6289 Sn12+ levels, plus the two correlated Sn13+ fine partners.
The complete bound E1-only graph has20 terminals, including the ground
state and19 excited levels. The archived `exact_E1_dark_bound_states` list
names only two previously unlisted J=11 odd endpoints; it is not the complete
terminal inventory. Flow/energy accounting already includes all20. The
original E1-only ledger keeps their internal energy explicitly. The lower-level `numerics.acyclic_rates.integrated_acyclic_rates`
computes unnormalized positive residence, edge counts and terminal yield in
a topological graph, retaining every supplied rate. It rejects cycles/upward
order violations and resource overflow. Three independent augmented-matrix
exponential, branching/energy and rejection controls pass; this mathematical
contract does not provide missing physical collisions.

At32eV, a single unpolarized capture from the lower Sn13+ fine partner gives
unbranched capture5.59190934e-10cm3/s and conditional narrow-model stable
recombination1.53708742e-10cm3/s. From the upper partner the values are
5.16763432e-10 and1.29531408e-10cm3/s. Conditional13.365--13.635nm photon-energy
moments are5.64371220e-9 and4.90059334e-9eV cm3/s, respectively. All six20/24/32eV
rows close particle and total photon/electron/terminal-energy ledgers to
3.350e-15 relative. These are collision-free single-capture observables,
not full ionic DR coefficients, laser-prepared populations or CE.

The same3 parents fail the necessary operational isolation check with the
completed radiative widths. The fraction of conditional capture flow reaching
the first such node is1.312%/1.257% at32eV (lower/upper initial ion), and up
to1.465% over the six temperature/parent cases. A separate first-unadmitted-node
graph records this flow without deleting it or renormalizing the admitted
remainder. It is an accounting diagnostic, **not a rigorous Fano error bound**.
No experimental spectrum or conversion efficiency is accepted from this graph.
Evidence: `conditional_cascade/result.json`, `radiative_graph.npz`,
`complete_window_E1/result.json`, `even_rates_refinement.json` in the active
local evidence directory above.

The separate `E1_terminal_M1_E2_core` calculation adds317 native finite-
frequency M1/E2 lines from these20 original terminals to every permitted
lower endpoint. Signed radial-pair tensor reconstruction agrees with the
complete original operator at every static control; only the algebraically
zero common filled-J=0-core rank>0 operator is omitted in the finite-frequency
composition. All8 completed old finite-frequency amplitudes/rates replay
bitwise. This is exact-core factorization, with no weak amplitude cutoff.
The19 decaying terminals have conditional lifetimes from1.34127e-4s to2.65549s,
far longer than15ns. Eventual radiative yield cannot be substituted for
pulse-window emission. Other-state M1/E2, two-photon decay, collisions and
experimental validation are still absent; two known strict He-like magnetic
cancellation/public-consumer expected failures remain in the45-pass operator
panel. See `terminal_M1_E2_completed_overlap_audit.json`.

### Finite capture time window

`numerics.acyclic_rates.finite_acyclic_rates` consumes the same complete
downward graph and a caller-declared increasing time grid. A positive
backward-Euler triangular solve accumulates all residence integrals and edge
counts, without fast-state elimination, clipping or renormalization. Rates
and time units must be reciprocal. The original eventual-flow API is unchanged;
both share graph/resource admission. Temporal accuracy requires independent
refinement; conservation and stiff stability are not error certificates.
Fourteen finite/eventual controls include independent dense matrix exponentials,
all branching/repeated-edge rewards, exact two-state survival, large stiffness,
arbitrary labels/tiny weights and complete rejection gates. The affected
source/kinetic/architecture panel passes67 checks with a private Numba cache;
the first shared-cache ReferenceError and mistyped test-path failure remain.

The complete supplied6291-state graph combines3165162 E1 lines,317 terminal
M1/E2 lines and2864 open Auger channels, in the same original bare-DC snapshot.
For instantaneous single capture at t=0 from the lower ion at32eV, its15ns
128/256/512-step geometric-grid controls pass the declared1% observable gates.
The last population L1/capture change is3.17953e-4; all-edge count L1/capture
changes1.58985e-4, and photon-energy changes1.58926e-4. These global norms do
not certify every tiny state's relative accuracy. Energy/particle ledgers
close within4.271e-13/4.962e-13 relative. The finest capture-weighted photon
energy is3.86746625e-8eV cm3/s versus4.08845686e-8 for infinite waiting on
this same graph, about5.41% less. The band coefficient5.64371220e-9eV cm3/s
is almost unchanged, while retained internal energy stays explicit.
This is a time-windowed conditional capture calculation, not a laser pulse
population solution or CE. Failed resonance isolation, missing collisions,
other multipoles, continuum coupling and transport still limit the physics.
Evidence: `finite_time_cascade_protocol.json`, `conditional_capture_15ns` and
`finite_time_population_private_cache_panel.log` under the active evidence root.

## Full CI orbital stationarity and reference-field dependence

An independent closed36-electron Sn14+ reference, keeping the same120Sn Fermi,
ns90/ks9/R60 and full8543-CSF Sn13+ DCB space, gives **12971.539870cm^-1**,
**207.460130cm^-1 low** relative to13179(8)cm^-1. Its full-H residual is below
2.729e-12Ha and independent3-CSF H control below1.365e-11Ha. This crosses
the experimental reference and **fails**; choosing the orbital target by
which gap looks closest would be a fit. This is remaining orbital/correlation
dependence, not a transferable additive correction or an uncertainty bound.

The next objective is a fixed equal-weight average of both ground fine
levels, retaining the same complete8543-CSF expansion and varying every
core/active radial component. The new
[blocked orbital functional](ATOMIC_STATE_DESIGN.md#explicit-complete-block-orbital-objective-15-october-2026)
has independent finite signed-density and actual optimizer controls. At the
actual Sn point, the native ground average replays exactly, RDM reconstruction
differs from the full-H energy by8.186e-12Ha, and constrained radial finite
differences agree with the analytic derivative within2.740e-10 and4.822e-9Ha,
below the predeclared1e-6Ha gate. Observed146.64s/3051MiB qualifies the
gradient at that point, **not stationarity**. Initial Numba optional-index
typing failures and the Python3.10 probe syntax failure remain in the logs;
no scientific tolerance was changed.

Common-orbital DC optimization uses the complete fixed nuclear
positive-energy subspace, with orbital/energy gates1e-7 and1e-10Ha. Accepted
checkpoints preserve the original projector/residual metric for checked
resume; a budget return remains unconverged. The first1804s budget saved
iteration32 with residual0.00381935, well above1e-7. A continuation imports
the exact original geometry/inverse metric and increases only the secant
history from10 to64; the physical target and both gates are unchanged.
Both DCB levels must be
recomputed after actual stationarity. Projector, radial/virtual/core
correlation, QED and experimental gates remain distinct. No experimental
energy, fitted radial parameter or scalar paper correction enters this
objective, and alternative orbitals are not spliced into original DC rates.

## Fixed-space native Sn13+ FSCC core-valence controls

An isolated fresh EXP-T backend is built from the pinned upstream with the
source-qualified noncanonical Fov*T2 attachment repair, exact axial option
and compact indices. The executable hash is
`017dd306f791714d6afc3d1e05f50821da2aabcea5ae7476b9a7f85844d10913`.
The original installed backend lacks this repair and is not substituted.
The current public-source panel passes53 checks with one axial check initially
unenabled; explicit axial replay passes6 checks including that former skip,
covering54 distinct checks. These include complex/physical-ket HEFF, signed
full equations, phase and all supported sectors, not an experimental benchmark.

The native control retains the exact archived closed36-electron Sn14+ DC
spinors,120Sn Fermi/ns90/ks9/R60, complete4d model multiplets and all4d/4f
virtual multiplets. Static DCB enters frozen, reference and active pieces
consistently; no QED or scalar paper shift is applied. `core18` correlates
3d10+4s2+4p6 over frozen18; `core8` correlates4s2+4p6 over frozen28. Both
levels come from the same joint complete physical-ket model matrix in each
calculation.
These named frozen-core spaces and other orbital targets are not additive
corrections to the original DC rates or to one another.

| Fixed model | Lower total energy (Ha) | Upper total energy (Ha) | Splitting (cm^-1) | Difference from13179(8)cm^-1 |
|---|---:|---:|---:|---:|
| core18 CCSD | -6115.257628245319 | -6115.198559156414 | 12964.166512 | -214.833488 |
| core8 CCSD | -6115.223128646376 | -6115.164047218886 | 12966.874519 | -212.125481 |
| core8 CCSDT | -6115.226681716101 | -6115.167620138284 | 12962.518019 | -216.481981 |

The source-qualified C1/axial core8 CCSDT comparison gives complete
physical matrix agreement7.106e-15Ha and bitwise equal core/lower/upper
energies. Observed wall times are141.29s for C1 and533.76s for axial, on
shared hardware; this is not a controlled speedup. Python-process high-water
RSS542/543MiB is not backend/process-tree memory. The full core18 CCSDT
preflight64,379,676,848bytes originally exceeds8GiB and is preserved. After
the complete representation control, its explicit backend cap increases to
60GiB (68GiB outer address space,72GiB available-memory admission), retaining
all original orbitals and equations. The completed full-core result is:

| Additional fixed model | Lower total energy (Ha) | Upper total energy (Ha) | Splitting (cm^-1) | Difference from13179(8)cm^-1 |
|---|---:|---:|---:|---:|
| core18 CCSDT, n4 virtuals | -6115.261937220506 | -6115.202882421198 | 12961.030309 | -217.969691 |

The full-core raw-equation bound is1.470e-9Ha; full multiplet/cross-J gates
pass. The same-core18 CCSD->CCSDT shift is-3.136204cm^-1. Core8 gives
-4.356500cm^-1. Both endpoint levels are recalculated; neither increment
closes the roughly200cm^-1 small-space experimental gap.

The nested next-shell control preserves every archived occupied/n4 coefficient
bitwise and adds all50 n5 spinors in the exact same closed36 reference field.
A complete native Fock projection agrees within1.137e-13Ha; constrained
retained-Galerkin eigen residuals are below6.274e-14Ha and full same-kappa
Gram defect2.221e-15. Initial padded-boundary and unrefined-root failures
remain preserved. Native boundary elimination and existing bordered-Newton
refinement correct those representations without changing the field or gate.
The full74-virtual core18 CCSD result is13076.402401cm^-1,102.597599cm^-1
low; both complete multiplets and raw-equation bound1.100e-9Ha pass. Its
+112.235889cm^-1 shift from n4 is a space-dependence diagnostic, not an error
bar or fitted correction. The full n5 C1 CCSDT estimate1.885TB remains
unadmitted. The later exact axial support count gives90,105,843,504bytes
for the identical18core/74virtual/full4d model. This permits a separate
resource-admission decision, not a completed triples calculation or a peak
memory guarantee. The corresponding n6 and complete n7 estimates remain
638.038GB and2.735TB; no physical space is silently reduced.
The completed nested n6 CCSD model keeps146 virtual spinors with separately
declared24GiB backend/32GiB outer caps. It gives13098.077452cm^-1,
80.922548cm^-1 low, a further+21.675050cm^-1 space shift. Its lower/upper
totals are-6115.360021942755/-6115.300342710717Ha.
The full n7 estimate138.195GB is not admitted. A separately named n7-lmax3
radial diagnostic retains every n4/n5/n6 orbital and adds only the explicitly
declared32 n7 s/p/d/f spinors (178 total virtuals,48GiB backend/64GiB outer).
The omitted n7 g/h/i space is not assigned zero or represented as complete n7.
The first full integral build stops at the independent2GiB export cap before
launching a backend. The complete file needs2,942,079,196bytes. A source-bound
repeat explicitly admits4GiB export, retains the default2GiB for other users,
and persists a full integral checkpoint before its backend solve; construction,
export and amplitude budgets are separate. The failed build/log is retained.
This named178-virtual model finishes at13106.729287cm^-1 without QED,
72.270713cm^-1 low. Its raw bound1.577e-9Ha and complete multiplets pass;
its+8.651835cm^-1 change from n6 is not an error bound.
Every completed experimental8cm^-1 gate still fails. No large-basis/triples
composite, paper Lamb/Breit scalar or new orbital result is spliced into rates.

The first two completed CCSD runs hit an outer mappingproxy JSON-reporting
error after all numerical gates passed; their immutable backend results and
physical matrices are finalized without rerunning the solvers. The original
failure logs and reporting-only fix receipt are retained. Evidence:
`Sn13_FSCC_preflight`, `Sn13_FSCC_core18_CCSD`, `Sn13_FSCC_core8_CCSD`,
`Sn13_FSCC_core8_CCSDT` and backend manifests in the active evidence directory.

## Native model-QED operator qualification

The separate local `QEDMOD_native_profile*` preparation builds an isolated
copy of QEDMOD's full local/nonlocal model potential, matching the native
empirical120Sn Fermi diffuseness/half-density radius and CODATA c. The original
reference source is untouched. Independent Fortran/native nuclear potentials
agree within1.422e-15 in maxabs(rV)/Z. The model provides s/p/d self-energy
and all-kappa Uehling/Wichmann-Kroll vacuum polarization; it is not a complete
radiative many-body or nuclear-uncertainty calculation. The existing public
`shabaev_local` correction remains a different local approximation.

A naive component mapping gives unphysical core self-energy despite numerical
matrix agreement and is rejected. Five independent same-nucleus hydrogenic
Dirac states, covering both signs of kappa, determine the single convention
Q_Fortran=-Q_Atomx. Whole-spinor relative L2 differences after that fixed unitary
are9.486e-8..8.108e-7; the unmapped differences are0.122..0.371. This sign is
not selected from a desired Lamb shift. Original Fortran logarithmic origin
extrapolation also returns NaN for some signed native radial/action products.
The independent action control therefore uses complete signed Simpson
integration on its declared model grid, without the unsupported logarithm
or an added native-wave normalization. All failures remain preserved.

With the component mapping fixed, every native same-kappa one-body matrix
entry, including offdiagonal core/virtual entries, agrees with independent
Fortran local/nonlocal action within the unchanged1e-8Ha gate; measured2048-
grid maxima are7.148e-11Ha(SE),8.851e-12Ha(Uehling),6.737e-16Ha(WK).
Doubling the model grid changes entries by at most4.492e-10Ha and changes the
single-particle4d model increment by1.093e-7cm^-1, below the predeclared0.1cm^-1
gate. The separate4096-point model-box3->6a0 control changes that increment
by7.603e-9cm^-1 and passes the same gate; complete matrix entry changes are
at most3.759e-10Ha. The computed26.132511cm^-1
orbital increment is a diagnostic, **not** a correction pasted into a spectrum.

`physics.csf.one_body_composition.add_bound_one_body` consumes an explicit
point-bound `CIEffectiveOneBody` matrix over every retained occupied/virtual/
frozen radial key. It updates h, Fock, frozen energy and reference energy,
leaving every two-body integral unchanged. Complete original radial/core/
nuclear binding, scalar coverage, resource and duplicate-inclusion guards
reject mismatches. Correlated-core self-energies are not re-added through
this all-electron path. Packed and native on-demand storage are explicit;
the on-demand composition retains its complete signed two-body evaluator
without a quartic copy. Export/backend storage is admitted separately.
No operator generator,
orbital relaxation or automatic correction default is introduced.
Two signed-phase full-CI controls compare with the independent native one-
body CI operator, including frozen-core and cross-radial entries, at1e-12Ha.
The affected finite panel passes41 with5 optional backend skips. With the
source-qualified corrected backend,56 live checks pass and the historical
unrepaired-backend strict-FCI xfail marker produces XPASS; running that one
case normally with `--runxfail` passes its unchanged full-FCI target. These
are57 distinct executed checks, not an all-green default-marker invocation.

The complete core18/n6 CCSD calculation with this native model-QED operator
finishes from an immutable419-file runtime source snapshot. It reruns all
amplitudes and both complete model multiplets, including the frozen ledger.
Its splitting is13125.806220cm^-1: the full-solver QED change is+27.728768cm^-1,
and the difference from13179(8)cm^-1 remains-53.193780cm^-1, **failing** the
unchanged8cm^-1 experimental gate. Lower/upper totals are
-6112.478251197551/-6112.418445623949Ha; the raw equation bound is1.429e-9Ha,
multiplet defects<=8.882e-15Ha and cross-J entries<=2.995e-15Ha.
No scalar paper Lamb/Breit increment is added. This remains a named fixed-space
model with unresolved radial/core/radiative and experimental accuracy.
The separately named n7-lmax3 preparation qualifies complete QED matrices
over all43 radial keys at2048/4096 model points; every old n6 matrix block
replays bitwise. Newly added orbitals never receive placeholder zero QED.
Evidence: `QEDMOD_component_convention`, `QEDMOD_native_matrix_qualification_v4`,
`QEDMOD_native_matrix_grid4096`, `QEDMOD_complete_native_matrix_grid_audit.json`,
`QEDMOD_model_domain_audit.json`, `QEDMOD_n7_lmax3_complete_matrix_audit.json`
and `Sn13_FSCC_core18_CCSD_C1_n6_native_QED` under the active evidence root.

The complete named n7-lmax3 QED solve using independently qualified512-irrep
axial blocks gives13134.550920cm^-1,44.449080cm^-1 below experiment: the
unchanged8cm^-1 gate **fails**. Its same-space QED change is27.821633cm^-1;
the n6-QED to n7-lmax3-QED basis change is8.744700cm^-1. Every full model
partner and raw-equation gate passes. No orbital diagnostic is pasted in.

All36 closed-reference electrons can also be correlated in the unchanged n6
virtual/QED model. The first axial run fails its derived raw-equation bound
at5.068e-8Ha versus the required1e-8Ha. Tightening amplitude iteration from
1e-11 to1e-13, while retaining the original raw/experimental gates, gives
13135.464848cm^-1 and raw bound5.760e-10Ha. Its actual core18->36 shift is
9.658629cm^-1; the experimental difference remains-43.535152cm^-1, **failed**.
This core increment is not added to the separate n7-lmax3 basis increment.
The subsequent actual joint all36-core/full-n7 solve gives
13143.815225cm^-1,35.184775cm^-1 below experiment, still **failing**8cm^-1.
Lower/upper totals are-6112.516075163851/-6112.456187535184Ha; the raw
equation bound is3.940e-10Ha, with complete multiplet/cross-J gates passed.
The core shift in full n7 is10.263658cm^-1; it differs from the n6 core
shift by0.605029cm^-1. This directly measures nonadditivity in these named
spaces. Its numerical proximity to the paper's rounded13144cm^-1 prediction
does not establish matching Gaussian/nuclear/virtual inputs or experimental
accuracy. The input-identity serialization failure before the first backend
launch is preserved; the repeat copies the immutable outer identity to a
plain dictionary without altering the operator.
The same-H C1 execution reaches its2400s timeout. Its failed state is
retained and supplies neither a spectrum nor a controlled speedup ratio.

The exact axial resource model and512-irrep backend are independently
[qualified](ATOMIC_FSCC.md) with all-coordinate counts, mixed-parity/raw-H
reconstruction, full high-j CCSD/CCSDT comparisons and independent FCI.
The default256 table and C1 estimates remain available. Complete full n7
s..i preparation retains every old coefficient, with native constrained root
residuals<3.588e-14Ha and Gram defect2.221e-15. Its49-key QED matrices pass
independent/grid controls and preserve all43 old blocks bitwise. The counted
CCSD working payload is16,242,283,264bytes versus the old C1 estimate
138,195,013,312bytes; full packed ERIs alone still need9,399,880,896bytes.
The first four-worker construction reaches its4GiB worker address-space
cap after155 completed slabs. A separately recorded continuation preserves
the original files, copies only qualified completed slabs, and recomputes
every unrecorded slab with explicit cache2048 and worker recycling after
eight slabs. All261 slabs and23,796,656 allowed queries then complete.
Full-file Fock/reference and duplicate-Hermiticity gates pass; the complete
ERI SHA256 is recorded in the prepared operator manifest. This is exact
operator construction, not a peak-memory or performance guarantee.

The subsequent complete n7/core18 QED solve gives13133.551567cm^-1,
45.448433cm^-1 low, **failing** the unchanged8cm^-1 experimental gate.
Lower/upper totals are-6112.493267693181/-6112.4334268291805Ha and the
raw equation bound is1.679e-9Ha. Both full multiplets and cross-J gates pass.
Adding the previously absent n7g/h/i orbitals changes the splitting by
-0.999353cm^-1 relative to the named lmax3 model. Closeness to experiment
does not justify selecting the truncated pool. The joint all36-core/full-n7
operator and two-state solve are completed separately as recorded above;
separate core and basis shifts are not added.
Evidence: `Sn13_complete_n7_native_QED_integrals_resume2048`,
`full_n7_memory_failure_and_cache_continuation.json`,
`Sn13_FSCC_core18_CCSD_axial512_full_n7_native_QED` and
`exact_axial_SDT_n5_n6_fulln7_resource_audit.json`.

Actual completed full-n7 p56 diagnostics show transverse potential-cache
recomputation, motivating an explicit existing array-cache128->2048 control.
All183063 allowed queries and69,257,200 ABI bytes for that complete slab
replay **bitwise**, as do every full Fock/h/frozen/reference entry. Observed
query time is27.45s versus45.29s in the earlier worker history, on shared
hardware with different cache histories; this is not a controlled speedup.
It qualifies cache-capacity correctness for later bounded construction, not
an experimental accuracy improvement or a change to any public default.

## Finite second-order core-hole operator controls

A separate research construction retains the complete zero/one/two/three-body
operators induced by Hermitian second-order P/Q elimination. Outgoing vertices
are grouped by their exact core-hole/outside-particle occupations. Only equal
groups can return to P; template Wick contractions preserve every signed
coefficient, exchange term and complex interference. The induced three-body
operator is **not** the RCCSDT cluster T3. Neither the quarantined legacy
screening bubble nor a fitted paper centroid shift is used.

Independent complete finite-Fock controls with0..4 active electrons reconstruct
every coefficient and the full four-active matrix; native complex-phase DC+B
Be controls agree within5.205e-18Ha. The first coarse finite-coupling curvature
control fails; three smaller steps pass the unchanged derivative gate.
These are finite mathematical/physical-operator checks, not Sn convergence.

The actual archived Sn12+ n4 basis then supplies35489 nonzero outgoing words,
465 core-hole groups and18,716,319 ordered products. This explicitly partial
model has30 closed-core and30 active spinors, with **no outside-n4 virtuals**.
H0 is declared as the diagonal Fock matrix of the fixed isotropic Kr4d2
ensemble, not asserted to equal the paper's canonical DHF reference. All
noncanonical/subtraction one-body vertices remain. Complete coefficients fit
a276,088,000byte array envelope; finite/Hermitian, exact M/parity and full
rank-sector J+ commutator gates pass. Measured commutator defects are below
4.219e-15Ha. The3.02s/1371.6MiB observation on shared hardware is not a
controlled performance ratio or the cost of a full CI spectrum.

A subsequent p6d2-only projection recalculates all nine levels/45 magnetic
partners. Its bare spectrum matches independent native CSF CI within
6.367e-12Ha. The unique J=3 excitation is9781.043cm^-1 in bare DC,
9416.780 with Sigma1,10089.589 with Sigma1+2 and9837.834 with Sigma1+2+3.
The induced three-body change is-251.755cm^-1; the last model still differs
from the experimental9786(30)cm^-1 by+51.834cm^-1 and **fails** that level's
gate. The bare coincidence is not used to select an incomplete operator.
Full22-configuration CI, outside-n4 core correlation, dressed transitions and
experimental EUV/CE remain pending. No partial energy correction is spliced
into the existing DC rates. Evidence under the active root:
`core_MBPT_connected_rank_finite_control_v2`, `core_MBPT_grouped_word_control`,
`native_Be_core_MBPT_word_control_v2`, `Sn12_native_core_MBPT_n4_coefficients`
and `Sn12_native_core_MBPT_singlecfg_control`.

The bounded numerical `numerics.fermion.canonical_body_matrix` consumes
supplied canonical0..3-body coefficient matrices and explicit fixed-N masks.
Exact spectator sums use the shared signed Fock action. Eleven independent
all-word/complex/selected-space/resource controls pass. The routine owns no
physical state, MBPT policy or scalable full-CI claim.

The separately admitted `numerics.canonical_sparse.canonical_body_sparse`
retains exactly the same supplied words in a CSC determinant matrix. An
occupied-annihilator lookup, exact Pauli constraints and combinatorial row
map avoid a dense determinant matrix. Only exact-zero coefficients are absent;
no cancellation, small term or non-Hermitian direction is screened. Complete
candidate work and a first count of every reachable output coordinate precede
matrix allocation. A full combinatorial row map can itself reject the request.
Independent exhaustive complex/all-word, spectator and cross-layout tests
qualify the action; the actual Sn p6d2 complete matrices for all three operator
hierarchies reproduce the prior direct results **bitwise**. The combined
affected numerical/angular/native-CI panel passes61 tests before the additional
counted-output budget check. Evidence: `Sn12_canonical_sparse_native_control`,
`canonical_sparse_affected_tests.log` and the426-file immutable source epoch
`runtime_source_epoch_20261016_canonical_sparse`.

The original22-configuration fold6 J=2/J=3 even sectors contain9714/11363
CSFs and69098/59370 fixed-M determinants. A first preallocation guard rejects
an over-combined angular/dense scratch subbudget. The separate retry retains
all states, unchanged12GiB process and6GiB sparse matrix caps, and admits
explicit simultaneous phase bounds. Full bare-matrix comparison precedes any
corrected root. Both full selected-sector n4-only computations now finish;
no all-J CI+MBPT spectrum, corrected rates, external-virtual convergence or
experimental alignment is claimed. See the preserved
`Sn12_fullfold6_projection_admission_v1_rejection.json`.

The actual support has69091/59364 determinants: a pattern-M count is an
upper bound and can contain rows identically zero in every J-CSF column.
The first assertion equating that upper bound to CSF support fails and is
preserved; the retry retains every CSF and checks the full Gram matrix and
every bare-H entry. Complete bare J=2/J=3 matrix errors are7.913e-11 and
7.276e-11Ha. The n4-only corrected matrices have Hermitian defects below
2.499e-16Ha and all12 selected-root residuals below1.603e-12Ha.

An actual J=3 cProfile record attributes333.82 of409.14s to sparse CSC
multivector operations. The same-point matrix-free J=2 action retains all
CSFs, all canonical words and the full determinant operator; it compares all
12 root energies with the completed matrix within1.819e-12Ha and their
complete invariant subspace within3.553e-15. Input canonical Hermiticity has
the full-Fock bound `sum_r binom(N,r)*max_column_L1(Gamma_r-Gamma_r†)`;
numerical eigenpair residuals are separately checked. Observed full/matrix-free
workflow times400.46/128.21s have different shared-machine histories and are
**not** a controlled end-to-end speedup. Evidence includes
`Sn12_fullfold6_J2_6_actual_profile.pstats`,
`Sn12_fullfold6_MBPT_matrixfree_J2_4_Qn4` and
`matrixfree_full_CI_MBPT_protocol.json`.

The next physical rung includes all50 n5 outside spinors. Its exact
core-hole/outside-particle Q has291103 words,73941 groups and48858931
products. All old n4 Fcore/H0 entries and outgoing words/denominators remain
bitwise equal; complete rank0..3 operators pass Hermiticity, exact M/parity
and J+ gates. This Q includes external valence correlation as well as core
correlation. Static Breit is then added as the complete native
`DCB-minus-DC` Ecore/Fcore/two-body operator on the same archived orbitals,
once, before rediagonalizing. DC-only MBPT coefficients remain unchanged;
mixed Breit-correlation diagrams are absent. A separate native p6d2 DCB
all-nine-level control agrees with independent CSF CI within1.183e-11Ha.

Same-point full-sector J=3 excitations are:

| Model, full fold6 selected sectors | Excitation, cm^-1 | Difference from9786(30), cm^-1 |
|---|---:|---:|
| Bare DC |9939.819568|+153.819568, FAIL|
| DC plus complete induced Sigma0..3, n4 Q only |10000.990532|+214.990532, FAIL|
| DC plus Sigma0..3, all n5 outside Q |10026.885982|+240.885982, FAIL|
| Static DCB plus DC Sigma0..3, all n5 outside Q |9648.852857|-137.147143, FAIL|

The last native Breit change is-378.033125cm^-1; it is not a borrowed scalar.
The model-QED matrix is now generated afresh on the original Sn12 V38 n4
orbitals, rather than using the closed36 Sn13 basis. All16 core/p/d/f radial
keys pass independent signed Fortran comparisons and2048-to4096 grid
refinement, with the largest grid difference4.491e-10Ha. Subsequent composed
states remain a separate gate. The complete n6 outside preparation preserves
all n4/n5 orbital bytes, but the first packed byte admission and on-demand
old-word-support check fail; neither failed run supplies accepted coefficients.
The packed retry explicitly raises the integral subcap2GiB to2560MiB for
2187354624 required bytes while keeping the12GiB process cap. A mistyped
runtime path imported the live tree; all426 source/tool hashes matched at that gate
the frozen epoch, but a fresh source-guarded replay is nevertheless required.
Those failures and the still-unresolved on-demand support difference are
preserved separately. No n6 coefficient, all-J spectrum or experimental
acceptance was inferred from preparation. The final packed v4 source-guarded
replay now reproduces every old n4/n5 numerator and denominator bitwise;
its complete outgoing-word SHA also matches the quarantined live-source run.
All962565 words,337144 groups and90051679 products then yield complete
rank0..3 coefficients with finite/Hermitian, exact M/parity and J+ gates
passed. Their counted array envelope513419456bytes fits the512MiB
coefficient cap. This finite n6-Q operator is not a converged external tail.

Composing the fresh Sn12 model-QED operator first fails its16MiB additional
payload cap because the packed ERI copy is owned. The64MiB explicit retry
retains the12GiB process cap and identical physics/gates. Full-sector static
Breit/model-QED/DC-Sigma calculations give9671.231640cm^-1 for n5 Q and
9281.979114cm^-1 for n6 Q, differing from9786(30) by-114.768360 and
-504.020886cm^-1: **both fail**. Lower/upper operator identities, QED
matrices and state/basis signatures match. The n5 full-solver QED change is
22.378783cm^-1; the separate one-orbital increment25.414821 is not used
in its place. Selected roots keep dominant overlap with original bare states.

The large variation triggers a discriminator before another radial scan.
With Kr36 frozen and two valence electrons, independent complete finite DC
CI in all n4d/f plus n5 or n5/n6 orbitals is compared to explicit full-P/Q
second order on the same Hamiltonian. Complex two-electron elements first
match exhaustive fermion words. Actual J=2/J=3 dimensions are150/109 at n5
and545/426 at n6, with full rotational checks. The second-order splitting
differs from exact FCI by2.521138 and1.387885cm^-1. This explicitly frozen
diagnostic supplies no correction to the full 4p-hole calculation.

A complete fixed-state action on original n5 full-CI states resolves the
n6 increment while retaining interference. The p6d2-only gap response is
+10.788373cm^-1, the remaining-sector response-27.162251 and the full cross
term-408.299249, summing to-424.673127. Rediagonalization changes this by
+35.420601, giving the actual-389.252526cm^-1. All signed configuration
row contributions remain. This is an off-diagonal configuration response,
not an independent additive error budget. Evidence under the active root:
Sn12_frozenKr_external_MBPT_vs_FCI,
Sn12_Qn6_increment_coherent_response and
Sn12_fullfold6_staticB_QED_external_Q_comparison.json.

The research Hermitian P/Q construction is not source-matched to the paper's
AMBiT Brillouin-Wigner operator. Verified original AMBiT source uses fixed
partial-wave valence reference energies, diagram-specific denominators and
separately selected high-valence diagrams; see
[the MBPT contract](https://github.com/drjuls/AMBiT/blob/fe1115aaed97b340e1f4d625d451b165c5bcd7a5/src/MBPT/MBPTCalculator.h)
and [Sigma3 implementation](https://github.com/drjuls/AMBiT/blob/fe1115aaed97b340e1f4d625d451b165c5bcd7a5/src/MBPT/Sigma3Calculator.cpp).
The driver has optional denominator/term skipping. These defaults are not
evidence of unpublished2026 input choices and are not adopted to force a
pass. Native full-electron off-diagonal P/Q controls and a convention
comparison precede any new reproduction claim.

The independent selected-column control now retains actual physical38-electron
Slater Q determinants. For both n5/n6, every selected2x2 entry agrees with
the canonical research operator within7.858e-14Ha. The largest n6 incremental
coupling connects p6d2 and p4d2f2 magnetic determinants; its shared Q paths
include p5df6g. One path has denominators-13.492329Ha and-0.100326Ha on
the two sides, respectively. Selected high-P column first-order Q norm
squared grows from0.109081 to0.303638. These are basis-column diagnostics,
not full CI eigenstate admixtures or an error bound. Evidence:
Sn12_native38_offdiagonal_PQ_v2 and
Sn12_native38_selected_denominator_diagnosis.json.

A separate native-integral consumer compiles the original AMBiT MBPT source
without changes and uses its actual one/two-body storage canonicalization.
The closed-Kr Fock supplies subtraction integrals; the explicitly declared
isotropic V38 diagonal supplies orbital energies. All original Atomx radial
coefficients remain. Exact ordered particle-hole phases reproduce the full
native bare selected Slater matrix within9.095e-13Ha. Independent complex
normal-ordering in every finite particle-number sector agrees within
2.221e-16Ha, and all900 one/189225 two-body active coefficients pass
Hermiticity, exact M/parity and J+ gates. There are no original skipped-term
warnings. These qualify Sigma1+2 only, not the paper's full reference model.

The original on-the-fly Sigma3 raw selected directions differ by0.015799Ha
at n6. They are retained as an unqualified directional diagnostic, never
averaged into a claimed full Hermitian correction. The original Hamiltonian
consumer applies configuration order, triangular assembly and leading-
configuration selection; the [AMBiT paper](https://arxiv.org/abs/1805.11265)
also explicitly describes this leading-configuration restriction. These
consumer rules and the unpublished2026 leading/reference input require
separate treatment before a source-matched Sigma3 reproduction claim.

Full original fold6 J=2/J=3 consumption of the qualified original-source
Sigma1+2, with the same native static Breit/model-QED, gives10361.703834
and10390.559989cm^-1 at n5/n6. Both fail9786(30), by+575.703834 and
+604.559989cm^-1. The finite n6 increment is+28.856155cm^-1. Every full
bare matrix, Gram and selected-root residual gate passes, but missing Sigma3
and unsaturated external space preclude full-MBPT or experimental acceptance.
Evidence: Sn12_native_AMBiT_BW_sigma12_qualification_v2 and
Sn12_fullfold6_original_BW_Sigma12_comparison.json. A fresh native statistical
DHF versus isotropic-H0 energy/occupied-orbital audit is separate; no scalar
reference change is silently transferred into these results. Its repeated
occupied orbitals differ by at most3.499e-10 in coefficient and the native
statistical-DHF residual is9.092e-12Ha. Actual4d3/2,5/2 orbital energies
are-9.452840660/-9.393991885Ha, compared with the earlier declared isotropic
H0 values-8.658505599/-8.602221368Ha. Differences-0.794335061 and
-0.791770517Ha are reference-model effects. Other occupied differences
are below8.45e-10Ha. Original AMBiT open-shell exchange has conditional
nonrelativistic scaling, including a distinct virtual-d extension. The new
source-oracle/native AOC-field protocol tests complete old d-field projected
matrices and occupied energy identity before regenerating the complete n5/n6
d multiplets. No field correction or experimental result is inferred from
this protocol. Evidence: Sn12_native_AOC_reference_audit_v2 and
Sn12_AOC_native_field_protocol.json. That source/native occupied-energy
gate subsequently fails: original NON_REL_SCALING expectations differ from
native statistical-jj DHF by-0.001996180/+0.001310778Ha. The distinct
averaging operators remain distinct, with no imported scalar energies.

A separate, explicitly native statistical-jj reference experiment now uses
each retained occupied orbital's actual native target-Fock expectation for
its diagonal H0 energy. The native4d target Fock supplies the conditional-d
virtual extension; all n5/n6 s..h keys remain, with only the four outside
d spinors regenerated. All old n4 coefficients stay bitwise. Occupied
expectations match fresh native DHF within8.447e-10Ha and allowed stationarity
residuals are below7.63e-13Ha. Four new d roots pass constrained residuals
below3.36e-14Ha and complete Gram2.443e-15; unprojected residuals up to
0.001754Ha remain explicit because the occupied constraints are retained.
This is a declared native reference extension, not original AMBiT AOC.

The new full182-spinor native integral construction keeps all old60-spinor
core Fock/numerators bitwise and rebuilds every outgoing denominator. Its
962555 words/337147 groups/90041901 products include a minimum coupled
absolute denominator0.009079827Ha without clipping or skipping. Complete
native-reference n5/n6 canonical0..3 coefficients pass Hermiticity, exact
M/parity and J+ gates; maximum J+ defect9.344e-13Ha. Independent full38-electron
Slater P/Q reproduces every selected2x2 canonical entry within7.006e-14Ha.
The selected signed offdiagonal is0.063434664/0.063713954Ha at n5/n6;
the small denominator remains in the operator. Both full22config selected
J2/J3 calculations pass complete bare-H/Gram/root gates but give8753.338132
and8765.259249cm^-1, errors-1032.661868/-1020.740751cm^-1 against9786(30).
The n6 increment is+11.921117cm^-1. All12 roots are compared with the original
bare eigenbasis; the lower and upper states retain maximum overlap with their
original lowest roots, minimum squared overlap0.998679. Thus this discrepancy
is not explained by replacing those roots with different bare states.

A preregistered same-H induced-rank diagnostic, keeping complete0..2body
but omitting induced3body, gives12128.507865cm^-1. Full re-diagonalization
with induced3body changes the gap by-3363.248616cm^-1. Both versions fail;
the incomplete rank2 version cannot replace rank3 by proximity. These are
induced second-order Sigma3 terms, not coupled-cluster T3. Their large effect
and near-resonant denominators require further correlation/reference work;
neither reference choice proves a converged paper or experimental result.
Evidence: Sn12_AOC_source_native_mismatch.json,
Sn12_native_consistent_AOC_preparation_v2,
Sn12_native_AOC_n6_outgoing, Sn12_native_AOC_n5/n6_coefficients,
Sn12_native_AOC_native38_PQ and Sn12_native_AOC_full_comparison.json.

A separate private two-particle FSCC bridge is qualified before Sn use.
Complex five-spinor/two-core finite controls include nonzero Fov; all core,
one-attachment and two-attachment roots agree with complete FCI within
3.11e-15Ha. Full independent Bloch wave operators distinguish the physical
ket convention (raw HEFF adjoint) from transpose, conjugate and unmodified
raw matrices: complete Bloch residuals are below1.23e-15Ha. Native Be
complete2s/2p1/2 multiplets then test all six two-particle model states.
C1/axial full matrices agree within8.89e-16Ha, and independent spinor complex
phase changes obey full matrix covariance within1.34e-15Ha. N2 core and
all four N3 roots match FCI within1.43e-14Ha. The N4 CCSDT-FCI defect
1.702e-6Ha is retained as omitted connected quadruple correlation, not an
exactness gate. No non-Hermitian HEFF averaging is used.

The preregistered native Sn12 n4 control uses the exact archived closed36
DCB operator, core18/frozen18/virtual24 and every45 active4d pair. A fresh
point-bound closed-reference QED4096 composition first matches all old n4
radials bitwise; no Sn12 AOC QED matrix or scalar shift is transferred.
Private0h2p CCSD gives9408.777175cm^-1, error-377.222825 FAIL30, with
raw bound9.361e-10Ha. Shared angular projection retains all45 determinants,
all9 fine levels and every M: Gram3.58e-15, crossJ/M7.47e-14Ha and
same-J all-M block differences2.14e-14Ha. Same-H CCSDT now gives9397.227492
cm^-1, error-388.772508 FAIL30. The actual rank increment is-11.549683
cm^-1; raw bound1.476e-9Ha and complete angular gates pass. It is not
transferred to a larger basis.
Public FSCC sector admission remains unchanged and rejects0h2p; these are
source-bound research consumers, not a newly released public capability.
Evidence: Sn12_two_valence_FSCC_finite_v3, Sn12_two_particle_native_control,
Sn12_two_particle_n4_protocol.json and Sn12_two_particle_n4_CCSD/CCSDT.

The separate all36-core/fulln7 two-attachment CCSD run retains every244
virtual and all45 active pairs at the same native DCB+49-key model-QED
point. Its full one-attachment model matrix and correlated core replay the
previous Sn13 run bitwise. Raw bound3.941e-10Ha, complete angular crossJ/M
7.11e-14Ha and all-M block1.07e-14Ha pass. The optical gap9709.739955
cm^-1 differs from9786(30) by-76.260045cm^-1, still FAIL30. Wall time
144.42s is an actual workflow measurement on shared hardware, not a paired
speedup. Resource admission retains the complete23.175GB estimate,32GiB
backend/64GiB AS and16GiB explicit export cap. The cap-only private bridge
variant repeats all native C1/axial/gauge controls before use.

Ground-configuration LS assignment has its own independent angular anchor.
Pauli-allowed equivalent-d-electron CG vectors span all45 determinants;
Gram4.33e-15 and independent second-quantized totalL2/S2/J2/Jz residuals
7.11e-15 pass. Before applying them to native signed matrices, the complete
k=0,2,4 Ck components establish the native Grant phase conversion within
1.23e-15; omitting it gives a0.712697 matrix defect. The original consumer's
unconverted native labels are explicitly withdrawn and retained. No native
Hamiltonian is changed. One-to-one LS assignment uses only angular model
overlap within J, never experimental energies. These normalized right-model
characters are not full correlated-wave populations or dressed properties.
At fulln7 CCSD, the independently assigned five measured levels are:

| Level | Native cm^-1 | Experiment difference cm^-1 | D2 ground-relative gate |
|---|---:|---:|---|
| 3F3 | 9709.739955 | -76.260045 | FAIL30 |
| 3F4 | 18522.651630 | -41.348370 | PASS50 |
| 1D2 | 25082.819089 | +244.819089 | FAIL25 |
| 3P2 | 39237.312028 | +193.312028 | FAIL30 |
| 1G4 | 40650.533319 | +932.533319 | FAIL44 |

TableIII of [Windberger2016](https://arxiv.org/pdf/1605.04236) distinguishes
ground-relative D2 from dispersive D1 and unmeasured levels; neither blank
Eexp cells nor paper predictions become experimental targets. Its FSCC
protocol uses a much larger active model and an intermediate Hamiltonian,
all-electron correlation, a large universal basis and self-consistent model
QED. Our boundn7/smallP/fixed-radial Fermi model is not that exact protocol.
A new same-H control enlarges only P from4d10 to4d+4f24 (45->276 pairs)
before attempting the paper-pattern154 active spinors. The preliminary
full154 resource estimate is60.400GB, but no154 solve/IH qualification is
implied. Evidence: Sn12_two_particle_fulln7_CCSD,
Sn12_two_particle_LS_native_phase_comparison.json,
Sn12_two_particle_LS_native_phase_correction.json and
Sn12_two_particle_fulln7_active24_protocol.json. No experimental spectrum
or CE is accepted.

The same-H adjacent-charge diagnostic gives native fulln7 Sn12/Sn13 IPs
257.922966/282.917847eV. The frozen NIST standards258(3)/282(3)eV are
explicitly theoretical limits; both satisfy their loose6eV auxiliary gate.
They do not establish measured optical, EUV or CE accuracy. Both charge-state
energies use the same complete input, operator and correlated reference;
Sn12_two_particle_NIST_IP_comparison.json retains the standards SHA/origins.

The expanded P24 unshifted DIIS8 solve fails its0h1p prerequisite at the
original120 iterations (last doubles update0.00090405); its partial amplitudes
are not an energy result, despite the backend returning zero exit status. A
pure numerical DIIS64 comparison keeps that full model and strict gates,
adding two conservatively counted amplitude vectors for each extra history
before admission. The separate original-source IH-IMMS qualification uses
imaginary attenuation exponent5, scale1 and an automatic main-energy frontier,
all fixed before experiment comparison. In complex finite fullFCI controls,
allmain zero shifts replay whole matrices bitwise; nonzero buffer shifts
preserve main roots within2.443e-14Ha. Native Be C1/axial controls preserve
main energies within6.67e-16Ha; the existing N4 omitted-quadruple defect
7.812e-8Ha remains. The full shifted-update bound explicitly includes the
maximum possible imaginary shift; the unshifted6|epsilon| bound alone is
not reused for buffers. A later Sn IH run must separately pass main-state
realness, complete angular symmetry, model overlap and experiment gates;
buffer roots are auxiliary rather than measured levels. EXP-T IH-IMMS is
not asserted to reproduce unpublished TRAFS settings. The actual native
P24 imaginary-IH run subsequently converges every amplitude sector but fails
main-root realness: maximum imaginary component0.003446535Ha against1e-8Ha.
All276 angular Gram/crossJ/M gates pass and main overlaps exceed0.995, but
real parts of these shifted roots cannot replace physical optical energies;
these are not physical resonance widths. The original assertion failure
and a separate diagnostic checkpoint retain every complex root. The physical
gate remains failed. The new exact-M/parity angular representation reproduces
every entry/transform of the dense276 control within7.106e-15Ha, independently
of this physical failure; numerical equivalence supplies no IH acceptance.
Evidence:
Sn12_two_particle_diis_native_control, Sn12_IH_finite and Sn12_IH_native.

A distinct source-numerics audit finds an actual tiny-numerator issue before
any Sn correction is inferred. Original `diveps.c` leaves a numerator
unchanged when its magnitude is below1e-14, rather than dividing it by its
nonzero diagonal denominator. In an analytic one-electron2-spinor case with
D=-0.001Ha and N=8e-15Ha, the backend reports zero converged update yet
decodes t1=8e-15 instead of-8e-12; amplitude error8.008e-12 and independent
wave residual8.008e-15Ha remain. A second N=2e-14 control divides correctly.
This is a numerical floor mechanism, not evidence of a Sn optical correction.
A private candidate changes only the four numerator skips to exact complex
zero tests. It must pass the analytic, original live and native full-matrix
gates before Sn use or any builder change. Old update bounds need the
retained-numerator skip-floor term as a separate conservative audit; zero
update alone does not bound the old physical residual by zero. Evidence:
EXPT_diveps_numerator_threshold and EXPT_exact_nonzero_division_protocol.json.

The exact-nonzero candidate now passes both analytic amplitudes,63 original
live controls, complete complex finite FCI/Bloch and native Be/42-spinor
C1/axial/gauge controls. Its new executable/source identity is retained; the
largest native42 old-model/core difference is9.095e-13Ha. It is admitted to
a separate same-H/fulln7/P24/DIIS64 retry, with unchanged strict gates. The
legacy DIIS64 run separately fails at160 iterations with doubles update2e-12.
No new optical result is yet inferred from the retry. The public optional
builder flag and corrected legacy residual-floor contract are documented in
[ATOMIC_FSCC](ATOMIC_FSCC.md#numerator-division-source-contract);74 public
affected tests pass with both actual binaries. A distinct427-file immutable
Python/tool epoch records this public adapter change; all physical operator
and integral source files remain unchanged. The already running retry is
explicitly bound to the old426 Python epoch plus the new qualified external
binary. Evidence: EXPT_exact_nonzero_qualification.json,
public_exact_nonzero_targets.log and
Sn12_two_particle_fulln7_active24_exact_nonzero_protocol.json.

An independent closed36-point frozen-Kr two-valence CI control now consumes
the same native fulln7 DCB+QED bank without copying a full rank4 tensor.
All24/74/146/244 virtuals at n4/n5/n6/n7 are retained. Complete J=2,3,4
CSF dimensions equal N(M)-N(M+1); full magnetic determinant spaces, Gram,
highest-weight, invariant-J action and every eigenpair are checked. An
independent complex Slater-word anchor is exact, and a fresh native n4
operator rebuild matches every retained ERI bitwise, core ledger exactly,
and Fcore within1.422e-14Ha. Largest selected n7 M-space dimension is1411;
the J3/J4 eigenpair residuals are below3.91e-14Ha. The twelve-block workflow
takes30.65s, peak1.669GiB RSS, on shared hardware; this is not a solver-wide
speedup or a reduced FSCC memory estimate.

| Level | Frozen-Kr exact valence n7 cm^-1 | Full-core CCSD n7 cm^-1 | Actual method difference cm^-1 |
|---|---:|---:|---:|
| 3F3 | 9301.304471 | 9709.739955 | +408.435484 |
| 3F4 | 17999.178595 | 18522.651630 | +523.473035 |
| 1D2 | 26189.653411 | 25082.819089 | -1106.834322 |
| 3P2 | 40105.151376 | 39237.312028 | -867.839349 |
| 1G4 | 42199.114760 | 40650.533319 | -1548.581441 |

These are separately solved excitation differences; core coupling and the
finite CCSD/model approximation remain entangled. They are not additive
corrections to another calculation. Frozen-valence n7-minus-n6 increments
are+6.249129/+3.836962/-39.953565/-29.971128/-72.957866cm^-1 in the
table's order; no tail error bound or continuum convergence follows.
The first consumer tried an unsupported empty occupied reference; the next
used the actual occupied Kr reference and froze it only in the CI space.
Its complete packed-copy count required a32MiB composition subcap inside
the unchanged24GiB AS cap. A subsequent Numba typing failure is retained;
the final consumer compiles the unchanged shared packed accessor, passes
integer offsets as an array and checks all625 complex finite queries before
native use. Evidence: Sn12_closed36_frozenKr_valence_FCI_v4 and
Sn12_closed36_frozenKr_vs_CCSD_n7_comparison.json.

The source-controlled fulln7 P10 exact-nonzero replay now completes all
three CCSD sectors (13/11/10 iterations). All45 angular/all-M gates and
the phase-bound LS assignment pass. Maximum nine-level excitation change
from the legacy source is1.170e-8cm^-1; the five measured-level verdicts
are unchanged (only3F4 passes its fixed D2 interval). The3F3 gap remains
9709.739954498cm^-1,76.260045502 below experiment. This is an actual Sn
source control, not an inference from small finite tests.

The fulln7 P24 exact-nonzero attempt converges0h0p in13 iterations and
0h1p in99, then reaches its original1800s backend wall limit during0h2p
iteration111. No0h2p result is accepted. Five prerequisite binary diagrams
are retained unchanged. A private restart consumer verifies the original
executable, input bytes, operator, integral hashes, model, active ordering
and every copied diagram; missing-file fallback is forbidden. All sectors
must reconverge at1e-13, with the unchanged1e-8Ha raw gate. Its first finite
attempt rejects JSON-list versus in-memory-tuple labels; persisted labels
and integral hashes are equal. The explicit tuple normalization retry
passes two complex finite Bloch anchors and native C1/axial/gauge full
matrices. A fresh P24 continuation reuses only converged0h0p/0h1p diagrams,
freshly sorts integrals and initializes0h2p from scratch. The original
40GiB backend/64GiB AS and1800s backend caps remain; no total-time speedup
or physical pass follows from checkpoint reuse. Evidence:
Sn12_two_particle_exact_nonzero_P10_source_comparison.json,
Sn12_two_particle_fulln7_active24_exact_nonzero_failure.json and
Sn12_exact_nonzero_resume_control_v2/result.json.

The original source CROP3 iteration is now separately qualified with DIIS
explicitly disabled. Complete complex finite Bloch and native Be C1/axial/
gauge matrices differ by at most4.219e-15Ha from the independently qualified
DIIS source. A raw-block-only consumer also checks every core, one-particle
and two-particle source block against those references; it does not allocate
a global11781x11781 matrix. Its first probe exposes an unguarded dense-matrix
assignment after suppressing that allocation, and is retained; the explicit
conditional-write retry passes. Raw-block results declare angular/all-root
realness pending, never a physical spectrum. The CROP memory admission keeps
the original32-array estimate and additionally counts all8 history/transient
amplitude/error vectors. The actual fulln7 P24 Sn CROP3 comparison is admitted
at32GiB backend/64GiB AS/1800s, before any11781-state deployment. Numerical
small-model success alone does not establish large-Sn convergence or a timing
improvement. Evidence: Sn12_exact_nonzero_crop3_control/result.json,
Sn12_exact_nonzero_crop3_blocks_control_v2/result.json and
Sn12_two_particle_fulln7_active24_crop3_protocol.json.

The actual Sn CROP3 trial stalls in0h1p: iterations21..28 retain raw printed
T1/T2 updates0.002002286634/0.078190611667, with less than1e-8 variation
across eight steps. It is stopped as a low-information failed numerical
branch; no enlarged-model CROP deployment follows. The DIIS64 continuation
also fails0h2p at its original160 iterations despite rounded zero printed
updates. The final flushed02 diagrams pass the independently finite/native
qualified all-sector reuse path, but a single fresh three-iteration
stationarity probe still lacks02 success. No repeat, convergence relaxation
or P24 splitting is accepted. Evidence: Sn12_two_particle_fulln7_active24_
exact_nonzero_resume_failure.json, Sn12_two_particle_fulln7_active24_
final_stationarity_failure.json and Sn12_two_particle_fulln7_active24_
crop3_stagnation_failure.json (join the wrapped filename components).

A complete native-AOC H0-word audit reconstructs all962555 denominators
exactly. Of18224 radial channels, five have a magnetic component with
abs(V/Delta)>=1;174 magnetic words cross that diagnostic ratio. The largest
is4d3/2+4f7/2 <->4p3/2+5g9/2, Delta=0.009079826647Ha and
abs(V/Delta)=3.509660. Such ratios alone do not prove a particular eigenstate
is strongly affected. Complete radial multiplets in these five channels
form612 words A; all remaining10131 words B in the same18 outside-particle
groups are retained. Shared signed normal ordering constructs all induced
ranks0..3 for AA, BB, full(A+B), and cross=full-AA-BB. Nothing is deleted
from the physical calculation.

On the actual complete-fold6 AOC n6 J2/J3 lowest states, the resulting
fixed-state splitting responses are AA=-38.088132, BB=-94.462174,
cross=-1043.601668, total=-1176.151974cm^-1. Each decomposition reconstructs
its complete coherent expectation within1e-10Ha. The p6d2 weights are
0.9741924/0.9739140. Direct AA action on each pure p6d2 component vanishes;
cross diagonal expectation vanishes, while cross off-diagonal action is
allowed and materially couples the other configurations. An initial
incorrect cross-action-zero assertion was withdrawn analytically before
interpretation; the original diagnostic was stopped and retained. These
are expectation diagnostics of the failed AOC operator, not additive energy
repairs, a proof of paper-specific intruders, or a justification for dropping
Sigma3/interference. Evidence: Sn12_native_AOC_near_resonant_words and
Sn12_native_AOC_resonant_partition_v2.

The next declared physical discriminator adds all26 directly Coulomb-reachable
single5g configurations to the complete22-configuration n4 P space,
with both5g j partners and every magnetic state. This adds1896004 fine
levels; combined selected J2/J3 CSF dimensions106896/133194 would require
85.14/132.18GiB for H alone. Dense CSF storage is not admitted. A common
fresh native DCB+18-key model-QED operator is prepared and independently
qualified before exact sparse fixed-M admission. All old16-key QED and
n4 body coefficients must replay, plus physical38-electron signed P/Q
Slater entries and complex phase covariance. The first composition rejects
a64MiB packed-copy subcap; the retry counts the full payload and raises
only that subcap to128MiB inside unchanged12GiB AS. The original real-array
Slater probe emits a complex-cast warning; an explicit complex probe retains
all components for requalification. This is a common bare-CI basis
comparison, with no extra MBPT or spliced correction and no solve claimed.
Evidence: Sn12_n4_plus_single5g_physical_protocol.json and
Sn12_n4_plus_single5g_preflight.json.

The fresh common operator is now qualified: all retained n4 Fcore/Ecore/ERIs
and old16-key model-QED entries match exactly. Both potential grids pass the
original Fortran check; maximum grid-doubling entry4.491e-10Ha. An explicit
complex48-determinant physical38-electron Slater anchor differs by at most
7.276e-12Ha, complex phase covariance7.277e-12Ha. The first real-array
warning is resolved by retaining complex components in v3; coefficient bytes
are unchanged from v2. Actual native48/N8 sparse/colex indexing repeats the
physical and complex-phase anchors within8.187e-12Ha.

Complete fixed-M masks now reuse shared PatternLayout/subshell enumeration,
without a giant CSF projector. Every mask checks N=8, M and even parity,
unique global native-spinor ordering and independent NR-J multiplicities.
The n4 M=2 dimension69100 exceeds the earlier highest-J2-pattern union69098
because the complete M space also contains patterns with only higher J;
the old9714-dimensional target-J CSF space was complete. An independent
initial p6d2 seed highest-weight check passes for both targets. The existing
exact canonical sparse engine counts full candidate work/output before
allocation; no matrix symmetry averaging or nonzero screening is used.
The complete expanded lower-state M=2 solve retains1086326 determinants,
507892914 stored coordinates and6099060276 owned CSC bytes. Its build takes
47.48s, whole sector76.61s; eigenpair residual1.197e-12Ha, highest-weight
Jplus norm1.698e-12, actual Hermiticity bilinear error9.437e-16Ha.
These shared-hardware measurements establish admitted exact storage and
correct selected J, not controlled speedup or experimental acceptance.
The complete upper-state M=3 solve retains979430 determinants,
447914312 coordinates/5378889468 owned CSC bytes, build40.56s/whole66.37s;
eigenpair residual5.231e-13Ha and highest-weight norm6.501e-12.
The common-H n4 P splitting9587.366125cm^-1 becomes9589.170066 with all
single5g directQ configurations: actual basis change+1.803941cm^-1.
The experimental error remains-196.829934cm^-1, **FAIL30**. Both ground and
upper energy decreases are separately recorded; no MBPT response is added.
Evidence: Sn12_n4_plus5g_operator_v3, Sn12_n4_plus5g_sparse_control,
Sn12_n4_plus5g_seed_control.json and Sn12_n4_plus5g_sparse_J2_4.

The native n5 RCCSDT comparator independently times out at its original3600s
limit. Its first iteration reports1908.1s and unconverged T3 update0.001187;
it supplies no splitting. Kernel perf/ptrace access is denied. A user-space
CPU sampler runs on the exact source/input but warns that its interval timer
changed; its percentages are diagnostic leads, **not** a certified hotspot or
speedup. The initial sampler loader failure is also retained. None of these
runtime failures authorizes adding a small-space triples correction to the
complete n7 CCSD result or loosening any optical gate.

A private exact block-matching experiment replaces internal multiplication
candidate scans with stable operand-1 prefix ranges and the existing exact
operand-2 coordinate lookup. Contraction order, restoration, values and
equations stay unchanged; one binary retains legacy/indexed A/B modes.
All52 live original targets pass under --runxfail. Complete real42-spinor,
high-j native C1/axial CCSD and full CCSDT core energies and10x10 model
matrices are **bitwise identical** between both modes and archived originals;
independent FCI errors remain below5.457e-12Ha. The prefix entry is72bytes
because the backend maximum rank is16; the earlier32byte estimate is
corrected before native use. About30.94MB conservative index/sort scratch
covers the measured214876-block operand; original88GiB backend/100GiB
process caps remain.

Actual complete n5 CCSD then replays energies bitwise and the original C1
full model within1.293e-15Ha; observed workflow5.60s. Nonperturbative n5
CCSDT completes in1332.19s, with the original amplitude/raw/multiplet gates
passed: maximum update8.806e-12, raw bound1.622e-9Ha and full multiplet/
cross-J defects below8.89e-15Ha. Its splitting13067.002698cm^-1 differs
from experiment by-111.997302cm^-1, **FAIL8**. The actual same-space
CCSDT-minus-CCSD change is-9.399703cm^-1. No small-space triples shift is
transferred to full n7. The first iteration57.4s has printed updates identical
to the original1908.1s iteration; different shared histories preclude a
controlled end-to-end ratio. The production builder/default remains unchanged
while this is a private experiment. Complete evidence is in
Sn13_FSCC_core18_CCSDT_axial512_n5_match_index/comparison.json.

## Measured exact angular reuse in the full CI objective

An actual complete8543-CSF objective profile shows repeated magnetic CSF
construction consumes about30.3s of59.7s under profiling; H and RDM independently
rebuild the same bases. The existing bounded angular-cache machinery now
reuses those signed bases for both consumers through
`MagneticCSFProjectionCache`. A128MiB default budget is part of the blocked
objective's resource admission; zero disables reuse. Radial/root coefficients
remain fresh, all weak entries are retained, and both full DC H blocks stay
unchanged. Standalone `SpinorCIHamiltonian`/`SpinorRadialDensity` require an
explicit cache to use this path.

The isolated actual checkpoint comparison gives original/cached-cold/cached-warm
full energy+gradient times43.69/31.64/22.39s, with accounted retained angular
payload53,487,820bytes and observed RSS about3022MiB. Both complete H hashes,
energy and every radial gradient coordinate are **bitwise equal**. These are
shared-machine observations, not a controlled speedup or a full stationary
solve timing. Seven independent cache/matrix/gradient controls include opened
core holes, radial phase reversal, DC/Breit, tiny/complex coefficients,
immutable wrapper ownership and oversized-entry behavior; the29-test affected
cache/CI/density/orbital panel passes.

The unconverged history64 run was explicitly interrupted at an accepted
checkpoint to apply this qualified numerical change, preserving its original
source, log, checkpoint and interruption record. The new source-bound solve
first replays that checkpoint's two complete H hashes, energy, original
residual value and metric hash, then continues unchanged1e-7/1e-10Ha gates.
Its3621s budget return saves94 accepted steps with residual4.45460e-7,
still above the original1e-7 gate. A later near-floor continuation is rejected
by the original checkpoint residual replay check before taking a step; complete
H hashes and energy match. Two Python hash seeds change full gradients by
up to6.346e-12Ha/coefficient while leaving H and energy bitwise equal; the
resulting residual difference7.642e-16 trips that very tight replay test.
Canonical complete radial-key ordering removes this source: all gradient
coordinates, energy, complete H hashes and original-metric residual are now
bitwise identical across both processes. No term or convergence gate changes.
A third-process continuation replays the canonical certificate exactly and
retains the original metric hash and1e-7/1e-10Ha gates. Near the subtraction
floor it uses the opt-in full-old-CI ground variational comparison and also
requires a decrease of the original residual. The first accepted step reduces
that residual4.45460e-7->3.63091e-7 while the raw deep-core energy difference
has the wrong sign. This is evidence for the numerical comparison, not a
floating-point proof or a stationarity/experimental pass. Every failed
checkpoint and original energy remains preserved.
The qualified canonical continuation now reaches the original stationarity
gate in10 accepted steps: residual7.04138e-8 below1e-7 and energy indicator
9.88402e-14Ha below1e-10. The original residual metric hash is unchanged.
Both complete DCB ground blocks are then recomputed from these common
stationary DC orbitals. Their energies are-6115.234010171020 and
-6115.173474499976Ha, with full-H residuals below2.729e-12Ha. The splitting
is13286.044087cm^-1, **107.044087cm^-1 high**: experimental8cm^-1 acceptance
still fails. DC stationarity does not mean stationary DCB orbitals or a
saturated core/virtual/no-pair model. The saved557.88s wall field measures
optimization before subsequent DCB reconstruction, not total workflow time.
No stationary orbitals or new-H rates are spliced into the old DC source.
Evidence: `full_CI_profile_v2`, `full_CI_projection_cache_probe`,
`full_CI_MCDHF_cached`, `blocked_canonical_accumulation_audit.json` and
`full_CI_MCDHF_canonical_variational_ground` under the active evidence root.

## Experimental comparison: current failed gates

The earlier spatial/experiment stage passed **165 checks**; the current finite
kinetic extension passes **191 affected checks** with an isolated Numba cache.
New independent controls cover self-consistent adjacent Saha ratios, EOS response,
immutable source identity, uniform-slab and Planck-bath transfer, cold-cell heating,
Milne photon reciprocity, quarter-wave/opaque Drude limits, passage through critical
density, conservative multi-charge spatial evolution, directional radiance and
absolute count calibration. These do not establish experimental agreement.

The replayable comparison is local evidence under
`~/playground/tasks/atomx-rccsdt-euv/build/euv_alignment_20261015`:

- Torretti et al. Fig. 4a provides a separately peak-normalized experimental
  spectrum. At its effective comparison assumptions T=32 eV, rho=0.002 g/cm³,
  LOS length=30 µm, the saved native **single Sn12+ bare-CI** directional spectrum
  peaks at **13.1096 nm** in the 12.5–14.5 nm window. Its distance from the
  conservative plot-derived experimental peak interval **13.40–13.65 nm** is
  at least **0.2904 nm**. This interval is not a measured one-sigma error bar;
  charge mixture, correlation and instrument response remain unmatched. Visible
  experimental curve segments are digitized with red-theory occlusions missing.
  No wavelength shift is applied and this is **not a matched experimental pass**.
- With the auxiliary NIST Sn XIII ground IP `(258) +/-3 eV`, **110933/115715**
  saved bare-CI levels lie above the nominal threshold. At 32 eV they carry
  **5.0007%** of this conditional single-ion partition. The NIST value is
  theoretical, and bare-CI energy/threshold uncertainty matters; this diagnoses
  candidate resonances, not their widths. Their naive promotion into stable
  multi-charge Saha partitions is forbidden by the new interface.
- Musikhin et al.'s open SparkLight dataset supplies three selected 1024x256
  EUV detector images, extracted with file CRC verification. The archive is
  only partially acquired; the three selected files are complete. They are
  counts, not joules: wavelength-fit coefficients, absolute gain and matched
  collection calibration are unavailable in those files. Its CTS conditions
  differ from the high-intensity EUV images and are not transferred between runs.
- Mostafa et al.'s 2023 **5%** claim has a **2025 calibration erratum**. The revised
  directly obtained CE is in the approximately 4% range; the extrapolation at
  100% enclosed beam energy is **4.7(5)%**. This extrapolation is not a raw point.
  Revised figures and matched disk target/pulse/2pi collection must define the
  CE benchmark; the old uncorrected title is not a numerical acceptance target.
- The public Zenodo STAR-1D data are **simulation**, not experiment. The best
  listed 2 µm model has CE=4.635%; this is only a reference definition/coverage
  check. Its emitting plasma reaches mean charge 14.7, so Sn11–14 alone need
  not close a useful charge window. Rounded duplicate x coordinates invalidate
  reconstruction of positive cell widths in the supplied 5.5/10.5 µm profiles;
  those wave diagnostics reject rather than inventing a mesh. The 2 µm profile
  is usable only as an optical-operator diagnostic with explicitly approximate
  collision inputs, not an independent source CE prediction.

Absolute CE uses 13.365–13.635 nm and the hemisphere facing the laser. The
calibration consumer requires the actual response, accumulated shots and angular
conversion; a peak normalization cannot meet this contract. Current status is
**EXPERIMENTAL_ALIGNMENT_NOT_PASSED**. Priority is resonance-safe multi-charge
atomic/rate coverage and prepared-target EOS/ablation, followed by the matched
corrected absolute experiment. No new empirical shift, fitted CE multiplier or
simulated datum is used as an experimental substitute.

References defining the physical target, rather than numerical acceptance of
this reduced model:

- [Mostafa et al., calibration erratum (2025)](https://doi.org/10.1063/5.0258427): corrected 2 µm CE reference and revised figures; supersedes the original 2023 calibration.
- [Musikhin et al. experiment and open dataset](https://doi.org/10.34770/ajd8-by34): raw EUV detector images, DOI 10.1063/5.0325233.
- [STAR-1D public model data](https://zenodo.org/records/20422714): simulated spectra/profiles, not an experimental CE benchmark.
- [Torretti et al., Nature Communications 11, 2334 (2020)](https://www.nature.com/articles/s41467-020-15678-y): multiply-excited-state contributions, population/EOS and opacity/transfer are separate from a line list.
- [Characterization of plasmas driven by laser wavelengths in the 1.064–10.6 μm range](https://arxiv.org/abs/2208.14223): coupled radiation hydrodynamics, Drude absorption and geometry-dependent trade-offs; its optimum cannot be transferred to the present slab.
- [Atomic Processes in Laser-Produced Tin Plasmas, PFR 21, 2401001 (2026)](https://doi.org/10.1585/pfr.21.2401001): collisional-radiative populations and radiation data in EUV-source modeling.

Documentation impact: this is the current owner of reduced-source conventions;
CI, capabilities, architecture, verification, testing and navigation link here.
General atomic solvers, historical sealed results and the single-core challenge's
atomic acceptance criteria are unchanged. Local source-bound runs and the
fixed-charge package live in `~/playground/tasks/atomx-rccsdt-euv/build/euv_source`;
the sealed spatial/experiment package is in
`~/playground/tasks/atomx-rccsdt-euv/build/euv_alignment_20261015`. Current finite
kinetics, real input-coverage audit and a distinct development package live in
`~/playground/tasks/atomx-rccsdt-euv/build/euv_nonlte_20261015`.


The optional numerator-division factory now also completes a fresh actual
public CMake build from the immutable427-file tool epoch. Its binary
156ad0b19fee65e116f0d0b02d60873e9d526e1f8314e3b26620bc3a11a301e4
passes the same74 affected live tests with --runxfail and no skips in22.73s.
Every source file matches the private qualified candidate except mult.c's
separately qualified producer-index experiment. This goes beyond earlier
patch-only replay; old Sn jobs keep their original E6 binary. Build AS/wall
monitoring began after initial launch, so no total-build resource/timing
certificate is claimed. Evidence: public_exact_nonzero_actual_build_
verification.json (join wrapped filename).


A fresh common-H coherent P/Q wave ledger separately replays each complete
sparse matrix by all data/indices/indptr hashes. Actual5g Q wavefunction
weights are0.000148423/0.000147023, not plasma populations. The computed
lowest Q-only energies lie6.073942/6.030259Ha above the retained P roots;
Q eigenpair residuals are below8.72e-12Ha and the full P/Q equations below
9.27e-13Ha. Signed P-relaxation, Q-energy and cross terms reconstruct each
nonperturbative energy shift within1e-9Ha. Thus the small H0word denominator
is not the physical gap from these low roots. This is a statement about
the declared model, not all excited EUV states or the paper reference.

The DC-only symmetric-MBPT18-group diagnostic evaluated on these **same bare
P states**, instead of its self-consistently rediagonalized states, gives
AA=+36.585461, BB=-125.657731, cross=-646.667565, full=-735.739835cm^-1
fixed-state gap responses. They differ from the earlier dressed-state
expectations; a state is not transferable by its label alone. Because the
MBPT vertices are DC-only while the explicit CI includes DCB/modelQED,
this comparison does not isolate denominator treatment as the sole cause.
No expectation component is an accepted energy repair. Evidence:
Sn12_single5g_coherent_ledger_comparison.json.

The exact optional hash row index now removes a separate combinatorial
storage barrier. Nineteen finite controls include independent complexN0..4
actions, a selected40-spinor/18-electron case, zero/vacuum/tiny values,
exact CSC byte equality and the installed SciPy int64-index promotion rule.
An initial Numba unsigned/signed index merge fails before arithmetic and
is retained; explicit signedint64 slots repair typing. Native1086326-state
Sn CSC values/indices/pointers for all507892914 coordinates then match the
qualified colex matrix byte for byte; its rowmap count falls from
1509395976 to50331648 bytes. These are storage and correctness measurements,
not a controlled solver speedup. A distinct426-file hash epoch changes only
the numerical sparse module; the subsequent CSC-admission epoch retains
identical assembly/lookup source and independently replays native40/N18
complex physical38 anchors byte for byte. Large-output admission now uses
8-byte row indices when SciPy requires them; no hidden promotion copy is
ignored. The public default stays colex; see the numerical architecture
and migration notes.

A new physically explicit core-valence discriminator unfreezes complete
3d10 at the original16-key native DCB/model-QED point. Twenty electrons,
including4s2, stay frozen; all18 correlated electrons occupy40 spinors.
Reclosing3d10 reproduces previous Ecore exactly, Fcore within3.198e-14Ha
and every retained n4ERI exactly. Independent physical38-electron Slater
and complex-phase/hash anchors pass within1.183e-11Ha. The retained P has
all22 original3d10/n4fold6 configurations; Q contains all33 directly
Coulomb/one-body reachable3d9/n4nine-electron patterns. No5g scalar change
or MBPT is added. Selected M=2/M=3 dimensions3558109/3172442 would need a
453521047200-byte colex rowmap, replaced by an exact100663296-byte hash
index. Dense CSF H storage remains inadmissible.

The first launch's missing PYTHONPATH is rejected by the source-epoch guard
before physics; the failure is retained. The correctly pinned continuation
reproduces the retained-P total energy within1e-12Ha and passes angular
and raw eigenpair gates. The complete lower/upper matrices contain
1860826293/1627444800 entries and own22344147956/19542027372 bytes.
Build times are663.61/235.00s; whole runs790.37/326.27s on shared hardware.
The upper run uses the separately qualified exact prefix-population index;
different J spaces and sources prohibit a paired speedup claim. Both keep
96GiB AS,64GiB sparse/work,80billion candidate work and3600s caps.
The common-H splitting moves9587.366125 to9580.973679cm^-1: the actual
core-hole increment is-6.392447cm^-1 and the experimental error-205.026321,
**FAIL30**. No scalar correction is combined with other models.

Three low roots from the upper run's hash-verified complete cached matrix
also pass explicit J-squared vector residuals below1.30e-11 and raw
energy residuals below3.86e-13Ha. Native Grant-phase LS anchors and
within-J character assignment, without energy matching, give3F4=18470.531302
and1G4=41899.012262cm^-1, errors-93.468698 and+2181.012262: **FAIL50/44**.
The first result serialization fails on a NumPy boolean after preserving
all states; a checkpoint replay recertifies H, states, J and decisions
before writing valid JSON. No eigensolve is silently repeated. Evidence:
Sn12_single3d_hole_common_H_comparison.json,
Sn12_single3d_cached_M3_levels_v2/result.json and the retained initial failure.

### Exact prefix rejection and the core-valence partition audit (2026-10-16)

The optional hash index derives prefix-population support from every declared
row mask. A candidate with a population absent from that exact support cannot
be a retained row. This prefilter uses no energies, coefficients or amplitudes;
full masks still decide every possible match. Nineteen finite checks pass in
2.58s. The1086326-state/507892914-entry native matrix is again bytewise equal
to the original colex output. On1024 deterministic kets from the actual
3558109-row3d-hole space, all2811244 Pauli-valid queries return exactly the
same row IDs;1766591 misses are rejected by the low10-bit support{9,10}.
Warm alternating lookup timings about0.083/0.030s are microkernel observations,
not whole-solver or controlled hardware speedups. Immutable source epochs and
native40/N18 physical Slater/phase replays bind this change separately.

A primary-source method audit corrects the interpretation of the earlier
generic native P/Q second-order results. [Reitsma et al. (2026)](https://journals.aps.org/pra/pdf/10.1103/v557-qqql)
uses complete n4 CI for valence correlation and second-order **core-valence**
MBPT, including effective three-body terms. Previous generic Q additionally
contained outside-n4 valence sectors. Those finite models and their failed
results remain diagnostics of their declared partition, not direct paper
reproductions. A later source/runtime audit below further corrects an overly
narrow interpretation: original CoreMBPT uses the full closed-Fermi core,
including4p valence holes, not only deepZn30. Thus a require-Zn30-hole
selector is also a distinct finite model, not the source's default core-only
partition. No prior output or failed gate is rewritten; differences between
these models cannot be attributed solely to denominator conventions.

A finite-model discriminator selects every Q group containing a Zn30 core hole,
before energies, retaining all coupled vertices and ranks0..3. All Hermitian,
forbidden-coordinate and complete-rank J-plus commutator gates pass. An
independent physical38-electron Slater Q-sector calculation agrees at every
selected2x2 entry within3.09e-15Ha(n5)/6.47e-14Ha(n6). The minimum retained
word denominator is0.449510842Ha; the previously identified0.009079827Ha
pure-valence denominator is outside this declared core-hole partition.

Complete n4 M=2 CI(69100 rows/177144455 entries/2126009864 owned bytes)
then gives six roots with eigenpair residuals below4.41e-12Ha, J-squared
vector residuals below9.51e-11, and Js{2,2,2,3,4,4}. Assignment is
bijective within J by native Grant-phase LS character, independently of
energy. The predeclared character threshold0.5 is retained.

| Target | Core-only n5(cm^-1) | Core-only n6(cm^-1) | n6 minus experiment(cm^-1) | Fixed gate |
|---|---:|---:|---:|---|
|3F3|9789.583125|9788.921430|+2.921430|PASS30, conditional finite model|
|3F4|18659.967357|18658.471115|+94.471115|FAIL50|
|1D2|24745.825881|24754.359401|-83.640599|FAIL25; LS character0.4712 is ambiguous|
|3P2|38743.097271|38752.515941|-291.484059|FAIL30|
|1G4|41259.462455|41238.462426|+1520.462426|FAIL44|

The two whole runs take24.35/24.33s with about2542MiB peak RSS on shared
hardware. One conditional interval pass does not establish five-level,
virtual-basis, paper, EUV or CE alignment. These bound n5/n6 calculations
still lack the paper's30spdfghi positive-energy basis and matching AMBiT
H0/subtraction/source conventions. No explicit3d/5g CI increment is added.
The original unexecuted driver contains a syntax error; v2 preserves it and
repairs only syntax/mask admission before either run. Evidence:
Sn12_native_AOC_core_only_n5_coefficients/result.json,
Sn12_native_AOC_core_only_n6_coefficients/result.json,
Sn12_core_only_fullM2_n5/result.json and Sn12_core_only_fullM2_n6/result.json
under the existing euv_resonance_alignment_20261015 evidence root.


The existing direct-single5g discriminator now also has all six low M=2 roots,
with **every** original/expanded CSC value/index/pointer hash equal to the
previously qualified matrices. Independent J-squared vector residuals stay
below9.42e-11 and eigenpair residuals below1.55e-12Ha. With the same native
Grant-phase LS assignment, complete single5g CI changes the ground-relative
3F3/3F4/1D2/3P2/1G4 levels by+1.803941/-3.018543/-24.440743/-19.054882/
-67.938879cm^-1. All five expanded values remain outside their original
experimental intervals. Thus this particular complete direct5g sector does
not explain the low1G4 error; other valence/core/continuum correlations remain
unbounded. No5g increment is added to the distinct core-only effective model.
The original/expanded six-root whole runs take5.93/107.71s on shared hardware,
not a controlled speed comparison. Evidence: Sn12_single5g_all_five_comparison.json
and Sn12_single5g_fullM2_alllevels_original/expanded result directories.


The nested **complete bound n7 s..i** core-hole-only rung preserves every
old radial coefficient/H0 energy bitwise and passes metric/stationarity gates.
Using on-demand native ERIs avoids the unused full280-spinor packed bank.
Two strict nonzero-support guards fail first and are retained. The271 absent
old words all have unequal kappa and are analytically forbidden scalar
one-body coordinates; their largest old cancellation remnant is1.180e-16Ha.
No magnitude threshold is applied. After proving this exact selection, every
old n6 rank0..3 coefficient replays within6.662e-16Ha with unchanged
coefficient tolerances. Full n7 independent physical38-electron Q assembly
contains1548351 actual rows and agrees with the supplied effective matrix
within2.868e-13Ha. All complete-rank angular/Hermitian gates pass.

The n7 operator retains1732456 words/899234 Q groups/36251808 ordered
products; its counted contraction envelope is710511552bytes. Checkpoint
continuation takes87.88s/about4174MiB RSS. Complete low-M2 CI then takes
25.68s and gives3F3=9787.877974(PASS30 conditional),3F4=18656.841363
(FAIL50),1D2=24763.035901(FAIL25 and character0.4710 ambiguous),
3P2=38761.546249(FAIL30),1G4=41226.954690(FAIL44), in cm^-1.
N7-minus-n6 shifts are-1.043456/-1.629752/+8.676500/+9.030309/-11.507736.
These finite shifts are not uncertainty bounds; the last four failed gates
and all continuum/paper/EUV/CE gaps remain. Evidence:
Sn12_core_only_n5_n6_n7_comparison.json,
Sn12_native_AOC_core_only_n7_coefficients_v3/all_old_rank_coefficient_replay.json
and its independent_physical_Q_sector_anchor.npz. No discarded failed run
is relabeled as the accepted continuation.


### Complete positive-box basis and exhaustive Q partition (2026-10-16)

A nested native n<=30/l<=6 complement now includes2534 spinors: all earlier
n<=7 coefficients/H0 energies stay bitwise, with192 new negative and107 new
positive radial keys. There are808 positive magnetic spinors. The maximum
box energy is78.235458Ha. All299 new requested roots keep projected residuals
below1.3e-12Ha and full radial Gram error1.617e-12. These are box-normalized
pseudostates, not scattering-flux functions or a radial/box convergence claim.
The bound-only accessor first runs out of bound roots; raw generalized roots
then fail the original residual gate. Both failed attempts remain. Reusing the
shared solver's constrained eigenpair refinement on the complete physical
branch repairs numerical accuracy without changing the basis or gate.

A full packed complex integral bank would need82440986444992bytes, and the
complete M/parity-compatible core-hole vertex loop has183730802 candidates.
No such quartic allocation is attempted. Exact Q-group exterior counts0,1,2
are exhaustive and disjoint for a two-body Hamiltonian. Matching Q occupations
alone contract at second order. Consequently these class operators add within
**one fixed Hamiltonian/H0 model**; they are not separately computed CI shifts.
The exterior0 class, including induced rank3, is invariant under this extension.
Exterior1 groups are streamed through the unchanged independently qualified
word kernel. Exterior2/corehole1 contributes only scalar Sigma1; exterior2/
corehole2 contributes only scalar Sigma0. Both use the existing signed
J-coupled pair kernel and shared Slater GEMM tables, retaining every declared
radial pair, multipole and Pauli/identical-pair factor.

The complete n7 replay checks every directed0..3-body coordinate. Streamed
exterior1 differs from the whole-word oracle by at most6.440e-15Ha; the
coupled exterior2 Sigma1/Sigma0 comparisons give3.383e-17/2.993e-15Ha.
Deterministic signed pair calls separately match the batched angular recipes
below4.337e-19Ha. Reassembling all Q classes agrees with original whole n7
coefficients within2.567e-13Ha and the independently assembled physical38
selected Q matrix within1.580e-14Ha. Hermitian, complete-rank Jplus and exact
M/parity-forbidden coordinate gates remain separate and pass.

The actual n30 exterior1 calculation retains4017906 words/682455 groups/
144902482 ordered products, at most2966 words in any streamed exterior-mode
batch. It takes118.85s/about1431MiB RSS. Coupled exterior2 Sigma1 and Sigma0
take2.49/2.07s/about410MiB RSS. Timings describe these stages on shared
hardware; no whole-solver or paired speedup is claimed.

A further independent native38-electron control keeps **all13 nominal-n30
radial multiplets** (98 new magnetic spinors) with the original n4/core,
including all core-hole Q of that158-spinor diagnostic. Packed native
integrals and explicit signed single/double Slater couplings provide a
consumer independent of streamed on-demand generation. Its343704 actual Q
states/344176 couplings agree at every selected2x2 entry within5.413e-16Ha.
This is a complete finite **n30-shell control**, not a brute-force independent
replay of the entire2534-spinor Q. Full composed n30 coefficient Hermiticity,
Jplus and forbidden-coordinate gates also pass; low-root experimental
comparison remains a distinct gate. Original AMBiT averaging/subtraction/
reference input identity and radial/box convergence remain unverified.

Evidence under the existing active root: Sn12_core_only_n30_preparation_v3
(including resource_preflight.json), Sn12_one_outside_corehole_stream_n7/n30,
Sn12_two_outside_corehole_coupled_n7/n30,
Sn12_two_outside_core_scalar_coupled_n7/n30,
Sn12_n30_shell_physical_Q_anchor and Sn12_corehole_Q_partition_n7/n30.
These private research drivers introduce no new public Atom dispatch or claim
of complete Sn rate, EUV emission or absolute CE validation.


The complete n30/l6 positive-box core-hole model now passes the six-root
numerical/J gates (whole CI21.40s). Its fixed-target results, in cm^-1, are:
3F3=9805.401202 (error+19.401202, conditional PASS30),
3F4=18684.896599 (+120.896599, FAIL50),
1D2=24868.770603 (+30.770603, FAIL25 and LS character0.4633 ambiguous),
3P2=38905.433466 (-138.566534, FAIL30),
1G4=41025.192980 (+1307.192980, FAIL44).
Adding the complete declared positive-box complement therefore changes actual
root predictions; it does not close the five-level experimental target.
N7-to-n30 changes are not grafted onto an old state or called error bars.
One further fixed-l<=6 n35 rung is preregistered to measure radial stability
(all five changes below1cm^-1 would be required); this does not license a
sequence of low-information scans or alter the failed experimental gates.
Evidence: Sn12_core_only_fullM2_n30/result.json and its immutable source/H
coefficient receipts. Overall status remains EXPERIMENTAL_ALIGNMENT_NOT_PASSED.


The predeclared **n35/l6** check completes with the same original n<=30 inputs.
Its new nominal-n35-shell physicalQ anchor agrees within8.882e-16Ha; the full
operator passes all rank/angular gates. Whole CI takes22.38s. The five values
are9792.557610/18662.388746/24852.326800/38872.919949/40991.785106cm^-1.
N35-minus-n30=-12.843592/-22.507853/-16.443804/-32.513517/-33.407874cm^-1;
all exceed the1cm^-1 stability gate. The1D2 energy enters its25cm^-1 interval,
but its0.4632 LS character still fails the predeclared identity gate; this is
not promoted to a new accepted target. Other experiment and convergence
gaps remain. Evidence: Sn12_core_only_n30_n35_radial_comparison.json.

### Explicit single and double 4s holes (2026-10-16)

The0.449510842Ha minimum **H0 word** denominator and vertex ratio0.306664
are identified specifically with a4s hole. These are basis-column diagnostics,
not the actual gap from a low eigenstate. Before attributing a failure to
perturbation theory, the new bare-CI discriminator reopens4s2 completely in
the original16-key native DCB/model-QED Hamiltonian. It freezes28=[Ar]3d10,
correlates10electrons/32spinors, and retains original22P plus all33single-
and35double-4s-hole directly coupled occupation patterns. All magnetic and
j partners remain, without model selection by experiment.

Reclosing4s2 recovers Ecore within2.729e-12Ha, Fcore1.066e-14Ha and every
old active ERI exactly. Independent physical38-electron P/Q1/Q2 Slater,
supplied-body/hash sparse and complex-phase controls pass within1.001e-11Ha.
The correctly source-bound426-file prefix-index epoch is distinct from the
old physical-coefficient epoch. Original-P six-root energies replay within
the unchanged1e-8Ha gate. The complete expanded M2=4 space has1477696 rows,
783884192 CSC entries and9412521092owned bytes; assembly83.67s/whole153.03s
on shared hardware, without a paired speedup claim. Every root and explicit
J-squared vector gate passes before native Grant-phase LS assignment.

|Target|Explicit4s(cm^-1)|Same-H4s increment(cm^-1)|Minus experiment(cm^-1)|Gate|
|---|---:|---:|---:|---|
|3F3|9662.155775|+74.789650|-123.844225|FAIL30|
|3F4|18528.678090|+46.188186|-35.321910|Conditional PASS50|
|1D2|25407.586471|-1307.721923|+569.586471|FAIL25; character0.4796 ambiguous|
|3P2|39518.315153|-1413.322092|+474.315153|FAIL30|
|1G4|41569.382630|-354.478053|+1851.382630|FAIL44|

The actual4s response is substantial and state dependent, but this particular
finite bare model still fails experiment. It has **no** complementary28-core/
exterior correction. Adding the old30-core MBPT shifts would doublecount4s
and mix operators; it is forbidden. A combined model must regenerate its
operator with the28-core/32-active partition and pass its own work/memory,
full-state and independent physical gates. Evidence: Sn12_4s_corehole_preflight,
Sn12_4s_corehole_operator, Sn12_4s_corehole_fullM2_alllevels_original/expanded
and Sn12_4s_corehole_all_five_comparison.json. The near-word ratio alone does
not establish a physical low-state resonance or the cause of the EUV blueshift.


### Separate V36 reference and exterior valence sector (2026-10-16)

The earlier full244-virtual frozenKr FCI and full-n7 FSCC archives use the
actual **closed36/V36** radial point, not the V38 statistical point used in
the preceding core-hole campaign. All common orbitals are compared before
reuse; no energy increment or amplitude is transferable between those points.
The new V36 full-n4 operator uses the same280-spinor disk source as that FCI.
Native on-demand full Fock/reference and every retained n4 ERI replay **exactly**.
A full38-electron Slater anchor verifies the noncontiguous occupied4s and
reordered4p wedge phases within5.457e-12Ha. This is an operator identity proof,
not a new experimental accuracy claim.

On that one V36 native DCB/49-key-model-QED Hamiltonian, the original22 n4
P configurations (69100 M2=4 rows) now couple coherently to all1395
frozenKr two-valence determinants in the full244-virtual n7 bank that have
at least one exterior orbital. Q keeps4p6; exterior configurations with4p
holes remain omitted. All45438 nonzero P-Q entries, Q-Q interactions and
interference are retained. New cross samples match independent native
DCB physical38 Slater values exactly; the full matrix has20032201 entries.
Raw six-root residuals stay below6.09e-13Ha and explicit J-squared below
7.76e-11. The P/expanded whole runs take2.17/18.81s on shared hardware;
this is a method/space discriminator, not a speedup comparison.

|Target|V36 n4P(cm^-1)|V36 P+external p6 sector(cm^-1)|Same-H change(cm^-1)|Final experiment gate|
|---|---:|---:|---:|---|
|3F3|9312.625743|9392.902198|+80.276455|FAIL30|
|3F4|18046.805730|18093.612942|+46.807211|FAIL50|
|1D2|26720.043495|26131.647319|-588.396176|FAIL25|
|3P2|40619.610740|40089.126009|-530.484730|FAIL30|
|1G4|41901.173195|40963.742128|-937.431067|FAIL44|

This measured exterior response matters, but no deep-core or V38 correction
is grafted onto it. A new V36 n35/l6 core-basis preparation independently
reproduces the old280-spinor DC Fock via radial field projection within
4.264e-13Ha, preserves every old radial byte and passes all old/new
stationarity and Gram gates (Gram9.962e-13). Its H0 is the actual closed36
field; there is no statistical-d averaging substitution. A first zeroexterior
producer still points one check at the V38 Fock reference and fails before
any coefficient use. The failure is retained; the continuation replaces it
with an independently reconstructed V36 field minus the explicit4p trace,
under the **unchanged1e-10Ha** field gate. New V36 coefficients and anchors
now pass: independent physical38-electron zeroexterior Q (19350 rows)
agrees within8.188e-16Ha; a separate158-spinor new-n35-shell control
(343810 Q rows) agrees within7.078e-16Ha. Every0..3 rank is generated
fresh in V36; no V38 coefficient enters the new operator. Its full3024-spinor
Q space is not claimed to have an independent brute-force replay.

Complete six-root M2 CI of that V36 n35/l6 core-only second-order model
takes20.97s/about2550MiB RSS. Raw eigenpair residuals are below2.412e-12Ha
and J-squared vector residuals below6.987e-11. All five experimental
energy gates fail:

|Target|V36 core-only n35(cm^-1)|Minus experiment(cm^-1)|Fixed acceptance|
|---|---:|---:|---|
|3F3|9429.984912|-356.015088|FAIL30|
|3F4|18174.849503|-389.150497|FAIL50|
|1D2|25773.813198|+935.813198|FAIL25 and identity(character0.4796)|
|3P2|39626.176415|+582.176415|FAIL30|
|1G4|41256.576736|+1538.576736|FAIL44|

The large remaining V36/V38 dependence is a negative finite-model
discriminator, not a reference-field convergence certificate. The V36
exterior-valence CI and V36 core-only MBPT tables remain different retained
spaces; their shifts must not be added. An independent original-AMBiT
particle-hole CI+Sigma1/2/3 calculation is the next convention control;
its specified inputs must remain distinct from the unavailable paper input.
Evidence: Sn12_V36_n4_operator, Sn12_V36_valence_sector_CI_original/expanded,
Sn12_V36_valence_sector_comparison.json, Sn12_V36_core_n35_preparation and
the retained Sn12_V36_core_zero_outside failure/continuation,
Sn12_V36_core_Q_partition_n35_v2 and Sn12_V36_core_only_fullM2_n35.

The2026 [primary paper](https://journals.aps.org/pra/pdf/10.1103/v557-qqql)
states that its supporting research data/software are not publicly available;
authors provide further information on request. A public-data search finds no
source input to replace that limitation. No author contact is sent. The local
finite source-qualified models remain distinct from a byte-matched paper run.

### Independent original AMBiT full particle-hole control (2026-10-16)

An independently specified original-source run now uses closed V36,
explicit Fermi120Sn half-density radius5.514213570173778fm/thickness2.3fm,
DKB/Reno90 splines/order9/R60 on4096 lattice points, all22 fold6
configurations as leading, no generated extra excitations, frozenZn30,
and30spdfghi MBPT orbitals. Denominator shift/floor are explicitly zero;
this input floor applies to Sigma1/2. Source Sigma3 retains its inherited
0.01Ha floor, which a complete deep/active orbital-energy check finds
inactive here (minimum absolute denominator1.868908973895Ha). Original
nonperturbative-term warnings remain a rejection condition. PureDC
and complete Sigma1+2+3 are separate runs; neither is the unpublished
author input. The original source-owned triangular/leading Sigma3 consumer
is used without replacing it by a Hermitian average of raw directions.

Preflight reproduces exactly9714/11363/11738 CSFs for J2/3/4
(32815 total); three dense matrices would occupy2890081672bytes.
Preflight takes170.91s and the pureDC run212.13s on shared hardware.
Only qualified identical orbital/angular caches are copied; no levels or
MBPT coefficients transfer between the bare and corrected runs.

The first model-projection export fails because original configuration
names have a default display prefix; the next exporter explicitly requests
Name(false). Both failures and corrections are retained. A subsequent raw
Davidson residual8.768e-8Ha fails the unchanged1e-8 gate. A separately
specified continuation refines the same complete saved matrices with eigsh
and recomputes projections from the exact original CSF map; it does not
change the Hamiltonian, root count, tolerances or state rule. Independent
full-matrix residuals are then below9.545e-14Ha. Source Ck angular gauge,
signed radial phases and the d2 model J-squared are checked before the
same within-J/LS>=0.5 target matching. The J-squared check independently
covers the model projection, not every original full-space CSF.

|Target|Native V36 pureDC(cm^-1)|Original AMBiT pureDC(cm^-1)|
|---|---:|---:|
|3F3|9672.048753|9672.049851|
|3F4|18693.581579|18693.583443|
|1D2|27029.220610|27029.221577|
|3P2|41343.648392|41343.650712|
|1G4|42457.616201|42457.617795|

Every excitation agrees within0.002321cm^-1 and the signed model LS
characters agree closely. This strongly qualifies the bare n4 CI comparison
at this point, but is not a full-integral or radial-byte identity certificate.
All five bare experimental intervals still fail.

A separate bounded performance control wraps only the original Sigma3
getter at link time and caches exact ordered six-state calls, including hole
flags. Original compiled main/libraries and matrix assembly remain unchanged.
100000 raw/cached bitwise comparisons (4596 nonzero),4096 runtime hit checks,
and explicit qualified update-invalidation controls pass. A first guard
failure is retained: linker wrapping does not intercept general C++ virtual
UpdateIntegrals dispatch. The continuation is therefore explicitly confined
to this one immutable operator after initial update, not a generic mutable
operator cache. At8million entries overflow computes the original function;
it does not remove a term. The uncached calculation remains an independent
control, and full saved-matrix byte equality is required when available.
These concurrent shared-host runs do not establish a controlled speedup.

The cached full V36 Sigma123 run completes in402.74s/about2992MiB RSS,
with2156946996 calls,2152774908 hits,4172088 unique misses,6248 bitwise
runtime checks and no overflow or source skip warnings. Independent saved-H
audits retain original Davidson failures8.803e-8/1.415e-7Ha in J3/4,
refine those identical matrices, and recompute the signed model projections.
The maximum final full-matrix residual is8.941e-9Ha, below the unchanged
1e-8 gate. This finite **DC-only** reference gives:

|Target|Original V36 DC+Sigma123(cm^-1)|Same-point MBPT change(cm^-1)|
|---|---:|---:|
|3F3|10083.310837|+411.260986|
|3F4|19280.640233|+587.056791|
|1D2|26070.630873|-958.590703|
|3P2|40763.285841|-580.364871|
|1G4|42270.871071|-186.746724|

All five experimental intervals fail and1D2 also fails identity(character
0.4771). The full uncached run reaches its original1800s cap before finishing
the first J2 matrix; that failed control is retained. A separate complete
J12 sector of the same22leading/core/basis model has311 CSFs. Its raw and
cached388132-byte matrices match exactly(SHA804314866aadd243dcb4a0c3aeeebe7470e0adc6b6cc1c675c45cc4c5266713f);
independent six-root residual is7.229e-15Ha. This is complete consumer
evidence in J12, not a large-J2/3/4 byte-equality certificate or a controlled
speedup measurement. Cached J12 retains1046856 calls/240690 unique misses.

An independently regenerated original averaged V38/subtraction control,
with explicit4d0.8/4d+1.2 occupations, completes in421.02s. Its fresh
preflight reproduces all CSF counts and100000 bitwise cache controls
pass(4594nonzero); inherited Sigma3 floor is inactive(minimum1.816407357494Ha).
Saved-H/CSF-map audits retain failed original Davidson roots and qualify
the same fixed six-state matching with maximum residual8.888e-9Ha:

|Target|Original V38 DC+Sigma123(cm^-1)|V38 minus V36(cm^-1)|
|---|---:|---:|
|3F3|10124.378270|+41.067433|
|3F4|19310.732781|+30.092548|
|1D2|25785.072527|-285.558347|
|3P2|40478.467534|-284.818307|
|1G4|41931.278753|-339.592319|

All experimental energy intervals fail;1D2 also fails identity. Only the
3F4 field variation lies within its original experimental interval, so the
five-level field-stability test fails. This is a finite independently
specified30spdfghi DC model, not a refutation of the unpublished paper input.

Runtime classification and a physical Q-path audit now qualify the source
partition itself: original core has12 radial keys/36 spinors, deep has
10/30, and hole has2/6(the4p shell). CoreMBPTCalculator loops over
core=deep+hole and uses InQSpace to reject only wholly valence intermediates.
The actual 4d3/2+4p1/2 ->5s1/2+4f5/2 antisymmetrized Coulomb coupling is
0.008412012287694828Ha and is retained in source core MBPT, while the
native require-Zn30-hole selector excludes it. Thus earlier native/source
differences also involve Q coverage, not only denominators or subtraction.
The existing native Zn30-hole results retain their numerical qualifications
as finite models; their interpretation as the source's core partition is
withdrawn. Source --use-valence separately adds no-nonvalence-hole
high-valence diagrams. The author's unpublished flag choices remain unknown.
Evidence: Sn12_AMBiT_core_partition_probe, actual original
OrbitalManager classification and Core/ValenceMBPTCalculator sources.

A fixed-N90 original all-positive-branch control is admitted to address
the different high-virtual coverage. Request100spdfghi deliberately exceeds
that finite basis capacity; the actual inventory has1170 radial states,
887 positive energies, maximum principal label96 and no original norm-based
spurious removals. It is not labeled a completed n100 calculation. The
preflight requires38088433 bare MBPT integrals below the predeclared200million
cap; all same22 CI dimensions pass. A separate source-owned simultaneous
static-Breit plus effective self-energy/Uehling model is prepared with the
same full-positive branch, fresh selfconsistent orbitals and all effective
operators. No native QEDMOD or another model's scalar correction is added.

The low-n comparison does not certify the high virtual basis. On the same
closed36 field, native/original30s pseudostates have energies75.459063/
1.192577Ha, and30i+ has34.763075/1.218176Ha, although common n4/n7 bound
energies closely agree. The original native outer knot intervals reach
10.4646bohr. A single preregistered exact knot-enrichment diagnostic keeps
all old knots and core functions, subdivides intervals ending above1bohr
to maximum width0.5bohr, and re-solves virtuals without transferring MBPT
coefficients. The197-spline embedding preserves every old function within
6.662e-16 and core energies within5.685e-14Ha, but richer-space core
stationarity fails at2.340e-5Ha versus1e-8. It is therefore not used as a
converged new core. A separate fresh solve through the public
install_physics_grid interface(hmax0.5/default hi/he) has180 splines and
actualR60.32336491849; it converges in12 iterations, with projected core
stationarity1.017e-11Ha and Gram1.959e-15. Low-n4 energies agree with
original AMBiT within1.052e-8Ha, but its n30 box states lie at0.1616--0.2262Ha.
The preregistered0.1Ha pseudostate-distribution comparison fails(maximum
1.2691Ha difference). Neither failure is relaxed or treated as an experimental
result. Matching n labels does not establish continuum coverage; integrated
core-response convergence requires its own qualification.

The N90 whole-positive source generator completes in605.67s but its
request100 notation produces a source basis-coverage warning. That output
remains unadmitted under the original warning gate. A separate exact-capacity
read replay uses90s91p92d93f94g95h96i: the full1170-orbital inventory and
every copied scientific basis/integral/H/level SHA stay unchanged, with no
reader warning. This repairs the representation of the predeclared available
branch, not the physical space or an experimental threshold. Strict saved-H
and CSF-map audits on those same matrices then pass(maximum4.080e-9Ha),
retaining raw Davidson failures and refinement:

|Target|Original V36 whole-positive N90 DC+sourceCore Sigma123(cm^-1)|Minus n30(cm^-1)|
|---|---:|---:|
|3F3|10092.455440|+9.144603|
|3F4|19293.396301|+12.756068|
|1D2|26182.629642|+111.998769|
|3P2|40898.933000|+135.647159|
|1G4|42031.663719|-239.207352|

All five experimental intervals fail and1D2 remains ambiguous(character
0.4684). The single preregistered whole-positive N60/N90 response comparison
passes; its all-five stability threshold is1cm^-1, independently of
experimental proximity. N90-minus-N60 changes are+0.047263/+0.079020/
+0.045080/+0.103079/+0.072393cm^-1 for the five ordered targets; maximum
0.103080cm^-1. N60 keeps all780 available orbitals using exact
60s61p62d63f64g65h66i capacity notation before full-H execution. The native
grid/distribution failures, source n30 response and full-positive response
are distinct qualifications. A matched finite n7 source core versus
core+high-valence(--use-valence) pair has matched field/basis bytes and
52807/68169 bare-integral admissions. Core completes395.00s without warnings;
the optional-valence producer completes473.16s but is rejected for six
original nonperturbative dropped-term warnings, including4f+4f ->4d+5g.
Examples have reduced-diagram numerator0.339939Ha^2/denominator-0.273171Ha
and0.639197/-0.324691. These are diagram diagnostics, not a measured
low-root Q gap or a universal perturbation-norm bound. The source already
finished when the warnings were reviewed; no child was terminated. Its
energy/state admission is stopped, and no skipped-term table is promoted
or exterior-FCI scalar increment added. A justified retained-space or
resolvent treatment is required before consuming that optional Q class.

Full nuclear-density source QED initialization reaches the original1800s
cap before HF iterations. The distinct original-default finite-RMS QED
model is now profiled separately; Fermi Dirac nuclear potential, static
Breit, electric/magnetic self-energy and Uehling remain explicit. Neither
QED nuclear approximation is relabeled as another model's validation.
Evidence: Sn12_AMBiT_original_V36_control (protocol, source manifest,
preflight admission, original logs/matrices/levels and three audit versions),
Sn12_V36_DC_n4_operator, Sn12_V36_DC_fullM2_levels and
Sn12_V36_native_original_AMBiT_DC_comparison.json under the active root.
