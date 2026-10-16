# Complete signed radial densities and gradients without angular tables

`CIRadialDensity(..., angular_storage='streamed')` now contracts the same exact
DC pair/core algebra directly into radial integral weights. It keeps the
ordered sparse CSF projection and streams the complete connection graph;
it retains neither an angular coefficient table nor a pair-by-state array.
`contract_bilinears(left, right)` returns one independent signed bilinear per
column pair in a single graph traversal. Exact zero alone is removed. No
coefficient threshold, configuration cut, radial cut or positivity replacement
enters these bilinears.

`CSFOrbitalFunctional(..., density_storage='streamed')` explicitly selects that
density representation. Both defaults remain cached. This density option
does not change the CI solver, extend DC gradients to Breit/QED, normalize raw
vectors, or create a full-spectrum CI result from selected roots. The legacy
ensemble `contract` retains a magnitude cutoff; complete derivative consumers
explicitly use unfiltered bilinears and their actual full CI norms.

The [113-test affected panel](test_receipt.json) passes, including21 new tests.
Independent complete Fock creation/annihilation actions, common-core phases,
multiple parentages, reordered CSFs, signed off-diagonal left/right vectors,
tiny coefficients below1e-15, batch work sharing, early cap rejection and
complete native RKB radial gradients are checked. The existing five-UID
forward derivative and public ensemble/response panels also pass. These finite
anchors do not establish Yb experimental accuracy.

## Source-bound physical qualification

The [predeclared contract](contract.json), [execution contract](execution_contract.json)
and [source review](source_review.json) preserve the previous complete model:
native69-electron Yb⁺, Z70/A175 Fermi nucleus, literal accepted13 R0, all31 radial
waves, original positive-energy projectors/metric, unshifted DC and complete
4f12 A173/B182 NR spaces. The five sectors contain44,707 /45,686 target CSFs.
No accepted13 correction is transferred to production terminal16.

The actual358-file source fingerprint is
`f8307f6a107546aac9f528bf8344f127df0584ed352b3acd55678f2d731b49bf`.
Only the radial-density implementation and its B-spline functional constructor
change from the archived [before source](source_before.json).
`source_current.tar.gz` preserves all358 actual dirty-source Python files;
every archived member is checked against the manifest, independently of HEAD.

Qualification first compares every signed radial key and all31×246 gradient
components against cached contractions on the complete original A1595/B2574
models, using the exact owned004 vectors. Full-class selected009 vectors are
then bound by their full signed basis signatures, A nesting, saved root/norm/
residual/overlap and the original radial/nuclear/metric identities. These
checks consume numerical selected-state ancestry; they do not newly verify
the full Hamiltonian residual or certify the omitted spectrum.

For each state, complete density weights reconstruct its owned DC energy.
The directional native gradient is checked against independently differentiated
two-slot ket-normalized one-body and four-slot Coulomb cell integrals. The
direction is the original complete negative preconditioned gradient, carried
through the unchanged projectors; it is not selected from NIST. Absolute-state
and line derivatives remain distinct.

Each once-only case has one CPU/BLAS thread,8 GiB address space and1200 s,
with at most three simultaneous complete-sector workers. The largest graph
has43,720,175 ordered connections and92,715 raw determinants. Complete radial
keys are bounded by384,706 before execution. First scientific failure stops
remaining work and preserves evidence; no gate, step, model or cap changes
follow a failed case. The resource revision4→8 GiB for the matched case was
made before any scientific execution, using the same bounded worker contract.

The original complete A/B qualification passes102/102 gates. Every signed
radial key is identical to its cached reference; all31×246 gradient components
agree to6.40e-14, and independent directional derivatives agree to7.01e-14 Ha.
Owned energy reconstruction differs by at most2.37e-11 Ha (~5.2e-6 cm^-1).
This point-specific arithmetic difference cannot account for the hundreds to
thousands of cm^-1 spectral gaps. The matched run takes158.45 s and937.67 MiB;
it is a correctness/feasibility observation, not a controlled speedup.

The [complete results](RESULTS.md) pass **357/357 live gates**:102 original-model
controls and255 complete-class controls. All five sectors finish once.
The [102-check saved-array review](aggregate001/result.json) also passes.
Complete-sector peak RSS is2,194.773 MiB (~2.143 GiB), with623-647 s per sector.
No angular table or new H/eigensystem is constructed for these full-class
gradient cases. Timing is an unpaired feasibility observation.

The complete five-state EOL uses the original0.2 weights and all31 waves.
Its original-metric preconditioned norms are0.189697/0.244299 for A/B, far
above1e-7. These fixed-point diagnostics do not qualify stationarity and do
not update the original terminal16 result. The forward integral comparison
differentiates fixed saved CI vectors; it does not newly check finite-step
CI response or replace a rediagonalized finite difference.

For all complete sectors, the new signed gradients agree with independent
two-/four-slot radial differentiation within7.91e-14 Ha. One-body and Coulomb
directional contributions are about-42.6/+42.5 Ha, leaving about-0.17 Ha.
Along the same original direction, D3/2 line derivatives change from
-8.865e-4/-7.595e-4 to-2.236e-4/-1.362e-4 Ha for A/B; D5/2 changes from
-2.213e-3/-2.135e-3 to-1.288e-3/-1.277e-3 Ha. These are local responses of
different model spaces, not finite relaxed shifts, improvements against NIST,
or evidence that the old direction is a new-model optimizer solution.

The matched case constructs its cached references before instrumenting the
streamed graph. Its five counted graph traversals refer to the streamed
phase; cached-reference construction adds work. The complete-class cases
construct no cached reference or angular table.

The [three failed ordinary line-FD gates](../physical_next_decision005/STATUS.md)
remain failed. Analytic density/gradient checks do not supersede those tests
or admit an optimizer continuation. Stationarity, radial/correlation
convergence, experimental accuracy, full-spectrum ownership and the full-element
database remain separate unfinished work.
