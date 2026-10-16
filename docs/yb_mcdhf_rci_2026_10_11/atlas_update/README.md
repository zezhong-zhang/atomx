# Yb⁺ MCDHF → RCI atlas update preparation

The existing periodic-table atlas now has a locally prepared Yb II comparison with eight source-bound models: native005 five-reference MCDHF Dirac–Coulomb and subsequent static-Breit RCI, followed by the 294, 468 and 531-CSF balanced-family fixed-orbital RCI spaces under both operators. All 32 added excitation rows, multiplet centroids/splittings, actual lower/upper energy changes and physical bare-character connections are copied from the frozen native results. The original two SD/dynamic models and eight historical rows remain available.

This preparation did not commit, push, deploy, register a project, alter access, or read credentials. Root owns publication to the existing site after checking the frozen [handoff manifest](handoff_manifest.json). The hosting manifest and all 6019 original NIST gzip files remain byte-identical.

## Scientific boundary

The original five-reference native005 MCDHF solution passed its unchanged numerical stationarity gates in 68 steps: original-metric orbital residual 5.143009363125294e−8 < 1e−7. This certifies that DC five-CSF optimization objective and its numerical solution. Its static-Breit RCI uses those optimized orbitals; it does not claim static-Breit orbital optimization.

The three larger spaces use the same 24 native005 radial orbitals and 69 electrons. Their complete signed fixed-orbital RCI results are conditional physical discriminators, not converged expanded-space MCDHF, full core–valence correlation, or experimental certification. No other backend or fitted correction supplies these results. The original Fermi A175 empirical rms nuclear model is retained; isotope-specific radius/mass has not been certified.

The final 531 static-Breit excitation energies, in D3/2, D5/2, P1/2, P3/2 order, are 25043.731646629, 25635.584202426, 25034.531130641 and 28263.092182802 cm⁻¹. Their uniquely matched local ASD references are 22960.80, 24332.69, 27061.82 and 30392.23 cm⁻¹, so all four fail the unchanged nominal 1% comparison. Final D/P splittings are 591.852555797 / 3228.561052161 cm⁻¹, compared with ASD 1371.89 / 3330.41 cm⁻¹. Relative comparison is not an uncertainty-qualified experimental certificate.

The latest 531 mixed-CI DC EOL probe is presented as one separate negative-status paragraph. Its original-metric residual is 0.036760718407053486; one accepted orbital step gives 0.024993298926343503, both above 1e−7. That step lowers the EOL average by 0.0002646866469149245 Ha and worsens every nominal DC experimental gap. Its diagnostic spectrum is not added as a ninth model and never replaces or corrects the fixed005 static-Breit spectra. Stationarity for the intermediate 294/468 objectives remains unassessed; no expanded model is certified.

## State identity and preserved corpus

[prepare.py](prepare.py) reads the retained `70-1.json.gz` and independently matches evaluated, finite, unflagged, unbracketed, unambiguous ASD entries by normalized configuration, 2J and parity using the frozen native005 normalizer and existing exporter record helper. It does not rely on old row positions or re-export/recompress the corpus. The unique `[Xe]4f14 6s` 2J=1, even ground is the common zero, ASD ID `070002.000001`. The four upper IDs are `070002.000009`, `.000010`, `.000012` and `.000015`. Missing ASD uncertainties remain null, not zero. Saved physical CI target binding supplies the calculated state identity independently of global eigenvalue-column order.

The 118 elements, 7021 species and existing 119394 level / 296998 line / 6019 ionization records stay unchanged. Only `dist/index.html` changes among the original 6022 tracked files. Ten new exact JSON evidence copies and a separate current-model overlay are added. The loader fetches the overlay only for Yb II, keeps immutable cached NIST payloads, and reports an explicit current-overlay failure while preserving historical models. Level/line filters, pagination, search, missing-data state and mobile access continue to work.

## Source chain and verification

- [contract.json](contract.json), [prechange_manifest.json](prechange_manifest.json), [prepared_stage002.json](prepared_stage002.json) and [prepared.json](prepared.json) bind the original site HEAD, old corpus and initial scientific evidence. The actual card-delivery receipt directly binds stage002 and baseline; stage002 inputs bind the balanced result and independent verification transitively.
- [contract_addendum_stage003.json](contract_addendum_stage003.json) preceded the display amendment. [amend_probe.py](amend_probe.py) checks the stage003/probe/card/independent chain, copies the exact JSON and explicitly verifies that all eight row/multiplet ledgers remain unchanged.
- [data_verification.json](data_verification.json): **6685 checks, all passed**, including every original gzip byte hash, immutable hosting/index, all exact scientific copies, unique physical ASD rematching, native energies/absolute energies/bare weights/deltas, unchanged gates, full centroid/splitting ledgers and failed mixed-CI status.
- [browser_result.json](browser_result.json): **29 checks, all passed**, local existing cached Playwright Chromium. Desktop/mobile, 10 models and 40 rows, latest actual energies, bounds, source links, exact download, cache revisit without duplication, original SD/dynamic rows, Ne observed-line filtering, 349-level Yb pagination, periodic search, missing species, viewport fit and explicit failed-overlay fallback were exercised. No JavaScript errors occurred. [desktop.png](desktop.png) and [mobile.png](mobile.png) record the inspected previews.
- Node syntax validation and Ruff for the four preparation/check scripts passed. Site `git diff --check` passed. These are data/presentation validations, not new scientific computations or performance measurements.

The retained [prepare_failure001.json](prepare_failure001.json) records an incorrect initial assumption that delivery directly bound the balanced sources; this was corrected to the actual transitive chain before any site mutation. The first [browser_failure001.json](browser_failure001.json) used JSON asset filenames as visible source labels; canonical source paths are the real visible labels, and the corrected check passes. Neither failure was repaired by altering scientific results or thresholds.

## Publication handoff

Site checkout: `/home/zzhang/.local/share/atomx-nist-atlas/site`. Existing pre-update HEAD: `d987d8e02b7395bb8c16afae7c10b7c5eb726dd6`. Hosting project identity remains the existing `.openai/hosting.json`, with `dist` as its static directory; keep the current sole-owner/private access scope. The local preview used only a temporary loopback HTTP server; stopping that owned process does not publish anything.

Root can package the existing `dist` directory with the Sites source helper after verifying [handoff_manifest.json](handoff_manifest.json). It contains **40,079,382 uncompressed static bytes**, safely below the skill's 256 MiB source limit; do not run the NIST exporter again. Include the changed page, overlay and new evidence assets, with all existing gzip files. Root-owned `root_push_source.py` and deployment receipts are deliberately outside this preparation's manifest. Publication requires its own remote receipt/readback; this local record does not claim a new live version.

## Documentation impact and scoped lesson

Type: additive current presentation update. Dependencies: immutable native005 and balanced replay results, the latest mixed-CI diagnostic, original local ASD, the existing static atlas renderer and existing Sites hosting. No Atomx public source, method, physical model, gate, NIST archive or hosting policy changed. Existing atlas deployment history is retained as history; root owns consolidated navigation and live-status updates. EARS is recorded in ignored `trace.md`.

No new global insight is required. This increment applies existing rules: a smaller-space MCDHF certificate must not promote a larger CI space, physical state rematching must survive root reordering, failed probes remain explicit, and presentation appendages must preserve original reference bytes. Trigger → when a later diagnostic changes certification status, append its exact source chain and update the current status separately; do not splice its unmatched operator/iteration into a previously frozen spectrum.
