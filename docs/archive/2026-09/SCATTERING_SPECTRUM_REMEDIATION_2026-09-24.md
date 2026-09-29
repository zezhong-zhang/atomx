# Scattering and spectrum remediation — 2026-09-24

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

This follows the [pre-fix audit](SCATTERING_SPECTRUM_REVIEW_2026-09-24.md) of atomx `f8e2827`. The original audit, its raw FAC output, and defect probes remain historical evidence. The current behavior is gated by `tests/test_scattering_spectrum_review_fixes.py` and the affected family tests. The density workflow still defaults to CI; statistical density averaging remains an explicit option.

| Audit finding | Current behavior |
|---|---|
| 1. Polarization normalization | Combined ±M populations are counted once. E1 analyzer intensities are derived from upper/lower J dipole amplitudes and satisfy isotropy and Stokes bounds. Other multipoles are rejected rather than inferred from rank-two alignment. |
| 2. Silent Auger failure | A failed continuum solve raises a contextual error in the determinant, jj-level, and configuration-average routes. A forbidden angular channel still contributes zero. |
| 3. Inconsistent R-matrix products | The optional empirical pole correction now recomputes S, T, cross sections, and collision strengths from the final K. The returned provenance identifies this as an empirical resonance model. |
| 4. Ignored CI spectrum selector | `line_resolution="ci"` requires a solved closed-shell result, generates supported configuration-pair line strengths, and records each group. Unsupported pairs and transitions without configuration labels retain their CRM lines with `ci_fallbacks` reasons. In a J-resolved `from_atom` CRM, replacing solved level populations by a statistical redistribution would be wrong, so those lines are retained explicitly. |
| 5. Wrong lifetime width | The damping rate includes all radiative branches and available autoionization loss of both levels. The optional Lorentzian Stark HWHM adds linearly. This remains a specified Lorentzian model, not a quasi-static microfield calculation. |
| 6. Optical weights mistaken for collision substates | Electron-impact M weights must be supplied from a collision calculation. The previous transverse-photon formula is available only as the named `photoabsorption` geometry; forbidden E1 triangles raise. Atomx does not yet compute FAC-style CETableMSub collision tensors. |
| 7. Circular resonance validation | The pole/fitter test is labeled empirical and no longer claims independent Burke resonance accuracy. A correlation-converged ab initio resonance benchmark is still required. |
| 8. RR Milne phase space | Both ordinary and Rydberg RR paths use relativistic electron momentum squared, `E*(2+alpha**2*E)`. |
| 9. EII hybrid seam | The default DW/BEB hybrid has a continuously differentiable blend over its switch interval and records the approximation. It is not FAC's Born completion; model error outside the seam still needs separate validation. |
| 10. Sampling-dependent line widths | Gaussian, Lorentzian, and narrow Voigt profiles use physical widths and bin-integrated deposition; narrow UTA arrays have no artificial grid-width floor. The grid approximates a finite-volume spectrum and should be treated as immutable once lines are added. |
| 11. Mixed DW orbital meshes | Each bound orbital is evaluated on its own B-spline mesh before interpolation. |
| 12. NaN free-free tail | The Born Gaunt factor uses scaled `k0e`, remaining finite when emissivity underflows to zero. |

The review also exposed a DR Hamiltonian limitation. The high-level B-spline resonance and automatic Ne-like `dr33` paths now construct a bare nuclear Dirac one-body matrix, fold inactive-core direct/exchange terms into it, and add active-active interactions once. Magnetic-projection pruning avoids constructing impossible determinant products. Low-level direct `ResonanceManifold` construction without `one_body` still uses approximate dressed orbital energies; callers requiring a first-principles active-space Hamiltonian must supply the core-folded matrix. The automatic DR route remains limited to supported Ne-like manifolds.

Against the **unchanged** clean-room FAC `dr33` reference column, recomputed atomx/FAC rates at 1, 5, 10, and 50 MK are 1.09046, 1.00740, 0.99458, and 0.98400 (4.62% RMS relative error). This is a four-temperature comparison for that sequence, not a general DR accuracy claim. The refreshed PI/RR panel retains its original FAC reference column; the largest absolute relative discrepancy among its 24 rows is 17.41%. The changed atomx values are implementation snapshots. The separate live FAC DW-excitation audit remains in the pre-fix report.

For 2,000 Gaussian lines on 10,001 wavelength points, warmed rendering measured 13.0 ms after localized finite-volume deposition versus 96.45 ms on the audited revision, about 7.4× faster on that workload. Both spectra conserve integrated power to approximately `1e-15` relative error. Voigt profiles retain algebraic wings and use a separate integration path; this Gaussian figure does not characterize Voigt throughput.

The fixes constrain observed errors, but the package is not certified for every ion or plasma regime. Independent convergence/reference work remains for open-shell level-resolved excitation, electron-impact magnetic sublevels, R-matrix resonance positions, arbitrary DR manifolds, and microfield Stark profiles. Those are missing physical models or validation domains, rather than defects that a local numerical patch can honestly declare solved.
