# Ca XVI 4f source and discrepancy recheck

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The completed H–Ca shooting comparison's largest absolute residual among
complete evaluated configuration centroids is Ca XVI `1s2 2s2 4f1`:

- Calculated excitation relative to the ground configuration centroid:
  6,078,904.064615503 cm⁻¹.
- Archived NIST centroid interval: 6,371,653.333333333 cm⁻¹.
- Difference: −292,749.26871783007 cm⁻¹.
- Two-grid change: 4.94e-8 cm⁻¹.

The [official NIST Ca XVI level table](https://physics.nist.gov/cgi-bin/ASD/energy1.pl?biblio=1&conf_out=1&j_out=1&lande_out=1&level_out=1&perc_out=1&spectrum=Ca+XVI&term_out=1)
lists both 4f fine-structure levels at 6,396,000 cm⁻¹ and the ground doublet
at 0 and 36,520 cm⁻¹. It identifies Sugar and Corliss (1985) as the primary
data source. These values agree with the local archive. The ground statistical
centroid is 24,346.666666666668 cm⁻¹, giving the interval above.

The largest discrepancy is therefore not explained by a transcription of these
energies or by the measured radial grid change. The historical independent
B-spline ledger differs from shooting by 9.74e-6 Ha for the excited total
energy and 1.05e-5 Ha for the ground total energy; their cancellation gives an
excitation difference of about 0.17 cm⁻¹. That comparison remains historical:
it is not a newly recomputed matched-source validation.

No NIST value has been changed or rejected. Agreement between two numerical
engines does not establish correctness of their shared physical model. Further
work must check the original level assignment/reference, term mixing and
correlated excitation/ionization limits, with independent external calculations.
This record must remain visible as a discrepancy rather than being removed
from the score because it is inconvenient.

## Independent FAC diagnostic

FAC 1.1.5 was run with ground-reference orbital optimization, two configurations
(`1s2 2s2 2p1`, `1s2 2s2 4f1`), point nucleus and Dirac–Coulomb only.
Breit, self energy, vacuum polarization and recoil were explicitly disabled.
The four levels have energies 0, 4.8613208, 758.003587 and 758.066954 eV
relative to the lowest level. Statistical weighting gives a centroid excitation
of 754.798916181 eV = 6,087,863.822321036 cm⁻¹, or −283,789.5110122971 cm⁻¹
relative to the NIST centroid interval.

The calculation finished normally; the final printed optimizer residual was
1.55e-14. Input, output table, optimizer log, binary hash and comparison are
retained in `output/nist_spectra/ca_xvi_fac_source_check/`.

This independent implementation reproduces the scale of the deficit. Its local
central field, frozen reference orbitals, nuclear model and missing corrections
differ from the shooting calculation, so it is not a matched-model benchmark
or proof of an erroneous NIST assignment. Original-reference and correlated
state checks remain required.

## Original compilations inspected

The [1979 calcium compilation](https://srd.nist.gov/jpcrdreprint/1.555609.pdf),
printed page 912 (PDF page 49), has no Ca XVI 4f entry. The
[1985 compilation cited by ASD](https://www.nist.gov/document/jpcrds2vol14pdf),
printed pages 99–100 (PDF pages 102–103), explicitly lists `2s2 4f`,
`2F odd`, J=5/2 and 7/2 jointly at 6,396,000 cm⁻¹. This confirms the
ASD number against the printed source; it is not an archive transcription error.
The source text attributes the 3d–4f doublet to Fawcett and Ridgeley (1981),
J. Phys. B 14, 203–208, DOI 10.1088/0022-3700/14/2/005.

A specific assignment concern now requires checking:

| Printed 1985 entry | Ca XVI | Ca XVIII |
|---|---:|---:|
| 3d J=5/2 level, cm⁻¹ | 4,664,000 | 5,384,200 |
| 4f level, cm⁻¹ | 6,396,000 | 7,115,200 |
| Difference, cm⁻¹ | 1,732,000 | 1,731,000 |

The Ca XVIII entries were visually checked on printed page 102 (PDF page 105).
Both stages cite the same Fawcett–Ridgeley paper. NIST's
[bibliography, entry 5314](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nbsspecialpublication363supp3.pdf)
indexes that paper under Ca XVIII and other Li-like ions, plus P XI–XII;
Ca XVI is not listed. This is suggestive of a cross-stage identification or
compilation problem, **not proof**. The original paper has not yet been
inspected; the ADS PDF endpoint returned HTTP 403. Neither the database values
nor the scoring policy have been changed on this evidence.

Downloaded source PDFs, hashes, visually checked page images and the structured
observations are retained in `output/nist_spectra/ca_xvi_original_references/`.
The lack of a numerical uncertainty on the 4f table row must not be replaced
with an assumed uncertainty from another stage or transition array.

The corresponding **configuration-centroid gaps**, which cancel the ground
configuration entirely, sharpen the diagnostic:

| 3d → 4f centroid gap, cm⁻¹ | Shooting DHF + corrections | NIST | Difference |
|---|---:|---:|---:|
| Ca XVI | 1,425,137.2265 | 1,732,800 | −307,662.7735 |
| Ca XVIII | 1,731,974.2270 | 1,732,200 | −225.7730 |

Thus the large Ca XVI outlier is not caused by subtracting the wrong ground
centroid. The Li-like Ca XVIII interval agrees closely on this scale, while
the two NIST stage intervals are nearly the same. Together with the bibliography
scope, this makes a reference assignment check the leading next action. It does
not justify adjusting DHF parameters to fit the suspect interval or declaring
all remaining DHF discrepancies solved. The source campaign identity and exact
numbers are saved in `cross_stage_centroid_gaps.json` beside the source scans.
