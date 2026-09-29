# Configuration-ordering audit — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The first20 recovered shooting snapshot has 260 observed lines whose computed
configuration-average upper-minus-lower energy is negative. Comparing these
with complete NIST configuration centroids, instead of individual term/J levels,
separates three cases:

| Assessment | Lines |
|---|---:|
| NIST configuration centroids also have negative ordering | 99 |
| NIST configuration manifold incomplete | 159 |
| Complete NIST centroids have opposite ordering | 2 |

The first category includes 54 Ar II lines. Overlapping configuration manifolds
can produce positive individual transitions even when their centroid interval
is negative. This does not establish that the configuration-centroid model
resolves those lines. The second category cannot decide centroid ordering from
the available evaluated records. Every original flag remains visible; the JSON
adds `ordering_assessment` and, when available, the NIST centroid interval.

## Targeted independent Cowan calculation

New RCN/RCN2 runs use the same unfitted HFR controls as the earlier pilot,
with all radial scales equal to one. No RCG calculation was performed for
this panel. All values below are configuration-centroid intervals in cm⁻¹.

| Interval | NIST | Atomx | Bare Cowan, CORRF=0 | Exported Cowan, CORRF=0 | Exported Cowan, CORRF=1 |
|---|---:|---:|---:|---:|---:|
| Mg I 3s3d → 3s4p | 653.982 | −234.387 | −234.838 | 295.131 | 464.370 |
| Al I 3s²4f → 3s²4d | 916.772 | −282.050 | −282.025 | −643.109 | −656.214 |

Both bare-code differences are smaller than the approximately 1.1 cm⁻¹
rounding resolution of Cowan's printed total-energy differences. This is strong
evidence against a shooting-specific explanation for these two ordering errors,
although the codes do not implement identical Hamiltonians. Cowan's approximate
correlation correction fixes Mg's ordering but worsens Al's. There is no basis
for a universal empirical shift or a universal replacement by Cowan energies.

## Correlation model required for Al I

Papoulia, Ekman and Jönsson's MCDHF/RCI study identifies strong interaction
between the 3s²nd ²D series and 3s3p² ²D, distributed over several states.
Their calculation includes valence and core–valence correlation. This supports
testing a multireference even-parity D space, with explicit state mixing and
active-space convergence, against the odd-parity F endpoint. It does not prove
that any one configuration addition will close the present gap.

Primary source: [Extended transition rates and lifetimes in Al I and Al II
from systematic multiconfiguration calculations](https://arxiv.org/html/1808.09478),
especially Sections 1 and 3.1. Neither NIST levels nor classifications were
changed by this audit.

## Reproduction and interface

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/benchmark_cowan_nist.py --panel ordering
```

The ordering panel has its own default directory,
`output/nist_spectra/cowan_ordering`, preserving the original Cowan pilot.
Its manifest retains the raw ledgers, inputs, output hashes, source/binary
hashes and script hash. The Cowan website panel selector exposes these records;
the shooting website summarizes the three ordering categories and retains the
assessment in each record's details. Matching/ordering and parser tests passed;
browser visual inspection remains outstanding.
