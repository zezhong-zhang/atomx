# Literature-to-implementation map

The papers and GRASP source below guide native Atomx development. No external
atomic-structure backend is run and no paper's numerical accuracy or speedup is
assigned to Atomx. Current source/input/model and the old failed gates remain
separate from literature evidence.

## Variational response

[X. Gonze, Physical Review A52,1086–1095 (1995),
doi:10.1103/PhysRevA.52.1086](https://doi.org/10.1103/PhysRevA.52.1086)
establishes variation–perturbation relations, including the2n+1 theorem with
constraints. First-order wavefunction information can therefore determine
third-order variational energy terms. The publisher abstract was checked;
the attempted publisher fulltext endpoint was inaccessible. The following
finite-real-CI specialization is our explicit derivation, independently tested,
not a claimed transcription of an unread equation from that paper.

Write H(t)=H0+tH1+t²H2/2+t³H3/6+… and normalize c(t). At t=0 choose
c.T c=1 and c.T c1=0. Differentiating Hc=Ec gives

```
E1 = c.T H1 c
Q(H0-E0)Q c1 = -Q H1 c,  Q = I-c c.T
E2 = c.T H2 c + 2 c1.T H1 c
E3/6 = c.T H3 c/6 + c1.T H2 c + c1.T(H1-E1) c1
```

The last two terms define B_CI. Hence the central derivative of the difference
between the optimized-CI energy and the fixed-c expectation has leading term
h² B_CI. H1/H2 are actual derivatives, not Taylor coefficients. H3 is not
calculated by the present response consumer; B_CI is not the whole central-FD
truncation term or a bound on its higher orders.

`numerics/variational_response.py` implements the complete finite-spectrum
solution of this projected equation, retaining every external-state denominator
and signed coupling. Degenerate/partial spectra reject. The physical consumer
binds basis, state, operator and parameter units. This finite dense diagnostic
does not provide a full4f12 matrix-free response solver or MCDHF Hessian.

## MCDHF orbital and CI separation

[C. Froese Fischer et al., Journal of Physics B49,182004 (2016),
doi:10.1088/0953-4075/49/18/182004](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=920054)
reviews variational multiconfiguration orbital optimization, multireference
single/double expansions and systematic convergence. Sections3.3 and4 discuss
orbital stationarity and the relativistic extension; section4.6 distinguishes
extended optimal-level and extended average-level treatments. This supports
keeping the weighted full-CI orbital objective and its physical stationary
conditions explicit, rather than inferring stationarity from stable energies.

The present diagnostic keeps the original five targets, equal weights,
projectors and metric. Polar-path derivatives include constraint acceleration;
they do not optimize or change the objective. The tiny difference between the
literal stored R0 and the mathematical polar path's zero point is measured,
and a fresh H0 is built instead of silently assigning old Hamiltonian ownership.

## GRASP source and efficient contractions

[Y. Li et al., Atoms11,12 (2023),
doi:10.3390/atoms11010012](https://strathprints.strath.ac.uk/84053/1/Li_etal_atoms_2023_Performance_tests_and_improvements_on_the_rmcdhf_and_rci_programs_of_GRASP.pdf)
separates CI diagonalization, orbital updates, angular coefficients, potentials
and Lagrange multipliers in performance analysis. Section3.2/Algorithm2
reorganizes coefficient and potential construction; repeated scalar searches
and repeated coefficient work are avoidable costs. This is a guide to
contraction structure, not evidence of an Atomx performance factor.

The official [CompAS GRASP repository](https://github.com/compas/grasp) was read
at commit9006157730a82ac839f2b4ff4e938bcba63a539e. Saved reference files and
hashes are listed in [grasp_source_reference.json](grasp_source_reference.json).
In rmcdhf90_mem/scf.f90, lines149–169 call MATRIX/NEWCO before SETLAG and
IMPROV; lines215–216 update the CI again. SETCOF contracts signed density
coefficients through DSUBRS. These files document that implementation's
scheduling; we do not claim they are the paper's newer FD variant.

Atomx reuses the existing native signed angular machinery, B-spline cell kernel,
raw Dirac/nuclear matrices and pair-potential construction. Its radial jet
consumer batches orders0–2 of each source/target density and includes all
one-body quotient terms and two-/four-slot cross terms. There is no imported
GRASP solver, dropped correlation class, fitted energy correction or copied
Fortran implementation in the new Python routines.
