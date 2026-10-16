# Closed header failure 001

The single admitted execution closed exit1 during `field_header:mixing`. The driver assumed C layout for every field; the original five-by-five CI mixing array has Fortran layout. This ignored field was header-only. No NPZ array body was decoded and no Fock, Dirac, SCF, CI or optimizer producer ran. Source/input/nucleus and the preceding four field headers passed. `actual/failure.json`, `actual/partial.json` and `execution.log` retain the failure unchanged.

The independent successor `../yb_high_l002/` declares all20 header layouts, established by stdlib ZIP/NPY-header reading with zero numerical imports/array-body reads. It changes only the exact header representation contract and provenance wording. Scientific fields, producers, gates, frozen model and resources remain unchanged; this directory is never reused or upgraded to pass.

Provenance scope: historical metadata marks the19 occupied nuclear orbitals asDHF; original driver bytes are not fully archived. Any new qualification concerns the owned frozen bank/currentV^N field, not a fresh stationary SCF atom or a high-l spectral response.
