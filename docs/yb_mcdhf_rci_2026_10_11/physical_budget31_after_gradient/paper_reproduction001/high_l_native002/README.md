# Native high-l capability pilot

This pilot qualifies existing Atomx radial, signed-CSF, CI and ownership APIs.
It does not reproduce Sahoo2025 electron affinities or certify Yb accuracy.
The bound [contract](contract.json) separates this current dirty source epoch
from the archived f043 Yb model. No shared scientific source is changed.

The hypothesis is that high angular momentum is already accepted by the native
operators; its absence from the old 31-orbital Yb bank is a model choice rather
than a parser or radial-engine limit. Source alone does not establish that claim.

The independent point-nucleus Z=10 control retains 5g, 6h and 7i, with both
κ partners for every shell. Six genuine native generalized Dirac solves share
one order-9 RKB basis: P/Q dimensions 102/103, 199 constrained coordinates,
requested radius 75 Bohr and actual native boundary 75.2351647362715 Bohr.
Sommerfeld energies and paired fine-structure splits are analytic anchors.
The native overlap metric, Gauss quadrature, electron branch and removed
boundary coefficients are checked separately. No SCF result is invented.

The first complete declared CI contains all six one-electron CSFs, spanning
66 magnetic states. A separate frozen-orbital two-electron 5g control retains
both κ=4 and −5: all 153 magnetic determinants and 17 coupled CSFs. A third
CI repeats exactly that model after a full signed κ=−5 orbital phase change.
The analytic DHD phase rule tests interference and the complete spectrum;
all ranks through the actually allowed g-channel rank 8 remain available.
This highly excited finite g pool is not a heliumlike ground state or resonance.

`CIRadialSnapshot` supplies owned immutable banks, and public native CI builds
retain genuine `CIHamiltonianSnapshot` owners. An explicit native `RadialState`
references the immutable six-wave bank, with no invented occupied density.
Every CI checks the original full H residual, full Gram matrix, trace, complete
multiplet count and actual operator/input bindings. No small-term pruning,
external backend, orbital optimizer, Breit, QED or recoil is introduced.

The run admits CPU 28 only, all numerical threads 1, 4 GiB address space,
120 s inner and 135 s outer time caps; host load and available memory are
checked before numerical imports. Prospective raw h/S storage is 4,073,856 bytes;
the three complete CI matrices need 4,912 bytes. The conservative working-set
estimate is 1 GiB. Producer-entry caps and native radial-cache diagnostics
record work without treating imported modules as executed producers.

Run [run.py](run.py) with `/usr/bin/python3 -I` for a source-only preparation.
Scientific execution uses the repository `.venv`, serial thread variables,
`taskset -c 28` and `timeout 135s`, followed by `--execute`.
The first failure is preserved; an existing `actual` directory rejects a rerun.
Complete waves, geometry, h/S and all three full CI eigensystems are saved.

The next Yb step must keep the original nucleus/geometry and freeze an actually
converged closed-core owner before building both κ partners via the existing
`build_virtual_basis` or explicit MCDHF spectator-orbital policy. Full source,
positive-energy projector, occupied metric-orthogonality, retained energy band,
determinants and memory must be admitted before expanding CI. Selecting one
root per channel here does not authorize dropping paper orbitals or excitations.
No Yb high-l calculation or paper-level correlation model is admitted by this pilot.

The predecessor [001 source failure](../high_l_native001/preexecution_failure.json)
stopped before numerical imports: an unrelated parallel EOM source changed.
This explicit successor keeps every physical model, budget and numerical gate;
356 current package sources remain bound, and importing the excluded EOM module
rejects before the first producer. All actually loaded Atomx sources are checked.

Documentation impact: this isolated capability record only; root owns integration,
current guides and EARS. Existing high-l tests and native APIs remain unchanged.
