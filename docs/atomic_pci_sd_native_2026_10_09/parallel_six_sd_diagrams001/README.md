Actual native signed SD diagram consumer, bounded finite anchor

`src/atomx/methods/sd_pair_diagrams.py` now consumes generic certified J/parity
interaction matrices for the complete scalar core doubles source, every
particle-particle ladder and every normal-ordered hole-hole ladder. This is
an actual SD subgraph, with full source and homogeneous coordinates. It is
not the complete native all-order SD map/solve, a valence map, or Sigma2.

The reference is a closed 1s2+2s2 determinant on fixed Be2+ DC orbitals, with
complete 3s,2p1/2,2p3/2 virtual multiplets: 12 magnetic spinors, 4 occupied,
10 scalar doubles, and all 495 four-electron determinants for the independent
Fermi-action oracle. This fixed reference is noncanonical and is not a newly
optimized four-electron DF solution. DC/static/dynamic use the same orbital
owner; dynamic uses the complete archived DC Fock energy differences.

For normalized antisymmetric pair creation operators A†(ab;JM), coordinates
are `sum_M A†(virtual;JM) A(occupied;JM)|core>/sqrt(2J+1)`. The source is
`sqrt(2J+1)*VJ[v,o]`; the complete ladders are `VJ[v,v] @ t` and
`t @ VJ[o,o]`. Linked doubles energy is the signed complex `source†t`.
No Coulomb electric C division or Coulomb-specific parity pruning occurs.
Forbidden coordinates are excluded only by exact J/parity/Pauli symmetry.

The independent oracle uses the signed public spinor ERI, complete magnetic
indices and elementary Fermi signs. It projects explicit all-virtual V for
PP. HH explicitly constructs core-normal-ordered Voo as `Voo - W + Eoo`,
where `W[p,q]=sum_i g[p,i,q,i]` and `Eoo=.5*sum_ij g[i,j,i,j]`, then evaluates
the full linked commutator. No metric renormalization, scalar fitting,
frozen-core transverse replacement or threshold adjustment is used.
The already independently validated pair angular owner is reused; the new
SD contraction is checked independently of its J block contractions.

Fock, h_eff, frozen scalar and reference energy remain in the original
certified owner, unchanged. HH normal ordering is internal to its subgraph;
it does not alter Fock/h or add a second frozen/reference contribution.
Remaining Fock/singles, mixed-rank/ring and other core diagrams must consume
their full signed interaction consistently before a complete SD solve may
be claimed. The same holds for all valence and Sigma2 maps.

pCI reference is archived under `reference/`: allcore-ci.f `term1` (lines
1828ff) contains its occupied-pair ladder, and `term21` (3211ff) its virtual
pair ladder. The new normalized-pair formulation is independently derived,
not a partial modification of pCI's Coulomb RINT/angular kernel.

Whole-diagram allocation/work is planned before block copies. J blocks are
reused for each occupied-pair column via matrix products; no quartic magnetic
amplitude tensor is allocated. Dense radial-pair integral blocks can still
be too large; the cap rejects them. This finite proof does not certify a
large production all-SD scaling result.

Replay: `.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_six_sd_diagrams001/probe.py --output <fresh-directory>`.
Hard limits: one CPU/BLAS worker, 4 GiB address space, 120 CPU seconds.
`inputs.npz` contains all signed h/F/g/constants and all basis and signed
amplitudes, plus original physical radial density inputs. Each run copies this
input, archives every loaded scientific source and reference source, and
preserves all arrays and original gates in result.json. Earlier runs and the
original contract remain. Current contract metadata explicitly enumerates
the same derived model dimensions for subsequent replay.

Original gates: all complete signed diagram arrays at 1e-11 Ha, scalar
metric at 1e-12. `tests/test_sd_pair_diagrams.py` also checks complete complex
multiplet phases, nonzero frozen/reference scalar covariance, immutable
outputs and whole-payload refusal. Highest-level certification remains false.
