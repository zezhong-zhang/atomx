# Same-orbital Yb II physical correction budget

This budget uses **only** native MCDHF run003's saved 24 radial spinors,
five CI states, A175 Fermi nucleus, Gauss workspace, and frozen 353-file Atomx
source. All calculations consume Atomx public physical operators. No orbitals,
states, SD results, external solver, or experimental fit are substituted.

The [contract](contract.json) was declared before execution. The
[actual calculation](same_orbital001/result.json) took 7.72 s and peaked at
779.56 MiB with one thread and hard limits of 4 GiB / 900 s. Timing is
uncontrolled. DC and static-Breit roots replay the saved arrays exactly;
finite-frequency transverse exchange with a zero energy map reproduces
static Breit exactly. Nine numeric consistency gates passed.

These are **diagnostic corrections**, not a certified spectrum. Original
orbital stationarity still fails: residual `2.6703e-6` against `1e-7`.
The original `1e-8` Ha estimated-energy-error and 1% experimental targets
remain unchanged. Corrections do not waive that failed stationarity gate.

Values below are correction differences between each upper state and the
same `6s₁/₂` lower state, in cm⁻¹. Large common core shifts are retained in
the ledger but cancel from these transition differences.

| Upper state | Current DC−NIST gap | Static Breit | Dynamic−static | Uehling model | WK model | SE overlap model | SE local model | NMS | SMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5d₃/₂ | +766.93 | +29.56 | −0.654 | +14.808 | −0.525 | +329.66 | −54.97 | +0.415 | −0.375 |
| 5d₅/₂ | −92.93 | −14.21 | −1.642 | +14.810 | −0.525 | +393.68 | −51.33 | +0.350 | −0.338 |
| 6p₁/₂ | −2042.56 | −35.67 | −0.372 | +14.099 | −0.495 | +309.68 | −51.32 | −0.303 | +0.296 |
| 6p₃/₂ | −2527.81 | −65.35 | −4.227 | +14.751 | −0.522 | +401.48 | −48.46 | −0.333 | +0.326 |

Dynamic Breit **replaces** static Breit; its total difference is separately
stored. Frequencies are the 24 diagonal EOL Lagrange energies reconstructed
from the saved same-state gradient and occupations, fixed throughout CI.
Each matrix element averages the two `|ε_a−ε_c|` vertex frequencies in the
real Coulomb-gauge one-photon operator. This omits finite-frequency orbital
relaxation, absorptive exchange, and higher-order QED.

The measured transverse increment is 0.37–4.23 cm⁻¹ on these transitions.
Uehling is about 14–15 cm⁻¹ and the local WK model about −0.5 cm⁻¹. The public
point-Coulomb Breit recoil consumers retain NMS and SMS separately: their
sum is below 0.041 cm⁻¹ here. The mass is the explicitly declared public
`175 × m_p/m_e` proxy, not a measured Yb isotope mass. Finite-size, radiative,
and higher-order recoil are absent.

**Self-energy coverage is a material physical gap.** The current default
Indelicato/overlap model gives +310–401 cm⁻¹ transition shifts, while the
local Shabaev model gives approximately −48 to −55 cm⁻¹. Their differences
are 361–450 cm⁻¹; this is model disagreement, **not** an uncertainty bound.
For the same `6s` spinor, the first model gives `−0.001588574` Ha and the
local model `+0.000244060` Ha. The default model warns that `6s`, `6p₁/₂`,
`6p₃/₂`, and both `4f` spinors fall outside tabulated coverage and use a
leading `Zα` estimate without finite-Z remainder or screened many-body terms.
At `Zα≈0.51`, this is not a precision all-order radiative calculation.
The requested public full nonlocal Shabaev operator raises
`NotImplementedError`; that missing operator is not replaced by a local
model and added to a supposedly complete spectrum.

Uehling/WK are point-source/local models applied to the unchanged Fermi
spinors; no finite-charge convolution or screened many-body VP is supplied.
The SE table model is diagonal in radial labels, while the local model
retains same-κ matrix coherence. Neither establishes complete screened SE.
All unsupported coverage is enumerated in the result.

The original prospective contract repeated an outdated local-SE class
docstring and incorrectly described the d channels as absent. Inspection
of the actual frozen `_reference_state_for_kappa` and nonzero saved d matrix
elements shows that `κ∈{−1,+1,−2,+2,−3}` is supported; f references and the
nonlocal projector remain absent. The
[coverage correction](same_orbital001/coverage_correction.json) records this
explicitly, retaining the original contract, raw results, and unchanged
array hash. Numeric gates pass; that original coverage description failed.
The [original executed script](same_orbital001/run_source.py) is preserved;
the current script differs only by import formatting/removing unused imports.

The evidence supports a specific order of work: finish the unchanged
stationarity gate; implement and test the missing within-symmetry
core-valence/core-hole correlation in this five-CSF baseline; improve SE
operator/table coverage and screening with an independent physics anchor.
Static Breit remains necessary. Dynamic transverse exchange, local VP and
recoil have now been measured on the same states and can be prioritized by
their actual differential effect. No claim that “QED is always small” or
that bigger radial cutoffs alone close the experimental gap is justified.

The complete [state ledger](same_orbital001/result.json),
[saved matrices and state corrections](same_orbital001/correction_arrays.npz),
[frequency map](same_orbital001/orbital_frequency_energies.json), and
[verification](verification.json) provide replayable numerical evidence.
The verification contains 125 passing array-ledger, source/input, calculation-AST,
and documentation checks. These check the archived evidence rather than establish
operator accuracy. Replay them without rerunning the physical solver:

```sh
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/correction_budget/verify.py
```
