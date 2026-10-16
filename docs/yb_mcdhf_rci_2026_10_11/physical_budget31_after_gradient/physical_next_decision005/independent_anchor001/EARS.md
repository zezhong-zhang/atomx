# Independent five-UID derivative anchor

Expectation: distinguish a derivative sign/factor/normalization defect from
the unresolved Yb finite-difference problem, using one small exact-CSF point.
No Yb run, cutoff scan, optimizer or production-source edit. Parent authorized
only the new `tests/test_mcdhf_state_resolved_derivative.py` shared test.

Action: Z4/3e, RKB ns35/nsp34/nsq35,8CSFs/7radial waves, five equal-weight UIDs
(1,+,0)/(3,+,0)/(5,+,0)/(1,−,0)/(3,−,0). Native-S/no-pair tangent. Forward dH
uses the raw ket-normalized two-slot one-body derivative and direct four-slot
Slater multilinearity, followed by exact-CSF forward assembly. The reverse
RDM/gradient route retains complete signed CI-coefficient norms cᵀc using
exact binary64 Fractions, then represents each nonzero coefficient once.
This independently tests derivative algebra, not an independent angular theory
or Coulomb implementation: those native physical kernels are intentionally shared.

Result:4/4 new tests pass in1.45s, CPU26/thread1/6GiB, source unchanged.
Each UID's one-/two-body directional mismatch≤2.924e−14 Ha. EOL forward
−0.3612418205132421 versus fullnorm gradient−0.36124182051310916. Signed
per-state scalings(2,.5,−1.7,3.1,−.3) change the complete gradient by≤4.17e−17.
One central rediagonalized polar FD at2e−5 differs by−2.3214e−8 Ha. Discarding
CI amplitude signs fails by−8.7754e−4 Ha, so the interference control is active.
The scaled vectors are a normalization representation control; they are not
passed to the native solver as purported normalized CI results.

Significance: no derivative factor/sign defect found at this small point.
Current/frozenf043 ASTs match for gradient_from_weights,energy_for_weights,
CIRadialDensity.contract/contract_bilinear. The raw polynomial gradient's
one-body ambient normal extension differs from the ket-normalized functional;
their exact native-S tangent derivatives coincide. Existing polar/reuse/stable-
energy tests also passed30/30 in1.23s before this new anchor. Yb stationarity,
original FD failures, large-space arithmetic and optimizer admission remain
unqualified. Parent owns current testing-guide/doc-impact/EARS integration;
this directory holds only the small executable anchor, actual result and receipt.
