# Actual complete-sector equivalence and independent saved-array audit

The root-approved successor invocation completed successfully. Its
[50 live checks](result.json) and
[1795 independent saved-array checks](saved_array_audit/verification.json)
passed. The independent audit performed no new Hamiltonian construction,
CI diagonalization, gradient evaluation, optimizer call, or Atomx import.
Preparation artifacts remain unchanged.

At the exact run002 accepted0160 point, the complete five-sector 261-CSF
representation reproduces the five target sectors of the full 531-CSF
model. Principal Hamiltonian blocks, complete retained-sector spectra,
five signed target vectors, EOL energy, coherent radial RDM, full 24-key
radial gradient, original tangent gradient, original preconditioned
gradient, residual, and energy indicator all agree with zero recorded
difference. The retained-to-omitted scalar-H coupling is exactly zero.
All 24 radial keys and all 2856 coordinates are preserved.

The five physical minor-overlap squared singular values differ from one
by at most 2.9754e-14. The independent coefficient-span check gives at most
4.88498e-15 deviation. Each actual target is isolated, with comparison
root [0]; the full near-degenerate-span selection rule was verified for
these saved spectra, but a genuinely degenerate target was not present
in this measurement. Signed vector and projector comparisons preserve
internal relative phases rather than comparing squared amplitudes alone.

| Complete target sector | CSFs | Sector minus full transition (Ha) |
| --- | --- | --- |
| 2J=1, even (lower) | 38 | lower reference |
| 2J=3, even | 59 | 0 |
| 2J=5, even | 62 | 0 |
| 2J=1, odd | 39 | 0 |
| 2J=3, odd | 63 | 0 |

The original projector and inverse blocks are byte-identical to the
original004 metric archive. The independent tangent and inverse-metric
array replay recovers the same residual, 4.717047646600387e-5, and energy
indicator, 6.8897239137767995e-9 Ha, for both representations. The residual
still fails 1e-7; the indicator passes 1e-8. This is representation
equivalence at a nonstationary finite-model point, not a new MCDHF
stationarity certificate or an experimental-gap improvement.

The audit independently checks every saved eigenpair's residual,
orthogonality and complete J/parity support; reconstructs local-root
mapping from basis rows and signed coefficients; verifies full versus
sector coherent CI density and the saved signed radial-RDM union; and
replays the original metric from saved arrays. Its physical radial
minor-overlap values and radial S-Gram certificate are explicitly
inherited from the bound live run and 988-check endpoint proof. They
were not re-contracted and no new Dirac/S matrix was constructed.

The successful invocation used two complete CI/gradient evaluations and
two explicit verification H builds. It recorded 7.82564 seconds and
836.97 MiB peak RSS for the combined uncontrolled diagnostic. Those
numbers establish neither an end-to-end speedup nor parallel efficiency.
The preceding failed invocation consumed another two CI/gradient calls
and two explicit H builds; its failure and resource/authorization records
are retained unchanged and are not relabelled as successful.

| Result artifact | SHA-256 |
| --- | --- |
| result.json | b4270213674272d5eb28c79670bdf872a0d9d7e4721e03c33220dcf8d2b4e8a4 |
| equivalence_arrays.npz | 48281f3515e535aac4993f3ed4395107cb1c4816fffed75cc5ba809798a4a927 |
| execution_binding.json | 4f61df585d1da202f2d4cefd114f5570b3cba53b1830cae52a3d56b1d6b620b8 |
| saved_array_audit/contract.json | 0179a2969ea4187b6a57a92b27f7147402316102b984751e0f72cd2d593e29ab |
| saved_array_audit/verify_arrays.py | d9d6cfabbd280f5004ce20d60d243dd5a3f0956be64a0a49cb51af988e9d6fb0 |
| saved_array_audit/verification.json | 56363aee43e70800df3144a3690630eeda40790e56ebe48cfee8d3a7a5999b92 |

The result supports using complete target-sector scalar DC/EOL machinery
for this declared point, basis, target and original metric. Extension to
another physical space must re-establish complete sector/key/target
identity. It certifies neither the omitted full spectrum/global ground,
transition operators connecting other sectors, static Breit equivalence,
nor complete core/radial/QED/recoil physics. Static Breit, core-double
and virtual-radial changes need their own matched-operator measurements;
old003 correction differences are not added here.
