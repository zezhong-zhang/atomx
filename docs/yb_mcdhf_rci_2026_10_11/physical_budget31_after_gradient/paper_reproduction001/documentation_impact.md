# Documentation impact: native paper-method development

This increment changes the existing native finite CC solver and adds a finite
EA consumer of its cluster machinery and the shared owned spinor integrals.
It does not adopt an external backend or change production MCDHF/RCI physics,
weights, spectra, nuclear parameters or acceptance gates.

| Owning guide | Update |
|---|---|
| [Atomic CC/FSCC](../../../ATOMIC_FSCC.md) | Separate native finite connected T3 and finite EOM-EA from external tensor FSCC and production Yb completeness |
| [Architecture capabilities](../../../ARCHITECTURE_CAPABILITIES.md) | Advertise implemented finite core/attachment ranks, real input and resource/provenance boundaries |
| [Testing](../../../TESTING.md) | Add independent cumulant/Jacobian, active lifted T3, omitted-rank and live native Be operator controls |
| [Native Yb](../../README.md), [physical priorities](../README.md) | Link measured development, preserve current production spectrum and failed gates |
| [Architecture design](../../../ARCHITECTURE_DESIGN.md) | Existing methods→physics/numerics direction applies; no facade dispatch, backend substitution or new architecture required |

The original source-only paper contract, frozen MCDHF evidence, older completion
reviews and high-l failed receipts remain unchanged. The current development
receipt explicitly records source-epoch changes. Verification is the39-test
affected panel, independent read-only review,127 actual high-l controls and
source/archive/link/whitespace checks. Original test/solver/cache/admission
failures are described in [results](RESULTS.md), with no numerical gate relaxed.

No new global insight is added: existing state/operator identity, signed active
coordinate validation and independent-reference rules cover these changes.
Cards, timers, publication, staging and commits remain outside this increment.
