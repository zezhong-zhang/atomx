The fixed-alpha saved-H diagnostic closed with exit0 in1.962591 seconds. All1066 diagnostic gates passed, including every1046-root projector span. The original semiseparable-route qualification remains FAIL; no source routing or production solver was changed.

Root296 remained a singleton under the original1e−7Ha rule. Its ordinary/candidate projector difference fell from the archived1.04436e−8 to2.30564e−9. The maximum over all1046 spans was2.33812e−9 at root298, below the unchanged1e−8 gate. Every shifted partition exactly matched its original unshifted partition: no additional roots were grouped together.

Restored eigenpairs passed against the original unshifted H: maximum column residual9.47838e−12Ha ordinary and9.26174e−12Ha candidate; both Gram errors2.88658e−15. Complete endpoint energy differences were at most5.45697e−12Ha. Five target projector differences were2.37e−13 to1.25e−12, with no target exemption from complete-CI gates.

Unshifted-to-shifted individual projector changes reached1.54e−8 and1.72e−8 within the respective endpoints. These observations support large common-energy centering as a numerical stabilization candidate; they do not prove a unique error decomposition or make the previous unshifted failure pass. A new private-source qualification must use the same centered dense solver on both ordinary and semiseparable paths, preserve complete H/roots/spans/gradients/metric/RDM/owner/anchors and original gates, and bind a new source epoch before any shared adoption.

Actual calls: two stored NPZ loads and exactly ten complete-block `evd` eigensolves;0 H/CI construction/gradient/Dirac/optimizer/GPU and0 other eigen/SVD/QR calls. The inputs, fixed alphas and original failed evidence remain unchanged. No controlled performance comparison was performed.

Result SHA `9e961e94f134fd859c89feb867cca3aad843fa4b3d5e027db9febce809c706c8`; shifted-array SHA `948f4c213ca077df6ca358b3b83b710a0d14b919b9858a38ec00185ccf8b3384`; root execution SHA `52a10893049acd80e37b58191da3d82dd59ba2a04f6ed958ce2a02c971e06aa8`.
