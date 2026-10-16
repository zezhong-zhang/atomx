# Semiseparable H route successor — actual closure

The sole root foreground run failed; the first failure and partial calls remain unchanged.
This closure reads JSON and hashes stored files; it performs no new physical calculation, array replay or timing.

The ordinary 62a109 epoch and private 608f8f9d candidate share the corrected inner-cell operator; the private candidate changes only the CI pair-potential route. Original route/derivative/stationarity/experimental failures remain historical failures.

anchors: stored pass=True; calls completed {"Gauss_setup_eigvalsh": 7, "basis_setup": 7, "density_band": 28, "factor_get": 92, "bank_get": 112, "direct_full_bank": 112, "direct_semiseparable": 92, "band_scalar": 170, "CI_source_pair": 17}
Stages (uncontrolled seconds): {"anchors_total": 12.816450280835852}

ordinary: stored pass=True; calls completed {"tracking_overlap": 3, "initial_hS": 7, "CI_internal_hS": 7, "density_band": 3242, "bank_get": 2372, "direct_full_bank": 2372, "CI_source_pair": 593, "band_scalar": 49054, "H": 1, "scipy_eigh": 5, "CI": 1, "tracking_SVD": 10, "tracking_resolution": 2, "factor_get": 2372, "direct_semiseparable": 2372, "gradient_band_expand": 2372, "gradient_source_pair": 593, "gradient": 1}
Stages (uncontrolled seconds): {"source_point_angular_setup": 28.036588602000847, "public_energy_trial_including_tracking_owner": 8.136886307038367, "complete_trial_eigenpair_Gram_owner_validation": 0.11571224988438189, "public_accepted_gradient_including_tracking": 3.0392219410277903, "complete_checked_compressed_archive_fsync_readback": 0.18268248299136758}

candidate: stored pass=True; calls completed {"tracking_overlap": 3, "initial_hS": 7, "CI_internal_hS": 7, "density_band": 3242, "factor_get": 4744, "direct_semiseparable": 4744, "CI_source_pair": 593, "band_scalar": 49054, "H": 1, "scipy_eigh": 5, "CI": 1, "tracking_SVD": 10, "tracking_resolution": 2, "gradient_band_expand": 2372, "gradient_source_pair": 593, "gradient": 1}
Stages (uncontrolled seconds): {"source_point_angular_setup": 27.613933203043416, "public_energy_trial_including_tracking_owner": 7.020295120077208, "complete_trial_eigenpair_Gram_owner_validation": 0.14995073713362217, "public_accepted_gradient_including_tracking": 2.1476443589199334, "complete_checked_compressed_archive_fsync_readback": 0.1815909759607166}

See completion.json for exact H/CI/gradient/UID/owner gates and stored file hashes. Full signed cluster gates remain in actual/comparison.json.

Fresh-process imports/JIT, cProfile, validation, source/input hashing and archive overhead make this an uncontrolled correctness/profile observation. It certifies no speedup, full-optimizer, parallel or GPU improvement; kernel getter counts include cache hits. Shared adoption needs a separate reviewed source change.
