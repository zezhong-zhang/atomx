"""The scalar/default QED entry point must surface analytic fallback coverage."""
import warnings

import numpy as np
import pytest

from atomx.corrections.qed import compute_se_energy


@pytest.mark.parametrize("method", ["auto", "mohr", "indelicato"])
def test_untabulated_yb_valence_warns_through_scalar_entrypoint(method):
    # No radial screening here: this gate checks coverage propagation,
    # not the validity of the high-Z leading-order hydrogenic estimate.
    with pytest.warns(RuntimeWarning, match="leading.*not precision QED"):
        result = compute_se_energy(
            [(np.ones(1), np.zeros(1), 1., -1, "6s1/2")],
            70, method=method,
        )
    assert np.isfinite(result)


def test_tabulated_valence_does_not_emit_fallback_warning():
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        compute_se_energy(
            [(np.ones(1), np.zeros(1), 1., 2, "5d3/2")], 70,
        )
    assert not emitted
