"""Independent SVD and finite-path checks of the real polar differential."""

import numpy as np
import pytest

from atomx.numerics.bspline.orbital_geometry import real_polar_retraction


@pytest.mark.parametrize("shape", [(7, 1), (8, 3), (4, 4)])
def test_polar_velocity_matches_svd_path_and_orthogonal_gauges(shape):
    rng = np.random.default_rng(718 + shape[1])
    b, k = rng.normal(size=shape), rng.normal(size=shape)
    y, velocity = real_polar_retraction(b, direction=k)

    def svd_polar(a):
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        return u @ vt

    np.testing.assert_allclose(y, svd_polar(b), atol=3e-14)
    np.testing.assert_allclose(real_polar_retraction(b), y, atol=0, rtol=0)
    for h in (1e-4, 1e-5):
        fd = (svd_polar(b + h * k) - svd_polar(b - h * k)) / (2 * h)
        np.testing.assert_allclose(velocity, fd, atol=4e-8, rtol=4e-8)
    np.testing.assert_allclose(y.T @ velocity + velocity.T @ y, 0, atol=3e-14)
    left, _ = np.linalg.qr(rng.normal(size=(shape[0], shape[0])))
    right, _ = np.linalg.qr(rng.normal(size=(shape[1], shape[1])))
    rotated_y, rotated_v = real_polar_retraction(
        left @ b @ right, direction=left @ k @ right
    )
    np.testing.assert_allclose(rotated_y, left @ y @ right, atol=3e-14)
    np.testing.assert_allclose(rotated_v, left @ velocity @ right, atol=3e-14)


def test_polar_preserves_internal_rotation_and_removes_radial_normal():
    b = np.eye(5)[:, :2]
    rotation = np.array([[0.0, 0.7], [-0.7, 0.0]])
    normal = np.array([[0.3, 0.2], [0.2, -0.4]])
    _, v = real_polar_retraction(b, direction=b @ (rotation + normal))
    np.testing.assert_allclose(v, b @ rotation, atol=2e-15)


@pytest.mark.parametrize(
    "b,k",
    [
        (np.zeros((3, 2)), None),
        (np.ones((2, 3)), None),
        (np.ones((3, 0)), None),
        (np.eye(2) * 1j, None),
        (np.array([[np.nan]]), None),
        (np.eye(2), np.ones((3, 2))),
        (np.eye(2), np.eye(2) * 1j),
        (np.eye(2), np.full((2, 2), np.inf)),
    ],
)
def test_polar_refuses_invalid_or_unsupported_inputs(b, k):
    with pytest.raises(ValueError):
        real_polar_retraction(b, direction=k)
