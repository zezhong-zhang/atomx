import math

import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.local_potential import (
    fac_local_central_potential_from_orbitals,
    fac_local_interaction_matrices_from_orbitals,
    local_central_interaction_matrices_from_radial_density,
    local_central_potential_from_radial_density,
    radial_density_from_orbitals,
    solve_orbitals_in_local_potential,
)

#: CODATA 2018 fine-structure constant: the closed form below must not depend
#: on an atomx constant.
_ALPHA = 1.0 / 137.035999084


def _dirac_coulomb(n: int, kappa: int, Z: float) -> float:
    """Exact point-Coulomb Dirac eigenvalue, in Hartree (see
    ``tests/test_dirac_hydrogenic.py`` for the derivation):

        gamma = sqrt(kappa^2 - (Z alpha)^2)
        N     = sqrt((n - |kappa| + gamma)^2 + (Z alpha)^2)
        E     = m c^2 (sqrt(1 - (Z alpha / N)^2) - 1)

    Used here because a ``-Z/r`` potential makes the solver's eigenvalues
    hydrogenic *and relativistic*: the Schrodinger value -Z^2/(2n^2) is
    1.3e-3 relative away at Z=10, so the closed form also proves the
    relativistic kinematics are wired in.
    """
    gamma = math.sqrt(kappa * kappa - (Z * _ALPHA) ** 2)
    N = math.sqrt((n - abs(kappa) + gamma) ** 2 + (Z * _ALPHA) ** 2)
    c = 1.0 / _ALPHA
    return c * c * (math.sqrt(1.0 - (Z * _ALPHA / N) ** 2) - 1.0)


def test_local_potential_solver_matches_nuclear_only_dirac_matrix():
    Z = 10
    ns = 36
    ks = 7
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.05 / Z, 40.0), ns, ks, kinetic_balance=True
    )
    r_table = np.unique(np.asarray(bsp_data["gr"], dtype=float).ravel())
    v_table = -float(Z) / r_table

    orbitals, returned_bsp, energies = solve_orbitals_in_local_potential(
        Z=Z,
        potential_r=r_table,
        potential_v=v_table,
        orbital_keys=((1, -1), (2, 1), (2, -2)),
        ns=ns,
        ks=ks,
        bsp_data=bsp_data,
    )

    assert returned_bsp["ns"] == bsp_data["ns"]
    assert {(1, -1), (2, 1), (2, -2)}.issubset(orbitals)

    for key in ((1, -1), (2, 1), (2, -2)):
        n, kappa = key
        # (a) Independent closed form: a pure -Z/r potential *is* the
        #     hydrogenic problem, so the eigenvalue must be the exact Dirac
        #     value.  Measured 2.0e-8 / 1.2e-8 / 2.3e-8 relative; gate 1e-7.
        closed = _dirac_coulomb(n, kappa, Z)
        assert energies[key] == pytest.approx(closed, rel=1e-7), (
            f"{key}: atomx={energies[key]!r} Dirac closed form={closed!r}"
        )
        assert energies[key] != pytest.approx(-Z * Z / (2.0 * n * n), rel=1e-6)
        # (b) And the same value as the direct matrix path (this part is the
        #     C-class indexing check: both sides share build_dirac_matrices).
        H, S = build_dirac_matrices(int(bsp_data["ns"]), kappa, Z, bsp_data)
        direct = solve_dirac(
            H,
            S,
            int(bsp_data["ns"]),
            kappa,
            nsp=bsp_data.get("nsp"),
            nsq=bsp_data.get("nsq"),
        )
        l = kappa if kappa > 0 else -kappa - 1
        expected_idx = n - l - 1
        _P, _Q, expected_energy = direct.get_orbital(expected_idx)
        assert energies[key] == pytest.approx(expected_energy, abs=1.0e-10)


def test_local_potential_solver_rejects_table_that_does_not_cover_basis_grid():
    Z = 4
    ns = 24
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 12.0), ns, ks, kinetic_balance=True
    )
    gr = np.asarray(bsp_data["gr"], dtype=float)
    r_min = float(np.min(gr))
    r_max = float(np.max(gr))
    r_table = np.linspace(r_min * 1.1, r_max * 0.9, 40)
    v_table = -float(Z) / r_table

    with pytest.raises(ValueError, match="cover the B-spline quadrature grid"):
        solve_orbitals_in_local_potential(
            Z=Z,
            potential_r=r_table,
            potential_v=v_table,
            orbital_keys=((1, -1),),
            ns=ns,
            ks=ks,
            bsp_data=bsp_data,
        )


def test_local_potential_solver_rejects_missing_requested_orbitals():
    Z = 6
    ns = 24
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 12.0), ns, ks, kinetic_balance=True
    )
    r_table = np.unique(np.asarray(bsp_data["gr"], dtype=float).ravel())
    v_table = -float(Z) / r_table

    with pytest.raises(ValueError, match="requested local-potential orbitals were not found"):
        solve_orbitals_in_local_potential(
            Z=Z,
            potential_r=r_table,
            potential_v=v_table,
            orbital_keys=((99, -1),),
            ns=ns,
            ks=ks,
            bsp_data=bsp_data,
        )


def test_local_potential_solver_rejects_duplicate_grid_points():
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_orbitals_in_local_potential(
            Z=2,
            potential_r=[0.1, 0.1, 1.0],
            potential_v=[-20.0, -20.0, -2.0],
            orbital_keys=((1, -1),),
            ns=20,
            ks=5,
        )


def test_local_potential_solver_rejects_nonfinite_potential_table():
    with pytest.raises(ValueError, match="potential_v values must be finite"):
        solve_orbitals_in_local_potential(
            Z=2,
            potential_r=[0.1, 0.5, 1.0],
            potential_v=[-20.0, np.nan, -2.0],
            orbital_keys=((1, -1),),
            ns=20,
            ks=5,
        )


def test_local_potential_solver_rejects_nonpositive_potential_r():
    with pytest.raises(ValueError, match="potential_r values must be positive"):
        solve_orbitals_in_local_potential(
            Z=2,
            potential_r=[0.0, 0.5, 1.0],
            potential_v=[-20.0, -4.0, -2.0],
            orbital_keys=((1, -1),),
            ns=20,
            ks=5,
        )



def test_local_potential_solver_rejects_nonphysical_orbital_keys_and_nuclear_callback():
    Z = 4
    ns = 24
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 12.0), ns, ks, kinetic_balance=True
    )
    r_table = np.unique(np.asarray(bsp_data["gr"], dtype=float).ravel())
    v_table = -float(Z) / r_table

    for key, message in [
        ((0, -1), "principal quantum numbers"),
        ((1, 0), "kappa values"),
        ((1, -2), "n > l"),
    ]:
        with pytest.raises(ValueError, match=message):
            solve_orbitals_in_local_potential(
                Z=Z,
                potential_r=r_table,
                potential_v=v_table,
                orbital_keys=(key,),
                ns=ns,
                ks=ks,
                bsp_data=bsp_data,
            )

    with pytest.raises(ValueError, match="nuclear_potential_func must return finite"):
        solve_orbitals_in_local_potential(
            Z=Z,
            potential_r=r_table,
            potential_v=v_table,
            orbital_keys=((1, -1),),
            ns=ns,
            ks=ks,
            bsp_data=bsp_data,
            nuclear_potential_func=lambda r: np.nan,
        )

def test_local_central_potential_rejects_nonphysical_density_inputs():
    gr = np.array([[0.2, 0.4], [0.8, 1.6]], dtype=float)
    grw = np.full_like(gr, 0.1)
    rho_rad = np.array([[0.3, 0.2], [0.1, 0.02]], dtype=float)

    bad_rho = rho_rad.copy()
    bad_rho[0, 1] = np.nan
    with pytest.raises(ValueError, match="rho_rad values must be finite"):
        local_central_potential_from_radial_density(
            Z=10, rho_rad=bad_rho, gr=gr, grw=grw, n_total=2.0
        )

    bad_rho = rho_rad.copy()
    bad_rho[1, 0] = -1.0e-8
    with pytest.raises(ValueError, match="rho_rad values must be non-negative"):
        local_central_potential_from_radial_density(
            Z=10, rho_rad=bad_rho, gr=gr, grw=grw, n_total=2.0
        )

    bad_gr = gr.copy()
    bad_gr[0, 0] = 0.0
    with pytest.raises(ValueError, match="gr values must be finite and positive"):
        local_central_potential_from_radial_density(
            Z=10, rho_rad=rho_rad, gr=bad_gr, grw=grw, n_total=2.0
        )

    with pytest.raises(ValueError, match="Z must be finite and positive"):
        local_central_potential_from_radial_density(
            Z=0, rho_rad=rho_rad, gr=gr, grw=grw, n_total=2.0
        )


def test_local_central_potential_normalizes_exchange_mode_spelling():
    gr = np.array([[0.2, 0.4], [0.8, 1.6]], dtype=float)
    grw = np.full_like(gr, 0.1)
    rho_rad = np.array([[0.3, 0.2], [0.1, 0.02]], dtype=float)

    fac = local_central_potential_from_radial_density(
        Z=10,
        rho_rad=rho_rad,
        gr=gr,
        grw=grw,
        n_total=2.0,
        exchange=" FAC-LOCAL ",
        fac_self_interaction_correction=False,
        latter_correction=False,
    )
    canonical = local_central_potential_from_radial_density(
        Z=10,
        rho_rad=rho_rad,
        gr=gr,
        grw=grw,
        n_total=2.0,
        exchange="fac-local",
        fac_self_interaction_correction=False,
        latter_correction=False,
    )

    for key in ("nuclear", "hartree", "exchange", "central"):
        np.testing.assert_allclose(fac[key], canonical[key])


def test_local_central_potential_components_have_fac_exchange_semantics():
    gr = np.array([[0.2, 0.4], [0.8, 1.6]], dtype=float)
    grw = np.full_like(gr, 0.1)
    rho_rad = np.array([[0.3, 0.2], [0.1, 0.02]], dtype=float)

    local = local_central_potential_from_radial_density(
        Z=10,
        rho_rad=rho_rad,
        gr=gr,
        grw=grw,
        n_total=2.0,
        exchange="fac",
        fac_self_interaction_correction=False,
        latter_correction=False,
    )

    from atomx.physics.local_exchange import fac_gu_exchange

    rho_vol = rho_rad / (4.0 * np.pi * gr * gr)
    expected_exchange = fac_gu_exchange(
        rho_vol, gr, Z=10.0, N_total=2.0, ihx=-1.0, hxs=1.0
    )
    np.testing.assert_allclose(local["exchange"], expected_exchange)
    np.testing.assert_allclose(
        local["central"], local["nuclear"] + local["hartree"] + local["exchange"]
    )
    assert np.all(local["hartree"] >= 0.0)
    assert np.all(local["exchange"] <= 0.0)


def test_fac_self_interaction_correction_drives_screening_toward_n_minus_one():
    r = np.linspace(0.02, 8.0, 400)
    raw = 2.0 * (1.0 - np.exp(-0.5 * r))
    from atomx.numerics.bspline.local_potential import _fac_csi_screening_correction

    corrected = _fac_csi_screening_correction(r, raw, n_total=2.0, ihx=-1.0, csi=0.55)

    assert corrected[-1] == pytest.approx(1.0, rel=2e-2)
    assert corrected[-1] < raw[-1]
    assert corrected[0] == pytest.approx(raw[0])


def test_radial_density_from_orbitals_respects_occupations_on_bspline_grid():
    ns = 20
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 8.0), ns, ks, kinetic_balance=True
    )
    nsp = int(bsp_data["nsp"])
    nsq = int(bsp_data["nsq"])
    P1 = np.zeros(nsp)
    Q1 = np.zeros(nsq)
    P2 = np.zeros(nsp)
    Q2 = np.zeros(nsq)
    P1[2] = 1.0
    Q1[3] = 0.25
    P2[4] = -0.5
    Q2[5] = 0.1
    orbitals = {
        (1, -1): (P1, Q1),
        (2, 1): (P2, Q2),
    }

    rho_all, n_all = radial_density_from_orbitals(orbitals, bsp_data)
    rho_occ, n_occ = radial_density_from_orbitals(
        orbitals, bsp_data, occupations={(1, -1): 2.0, (2, 1): 0.0}
    )

    assert n_all == pytest.approx(2.0)
    assert n_occ == pytest.approx(2.0)
    assert np.all(rho_all >= 0.0)
    assert np.all(rho_occ >= 0.0)
    assert not np.allclose(rho_all, rho_occ)



def test_radial_density_from_orbitals_rejects_invalid_physics_inputs():
    ns = 20
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 8.0), ns, ks, kinetic_balance=True
    )
    nsp = int(bsp_data["nsp"])
    nsq = int(bsp_data["nsq"])
    P = np.zeros(nsp)
    Q = np.zeros(nsq)
    P[2] = 1.0
    orbitals = {(1, -1): (P, Q)}

    with pytest.raises(ValueError, match="occupations must be finite and non-negative"):
        radial_density_from_orbitals(
            orbitals, bsp_data, occupations={(1, -1): -1.0}
        )

    with pytest.raises(ValueError, match="occupations must be finite and non-negative"):
        radial_density_from_orbitals(
            orbitals, bsp_data, occupations={(1, -1): np.nan}
        )

    bad_P = P.copy()
    bad_P[1] = np.inf
    with pytest.raises(ValueError, match="orbital coefficients must be finite"):
        radial_density_from_orbitals({(1, -1): (bad_P, Q)}, bsp_data)

    with pytest.raises(ValueError, match="shorter than the B-spline basis"):
        radial_density_from_orbitals({(1, -1): (P[:-1], Q)}, bsp_data)
    with pytest.raises(ValueError, match="shorter than the B-spline basis"):
        radial_density_from_orbitals({(1, -1): (P, Q[:-1])}, bsp_data)

    rho, n_total = radial_density_from_orbitals(
        orbitals, bsp_data, occupations={(1, -1): 0.0}
    )
    assert n_total == 0.0
    assert np.all(rho == 0.0)

def test_local_central_interaction_matrices_project_central_minus_nuclear():
    """API contract (C-degenerate by construction): this re-runs the
    source's own ``0.5(A + A^T)`` Galerkin projection of
    ``central - nuclear`` with ``grw * delta_v``, so it detects an indexing
    or weighting change, not a physics error.

    The independent statement added here is the closed form for the nuclear
    channel: for a ``-Z/r`` nucleus that channel *is* ``-Z/r`` exactly on the
    quadrature grid (measured 1.8e-16 relative, i.e. round-off).
    """
    from atomx.numerics.bspline.integrals import _assemble_integral

    ns = 20
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 8.0), ns, ks, kinetic_balance=True
    )
    gr = np.asarray(bsp_data["gr"], dtype=float)
    rho_rad = 0.4 * np.exp(-1.2 * gr)

    local = local_central_interaction_matrices_from_radial_density(
        Z=10,
        rho_rad=rho_rad,
        bsp_data=bsp_data,
        n_total=2.0,
        exchange="fac-local",
        fac_self_interaction_correction=True,
        latter_correction=True,
    )

    # Closed form, independent of the density/exchange model.
    np.testing.assert_allclose(local["nuclear"], -10.0 / gr, rtol=1.0e-13)

    delta_v = np.asarray(local["central"]) - np.asarray(local["nuclear"])
    nv = int(bsp_data["nv"])
    ksp = int(bsp_data["ksp"])
    ksq = int(bsp_data["ksq"])
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data["bsp_q"]
    expected_pp = _assemble_integral(
        nv, ksp, ksp, ns, bsp_p, bsp_p, np.asarray(bsp_data["grw"]) * delta_v
    )
    expected_qq = _assemble_integral(
        nv, ksq, ksq, ns, bsp_q, bsp_q, np.asarray(bsp_data["grw"]) * delta_v
    )
    expected_pp = 0.5 * (expected_pp + expected_pp.T)
    expected_qq = 0.5 * (expected_qq + expected_qq.T)

    np.testing.assert_allclose(local["V_PP"], expected_pp)
    np.testing.assert_allclose(local["V_QQ"], expected_qq)
    np.testing.assert_allclose(local["V_PP"], np.asarray(local["V_PP"]).T)
    np.testing.assert_allclose(local["V_QQ"], np.asarray(local["V_QQ"]).T)


def test_fac_local_potential_from_orbitals_matches_density_builder():
    """API contract (C-degenerate by construction):
    ``fac_local_central_potential_from_orbitals`` *is*
    ``radial_density_from_orbitals`` followed by
    ``local_central_potential_from_radial_density``
    (``local_potential.py:294-303``), and the test performs the same two
    calls — it gates the wrapper's wiring, not new physics.

    The independent closed-form assertion is the nuclear channel = ``-Z/r``.
    """
    ns = 20
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 8.0), ns, ks, kinetic_balance=True
    )
    nsp = int(bsp_data["nsp"])
    nsq = int(bsp_data["nsq"])
    P = np.zeros(nsp)
    Q = np.zeros(nsq)
    P[2] = 1.0
    P[3] = -0.35
    Q[3] = 0.2
    orbitals = {(1, -1): (P, Q)}
    occupations = {(1, -1): 2.0}

    rho_rad, n_total = radial_density_from_orbitals(
        orbitals, bsp_data, occupations=occupations
    )
    direct = local_central_potential_from_radial_density(
        Z=10,
        rho_rad=rho_rad,
        gr=np.asarray(bsp_data["gr"], dtype=float),
        grw=np.asarray(bsp_data["grw"], dtype=float),
        n_total=n_total,
        exchange="fac",
        fac_self_interaction_correction=True,
        latter_correction=True,
    )
    from_orbitals = fac_local_central_potential_from_orbitals(
        Z=10,
        orbitals_dict=orbitals,
        bsp_data=bsp_data,
        occupations=occupations,
        latter_correction=True,
    )

    assert from_orbitals["n_total"] == pytest.approx(2.0)
    for key in ("nuclear", "hartree", "exchange", "central"):
        np.testing.assert_allclose(from_orbitals[key], direct[key])
    # Independent closed form: the nuclear channel is exactly -Z/r.
    gr = np.asarray(bsp_data["gr"], dtype=float)
    np.testing.assert_allclose(from_orbitals["nuclear"], -10.0 / gr,
                               rtol=1.0e-13)


def test_fac_local_interaction_matrices_from_orbitals_matches_density_path():
    ns = 20
    ks = 6
    bsp_data = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 8.0), ns, ks, kinetic_balance=True
    )
    nsp = int(bsp_data["nsp"])
    nsq = int(bsp_data["nsq"])
    P = np.zeros(nsp)
    Q = np.zeros(nsq)
    P[2] = 1.0
    P[4] = -0.2
    Q[3] = 0.15
    orbitals = {(1, -1): (P, Q)}
    occupations = {(1, -1): 2.0}

    rho_rad, n_total = radial_density_from_orbitals(
        orbitals, bsp_data, occupations=occupations
    )
    direct = local_central_interaction_matrices_from_radial_density(
        Z=10,
        rho_rad=rho_rad,
        bsp_data=bsp_data,
        n_total=n_total,
        exchange="fac",
        latter_correction=True,
    )
    from_orbitals = fac_local_interaction_matrices_from_orbitals(
        Z=10,
        orbitals_dict=orbitals,
        bsp_data=bsp_data,
        occupations=occupations,
        latter_correction=True,
    )

    assert from_orbitals["n_total"] == pytest.approx(2.0)
    for key in ("nuclear", "hartree", "exchange", "central", "V_PP", "V_QQ"):
        np.testing.assert_allclose(from_orbitals[key], direct[key])
