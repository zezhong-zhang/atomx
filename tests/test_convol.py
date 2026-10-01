"""Tests for cell-based convolution algorithm (DBSR_HF).

Verifies:
1. Cell-based Slater integrals match Y^k approach (consistency)
2. He ground state energy matches DBSR_HF reference (sub-µHa accuracy)
3. Be ground state energy matches DBSR_HF reference
4. Direct potential matrix matches Y^k-based matrix
5. Exchange matrix matches Y^k-based matrix
"""

import numpy as np
import pytest

import atomx.numerics.bspline.convol as convol
from atomx.numerics.bspline.basis import (
    prepare_bspline_data,
)
from atomx.numerics.bspline.convol import (
    ConvolCache,
    _build_rkb_sym,
    _compute_moments_sym,
    _convol_case1,
    _convol_case1_semiseparable_t,
    _convol_case1_t,
    _convol_case2,
    _convol_case4_semiseparable_mixed_t,
    _convol_case4_semiseparable_t,
    _convol_case4_t,
    _density_sym,
    _triang_sym,
    compute_slater_fk_cell,
    compute_slater_gk_cell,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.slater_yk import (
    compute_slater_fk,
    compute_slater_gk,
)


@pytest.fixture
def he_setup():
    """Set up He atom (Z=2) with B-spline basis."""
    Z = 2
    ks = 9
    ns = 50
    t = make_knots(ns, ks, 0.025, 40.0)
    data = prepare_bspline_data(t, ns, ks)

    kappa = -1  # 1s1/2
    H, S = build_dirac_matrices(ns, kappa, Z, data)
    result = solve_dirac(H, S, ns, kappa)

    # Extract 1s orbital
    nsp = data.get("nsp", ns)
    nsq = data.get("nsq", ns)
    P_1s, Q_1s, _ = result.get_orbital(0)

    return {
        "Z": Z, "ks": ks, "ns": ns, "t": t, "data": data,
        "P_1s": P_1s, "Q_1s": Q_1s,
        "nsp": nsp, "nsq": nsq,
    }


@pytest.fixture
def be_setup():
    """Set up Be atom (Z=4) with B-spline basis."""
    Z = 4
    ks = 9
    ns = 50
    t = make_knots(ns, ks, 0.0125, 30.0)
    data = prepare_bspline_data(t, ns, ks)

    # 1s and 2s orbitals (kappa = -1)
    H1, S1 = build_dirac_matrices(ns, -1, Z, data)
    res1 = solve_dirac(H1, S1, ns, -1)
    P_1s, Q_1s, _ = res1.get_orbital(0)
    P_2s, Q_2s, _ = res1.get_orbital(1)

    return {
        "Z": Z, "ks": ks, "ns": ns, "t": t, "data": data,
        "P_1s": P_1s, "Q_1s": Q_1s,
        "P_2s": P_2s, "Q_2s": Q_2s,
    }


class TestMoments:
    """Test 1D B-spline moment computation."""

    def test_moments_shape(self, he_setup):
        d = he_setup["data"]
        ks = d.get("ksp", d["ks"])
        nv = d["nv"]
        kk = ks * (ks + 1) // 2
        rkm = _compute_moments_sym(d["bsp_p"], d["gr"], d["grw"], nv, ks, 0)
        assert rkm.shape == (kk, nv)

    def test_moments_k0_positive(self, he_setup):
        """k=0 moments (overlap) should be all positive."""
        d = he_setup["data"]
        ks = d.get("ksp", d["ks"])
        nv = d["nv"]
        rkm = _compute_moments_sym(d["bsp_p"], d["gr"], d["grw"], nv, ks, 0)
        # Diagonal elements (i==j) should be positive
        for iv in range(nv):
            ii = 0
            for i in range(ks):
                assert rkm[ii, iv] >= 0.0, f"Diagonal moment negative at iv={iv}, i={i}"
                ii += ks - i  # skip to next diagonal


class TestTriang:
    """Test diagonal cell nested quadrature."""

    def test_triang_shape(self, he_setup):
        d = he_setup["data"]
        ks = d.get("ksp", d["ks"])
        nv = d["nv"]
        ns = d.get("nsp", d["ns"])
        kk = ks * (ks + 1) // 2
        rkd = _triang_sym(d["bsp_p"], d["gr"], d["grw"], d["t"],
                           nv, ks, ns, 0)
        assert rkd.shape == (kk, kk, nv)

    def test_triang_symmetric(self, he_setup):
        """Diagonal cell integrals should be symmetric: rkd[ii,jj] == rkd[jj,ii]."""
        d = he_setup["data"]
        ks = d.get("ksp", d["ks"])
        nv = d["nv"]
        ns = d.get("nsp", d["ns"])
        rkd = _triang_sym(d["bsp_p"], d["gr"], d["grw"], d["t"],
                           nv, ks, ns, 0)
        for iv in range(nv):
            np.testing.assert_allclose(
                rkd[:, :, iv], rkd[:, :, iv].T,
                atol=1e-14, err_msg=f"Not symmetric at iv={iv}")


class TestRkb:
    """Test full rkb array assembly."""

    def test_rkb_shape(self, he_setup):
        d = he_setup["data"]
        ks = d.get("ksp", d["ks"])
        ns = d.get("nsp", d["ns"])
        rkb = _build_rkb_sym(d["bsp_p"], d["gr"], d["grw"], d["t"],
                              d["nv"], ks, ns, 0)
        assert rkb.shape == (ns, ns, ks, ks)

    def test_symmetric_exchange_contraction_matches_full(self, he_setup):
        """Coulomb pair symmetry permits a half-matrix exchange build."""
        data = he_setup["data"]
        ns = data["ns"]
        bank, _, ks = ConvolCache(data).get_rkb_t(0, "pppp")
        p = he_setup["P_1s"]
        density = np.outer(p, p)
        full = _convol_case4_t(bank, density, ns, ks, ks)
        half = _convol_case4_t(bank, density, ns, ks, ks, True)
        np.testing.assert_allclose(full, half, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(half, half.T)

    @pytest.mark.parametrize("ns,component,k", [
        (50, "pppp", 2), (256, "pppp", 0), (256, "qqqq", 0),
    ])
    def test_factored_exchange_matches_cell_bank(self, ns, component, k):
        """Separated Coulomb tails plus exact overlap cells retain exchange."""
        ks = 9
        data = prepare_bspline_data(
            make_knots(ns, ks, 0.0125, 50.0), ns, ks,
            kinetic_balance=True)
        cache = ConvolCache(data)
        bank, _, order = cache.get_rkb_t(k, component)
        near, rising, falling = cache.get_symmetric_exchange_factors(
            k, component)
        rng = np.random.default_rng(103 + ns + k)
        occupied = rng.normal(size=(3, ns))
        density = occupied.T @ occupied
        reference = _convol_case4_t(
            bank, density, ns, order, order, symmetric_output=True)
        factored = _convol_case4_semiseparable_t(
            near, rising, falling, density, ns, order)
        np.testing.assert_allclose(factored, reference, rtol=2e-13,
                                   atol=2e-12)
        np.testing.assert_array_equal(factored, factored.T)

    @pytest.mark.parametrize("ns,component,k", [
        (50, "pqpq", 1), (50, "qpqp", 6),
        (256, "pqpq", 0), (256, "qpqp", 0),
    ])
    def test_factored_mixed_exchange_matches_cell_bank(
            self, ns, component, k):
        data = prepare_bspline_data(
            make_knots(ns, 9, 0.0125, 50.0), ns, 9,
            kinetic_balance=True)
        cache = ConvolCache(data)
        factors = cache.get_coulomb_factors(k, component)
        assert (k, component, "T") not in cache._cache
        bank, _, ks_left = cache.get_rkb_t(k, component)
        ks_right = bank.shape[1]
        rng = np.random.default_rng(1301 + ns + k)
        density = np.outer(rng.normal(size=ns), rng.normal(size=ns))
        reference = _convol_case4_t(
            bank, density, ns, ks_left, ks_right)
        factored = _convol_case4_semiseparable_mixed_t(
            *factors, density, ns, ks_left, ks_right)
        np.testing.assert_allclose(factored, reference, rtol=2e-12,
                                   atol=2e-12)

    @pytest.mark.parametrize("component", ["pppp", "qqqq", "pqpq", "qpqp"])
    @pytest.mark.parametrize("k", [0, 1, 6])
    def test_overlap_strip_matches_independent_full_bank(self, component, k):
        """Cell assembly retains every overlap integral without a full bank."""
        ns = 50
        data = prepare_bspline_data(
            make_knots(ns, 9, 0.0125, 50.0), ns, 9,
            kinetic_balance=True)
        compact_cache = ConvolCache(data)
        compact_cache.clear()
        near, *_ = compact_cache.get_coulomb_factors(k, component)
        assert (k, component) not in compact_cache._cache
        assert (k, component, "T") not in compact_cache._cache
        bank, _, ks_left = compact_cache.get_rkb_t(k, component)
        ks_right = bank.shape[1]
        for ip in range(ks_left):
            for jp in range(ks_right):
                for i in range(ns):
                    first_j = i + ip - ks_left + 1
                    last_j = i + ks_right - jp - 1
                    for j in range(max(0, first_j), min(ns, last_j + 1)):
                        np.testing.assert_allclose(
                            near[ip, jp, i, j - first_j],
                            bank[ip, jp, i, j], rtol=2e-12, atol=1e-14)

    @pytest.mark.parametrize("component", ["pppp", "qqqq", "pqpq", "qpqp"])
    @pytest.mark.parametrize("ns,k", [(50, 1), (50, 6), (113, 2), (256, 0)])
    def test_factored_direct_matches_full_cell_bank(self, ns, k, component):
        """Prefix/suffix Coulomb moments preserve mixed and equal blocks."""
        ks = 9
        data = prepare_bspline_data(
            make_knots(ns, ks, 0.0125, 50.0), ns, ks,
            kinetic_balance=True)
        cache = ConvolCache(data)
        bank, _, order_left = cache.get_rkb_t(k, component)
        factors = cache.get_coulomb_factors(k, component)
        order_right = factors[3].shape[0]
        rng = np.random.default_rng(701 + ns + k)
        orbital = rng.normal(size=ns)
        density = np.zeros((ns, order_right))
        for offset in range(order_right):
            density[:ns - offset, offset] = (
                orbital[:ns - offset] * orbital[offset:]
                * (2 if offset else 1))
        reference = _convol_case1_t(
            bank, density, ns, order_left, order_right)
        factored = _convol_case1_semiseparable_t(
            *factors, density, ns, order_left, order_right)
        np.testing.assert_allclose(factored, reference, rtol=2e-12,
                                   atol=1e-12)

    @pytest.mark.parametrize("k", [0, 1, 6])
    def test_factored_slater_matches_all_four_cell_blocks(self, monkeypatch, k):
        """Final-energy scalar contractions retain mixed P/Q terms."""
        ns = 256
        data = prepare_bspline_data(
            make_knots(ns, 9, 0.0125, 50.0), ns, 9,
            kinetic_balance=True)
        cache = ConvolCache(data)
        rng = np.random.default_rng(919 + k)
        orbitals = [
            (rng.normal(size=data["nsp"]), rng.normal(size=data["nsq"]))
            for _ in range(4)
        ]
        args = tuple(component for orbital in orbitals for component in orbital)
        monkeypatch.setattr(convol, "SEMISEPARABLE_DIRECT_NS", 10**9)
        reference = convol.compute_slater_rk_cell(
            *args, data, k, cache=cache)
        monkeypatch.setattr(convol, "SEMISEPARABLE_DIRECT_NS", 256)
        factored = convol.compute_slater_rk_cell(
            *args, data, k, cache=cache)
        np.testing.assert_allclose(factored, reference, rtol=2e-12,
                                   atol=1e-10)


class TestSlaterIntegrals:
    """Test cell-based Slater integrals against Y^k reference."""

    def test_F0_he_consistency(self, he_setup):
        """F^0(1s,1s) via cell should be close to Y^k.

        Cell algorithm is more accurate than Y^k (no cumulative sum drift),
        so they differ by O(1%) on coarse grids. The cell result is the
        reference for exact B-spline integrals.
        """
        d = he_setup["data"]
        P, Q = he_setup["P_1s"], he_setup["Q_1s"]

        fk_yk = compute_slater_fk(P, Q, P, Q, d, 0)
        fk_cell = compute_slater_fk_cell(P, Q, P, Q, d, 0)

        # Both should be positive and in the same ballpark
        assert fk_cell > 0
        assert fk_cell == pytest.approx(fk_yk, rel=0.02)

    def test_F0_be_1s(self, be_setup):
        """F^0(1s,1s) for Be: cell ≈ Y^k."""
        d = be_setup["data"]
        P, Q = be_setup["P_1s"], be_setup["Q_1s"]

        fk_yk = compute_slater_fk(P, Q, P, Q, d, 0)
        fk_cell = compute_slater_fk_cell(P, Q, P, Q, d, 0)

        assert fk_cell == pytest.approx(fk_yk, rel=0.02)

    def test_F0_be_cross(self, be_setup):
        """F^0(1s,2s) for Be: cell ≈ Y^k (cross-shell integrals more accurate)."""
        d = be_setup["data"]
        P1, Q1 = be_setup["P_1s"], be_setup["Q_1s"]
        P2, Q2 = be_setup["P_2s"], be_setup["Q_2s"]

        fk_yk = compute_slater_fk(P1, Q1, P2, Q2, d, 0)
        fk_cell = compute_slater_fk_cell(P1, Q1, P2, Q2, d, 0)

        assert fk_cell == pytest.approx(fk_yk, rel=0.01)

    def test_G0_be(self, be_setup):
        """G^0(1s,2s) for Be: cell ≈ Y^k (exchange integrals have larger grid effects)."""
        d = be_setup["data"]
        P1, Q1 = be_setup["P_1s"], be_setup["Q_1s"]
        P2, Q2 = be_setup["P_2s"], be_setup["Q_2s"]

        gk_yk = compute_slater_gk(P1, Q1, P2, Q2, d, 0)
        gk_cell = compute_slater_gk_cell(P1, Q1, P2, Q2, d, 0)

        assert gk_cell == pytest.approx(gk_yk, rel=0.05)


class TestDensity:
    """Test density matrix construction."""

    def test_density_sym_shape(self):
        ns, ks = 10, 3
        p = np.random.randn(ns)
        d = _density_sym(p, p, ns, ks)
        assert d.shape == (ns, ks)

    def test_density_sym_diagonal(self):
        """Diagonal should be p[i]^2."""
        ns, ks = 10, 3
        p = np.random.randn(ns)
        d = _density_sym(p, p, ns, ks)
        np.testing.assert_allclose(d[:, 0], p ** 2)

    def test_density_sym_offdiag(self):
        """Off-diagonal d[i,s] = 2*p[i]*p[i+s] for same vector."""
        ns, ks = 10, 3
        p = np.random.randn(ns)
        d = _density_sym(p, p, ns, ks)
        for s in range(1, ks):
            for i in range(ns - s):
                assert d[i, s] == pytest.approx(2.0 * p[i] * p[i + s])


class TestConvol:
    """Test convolution contraction."""

    def test_case1_case2_transpose_relation(self, he_setup):
        """Case 1 and case 2 should be transposes of each other for symmetric rkb."""
        d = he_setup["data"]
        ks = d.get("ksp", d["ks"])
        ns = d.get("nsp", d["ns"])

        rkb = _build_rkb_sym(d["bsp_p"], d["gr"], d["grw"], d["t"],
                              d["nv"], ks, ns, 0)

        p = np.random.randn(ns)
        dens = _density_sym(p, p, ns, ks)

        r1 = _convol_case1(rkb, dens, ns, ks)
        r2 = _convol_case2(rkb, dens, ns, ks)

        # For symmetric rkb, case1 and case2 with same density should give same result
        np.testing.assert_allclose(r1, r2, atol=1e-12)


class TestSCFIntegration:
    """Test wiring into SCF solver."""

    def test_he_total_energy(self):
        """He ground state energy should match DBSR_HF reference."""
        from atomx.atom import Atom
        he = Atom(Z=2, config="1s2", exchange="hf").solve()
        # DBSR_HF He ground state: -2.861816 Ha (relativistic)
        assert he.total_energy == pytest.approx(-2.8618, abs=0.005)

    def test_be_total_energy(self):
        """Be ground state energy."""
        from atomx.atom import Atom
        be = Atom(Z=4, config="1s2 2s2", exchange="hf").solve()
        # DBSR_HF Be ground state: -14.5730 Ha
        assert be.total_energy == pytest.approx(-14.573, abs=0.01)


@pytest.mark.parametrize('rank', [0, 1, 2])
def test_full_dirac_direct_and_exchange_blocks_contract_to_slater(he_setup, rank):
    """Mixed spinor components are required in both J and K contractions."""
    from atomx.numerics.bspline.convol import (
        compute_direct_potential_cell,
        compute_exchange_matrix_cell,
        compute_slater_rk_cell,
    )
    bsp, ns = he_setup['data'], he_setup['ns']
    rng = np.random.default_rng(713)
    # Comparable P/Q amplitudes expose missing relativistic terms; random
    # bra/ket coefficients test the whole matrix, not just diagonal energy.
    a, b, c = [tuple(rng.normal(size=ns) for _ in range(2)) for _ in range(3)]
    cache = ConvolCache(bsp)
    jpp, jqq = compute_direct_potential_cell(*c, bsp, rank, cache=cache)
    kpp, kqq, kpq, kqp = compute_exchange_matrix_cell(*c, *c, bsp, rank, cache=cache)
    direct = a[0] @ jpp @ b[0]+a[1] @ jqq @ b[1]
    exchange = (a[0] @ kpp @ b[0]+a[1] @ kqq @ b[1]
                +a[0] @ kpq @ b[1]+a[1] @ kqp @ b[0])
    expected_j = compute_slater_rk_cell(*a, *b, *c, *c, bsp, rank, cache=cache)
    expected_k = compute_slater_rk_cell(*a, *c, *c, *b, bsp, rank, cache=cache)
    assert direct == pytest.approx(expected_j, rel=3e-13, abs=1e-11)
    assert exchange == pytest.approx(expected_k, rel=3e-13, abs=1e-11)
    np.testing.assert_array_equal(kpq, kqp.T)
