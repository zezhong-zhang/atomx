"""Phase-7 scaffold smoke tests.

Each planned calculator exposes a `compute` entry point that raises
``NotImplementedError`` with a tracking-issue reference until the
implementation lands.  These tests pin that contract so the
scaffolds stay loud (importable + documented raise) and can't
silently become no-ops.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# methods/ scaffolds
# ---------------------------------------------------------------------------


def test_mbpt_scaffold_import_and_raises():
    from atomx.methods import mbpt

    assert hasattr(mbpt, "MBPTResult")
    with pytest.raises(NotImplementedError, match="mbpt"):
        mbpt.compute()


def test_coupled_cluster_scaffold_import_and_raises():
    from atomx.methods import coupled_cluster

    assert hasattr(coupled_cluster, "CoupledClusterResult")
    with pytest.raises(NotImplementedError, match="ccsd"):
        coupled_cluster.compute()


def test_rpa_scaffold_import_and_raises():
    from atomx.methods import rpa

    assert hasattr(rpa, "RPAResult")
    with pytest.raises(NotImplementedError, match="rpa"):
        rpa.compute()


# ---------------------------------------------------------------------------
# prop/ scaffolds
# ---------------------------------------------------------------------------


def test_polarizability_scaffold_import_and_raises():
    from atomx.prop import polarizability

    assert hasattr(polarizability, "PolarizabilityResult")
    with pytest.raises(NotImplementedError, match="polarizability"):
        polarizability.compute()


def test_g_factor_scaffold_import_and_raises():
    from atomx.prop import g_factor

    assert hasattr(g_factor, "GFactorResult")
    with pytest.raises(NotImplementedError, match="g_factor"):
        g_factor.compute()


def test_pnc_scaffold_import_and_raises():
    from atomx.prop import pnc

    assert hasattr(pnc, "PNCResult")
    with pytest.raises(NotImplementedError, match="pnc"):
        pnc.compute()


def test_edm_scaffold_import_and_raises():
    from atomx.prop import edm

    assert hasattr(edm, "EDMResult")
    with pytest.raises(NotImplementedError, match="edm"):
        edm.compute()


# ---------------------------------------------------------------------------
# scattering/ scaffolds
# ---------------------------------------------------------------------------


def test_r_matrix_scaffold_import_and_raises():
    from atomx.scattering import r_matrix

    assert hasattr(r_matrix, "RMatrixResult")
    with pytest.raises(NotImplementedError, match="r_matrix"):
        r_matrix.compute()
