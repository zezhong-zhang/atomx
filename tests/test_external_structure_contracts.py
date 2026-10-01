"""External labels, printed precision, and ensemble identity must match."""

import pytest

from atomx.reference_data.structure_outputs import (
    dbsr_orbital_label,
    orbital_differences,
    parse_dbsr_configurations,
    parse_dbsr_summary,
    parse_radial_summary,
    rounding_bound,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1s", "1s1/2"),
        ("2p-", "2p1/2"),
        ("2p", "2p3/2"),
        ("3d-", "3d3/2"),
        ("3d", "3d5/2"),
        ("4f", "4f7/2"),
        ("5g-", "5g7/2"),
    ],
)
def test_dbsr_j_identity(raw, expected):
    assert dbsr_orbital_label(raw) == expected


def test_summary_uses_final_coulomb_energy_and_preserves_j():
    text = """Etotal = -100.0
  nl         e(nl)          dmp         ns      <r>      max_r
  3d-   -10.5   1.0D-09  64  0.300  9.5
  3d    -10.1   2.0D-09  64  0.310  9.5
Orbital convergence: 2.0D-09
Energy  convergence: 1.0D-12
Coulomb energy: -102.123456789012345 au
Sum: -101.0
"""
    parsed = parse_dbsr_summary(text)
    assert parsed["orbital_energies_Ha"] == {"3d3/2": -10.5, "3d5/2": -10.1}
    assert parsed["total_energy_Ha"] == float("-102.123456789012345")
    assert parsed["residuals"]["orbital"] == 2e-9
    assert parsed["printed_energy_rounding_Ha"] == 5e-16
    assert parsed["printed_radius_rounding_bohr"] == {"3d3/2": 0.0005, "3d5/2": 0.0005}
    assert parsed["orbital_fock_expectations_Ha"] == {}
    final = (
        text
        + """\nFinal orbital Fock expectations (au):
  3d- -10.49999999
  3d  -10.10000001
"""
    )
    diagnostic = parse_dbsr_summary(final)
    assert diagnostic["orbital_fock_expectations_Ha"] == {
        "3d3/2": -10.49999999,
        "3d5/2": -10.10000001,
    }
    assert diagnostic["orbital_energies_Ha"] == parsed["orbital_energies_Ha"]
    with pytest.raises(ValueError, match="every orbital"):
        parse_dbsr_summary(final.replace("  3d  -10.10000001", ""))


def test_dbsr_actual_carbon_ensemble_is_not_nr_statistical_average():
    shells, configs = parse_dbsr_configurations("""Core subshells:
  1s 2s
Peel subshells:
  2p- 2p
CSF(s):
  2p-( 1) 2p ( 1) 1.0000
*
""")
    assert [s.occupation for s in shells] == [2, 2, 1, 1]
    assert len(configs) == 1
    assert configs[0].weight == 1


def test_dbsr_ensemble_aligns_shells_and_normalizes_pair_weights():
    from atomx.config.jj_coupling import compute_WW

    text = """Core subshells:
  1s
Peel subshells:
  2s 2p- 2p
CSF(s):
  2s(1) 2p-(1) 2.0000
  2p(2) 1.0000
*
"""
    shells, configs = parse_dbsr_configurations(text)
    assert sum(c.weight for c in configs) == pytest.approx(1.0)
    for c in configs:
        assert [(s.n, s.kappa) for s in c.subshells] == [(s.n, s.kappa) for s in shells]
        assert sum(s.occupation for s in c.subshells) == 4
    p3 = next(i for i, s in enumerate(shells) if s.kappa == -2)
    assert compute_WW(p3, p3, configs) == pytest.approx(2 / 3)
    assert shells[p3].occupation == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="electron counts"):
        parse_dbsr_configurations(text.replace("2p(2)", "2p(1)"))
    with pytest.raises(ValueError, match="capacity"):
        parse_dbsr_configurations(text.replace("2p-(1)", "2p-(3)"))


def test_radial_printing_precision_is_not_experimental_uncertainty():
    parsed = parse_radial_summary("""Shell: 2p 3/2 (q = 4.000), E = -8.0000000E-01 Eh
Total binding energy = -7.4466717E+03 Eh = -2.0E+05 eV
""")
    assert parsed["printed_energy_rounding_Ha"] == 5e-5
    assert parsed["orbital_energies_Ha"] == {"2p3/2": -0.8}
    assert rounding_bound("-1.2800000D+02") == 5e-6


def test_missing_orbitals_cannot_be_counted_as_zero_error():
    report = orbital_differences({"3d3/2": -10.0}, {"3d5/2": -10.0})
    assert report["max_abs_Ha"] is None
    assert report["missing_calculated"] == ["3d5/2"]


@pytest.mark.parametrize(
    "controls",
    [
        dict(reference_tail_tol=1e-7),
        dict(reference_orbital_tol=float("nan")),
        dict(atomx_orbital_tol=0),
        dict(atomx_n_grid=100.5),
        dict(grid={"hi": float("inf")}),
    ],
)
def test_benchmark_rejects_inconsistent_precision_before_launch(tmp_path, controls):
    from benchmarks.bench_reference_structure import dbsr_case

    with pytest.raises(ValueError):
        dbsr_case(10, tmp_path / "unused", tmp_path, "shooting", 1, **controls)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "parser", [parse_dbsr_summary, parse_radial_summary, parse_dbsr_configurations]
)
def test_failed_or_truncated_output_is_rejected(parser):
    with pytest.raises(ValueError):
        parser("computation stopped")
