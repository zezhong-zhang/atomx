"""Regression checks for human-readable atomic calculation logs."""

from atomx.log import CalculationLog
from atomx.methods.dirac_fock import SCFResult


def test_log_accepts_text_correction_metadata(tmp_path):
    result = SCFResult(
        converged=True,
        iterations=27,
        residual=5e-5,
        total_energy=-24.0,
        orbital_energies={"1s1/2": -10.0},
        subshells=[],
        energy_components={
            "E_Breit": 0.001,
            "E_SE_backend": "mohr",
        },
    )
    path = tmp_path / "calculation.log"
    log = CalculationLog(str(path), verbose=False)
    log.converged(result)
    log.close()

    content = path.read_text()
    assert "SCF CONVERGED in 27 iterations" in content
    assert "E_Breit" in content
    assert "E_SE_backend" in content and "mohr" in content
