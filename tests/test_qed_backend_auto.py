from atomx.corrections.qed import resolve_se_method
from atomx.methods.dirac_fock import DFSolver, helium_subshells


def test_auto_selects_mss_mohr_for_low_z():
    assert resolve_se_method("auto", 36) == "mohr"


def test_auto_selects_indelicato_for_medium_and_high_z():
    assert resolve_se_method("auto", 37) == "indelicato"
    assert resolve_se_method("auto", 92) == "indelicato"


def test_mss_alias_preserves_mohr_backend_selection():
    assert resolve_se_method("mss", 92) == "mohr"


def test_dfsolver_default_qed_backend_is_auto():
    solver = DFSolver(
        Z=2,
        subshells=helium_subshells(),
        ns=40,
        ks=9,
        exchange="hf",
        include_qed=True,
    )
    assert solver.qed_se_method == "auto"
