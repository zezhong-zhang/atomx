"""The public input surface of :class:`atomx.Atom`.

``Atom.__init__`` answers ten physical questions — one argument each.  Those
ten names are frozen here so a future addition/removal fails loudly, and
every keyword this constructor ever accepted *before* the canonical surface
(28 flat keywords plus the six grouped dataclasses of the interim design) is
checked to fail with a message that names its replacement.  There are no
aliases: the old spelling must not bind, and it must not raise a bare
``TypeError`` either.

The physics wired behind the grouped objects (plasma screening, the radial
box, the multi-configuration self-energy backend, the Slater-integral
backend) is measured in ``tests/test_atom_inputs_wired.py``.
"""

from __future__ import annotations

import inspect
import re
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import (  # noqa: E402
    Atom,
    Basis,
    Corrections,
    Correlation,
    Environment,
    Exchange,
    Sampling,
)
from atomx.atom_inputs import (  # noqa: E402
    CORRECTION_PRESETS,
    ENVIRONMENT_MODELS,
    EXCHANGE_MODELS,
    SE_BACKENDS,
)

#: The canonical surface, in signature order.  Adding an argument is a
#: design decision, not a detail: it must be justified here first.
CANONICAL = ("Z", "config", "charge", "nuclear", "exchange", "corrections",
             "correlation", "engine", "basis", "environment")

#: Retired keyword -> (example value, tokens the message must name).
RETIRED = {
    "configs": (["1s2 2s2"], ["config=[", "list"]),
    "numerics": ("shooting", ["engine="]),
    "breit": (False, ["corrections="]),
    "qed": (False, ["corrections="]),
    "qed_backend": ("mss", ["corrections="]),
    "correction_stack": (None, ["corrections="]),
    "alpha_x": (1.0, ["exchange="]),
    "slater_method": ("power_series", ["exchange="]),
    "ns": (120, ["basis="]),
    "ks": (7, ["basis="]),
    "r_max": (80.0, ["basis="]),
    "grid_hi": (0.02, ["basis="]),
    "grid_he": (0.10, ["basis="]),
    "grid_hmax": (1.0, ["basis="]),
    "starting_potential": ("tfda", ["basis="]),
    "sampling": (Sampling.high, ["basis="]),
    "averaging": ("statistical", ["correlation="]),
    "radial_optimization": ("fac_frozen", ["correlation="]),
    "j_resolved_csfs": (True, ["correlation="]),
    "per_block_orbital_sets": (True, ["correlation="]),
    "plasma_ne": (1e22, ["environment="]),
    "plasma_Te": (100.0, ["environment="]),
    # The interim six-group design, replaced by the ten axes.
    "system": (None, ["charge=", "nuclear="]),
    "electronic": (None, ["config="]),
    "interaction": (None, ["exchange=", "corrections="]),
    "ci": (None, ["correlation="]),
}


def _params():
    return list(inspect.signature(Atom.__init__).parameters)


# ----------------------------------------------------------------------
# Mechanical surface freeze
# ----------------------------------------------------------------------

def test_signature_is_exactly_the_canonical_surface():
    """Ten arguments, in this order, plus the retired-name catcher."""
    params = [p for p in _params() if p != "self"]
    assert params[:len(CANONICAL)] == list(CANONICAL), params
    assert params[len(CANONICAL):] == ["retired"], params


def test_only_config_is_positional():
    """Everything after ``config`` is keyword-only.

    The old signature had 28 positional-or-keyword parameters; a positional
    call written against it would now bind ``charge`` as ``ns`` and silently
    solve a different atom.  Keyword-only makes that a loud TypeError.
    """
    sig = inspect.signature(Atom.__init__)
    kinds = {name: p.kind for name, p in sig.parameters.items()
             if name not in ("self", "retired")}
    assert kinds["Z"] == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert kinds["config"] == inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in CANONICAL[2:]:
        assert kinds[name] == inspect.Parameter.KEYWORD_ONLY, name
    Atom(10, "1s2 2s2 2p6")          # the two positional slots work
    with pytest.raises(TypeError):
        Atom(10, "1s2 2s2 2p6", 0, "fermi")


def test_every_argument_is_documented_in_the_class_docstring():
    """Every canonical argument has a ``name : type`` entry."""
    doc = Atom.__doc__ or ""
    missing = [name for name in CANONICAL
               if not re.search(rf"^\s*{name} : ", doc, re.M)]
    assert not missing, f"undocumented Atom arguments: {missing}"


# ----------------------------------------------------------------------
# Retired names must fail helpfully
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name,value,tokens", [
    pytest.param(n, v, t, id=n) for n, (v, t) in RETIRED.items()
])
def test_retired_keyword_names_its_replacement(name, value, tokens):
    with pytest.raises(TypeError) as excinfo:
        Atom(10, **{name: value})
    message = str(excinfo.value)
    assert name in message, message
    assert "retired" in message, message
    for token in tokens:
        assert token in message, f"{name}: {token!r} not in {message!r}"
    assert "MIGRATION" in message, message


def test_unknown_keyword_lists_the_valid_ones():
    with pytest.raises(TypeError) as excinfo:
        Atom(10, conflg="1s2 2s2 2p6")
    message = str(excinfo.value)
    assert "conflg" in message
    for name in CANONICAL:
        assert name in message, name


def test_retired_names_are_not_parameters():
    params = set(_params())
    for name in RETIRED:
        assert name not in params, f"{name} must not be an Atom parameter"


def test_retired_group_dataclasses_are_gone():
    """The interim six-group design leaves no second way to say anything."""
    import atomx
    import atomx.atom_inputs as inputs
    for name in ("System", "ElectronicStructure", "Interaction",
                 "CIProtocol", "GROUPED_INPUTS", "merge_grouped_input"):
        assert not hasattr(inputs, name), name
    for name in ("System", "ElectronicStructure", "Interaction", "CIProtocol"):
        assert not hasattr(atomx, name), name


# ----------------------------------------------------------------------
# The value objects
# ----------------------------------------------------------------------

@pytest.mark.parametrize("obj", [
    Basis(), Corrections(), Correlation(), Environment(), Exchange(),
])
def test_objects_are_frozen_dataclasses(obj):
    assert is_dataclass(obj)
    name = fields(obj)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(obj, name, 1)


def test_exchange_validates_its_own_model():
    assert Exchange("slater", alpha_x=1.0).is_local
    assert Exchange("tfda").is_parametric
    with pytest.raises(ValueError, match="alpha_x"):
        Exchange("gu", alpha_x=0.7)
    with pytest.raises(ValueError, match="model must be one of"):
        Exchange("xalpha")
    with pytest.raises(ValueError, match="slater_method"):
        Exchange("hf", slater_method="power_series")
    with pytest.raises(ValueError, match="slater_method"):
        Exchange("hf", slater_method="nonsense")


def test_corrections_rejects_a_backend_that_cannot_run():
    """``qed_backend`` is only read when the self energy runs."""
    with pytest.raises(ValueError, match="qed=False"):
        Corrections(qed=False, qed_backend="mss")
    Corrections(qed=False, qed_backend="auto")   # the default is fine
    with pytest.raises(ValueError, match="qed_backend must be one of"):
        Corrections(qed_backend="nonsense")
    assert set(CORRECTION_PRESETS) == {"none", "breit", "qed", "breit+qed"}
    assert set(SE_BACKENDS) >= {"auto", "mss", "mohr", "indelicato", "welton"}


def test_corrections_stack_is_authoritative():
    from atomx.corrections.stack import CorrectionStack
    stack = CorrectionStack(se_backend="mohr", include_vp=True,
                            include_breit=False)
    with pytest.raises(ValueError, match="contradicts the correction stack"):
        Atom(10, corrections=Corrections(stack=stack, breit=True))
    with pytest.raises(ValueError, match="contradicts the correction stack"):
        Atom(10, corrections=Corrections(stack=stack, qed=False))
    atom = Atom(10, corrections=Corrections(stack=stack))
    assert (atom.breit, atom.qed) == (False, True)
    assert atom.correction_stack is stack


def test_correlation_rejects_an_unknown_expansion():
    with pytest.raises(ValueError, match="expansion"):
        Correlation("spdf")
    with pytest.raises(ValueError, match="averaging"):
        Correlation(averaging="magic")
    with pytest.raises(ValueError, match="radial_optimization"):
        Correlation(radial_optimization="guess")


def test_basis_and_environment_validate():
    with pytest.raises(ValueError, match="ks"):
        Basis(ks=2)
    with pytest.raises(ValueError, match="r_max"):
        Basis(r_max=-1.0)
    with pytest.raises(ValueError, match="starting_potential"):
        Basis(starting_potential="random")
    with pytest.raises(ValueError, match="plasma_Te"):
        Environment(plasma_ne=1e22)
    with pytest.raises(ValueError, match="plasma_ne"):
        Environment(plasma_Te=10.0)
    with pytest.raises(ValueError, match="screening"):
        Environment(screening="ion_sphere")
    assert set(ENVIRONMENT_MODELS) == {"debye", "stewart_pyatt"}
    assert set(EXCHANGE_MODELS) == {"hf", "slater", "gu", "tfda", "gsz"}


@pytest.mark.parametrize("bad", [3.5, 12])
def test_grouped_arguments_reject_the_wrong_type(bad):
    with pytest.raises(TypeError, match="Basis"):
        Atom(10, basis=bad)
    with pytest.raises(TypeError, match="Environment"):
        Atom(10, environment=bad)
    with pytest.raises(TypeError, match="Exchange"):
        Atom(10, exchange=bad)
    with pytest.raises(TypeError, match="Correlation"):
        Atom(10, correlation=bad)
    with pytest.raises(TypeError, match="Corrections"):
        Atom(10, corrections=bad)


# ----------------------------------------------------------------------
# Resolution: what the constructor actually chose
# ----------------------------------------------------------------------

def test_string_forms_resolve_to_the_object_forms():
    env = Environment(plasma_ne=1e22, plasma_Te=10.0)
    a = Atom(10, exchange="slater", corrections="none", correlation="sd",
             environment=env)
    b = Atom(10, exchange=Exchange("slater"),
             corrections=Corrections(breit=False, qed=False),
             correlation=Correlation(expansion="sd"), environment=env)
    assert a.resolved_inputs == b.resolved_inputs
    assert (a.exchange_model, a.breit, a.qed, a.correlation, a.plasma_ne) \
        == (b.exchange_model, b.breit, b.qed, b.correlation, b.plasma_ne)


def test_correction_presets_map_to_the_booleans():
    for preset, (breit, qed) in CORRECTION_PRESETS.items():
        atom = Atom(10, corrections=preset)
        assert (atom.breit, atom.qed) == (breit, qed), preset
    default = Atom(10)
    assert (default.breit, default.qed) == (True, True)
    backend = Atom(10, corrections=Corrections(qed_backend="mohr"))
    assert backend.qed_backend == "mohr"


def test_engine_default_follows_the_exchange_model():
    assert Atom(10).engine == "shooting"
    assert Atom(10, exchange="slater").engine == "shooting"
    assert Atom(10, exchange="gu").engine == "shooting"
    assert Atom(10, exchange="tfda").engine == "bspline"
    assert Atom(10, exchange="gsz").engine == "bspline"
    assert Atom(10, engine="bspline").engine == "bspline"
    with pytest.raises(ValueError, match="engine must be"):
        Atom(10, engine="finite-difference")
    with pytest.raises(ValueError, match="requires engine='bspline'"):
        Atom(10, engine="shooting", exchange="tfda")


def test_slater_method_only_exists_where_it_is_read():
    """The B-spline Galerkin path has its own Slater quadrature."""
    with pytest.raises(ValueError, match="SHOOTING engine"):
        Atom(26, charge=16, engine="bspline",
             exchange=Exchange("slater", slater_method="power_series"))
    # Accepted on the engine that reads it; the default stays silent.
    atom = Atom(26, charge=16, engine="shooting",
                exchange=Exchange("slater", slater_method="power_series"))
    assert atom.slater_method == "power_series"
    # The default is resolved, but never rejected on the engine that reads
    # it: only an EXPLICIT backend can be dead on the B-spline path.
    assert Atom(26, charge=16, exchange="slater").slater_method == "slag6"


def test_multi_configuration_list_yields_to_bspline():
    atom = Atom(4, config=["1s2 2s2", "1s2 2p2"])
    assert atom.engine == "bspline"
    with pytest.raises(ValueError, match="multi-configuration"):
        Atom(4, config=["1s2 2s2", "1s2 2p2"], engine="shooting")
    # A local exchange model on the multi-configuration path is caught
    # when the MCDHF solver is selected (solve(), not construction).
    with pytest.raises(ValueError, match="multi-configuration"):
        Atom(4, config=["1s2 2s2", "1s2 2p2"], exchange="slater").solve()


def test_config_list_and_expansion_are_mutually_exclusive():
    with pytest.raises(ValueError, match="already lists the configurations"):
        Atom(4, config=["1s2 2s2", "1s2 2p2"], correlation="sd")


def test_single_element_list_still_means_multi_configuration():
    assert Atom(10, config="1s2 2s2 2p6")._multi_config is False
    assert Atom(10, config=["1s2 2s2 2p6"])._multi_config is True


def test_sampling_tier_presets_the_basis_fields():
    atom = Atom(10, basis=Basis(sampling=Sampling.high))
    assert atom.ns == 200
    assert atom.ks == 9
    # Explicit Basis fields win over the tier.
    other = Atom(10, basis=Basis(ns=77, sampling=Sampling.high))
    assert other.ns == 77


def test_tier_fields_reach_the_shooting_grid():
    default = Atom(10, engine="shooting")
    assert default._n_grid is None
    assert "n_grid=1000" in default.resolved_inputs
    tiered = Atom(10, engine="shooting",
                  basis=Basis(sampling=Sampling(n_grid=1500)))
    assert tiered._n_grid == 1500
    assert "n_grid=1500" in tiered.resolved_inputs


def test_grid_overrides_still_need_the_bspline_physics_grid():
    with pytest.raises(ValueError, match="shooting engine"):
        Atom(10, engine="shooting", basis=Basis(grid_hi=0.02))
    with pytest.raises(ValueError, match="replaces"):
        Atom(10, engine="bspline", basis=Basis(ns=150, grid_hi=0.02))
    # Accepted where it means something.
    assert Atom(10, engine="bspline",
                basis=Basis(grid_hi=0.02, grid_he=0.10)).grid_hi == 0.02


def test_multi_configuration_flags_need_a_multi_configuration():
    with pytest.raises(ValueError, match="multi-configuration"):
        Atom(7, correlation=Correlation(j_resolved_csfs=True))
    with pytest.raises(ValueError, match="multi-configuration"):
        Atom(7, correlation=Correlation(per_block_orbital_sets=True))
    assert Atom(7, correlation=Correlation(
        j_resolved_csfs=True, expansion="sd")).j_resolved_csfs


def test_explicit_ci_averaging_needs_hf_exchange():
    with pytest.raises(ValueError, match="averaging='ci' requires"):
        Atom(10, exchange="slater", correlation=Correlation(averaging="ci"))
    # The defaulted "ci" falls back quietly instead (see solve()).
    assert Atom(10, exchange="slater").averaging == "ci"


def test_stewart_pyatt_needs_a_charged_ion():
    env = Environment(plasma_ne=1e23, plasma_Te=50.0,
                      screening="stewart_pyatt")
    with pytest.raises(ValueError, match="ion-sphere"):
        Atom(26, environment=env)
    assert Atom(26, charge=16, environment=env).screening_model \
        == "stewart_pyatt"


def test_removed_legacy_exchange_strings_name_the_replacement():
    with pytest.raises(ValueError, match="was removed"):
        Atom(26, exchange="fac")
    with pytest.raises(ValueError, match="engine='shooting'"):
        Atom(26, exchange="shooting")


def test_type_enum_members_are_still_accepted_as_values():
    """``atomx.types.Exchange`` is the method API's enum, not a spelling."""
    from atomx.types import Exchange as LegacyExchange
    atom = Atom(26, charge=16, exchange=LegacyExchange.SLATER)
    assert atom.exchange_model == "slater"
    assert atom.engine == "shooting"


def test_resolved_inputs_reports_the_wired_inputs():
    atom = Atom(10, charge=2, basis=Basis(r_max=88.0),
                environment=Environment(plasma_ne=1e22, plasma_Te=100.0,
                                        screening="stewart_pyatt"),
                corrections="none")
    text = atom.resolved_inputs
    assert "r_max=88.0" in text
    assert "breit=off" in text
    assert "plasma(ne=1e+22, Te=100.0, stewart_pyatt)" in text
