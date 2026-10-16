"""Standard-library frozen tracker instrument seam; zero physical imports."""
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


def exercise(module):
    checks, calls, physical_calls = [], [], []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)

    pair = (object(), object())

    @dataclass(frozen=True)
    class FrozenTracker:
        response: tuple

        def resolve(self, *args, **kwargs):
            physical_calls.append((self, args, kwargs.copy()))
            return self.response

    target, other = FrozenTracker(pair), FrozenTracker(pair)
    check("mock_frozen_and_equal_but_distinct", target == other and target is not other)
    frozen_assignment_refused = False
    try:
        target.resolve = lambda: None
    except FrozenInstanceError:
        frozen_assignment_refused = True
    check("original_instance_assignment_refused", frozen_assignment_refused)
    original = FrozenTracker.resolve

    def counted(self, *args, **kwargs):
        calls.append((self, args, kwargs.copy()))
        return original(self, *args, **kwargs)

    FrozenTracker.resolve = counted
    capture = {}
    previous = module.install_resolution_capture(FrozenTracker, target, capture)
    check("previous_existing_counter_preserved", previous is counted)
    arguments = (object(), object(), object(), object())
    result = target.resolve(*arguments, complete=True)
    check("target_exact_original_result_returned", result is pair)
    check("target_captured_exact_return_elements", capture == {"weights": pair[0], "bindings": pair[1]})
    snapshot = capture.copy()
    result_other = other.resolve(*arguments, complete=False)
    check("non_target_exact_original_result_returned", result_other is pair)
    check("equal_non_target_does_not_capture", capture == snapshot)
    token = object()
    result2 = target.resolve(token, final=True)
    check("second_target_result_unchanged", result2 is pair)
    result3 = other.resolve(token, final=False)
    check("second_non_target_result_unchanged", result3 is pair and capture == snapshot)
    check("existing_counter_four_calls_once_each", len(calls) == len(physical_calls) == 4)
    expected = [(target, arguments, {"complete": True}), (other, arguments, {"complete": False}),
                (target, (token,), {"final": True}), (other, (token,), {"final": False})]
    check("self_args_kwargs_exactly_forwarded", all(a[0] is b[0] and a[1:] == b[1:]
          for a, b in zip(calls, expected, strict=True)))
    check("original_method_called_exactly_with_same_inputs", all(a[0] is b[0] and a[1:] == b[1:]
          for a, b in zip(physical_calls, expected, strict=True)))
    check("frozen_target_fields_unchanged", vars(target) == {"response": pair})
    check("frozen_other_fields_unchanged", vars(other) == {"response": pair})
    check("no_instance_resolve_attribute_installed", "resolve" not in vars(target) and "resolve" not in vars(other))
    check("no_numpy_atomx_import", not any(name == "numpy" or name.startswith("numpy.") or name == "atomx"
          or name.startswith("atomx.") for name in sys.modules))
    return {"all_mock_checks_passed": True, "checks": checks, "mock_counter_calls": 4,
            "real_CIBasisTracking_import": False, "physical_calls": 0,
            "NumPy_Atomx_imports": 0, "NPZ_bytes_or_arrays_read": 0}


def main():
    spec = importlib.util.spec_from_file_location("repair_capture_stdlib_only", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    result = exercise(module)
    (HERE / "instrumentation_mock_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['checks'])} frozen mock controls passed; zero physical calls")


if __name__ == "__main__":
    main()
