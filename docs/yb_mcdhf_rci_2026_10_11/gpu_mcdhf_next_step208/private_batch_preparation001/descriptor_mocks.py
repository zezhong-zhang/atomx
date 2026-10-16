"""Standard-library descriptor admission only: zero arrays/kernels/imports."""

import importlib.abc
import importlib.util
import sys
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path


class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in {"numpy", "numba", "cupy", "atomx", "scipy"}:
            raise RuntimeError(f"numerical import forbidden during mocks: {fullname}")
        return None


@dataclass(frozen=True)
class Flags:
    c_contiguous: bool = True


@dataclass(frozen=True)
class Descriptor:
    shape: tuple
    dtype: str = "float64"
    flags: Flags = Flags()


def run():
    forbidden = {"numpy", "numba", "cupy", "atomx", "scipy"}
    if any(name.split(".")[0] in forbidden for name in sys.modules):
        raise RuntimeError("mocks require a clean standard-library process")
    blocker = NoNumericalImports()
    sys.meta_path.insert(0, blocker)
    checks = []
    try:
        path = Path(__file__).with_name("batch_executor.py")
        spec = importlib.util.spec_from_file_location("private_native_batch_preparation", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        def check(name, condition):
            checks.append({"name": name, "passed": bool(condition)})
            if not condition:
                raise AssertionError(name)

        def refusal(name, action, exception=ValueError):
            try:
                action()
            except exception:
                check(name, True)
            else:
                check(name, False)

        keys = ((1, -1), (2, -1))
        jobs = ((6, keys[0], keys[1]), (0, keys[0], keys[0]))
        plan = module.BatchPlan("1" * 64, "2" * 64, module._digest(jobs),
                                123, (5, 6), (122, 123), keys, jobs)
        check("unequal_RKB_orders_extent_admitted", plan.orders == (5, 6))
        check("source_insertion_order_not_sorted", plan.jobs == jobs)
        check("exact_four_components_canonical", module.COMPONENTS ==
              ("pppp", "pqpq", "qpqp", "qqqq"))
        check("unique_pair_pool_keeps_order", plan.pairs == ((keys[0], keys[1]), (keys[0], keys[0])))
        check("factor_rank_pool_order", plan.ranks == (0, 6))
        refusal("ns95_no_fallback", lambda: replace(plan, ns=95))
        refusal("orders_not_tuple", lambda: replace(plan, orders=[5, 6]))
        refusal("bool_is_not_ns", lambda: replace(plan, ns=True))
        refusal("over_order_budget", lambda: replace(plan, orders=(10, 6)))
        refusal("extent_less_than_order", lambda: replace(plan, extents=(4, 123)))
        refusal("extent_greater_than_ns", lambda: replace(plan, extents=(122, 124)))
        refusal("invalid_source_SHA", lambda: replace(plan, source_sha256="not-an-epoch"))
        refusal("duplicate_keys", lambda: replace(plan, keys=(keys[0], keys[0])))
        refusal("source_union_SHA_mismatch", lambda: replace(plan, signed_source_union_sha256="3" * 64))
        for label, invalid in (
            ("negative_rank", ((-1, keys[0], keys[0]),)),
            ("rank7", ((7, keys[0], keys[0]),)),
            ("duplicate_source", (jobs[0], jobs[0])),
            ("wrong_source_order", ((0, keys[1], keys[0]),)),
            ("unknown_key", ((0, keys[0], (3, -1)),)),
        ):
            refusal(label, lambda invalid=invalid: replace(
                plan, jobs=invalid, signed_source_union_sha256=module._digest(invalid)))
        refusal("chunk129", lambda: replace(plan, chunk_sources=129))
        refusal("frozen_plan_mutation", lambda: setattr(plan, "ns", 96), FrozenInstanceError)
        module.require_descriptor(Descriptor((123, 5)), (123, 5))
        check("exact_float64_descriptor", True)
        for label, descriptor in (
            ("no_float32_cast", Descriptor((123, 5), "float32")),
            ("no_complex_cast", Descriptor((123, 5), "complex128")),
            ("no_noncontiguous_copy", Descriptor((123, 5), flags=Flags(False))),
            ("no_shape_clip", Descriptor((123, 6))),
            ("no_big_endian_cast", Descriptor((123, 5), ">f8")),
        ):
            refusal(label, lambda descriptor=descriptor: module.require_descriptor(descriptor, (123, 5)))
        refusal("snapshot_only_through_owned_factory", lambda: module.DensitySnapshot(), TypeError)
        all_keys = tuple((n, -1) for n in range(1, 32))
        all_jobs = tuple((rank, a, b) for rank in range(7)
                         for i, a in enumerate(all_keys) for b in all_keys[i:])
        worst = module.BatchPlan("1" * 64, "2" * 64, module._digest(all_jobs),
                                 123, (9, 9), (123, 123), all_keys, all_jobs)
        memory = module.resource_preflight(worst)
        check("full3472_sources_no_amplitude_cutoff", memory["source_jobs"] == 3472)
        check("full13888_component_jobs", memory["component_jobs"] == 13888)
        check("independent_component_scan_memory", memory["chunk_prefix_suffix_bytes"] == 9142272)
        check("factor_memory_complete_four", memory["factor_bytes"] == 38930976)
        check("source_density_H2D_complete", memory["unique_density_bytes"] == 8785152)
        check("full_D2H_inclusive", memory["all_band_D2H_bytes_upper"] == 122992128)
        check("owned_payload_within512MiB", max(memory["owned_host_bytes_upper"],
                                               memory["owned_device_bytes_upper"]) < module.MAX_OWNED_BYTES)
        check("no_numerical_modules_loaded", not any(
            name.split(".")[0] in forbidden for name in sys.modules))
        return {"checks_count": len(checks), "all_passed": all(c["passed"] for c in checks),
                "checks": checks, "mechanical31_worst_case": memory,
                "numerical_imports": 0, "arrays_or_NPZ": 0, "numerical_calls": 0,
                "scope": "descriptor/mock/source-admission only, not FP64 or CPU/GPU qualification"}
    finally:
        sys.meta_path.remove(blocker)


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
