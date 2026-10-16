"""Isolated owned Gram-cache candidate; unchanged angular/norm equations."""

from collections import OrderedDict

CACHE_ENTRIES = 256
CACHE_ARRAY_BYTES = 4 * 1024**2


class BoundGrams:
    """Per-consumer cache with measured array residency and explicit limits."""

    def __init__(
        self, uncached, *, max_entries=CACHE_ENTRIES, max_bytes=CACHE_ARRAY_BYTES
    ):
        if any(
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            for limit in (max_entries, max_bytes)
        ):
            raise ValueError(
                "positive integer cache entry and array-byte caps required"
            )
        self.uncached = uncached
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_bytes = self.hits = self.misses = (
            self.evictions
        ) = 0

    def __call__(self, *key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        value = self.uncached(*key)
        payload = sum(array.nbytes for array in value)
        if payload > self.max_bytes:
            raise MemoryError("complete Gram matrix exceeds declared cache allowance")
        while self.cache and (
            len(self.cache) >= self.max_entries
            or self.resident_bytes + payload > self.max_bytes
        ):
            _, old = self.cache.popitem(last=False)
            self.resident_bytes -= sum(array.nbytes for array in old)
            self.evictions += 1
        self.cache[key] = value
        self.resident_bytes += payload
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return value


def plan_scratch(core_plan, valence_plan, key):
    """Separate complete views/metadata, bounded cache and one quartet work."""
    d = core_plan["dimensions"]
    maxrank = d["KK"] + 1
    virtual = set(core_plan["virtual_keys"])
    maxradial = max(sum(k in virtual for k in group) for group in core_plan["groups"])
    maxmag = 2 * max(
        abs(k) for _, k in core_plan["occupied_keys"] + core_plan["virtual_keys"]
    )
    singles = sum(
        sum(key in virtual for key in group)
        for occupied in core_plan["occupied_keys"]
        for group in core_plan["groups"]
        if occupied in group
    ) + len(valence_plan["single_keys"][key])
    # Every decoded rank/channel is a zero-copy radial view, with Python
    # key/dict/view metadata reserved explicitly. No array cube is cached.
    view_metadata = 512 * (
        len(core_plan["channels"]) + len(valence_plan["channels"][key])
    )
    single_metadata = 256 * singles
    group_metadata = 256 * (
        len(virtual) + len(core_plan["occupied_keys"]) + len(core_plan["groups"])
    )
    cache = CACHE_ARRAY_BYTES + 1024 * CACHE_ENTRIES
    # All complete magnetic rows for one ordered quartet, including Python
    # coefficient lists, FP64 conversion and all three signed Gram outputs.
    # Include a complete newly constructed Gram triple and a previous
    # triple still held in the norm frame while replacement is evaluated.
    angular = maxmag**3 * (256 + 128 * maxrank) + 48 * maxrank**2
    radial = 64 * maxrank * maxradial**2
    required = (
        2 * 1024**2
        + view_metadata
        + single_metadata
        + group_metadata
        + cache
        + angular
        + radial
    )
    return dict(
        required_bytes=required,
        view_metadata_bytes=view_metadata,
        single_metadata_bytes=single_metadata,
        group_metadata_bytes=group_metadata,
        angular_cache_array_cap_bytes=CACHE_ARRAY_BYTES,
        angular_cache_metadata_bound_bytes=1024 * CACHE_ENTRIES,
        angular_quartet_transient_bound_bytes=angular,
        radial_quartet_transient_bound_bytes=radial,
        cache_entries=CACHE_ENTRIES,
        maximum_rank_count=maxrank,
        maximum_radial_group_count=maxradial,
        maximum_magnetic_multiplet_count=maxmag,
        cached_Gram_array_bound_bytes=min(
            CACHE_ARRAY_BYTES, CACHE_ENTRIES * 24 * maxrank**2
        ),
        scope="allocated norm payload and conservative Python view/cache metadata; excludes interpreter/Numba/shared Wigner-cache runtime RSS",
    )


def specialize(original):
    """Generate a reviewable small patch, with source-pattern checks."""
    if original.count("@lru_cache(maxsize=4096)\ndef grams(") != 1:
        raise ValueError("one original angular cache definition required")
    original = original.replace("from functools import lru_cache\n\n", "", 1)
    original = original.replace("@lru_cache(maxsize=4096)\ndef grams(", "def grams(", 1)
    begin = original.index("    kappas = {k for _, k in core_plan[")
    end = original.index("    if (\n        isinstance(memory_bytes", begin)
    replacement = (
        "    resource_plan = plan_scratch(core_plan, valence_plan, key)\n"
        "    extra = resource_plan['required_bytes']\n"
        "    angular_grams = BoundGrams(grams)\n"
    )
    changed = original[:begin] + replacement + original[end:]
    prefix, body = changed.split("def norm_components(", 1)
    if body.count(" = grams(") != 3:
        raise ValueError("complete original three Gram call sites required")
    body = body.replace(" = grams(", " = angular_grams(")
    marker = "        extra_scratch_bound_bytes=extra,\n"
    if body.count(marker) != 1:
        raise ValueError("one complete original norm resource return required")
    body = body.replace(
        marker,
        marker
        + "        resource_plan=resource_plan,\n"
        + "        angular_cache=dict(hits=angular_grams.hits, misses=angular_grams.misses, evictions=angular_grams.evictions, peak_array_bytes=angular_grams.peak_bytes),\n",
    )
    changed = prefix + "def norm_components(" + body
    # The proposed patch is self-contained: the private consumer needs no
    # new runtime dependency on this isolated experiment directory.
    from pathlib import Path

    declarations = Path(__file__).read_text().split("\ndef specialize(", 1)[0]
    marker = "def projections(kappa):"
    if changed.count(marker) != 1:
        raise ValueError("one original complete angular projection definition required")
    return changed.replace(marker, declarations + "\n\n" + marker, 1)
