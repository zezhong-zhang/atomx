"""Private complete homogeneous diagram decomposition and stage observations.

The original source-bound COMMON layout and symbol extent are checked before
reading any native array. Outputs are compared with the actual full map; no
component timing can replace its numerical or physical gates.
"""

import ctypes
import hashlib
import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np


def profile(valence, key, energy, vector, out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    dims = valence.core.plan["dimensions"]
    shape = (dims["NXX"], dims["NXX"], valence.plan["nvh"])
    size, nx = int(np.prod(shape)), dims["NX"]
    count = valence.plan["raw_amplitudes"][tuple(key)]
    bound = (
        valence.plan["complete_payload_bound_bytes"]
        + 8 * (20 * size + 12 * count)
        + 3 * valence.core.source.nbytes
    )
    if bound > valence.core.memory_bytes:
        raise MemoryError("complete homogeneous profiling payload exceeds host cap")
    lib = valence.core._backend
    library = Path(lib._name)
    symbols = subprocess.check_output(
        ["nm", "-S", "--defined-only", str(library)], text=True
    )
    extents = {
        row.split()[-1]: int(row.split()[1], 16)
        for row in symbols.splitlines()
        if len(row.split()) == 4
    }
    if any(extents.get(name) != 8 * (size + nx) for name in ("rhovout_", "drhovout_")):
        raise ValueError("source-bound complete homogeneous COMMON extent differs")
    source = (library.parent / "valence_diagrams.f").read_text()
    if "COMMON /rhovout/ xvo(nxx,nxx,nvh),rvo(nx)" not in source:
        raise ValueError("actual original homogeneous COMMON layout not bound")
    common = [
        np.ctypeslib.as_array((ctypes.c_double * (size + nx)).in_dll(lib, name))
        for name in ("rhovout_", "drhovout_")
    ]
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags=("F_CONTIGUOUS", "ALIGNED"))
    term = lib.atomx_val_term
    term.argtypes, term.restype = [ctypes.c_int, ctypes.c_double, array, array], None
    fock = lib.atomx_val_fock_map
    fock.argtypes, fock.restype = [ctypes.c_double, *([array] * 6)], None
    outin, singles = lib.pci_val_outin_, lib.pci_val_terma4_
    outin.argtypes, outin.restype = [], None
    singles.argtypes, singles.restype = [ctypes.POINTER(ctypes.c_double)], None
    records = {}

    def timed(name, function):
        begin = time.perf_counter()
        result = function()
        records[name] = dict(
            seconds=time.perf_counter() - begin, load_after=os.getloadavg()
        )
        return result

    with valence.core._lock:
        context = valence._initialize(key, energy)
        expected = timed(
            "complete_homogeneous_value",
            lambda: valence._evaluate_value(energy, vector, context),
        )
        index, (d, dd, _, _), _ = valence._load_inputs(vector, None, context)
        zero = np.zeros(shape, order="F")
        parts = {}
        common[0][:size] = d.ravel(order="F")
        common[0][size:] = 0
        common[1][:] = 0
        outin()
        e = ctypes.c_double(energy)
        timed("terma4_full_legacy_response", lambda: singles(ctypes.byref(e)))
        parts["terma4"] = (
            common[0][size:].copy(),
            common[0][:size].reshape(shape, order="F")[index].copy(),
        )
        for number, label in [(2, "term2"), (4, "term41")]:
            output = np.zeros(shape, order="F")
            timed(
                label + "_full_legacy_response",
                partial(term, number, energy, d, output),
            )
            parts[label] = (np.zeros(nx), output[index])

        def fock_output(input_data):
            rs, drs = np.zeros(nx), np.zeros(nx)
            rd, drd = np.zeros(shape, order="F"), np.zeros(shape, order="F")
            fock(energy, input_data, dd, rs, rd, drs, drd)
            return rs.copy(), rd[index]

        fixed = timed("fock_zero_full_core_response", lambda: fock_output(zero))
        full = timed("fock_vector_full_core_response", lambda: fock_output(d))
        parts["homogeneous_fock"] = tuple(
            a - b for a, b in zip(full, fixed, strict=True)
        )
        summed = tuple(sum(part[i] for part in parts.values()) for i in range(2))
        errors = [
            float(np.max(abs(a - b))) for a, b in zip(expected, summed, strict=True)
        ]
        if (
            not all(np.all(np.isfinite(a)) for a in (*expected, *summed))
            or max(errors) > 1e-11
        ):
            raise AssertionError("complete signed homogeneous decomposition failed")
        np.savez_compressed(
            out / "outputs.npz",
            vector=vector,
            expected_s=expected[0],
            expected_d=expected[1],
            reconstructed_s=summed[0],
            reconstructed_d=summed[1],
            **{
                label + suffix: pair[i]
                for label, pair in parts.items()
                for i, suffix in enumerate(("_s", "_d"))
            },
        )
    record = dict(
        status="pass",
        key=key,
        energy_Ha=energy,
        complete_signed_errors_Ha=errors,
        payload_bound_bytes=bound,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        common_extents_bytes={
            name: extents[name] for name in ("rhovout_", "drhovout_")
        },
        observations=records,
        timing_scope="single observed stage executions under recorded load; component term2/terma4/term41 retain legacy full-response work; Fock source subtraction is diagnostic only; no controlled or summed timing claim",
        role="complete homogeneous decomposition, not a solve, convergence, or experimental result",
    )
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
