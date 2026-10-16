"""Complete run001 initialization diagnosis; no physical solution claim."""

import ctypes
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atomx.methods.core_sd import CoreSD  # noqa: E402
from atomx.methods.valence_sd import plan_valence_sd  # noqa: E402
from atomx.potentials.nuclear import make_nuclear_potential  # noqa: E402


def main():
    out = OWNER / "yb_valence_init002"
    out.mkdir(exist_ok=False)
    source = OWNER / "yb_full_run008"
    backend = OWNER / "yb_full_valence001/backend"
    manifest = json.loads((backend / "manifest.json").read_text())
    plan = json.loads((source / "yb_full_plan.json").read_text())
    occupied, virtual = (tuple(map(tuple, plan[k])) for k in ("occupied_keys", "virtual_keys"))
    contract = dict(
        hypothesis="Angular-only rhoval passed; compare full init and repeated rhoval on identical complete state to locate mutation/layout mismatch",
        fixed="original68core414virtual Yb/V68/ns123/Fermi/l<=3/5Ha, original signed core amplitudes and failed001 executable",
        gate="all expected channels and ordering exact, inspect complete COMMON states before/after; no solve/precision claim",
        stop="retain mismatches; change no angular space or numerical criteria",
        cap_bytes=32 * 1024**3,
        load=os.getloadavg(),
        backend=manifest,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    shutil.copy2(__file__, out / "driver.py")
    with np.load(source / "yb_full_inputs.npz", allow_pickle=False) as data:
        workspace = {k.removeprefix("workspace_"): data[k].copy() for k in data if k.startswith("workspace_")}
        workspace = {k: v.item() if v.ndim == 0 else v for k, v in workspace.items()}
        waves = {key: (data[f"P_{key[0]}_{key[1]}"].copy(), data[f"Q_{key[0]}_{key[1]}"].copy()) for key in occupied + virtual}
    before = time.perf_counter()
    core = CoreSD(waves, workspace, 70, 123, occupied_keys=occupied, virtual_keys=virtual,
                  nuclear_potential=make_nuclear_potential(70, "fermi"), backend_directory=backend,
                  memory_bytes=32 * 1024**3, term21_threads=8, term21_mode="streamed",
                  core_singles_blas=True, core_single_to_doubles_blas=True, radial_representation="polynomial")
    print(json.dumps(dict(stage="prepared", seconds=time.perf_counter()-before)), flush=True)
    lib = core._backend
    d = core.plan["dimensions"]
    nn, nhf = d["NN"], d["NHF"]
    core_common = np.ctypeslib.as_array((ctypes.c_int * (2 * nn + 1)).in_dll(lib, "core_"))
    # The integer tail follows all three original REAL*8 arrays.
    tail = ctypes.addressof((ctypes.c_char).in_dll(lib, "core_")) + 8 * (2 * nhf * nn + nn)
    core_common = np.ctypeslib.as_array((ctypes.c_int * (2 * nn + 1)).from_address(tail))
    ldata = np.ctypeslib.as_array((ctypes.c_int * 4).in_dll(lib, "ldata_"))
    val_tail = ctypes.addressof((ctypes.c_char).in_dll(lib, "val_")) + 8 * (2 * nhf + 1)
    val = np.ctypeslib.as_array((ctypes.c_int * 2).from_address(val_tail))
    def state():
        return dict(core=core_common.tolist(), ldata=ldata.tolist(), val=val.tolist())
    states = dict(before=state())
    keys = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
    vp = plan_valence_sd(core.plan, keys)
    groups = core.plan["groups"]
    key = keys[0]
    slot = next(g.index(key) for g in groups if key in g)
    with np.load(source / "yb_full_solution.npz", allow_pickle=False) as data:
        seed = np.asfortranarray(data["amplitudes"])
    init = lib.atomx_val_init
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")]
    init.restype = None
    init(slot + 1, key[1], len(seed), seed)
    states["after_full_init"] = state()
    def capture(name):
        table = np.zeros((4, vp["nvh"]), dtype=np.int32, order="F")
        values = np.zeros((d["NXX"], d["NXX"], vp["nvh"]), order="F")
        e = np.zeros(1)
        core._call("atomx_val_info", [table, values, e])
        count = np.zeros(1, dtype=np.int32)
        core._call("atomx_val_count", [count])
        expected = np.asarray(vp["channels"][key]).T + 1
        expected[3] -= 1
        np.savez_compressed(out / f"{name}.npz", native=table, expected=expected, count=count)
        record = dict(name=name, count=int(count[0]), expected=len(expected.T), match=bool(np.array_equal(table[:, :expected.shape[1]], expected)), first=table[:, :8].tolist(), energy=float(e[0]), state=state())
        (out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    capture("full_init")
    k, count = ctypes.c_int(0), ctypes.c_int()
    lib.pci_val_rhoval_(ctypes.byref(k), ctypes.byref(count))
    states["after_repeat_rhoval"] = state()
    capture("repeat_rhoval")
    (out / "states.json").write_text(json.dumps(states, indent=2) + "\n")


if __name__ == "__main__":
    main()
