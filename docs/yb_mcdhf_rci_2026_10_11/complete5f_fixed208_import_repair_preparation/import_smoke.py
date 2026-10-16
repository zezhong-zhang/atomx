"""Optional bounded imports only; separate root authorization required, no data."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads(args.binding.read_text())
    contract = json.loads((HERE / "contract.json").read_text())
    preparation = json.loads((HERE / "preparation.json").read_text())
    expected = {"import_smoke_authorized_by_root": True, "scientific_execution_authorized": False,
        "smoke_driver_sha256": sha(Path(__file__).resolve()), "science_driver_sha256": sha(HERE / "run.py"),
        "contract_sha256": sha(HERE / "contract.json"), "preparation_sha256": sha(HERE / "preparation.json"),
        "CPU": 31, "threads": 1, "address_space_GiB": 3.95, "wall_seconds": 30, "outer_wall_seconds": 45}
    if any(seal.get(k) != v for k, v in expected.items()):
        raise ValueError("Separate exactroot import-smoke authorization required; no imports")
    source_root = ROOT / contract["source"]["candidate_archive"]
    for name, value in contract["inputs_sha256"].items():
        if sha(ROOT / name) != value:
            raise ValueError("Input changed before smoke: " + name)
    identity = json.loads((ROOT / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    for name, value in identity["all_atomx_python_sha256"].items():
        if sha(source_root / Path(name).relative_to("src")) != value:
            raise ValueError("Whole356 frozen source changed")
    output = HERE / "import_smoke_actual"
    output.mkdir(exist_ok=False)
    (output / "executed_smoke_source.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    rows = []

    def stop(*_):
        raise TimeoutError("Bounded30s import-smoke cap")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(30)
    try:
        if 31 not in os.sched_getaffinity(0):
            raise RuntimeError("CPU31 unavailable; no silentCPU change")
        os.sched_setaffinity(0, {31})
        limit = int(3.95 * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[name] = "1"
        cache = output / "numba_cache"
        cache.mkdir()
        os.environ["NUMBA_CACHE_DIR"] = str(cache)
        if any(name == "atomx" or name.startswith("atomx.") for name in sys.modules):
            raise ValueError("Fresh process required")
        sys.path.insert(0, str(source_root))
        import numpy as np

        def no_array_load(*_, **__):
            raise RuntimeError("Smoke forbids scientific NPZ/data reads")

        np.load = no_array_load
        for row in preparation["static_import_bindings"]:
            module = importlib.import_module(row["module"])
            if row["kind"] == "submodule":
                imported = importlib.import_module(row["module"] + "." + row["import_name"])
                actual_path = Path(imported.__file__).resolve()
            else:
                getattr(module, row["import_name"])
                actual_path = Path(module.__file__).resolve()
            expected_path = (ROOT / row["source_path"]).resolve()
            if actual_path != expected_path or sha(actual_path) != row["source_sha256"]:
                raise ValueError("Frozen import consumer escaped: " + row["module"])
            rows.append({"module": row["module"], "name": row["import_name"], "passed": True})
        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(source_root.resolve())):
                raise ValueError("Transitive Atomx import escaped frozen356")
        result = {"all_imports_passed": True, "imports": rows, "scientific_calls": 0, "NPZ_array_reads": 0,
                  "source_map_sha256": contract["fixed"]["source_map_sha256"], "binding_sha256": sha(args.binding),
                  "scope": "Import names/actualfiles only; no CSF/nuclear/cache/solver/array constructor consumer calls or physics assurance",
                  "elapsed_uncontrolled_s": time.monotonic() - started}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "completed_imports": rows, "scientific_calls": 0,
            "no_retry": True, "elapsed_uncontrolled_s": time.monotonic() - started}, indent=2) + "\n")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
