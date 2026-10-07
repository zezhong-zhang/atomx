"""Paired, load-guarded Gauss-rule timing; no atomic solve or physical claim.

Run with BLAS threads pinned to one. The baseline and candidate retain the
same quadrature order and interval; validate moments separately from timing.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from atomx.numerics.quadrature import _gauss_legendre_rule, gauss_legendre_rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=2000)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    if not (1 <= args.order <= 4000 and args.pairs >= 1):
        parser.error("require order in [1,4000] and at least one pair")
    # Baseline's dense companion alone needs O(order²) storage; cap 128 MiB.
    cpu_count = os.cpu_count() or 1
    max_load = 0.5 * cpu_count
    rows = []
    with threadpool_limits(limits=1):
        for pair in range(args.pairs):
            row = dict(pair=pair, load_before=os.getloadavg())
            rules = {}
            kinds = ("numpy", "shared") if pair % 2 == 0 else ("shared", "numpy")
            for kind in kinds:
                if kind == "shared":
                    _gauss_legendre_rule.cache_clear()
                start = time.perf_counter()
                rules[kind] = (np.polynomial.legendre.leggauss(args.order)
                               if kind == "numpy" else gauss_legendre_rule(args.order))
                row[kind + "_cold_s"] = time.perf_counter() - start
            x0, w0 = rules["numpy"]
            x1, w1 = rules["shared"]
            row["max_node_difference"] = float(np.max(np.abs(x0-x1)))
            row["max_weight_difference"] = float(np.max(np.abs(w0-w1)))
            errors = []
            for x, w in rules.values():
                if not np.all(w > 0):
                    raise RuntimeError("nonpositive Gauss weights")
                for degree in (0, 2, 8, 2*args.order-2):
                    if degree >= 2*args.order:
                        continue
                    errors.append(abs(float(w @ x**degree) - 2.0/(degree+1)))
            row["max_moment_error"] = max(errors)
            if row["max_moment_error"] > 2e-12:
                raise RuntimeError("polynomial moment gate failed")
            start = time.perf_counter()
            for _ in range(1000):
                gauss_legendre_rule(args.order)
            row["shared_warm_s_per_call"] = (time.perf_counter()-start)/1000
            row["load_after"] = os.getloadavg()
            rows.append(row)
    load_pass = all(max(r["load_before"][0], r["load_after"][0]) <= max_load for r in rows)
    result = dict(order=args.order, pairs=args.pairs, cpu_count=cpu_count,
                  max_load=max_load, load_guard_passed=load_pass,
                  numpy_version=np.__version__, scipy_version=scipy.__version__, rows=rows)
    if load_pass:
        result["median_cold_ratio"] = float(np.median([
            r["numpy_cold_s"]/r["shared_cold_s"] for r in rows]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
