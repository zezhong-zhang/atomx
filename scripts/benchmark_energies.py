#!/usr/bin/env python3
"""Benchmark atomx DHF energies against reference Table S12.

Computes ground-state Dirac-Hartree-Fock energies for Z=2..86 and
compares with the high-accuracy reference from energy_level.csv.
"""

import sys
import csv
import time
import numpy as np

# Parse reference data from CSV
REF_DATA = {}
csv_path = "/Users/zhangzz/code/ASAP/energy_level.csv"
with open(csv_path) as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 3:
            continue
        try:
            z = int(row[0].strip())
            elem = row[1].strip()
            e_ref = float(row[2].strip())
            REF_DATA[z] = (elem, e_ref)
        except (ValueError, IndexError):
            continue

# Import atomx
sys.path.insert(0, "src")
from atomx.atom import Atom

# Elements to compute (skip Z=1 hydrogen, start from He)
z_list = sorted(z for z in REF_DATA.keys() if 2 <= z <= 86)

print(f"{'Z':>3}  {'Elem':>4}  {'E_atomx':>16}  {'E_ref':>16}  {'Diff(Ha)':>12}  {'Diff(mHa)':>10}  {'Status':>8}  {'Time':>6}")
print("-" * 95)

results = []
errors = []
for z in z_list:
    elem, e_ref = REF_DATA[z]
    t0 = time.time()
    try:
        atom = Atom(Z=z, exchange="hf", nuclear="fermi")
        atom.solve(max_iter=80, tol=1e-7)
        dt = time.time() - t0

        if not atom.converged:
            status = "NOCONV"
            e_calc = atom.total_energy
        else:
            status = "OK"
            e_calc = atom.total_energy

        diff = e_calc - e_ref
        diff_mha = diff * 1e3
        results.append((z, elem, e_calc, e_ref, diff))
        print(f"{z:3d}  {elem:>4}  {e_calc:16.8f}  {e_ref:16.8f}  {diff:12.6f}  {diff_mha:10.3f}  {status:>8}  {dt:5.1f}s")
    except Exception as ex:
        dt = time.time() - t0
        errors.append((z, elem, str(ex)[:60]))
        print(f"{z:3d}  {elem:>4}  {'ERROR':>16}  {e_ref:16.8f}  {'---':>12}  {'---':>10}  {'FAIL':>8}  {dt:5.1f}s")
        print(f"     -> {str(ex)[:80]}")

# Summary
if results:
    diffs = [abs(r[4]) for r in results if r[4] is not None]
    print()
    print(f"Computed: {len(results)}/{len(z_list)} elements")
    print(f"Errors:   {len(errors)}")
    if diffs:
        print(f"Mean |diff|: {np.mean(diffs):.6f} Ha = {np.mean(diffs)*1e3:.3f} mHa")
        print(f"Max  |diff|: {np.max(diffs):.6f} Ha = {np.max(diffs)*1e3:.3f} mHa")
        print(f"RMS  |diff|: {np.sqrt(np.mean(np.array(diffs)**2)):.6f} Ha")

if errors:
    print(f"\nFailed elements:")
    for z, elem, msg in errors:
        print(f"  Z={z} ({elem}): {msg}")
