"""Source-bound strong-rotation raw-residual diagnosis, no tolerance changes."""
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'reduced_monovalent002'
OUT.mkdir(exist_ok=False)
contract = dict(hypothesis='Find whether strong-rotation raw gate failure is a state-dependent map discrepancy, denominator/coordinate conversion or cancellation',fixed='same .37rad rotated complete twelve-spinor core and original reduced backend, original signed1e-11/raw1e-10/amplitude1e-10 gates',stop='No production promotion before diagnosis and unchanged-gate proof')
(OUT/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
files=list((ROOT/'src/atomx').rglob('*.py'))+list((ROOT/'src/atomx/methods').glob('_pci_*.f90'))+[ROOT/'tests/test_reduced_monovalent_sd.py',ROOT/'tests/test_monovalent_sd.py',Path(__file__)]
hashes={}
for path in files:
 rel=path.relative_to(ROOT); dest=OUT/'source'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest); hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
sys.path.insert(0,str(ROOT/'tests'))
helpers=runpy.run_path(str(ROOT/'tests/test_reduced_monovalent_sd.py'))
class Factory:
 def mktemp(self,name):
  path=OUT/name; path.mkdir(); return path
core,result,problem=helpers['reduced_core'].__wrapped__(Factory(),SimpleNamespace(param='rotated'))
key=(2,-1)
v=core.integrals.labels.index((2,-1,-1))
finite=helpers['FiniteMonovalentSD'](core,result,v,work_cap=200_000_000)
reference=finite.solve()
actual=problem.solve(key,residual_tolerance_Ha=1e-12)
physical=helpers['expanded'](problem,key,actual.amplitudes,finite)
independent,_=helpers['dense_linked'](finite,physical)
energy,rhs=problem.evaluate(key,actual.amplitudes)
projected=helpers['expanded'](problem,key,rhs,finite)+finite.diagonal_gaps_Ha*physical
raw=independent[1:]-actual.correlation_energy_Ha*physical
native_raw=projected-actual.correlation_energy_Ha*physical
report=dict(native_declared_raw=actual.residual_max_Ha,independent_raw=float(max(abs(raw))),expanded_native_raw=float(max(abs(native_raw))),map_at_root_error=float(max(abs(projected-independent[1:]))),energy_map_error=energy+finite.orbital_energy_Ha-independent[0],root_error=actual.correlation_energy_Ha-reference.correlation_energy_Ha,amplitude_error=float(max(abs(physical-reference.amplitudes))),large_rows=[dict(row=i,coordinate=str(finite.excitations[i][:2]),raw=float(value),native=float(native_raw[i]),map_difference=float(projected[i]-independent[i+1])) for i,value in enumerate(raw) if abs(value)>1e-11],continuation=actual.continuation_path)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(OUT/'arrays.npz',reduced=actual.amplitudes,physical=physical,finite_reference=reference.amplitudes,independent=independent,expanded_native=projected,raw=raw,native_raw=native_raw,rhs=rhs)
print(json.dumps(report),flush=True)
