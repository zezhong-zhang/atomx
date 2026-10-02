import importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks')]
from atomx.config.jj_coupling import generate_jj_average
from atomx.methods.dirac_fock import DFSolver
from atomx.constants import HARTREE_EV
from atomx.scattering.models import ContinuumModel
from atomx.scattering.radiative_recombination import _shooting_photoionization_cross_section
import bench_systematic_xray as b

ref=json.loads((ROOT/'docs/systematic_xray_2026_10_02/references.json').read_text())
for el in ('C','Ne'):
 for exchange in ('hf','gu'):
  results={}
  for hole in (None,'1s','2p'):
   shells=dict(b.CASES[el][1])
   if hole:shells[hole]-=1
   subs,cfgs=generate_jj_average(b.config(shells))
   solver=DFSolver(b.CASES[el][0],subs,jj_configs=cfgs,exchange=exchange,ns=180,ks=9,r_max=80,nuclear='point',starting_potential='tfda')
   r=solver.solve(max_iter=300,tol=1e-9,damping=.5,mode='orbital')
   assert r.converged
   results[hole]=r
  delta=(results['1s'].total_energy-results['2p'].total_energy)*HARTREE_EV
  rad=b.radiative(results['1s'],'1s',{'2p->1s':delta})['radiative_per_s']
  pi=b.pi(results[None],ref[el]['photons_eV'],16000)
  print(el,exchange,'deltaSCF',delta,'rad/FAC',rad/ref[el]['decay']['1s']['fac']['radiative_per_s'],'PI/FAC',[(s,[round(x/y,4) for x,y in zip(v,ref[el]['fac_pi'][s])]) for s,v in pi.items()],flush=True)
