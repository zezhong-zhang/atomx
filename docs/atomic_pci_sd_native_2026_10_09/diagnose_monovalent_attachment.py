"""Resolve finite attachment commutator vs literal SD equation differences."""
import hashlib,json,os,runpy,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OWNER=Path(__file__).resolve().parent
OUT=OWNER/'monovalent_sd002'
OUT.mkdir(exist_ok=False)
for p in list((ROOT/'src/atomx').rglob('*.py'))+[ROOT/'tests/test_monovalent_sd.py',Path(__file__).resolve()]:
 t=OUT/'source'/p.relative_to(ROOT);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
(OUT/'contract.json').write_text(json.dumps({'question':'Isolate core/valence and particle-v contributions to failed canonical Eq9 gate; no tolerance change or promotion','gates_Ha':{'signed_action':1e-11},'load':os.getloadavg()},indent=2))
helpers=runpy.run_path(str(ROOT/'tests/test_monovalent_sd.py'))
class Factory:
 def mktemp(self,name):
  p=OUT/name;p.mkdir();return p
core,result=helpers['live_core'].__wrapped__(Factory())
problem=helpers['problem_for']((core,result))
from atomx.numerics.fermion import apply_terms
n=len(problem.integrals.labels)
g=np.empty((n,)*4)
from itertools import product
for idx in product(range(n),repeat=4):g[idx]=.5*problem.integrals.eri(*idx).real
f=problem.integrals.fock_Ha.real; off=f-np.diag(np.diag(f))
ft=tuple((off[p,q],((p,True),(q,False))) for p,q in product(range(n),repeat=2) if off[p,q]!=0)
records=[];arrays={}
original_core=problem.core_terms
rng=np.random.default_rng(41319)
for case in ('core','singles','doubles_v','doubles_not_v','all'):
 amps=rng.normal(size=problem.plan['amplitudes'])*.013
 if case=='core':amps*=0
 elif case=='singles':amps[problem.plan['singles']:]=0
 elif case=='doubles_v':
  for i,row in enumerate(problem.excitations):
   if len(row[0])!=2 or problem.valence_index not in row[0]:amps[i]=0
 elif case=='doubles_not_v':
  for i,row in enumerate(problem.excitations):
   if len(row[0])!=2 or problem.valence_index in row[0]:amps[i]=0
 problem.core_terms=original_core if case in ('core','all') else ()
 cs,cd,vs,vd=helpers['amplitude_tensors'](problem,amps)
 rs,rd=helpers['paper_rhs'](g,cs,cd,vs,vd,problem.valence_index)
 paper=[rs[problem.valence_index]+problem.orbital_energy_Ha]
 for particles,holes,_ in problem.excitations:
  paper.append(rs[particles[0]] if len(particles)==1 else rd[particles[0],particles[1],holes[1]])
 paper=np.array(paper);paper[1:]+=problem.diagonal_gaps_Ha*amps
 fock,_=helpers['dense_linked'](problem,amps,ft)
 actual=problem.linked_action(amps)
 diff=actual-paper-fock
 arrays[case+'_difference']=diff;arrays[case+'_amplitudes']=amps
 records.append({'case':case,'max_difference_Ha':float(np.max(abs(diff))),'nonzero_above_gate':[(i,(('ref',) if i==0 else problem.excitations[i-1][:2]),float(x)) for i,x in enumerate(diff) if abs(x)>1e-11]})
np.savez_compressed(OUT/'arrays.npz',**arrays)
(OUT/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
