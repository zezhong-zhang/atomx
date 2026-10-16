"""Source-bound numerical/space/experiment gates, matched physical multiplets."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from atomx.methods.fscc import read_heff,resource_estimate,load_integrals
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NAMES={(6,-1):'6s1/2',(6,1):'6p1/2',(6,-2):'6p3/2',(5,2):'5d3/2',(5,-3):'5d5/2'}
EXPERIMENT={'5d3/2':22960.80,'5d5/2':24332.69,'6p1/2':27061.82,'6p3/2':30392.23}
records=[]
for name in ['yb_o14_r1','yb_o22_r1','yb_o22_r2','yb_o68_r2','yb_o14_r1_ccsdt']:
 p=HERE/name
 if not (p/'result.json').exists():
  raise RuntimeError('required run not complete '+name)
 req=json.loads((p/'request.json').read_text());sol=json.loads((p/'result.json').read_text())
 h=read_heff(p/'scratch/HEFF')['0h1p',1];labels=req['integrals']['labels'];act=req['active_indices'];groups={}
 for i,idx in enumerate(act):
  n,k,m=labels[idx];groups.setdefault((n,k),[]).append(i)
 if set(groups)!=set(NAMES):raise RuntimeError('model-space multiplets differ')
 scalar_error=float(np.max(abs(h-np.diag(np.diag(h)))))
 energies={NAMES[k]:float(np.mean(np.diag(h)[v]).real) for k,v in groups.items()}
 scalar_error=max(scalar_error,max(float(np.max(abs(np.diag(h)[v]-energies[NAMES[k]]))) for k,v in groups.items()))
 if scalar_error>1e-9:raise RuntimeError('M/parity/J scalar multiplet gate failed')
 ground=energies['6s1/2'];trans={k:(e-ground)*219474.6313632 for k,e in energies.items()}
 errors={k:trans[k]-v for k,v in EXPERIMENT.items()}
 maxpercent=max(abs(errors[k]/EXPERIMENT[k])*100 for k in EXPERIMENT)
 log=(p/'cc.log').read_text();norms=re.findall(r'Norm \|T\{([^}]+)\}_([123])\| = (\S+)',log)
 peak=re.findall(r'max memory usage = .*? = ([\d.]+) Gb',log)
 records.append(dict(run=name,model=req['model'],occupied=req['resources']['occupied'],virtual=req['resources']['virtual'],core_independent_amplitudes=req['resources']['core_independent_amplitudes'],core_energy_Ha=sol['core_energy_Ha'],valence_attachment_energies_Ha=energies,total_state_energies_Ha={k:sol['core_energy_Ha']+e for k,e in energies.items()},transition_cm=trans,experiment_cm=EXPERIMENT,residual_cm=errors,max_abs_percent=maxpercent,numerical_pass=True,experimental_1percent_gate=maxpercent<1,multiplet_scalar_error_Ha=scalar_error,amplitude_update=sol['amplitude_update_max'],equation_residual_bound_Ha=sol['equation_residual_bound_Ha'],cluster_norms={f'{s}_rank{rank}':float(v) for s,rank,v in norms},backend_peak_GiB=float(peak[-1]) if peak else None,binary_sha256=req['executable_sha256'],integrals_sha256=req['integrals']['files']))
controls=[]
for left,right,label in [(0,1,'open5s5p:14to22core,18virtuals'),(1,2,'18to36virtuals,22core'),(2,3,'openall68core,36virtuals'),(0,4,'CCSDtoCCSDT:14core,18virtuals')]:
 a,b=records[left],records[right]
 controls.append(dict(control=label,core_energy_shift_Ha=b['core_energy_Ha']-a['core_energy_Ha'],total_state_shift_Ha={k:b['total_state_energies_Ha'][k]-a['total_state_energies_Ha'][k] for k in NAMES.values()},transition_shift_cm={k:b['transition_cm'][k]-a['transition_cm'][k] for k in NAMES.values()}))
r=dict(Yb=records,matched_controls=controls,native_anchors=json.loads((HERE/'native_anchors.json').read_text()),Ne=json.loads((HERE/'ne_full_band_summary.json').read_text()),Yb_complete_band=json.loads((HERE/'yb_complete_band_preflight.json').read_text()),operator_coverage='Dirac-Coulomb only; same Fermi nucleus and V68 orbitals for all Yb models; no Breit/QED/recoil appended',experiment_source='Retained measured ASD source and term assignments from rare_earth_yb_spectrum_2026_10_08; not refreshed or fitted',space_1cm_gate=False)
(HERE/'results.json').write_text(json.dumps(r,indent=2)+'\n')
# The independent rank-zero inputs/outputs are kept even when their accuracy fails.
source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/atomx').rglob('*.py'))}
before=json.loads((HERE/'before_source_sha256.json').read_text())
(HERE/'source_identity_chain.json').write_text(json.dumps(dict(before_count=len(before),final_count=len(source),final=source,modified={k:dict(before=before.get(k),after=v) for k,v in source.items() if before.get(k)!=v},old_shared_physical_kernel_changes=[k for k in before if (k.startswith('src/atomx/physics/') or k.startswith('src/atomx/angular/') or k.startswith('src/atomx/numerics/')) and before[k]!=source.get(k)],note='New guards/serialization were refined after large captures. Integral bytes and the same executable bind electronic replay; common native physical kernels unchanged.'),indent=2)+'\n')
print(json.dumps(dict(Yb=[{k:r[k] for k in ('run','max_abs_percent','transition_cm','equation_residual_bound_Ha')} for r in records],controls=controls),indent=2))
