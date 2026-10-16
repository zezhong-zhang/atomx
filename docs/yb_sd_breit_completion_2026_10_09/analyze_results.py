"""Common-ground transitions; no experimental centering or transplanted shifts."""
import json
from pathlib import Path
import sys
from atomx.constants import HARTREE_CM1
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'yb_diagnosis_repair_2026_10_08'))
from common import EXPERIMENT
rows={label:json.loads((HERE/(label.replace('/','_')+'_result.json')).read_text()) for label in EXPERIMENT}
g=rows['6s_1/2']
for label,r in rows.items():
    r['transitions_cm1']={model:(r[model]['total_Ha']-g[model]['total_Ha'])*HARTREE_CM1 for model in ('one_hole_DC','DC','DCB')}
    r['experimental_cm1']=EXPERIMENT[label]
    r['residuals_cm1']={model:energy-EXPERIMENT[label] for model,energy in r['transitions_cm1'].items()}
    controls=r['restricted_H0_control']; gc=g['restricted_H0_control']
    r['restricted_B1_transition_cm1']=(controls['first_order_breit_Ha']-gc['first_order_breit_Ha'])*HARTREE_CM1
    r['restricted_CB2_transition_cm1']=(controls['coulomb_breit_Ha']-gc['coulomb_breit_Ha'])*HARTREE_CM1
    r['correlated_first_B_transition_cm1']=r['first_order_correlated_Breit_cm1']-g['first_order_correlated_Breit_cm1']
    r['full_B_transition_cm1']=r['full_Breit_response_cm1']-g['full_Breit_response_cm1']
    r['nonlinear_B_transition_cm1']=r['nonlinear_Breit_cm1']-g['nonlinear_Breit_cm1']
models=('one_hole_DC','DC','DCB')
maxerr={m:max(abs(r['residuals_cm1'][m])/r['experimental_cm1'] for label,r in rows.items() if label!='6s_1/2') for m in models}
splittings={tag:{**{m:rows[b]['transitions_cm1'][m]-rows[a]['transitions_cm1'][m] for m in models},'experiment':EXPERIMENT[b]-EXPERIMENT[a]} for tag,a,b in [('5d','5d_3/2','5d_5/2'),('6p','6p_1/2','6p_3/2')]}
result=dict(rows=rows,max_excitation_relative_error=maxerr,splittings_cm1=splittings,
    numerical_pass=all(r['numerical_pass'] for r in rows.values()),
    experimental_1_percent_pass=maxerr['DCB']<.01,
    max_two_hole_transition_change_cm1=max(abs(r['transitions_cm1']['DC']-r['transitions_cm1']['one_hole_DC']) for r in rows.values()),
    space_convergence_1_cm1_pass=max(abs(r['transitions_cm1']['DC']-r['transitions_cm1']['one_hole_DC']) for r in rows.values())<1,
    limits='finite NR-CISD in five outer spinors, frozen Xe and orbitals; not linearized all-order SD/CC/FS-CCSDT; restricted CB2 not full atomic B2')
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
for label,r in rows.items():
    print(label,r['transitions_cm1'],r['restricted_CB2_transition_cm1'],r['full_B_transition_cm1'])
