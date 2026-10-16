"""Matched spherical native streaming inputs, retained and expanded Yb pool."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT/'docs/atomic_allorder_fscc_2026_10_09'
sys.path.insert(0, str(ROOT/'docs/yb_diagnosis_repair_2026_10_08'))
from common import load_reference
from atomx.physics.csf.spinor_integrals import build_spinor_integrals
from atomx.methods.fscc import compute, export_integrals, load_integrals, read_heff, resource_estimate

NAMES={(6,-1):'6s1/2',(6,1):'6p1/2',(6,-2):'6p3/2',(5,2):'5d3/2',(5,-3):'5d5/2'}
EXPERIMENT={'5d3/2':22960.80,'5d5/2':24332.69,'6p1/2':27061.82,'6p3/2':30392.23}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--radials', type=int, choices=[2,3,4], required=True)
    ap.add_argument('--representation', choices=['packed','on_demand'], default='on_demand')
    ap.add_argument('--action', choices=['solve','export'], default='solve')
    ap.add_argument('--name', required=True)
    args=ap.parse_args()
    start=time.monotonic()
    core,*_=load_reference()
    from atomx.methods.mbpt import build_virtual_basis
    od={k:v[:2] for k,v in core.orbitals_by_subshell().items()}
    frozen=tuple(k for k in od if k not in ((4,3),(4,-4),(5,-1),(5,1),(5,-2)))
    occupied=tuple(k for k in od if k not in frozen)
    vir=[]; eps={}
    for k in (-1,1,-2,2,-3):
        _,states=build_virtual_basis(core,k,e_max=5,potential_mode='VN',orthogonal_to_occupied=True)
        n=max(n for n,k0 in od if k0==k)+1
        assert len(states)>=args.radials
        for j,(P,Q,e) in enumerate(states[:args.radials]):
            key=n+j,k; od[key]=P,Q; vir.append(key); eps[str(key)]=e
    a=build_spinor_integrals(od,core._bsp_data,70,123,occupied_keys=occupied,
        virtual_keys=vir,frozen_keys=frozen,nuclear_potential=core._nuclear_potential,
        max_integral_bytes=2*1024**3,representation=args.representation)
    construction_seconds=time.monotonic()-start
    active=tuple(i for i,(n,k,m) in enumerate(a.labels) if (n,k) in vir[::args.radials])
    estimate=resource_estimate(a,sector='0h1p',active_count=len(active))
    pre=dict(radials=args.radials,representation=args.representation,active_indices=active,
        virtual_definition=f'lowest {args.radials} VN complement eigenstates per s/p/d kappa, not a full band',
        virtual_energies_Ha=eps,construction_seconds=construction_seconds,resources=estimate,
        owned_one_body_bytes=a.one_body_Ha.nbytes+a.fock_Ha.nbytes)
    (HERE/(args.name+'_preflight.json')).write_text(json.dumps(pre,indent=2)+'\n')
    print(json.dumps(pre),flush=True)
    directory=HERE/args.name
    if args.action=='export':
        manifest=export_integrals(a,directory)
        reference_directory = (OLD/'yb_o22_r2' if args.radials==2
                               else HERE/'yb22_v54_stream' if args.radials==3 else None)
        old=load_integrals(reference_directory) if reference_directory else None
        checks=dict(eri_sha_matches_old=manifest['files']['ATOMX_TWO']==json.loads(
            (reference_directory/'integrals.json').read_text())['files']['ATOMX_TWO']) if old else {}
        if old:
            checks.update(fock_max_error_Ha=float(np.max(abs(old.fock_Ha-a.fock_Ha))),
                one_body_max_error_Ha=float(np.max(abs(old.one_body_Ha-a.one_body_Ha))))
            assert checks['eri_sha_matches_old'] and checks['fock_max_error_Ha']<1e-12
            assert checks['one_body_max_error_Ha']<1e-12
        summary=dict(preflight=pre,checks=checks,seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    else:
        base=Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
        env=dict(os.environ)
        env['LD_LIBRARY_PATH']=str(base/'toolchain/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
        executable=env.get('ATOMX_EXPT_EXECUTABLE',str(base/'EXP-T-atomx-build/expt.x'))
        result=compute(a,executable=executable,directory=directory,
            sector='0h1p',active_indices=active,tolerance=1e-13,residual_bound_tolerance_Ha=1e-8,
            memory_bytes=32*1024**3,tile_size=16,threads=2,timeout_seconds=1800,environment=env)
        heff=read_heff(directory/'scratch/HEFF')['0h1p',1]
        groups={}
        for j,i in enumerate(active):groups.setdefault(a.labels[i][:2],[]).append(j)
        assert set(groups)==set(NAMES)
        attach={NAMES[k]:float(np.diag(heff)[v].real.mean()) for k,v in groups.items()}
        scalar_error=float(np.max(abs(heff-np.diag(np.diag(heff)))))
        scalar_error=max(scalar_error,max(float(np.max(abs(np.diag(heff)[v]-attach[NAMES[k]]))) for k,v in groups.items()))
        assert scalar_error<1e-9
        transition={k:(e-attach['6s1/2'])*219474.6313632 for k,e in attach.items()}
        old=json.loads((OLD/'results.json').read_text())['Yb'][2]
        shifts={k:transition[k]-old['transition_cm'][k] for k in transition}
        error=max(abs(transition[k]-e)/e for k,e in EXPERIMENT.items())
        summary=dict(preflight=pre,core_energy_Ha=result.core_energy_Ha,attachment_energies_Ha=attach,
            total_state_energies_Ha={k:result.core_energy_Ha+e for k,e in attach.items()},
            transition_cm=transition,shift_from_36virtual_cm=shifts,max_experiment_fraction=error,
            experimental_gate=error<.01,space_gate=(max(abs(v) for v in shifts.values())<1) if args.radials>2 else None,
            raw_update=result.amplitude_update_max,equation_bound_Ha=result.equation_residual_bound_Ha,
            multiplet_scalar_error_Ha=scalar_error,seconds=time.monotonic()-start,
            parent_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if args.radials==2:
            saved=json.loads((OLD/'yb_o22_r2/result.json').read_text())
            summary['saved_model_energy_error_Ha']=float(np.max(abs(np.array(result.sector_energies_Ha)-saved['sector_energies_Ha'])))
            assert summary['saved_model_energy_error_Ha']<1e-10
    (HERE/(args.name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
