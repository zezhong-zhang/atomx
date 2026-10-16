"""Same complete Ne window: bounded export measurement or actual SD replay."""
import argparse
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.fscc import atomic_integrals, compute, export_integrals

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'atomic_allorder_fscc_2026_10_09'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--representation',choices=['packed','on_demand'],required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--solve',action='store_true')
    args=ap.parse_args()
    start=time.monotonic(); load_before=os.getloadavg()
    core=DFSolver(Z=10,subshells=[Subshell(1,-1,2),Subshell(2,-1,2),Subshell(2,1,2),Subshell(2,-2,4)],ns=40,ks=7).solve()
    a=atomic_integrals(core,virtual_lmax=2,virtual_energy_max_Ha=1,
        representation=args.representation,max_integral_bytes=2*1024**3)
    built=time.monotonic()
    directory=HERE/args.name
    if args.solve:
        base=Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
        env=dict(os.environ)
        env['LD_LIBRARY_PATH']=str(base/'toolchain/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
        executable=env.get('ATOMX_EXPT_EXECUTABLE',str(base/'EXP-T-atomx-build/expt.x'))
        result=compute(a,executable=executable,directory=directory,
            tolerance=1e-13,residual_bound_tolerance_Ha=1e-8,memory_bytes=32*1024**3,
            tile_size=32,threads=2,timeout_seconds=600,environment=env)
        old=json.loads((OLD/'ne_full_band_summary.json').read_text())
        prior=old['core_energy_Ha']
        error=abs(result.core_energy_Ha-prior)
        assert error<1e-10
        solved=dict(energy_Ha=result.core_energy_Ha,old_energy_error_Ha=error,
            raw_update=result.amplitude_update_max,equation_bound_Ha=result.equation_residual_bound_Ha)
        manifest=json.loads((directory/'integrals.json').read_text())
    else:
        manifest=export_integrals(a,directory)
        solved=None
    end=time.monotonic()
    reference=json.loads((OLD/'ne_full_band/integrals.json').read_text())
    with (OLD/'ne_full_band/ATOMX_ONE').open('rb') as stream:
        stream.seek(8)
        n,_,_=struct.unpack('<idd',stream.read(20))
        stream.seek(28+12*n)
        fock=np.fromfile(stream,dtype='<c16').reshape(n,n)
    assert tuple(map(tuple,reference['labels']))==a.labels
    assert manifest['files']['ATOMX_TWO']==reference['files']['ATOMX_TWO']
    fock_error=float(np.max(abs(fock-a.fock_Ha)))
    assert fock_error<1e-12
    record=dict(representation=args.representation,spinors=len(a.labels),virtual=len(a.labels)-a.noccupied,
        construction_seconds=built-start,export_or_solve_seconds=end-built,
        end_to_end_seconds=end-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        load_before=load_before,load_after=os.getloadavg(),producer=a.diagnostics(),
        eri_sha_matches_old=True,fock_error_Ha=fock_error,solved=solved,
        timing_scope='Core/complete-band construction and integral export; backend included only when solved is present')
    (HERE/(args.name+'_summary.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
