"""Matched integrals, independent upstream diagram sums; no radial accuracy claim."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time
from unittest.mock import patch

import numpy as np
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.mbpt_matrix import build_mbpt2_matrix_self_energy
from atomx.numerics.bspline.convol import ConvolCache, build_slater_rk_bilinear, compute_slater_rk_table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT.parent/'atomx-references/2026-10-09-ci-fscc'
SRC = BASE/'AMBiT'
BUILD = BASE/'AMBiT-build/src'
KAPPAS = (-1, 1, -2, 2, -3)

def key(k, a, b, c, d):
    x, y = sorted((tuple(sorted((a,c))), tuple(sorted((b,d)))))
    return (k, *x, *y)

def compile_probe():
    flags = (BUILD/'CMakeFiles/ambit.dir/flags.make').read_text()
    options=[]
    for line in flags.splitlines():
        if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
            options += shlex.split(line.split('=',1)[1])
    obj=HERE/'ambit_sigma_probe.o'
    compile_cmd=['/usr/bin/c++', *options, '-c', str(HERE/'ambit_sigma_probe.cpp'), '-o', str(obj)]
    link=shlex.split((BUILD/'CMakeFiles/ambit.dir/link.txt').read_text())
    start=link.index('CMakeFiles/ambit.dir/ambit.cpp.o')
    link[start]=str(obj)
    link[link.index('-o')+1]=str(HERE/'ambit_sigma_probe')
    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    lib=BASE/'toolchain/usr/lib/x86_64-linux-gnu'
    env['LD_LIBRARY_PATH']=str(lib)+':'+str(lib/'openblas-serial')+':'+env.get('LD_LIBRARY_PATH','')
    with (HERE/'probe_build.log').open('w') as out:
        for cmd in (compile_cmd,link):
            out.write(shlex.join(cmd)+'\n'); out.flush()
            subprocess.run(cmd,cwd=BUILD,env=env,stdout=out,stderr=out,check=True,timeout=300)
    return env

def main():
    resource.setrlimit(resource.RLIMIT_AS,(4096*1024**2,4096*1024**2))
    started=time.monotonic()
    core=DFSolver(Z=10,subshells=[Subshell(1,-1,2),Subshell(2,-1,2),
                                Subshell(2,1,2),Subshell(2,-2,4)],ns=40,ks=7).solve()
    if not core.converged: raise RuntimeError('unconverged native reference')
    data=core._bsp_data
    states=[]; rows=[]; by_array={}; arrays={}
    for sh in core.subshells:
        P,Q,_=core.orbitals_by_subshell()[sh.n,sh.kappa]
        e=core.orbital_energies[sh.label]
        i=len(states); states.append((P,Q,sh.kappa,e)); rows.append((i,sh.n,sh.kappa,e,1))
        by_array[id(P),id(Q)]=i
    core_ids=list(range(len(states)))
    virtual_ids=[]; blocks={}
    for k in KAPPAS:
        _, raw=build_virtual_basis(core,k,e_max=50.,potential_mode='VN',orthogonal_to_occupied=True)
        blocks[k]=raw[:3]
        for j,(P,Q,e) in enumerate(blocks[k]):
            i=len(states); states.append((P,Q,k,e)); rows.append((i,3+j,k,e,0))
            by_array[id(P),id(Q)]=i; virtual_ids.append(i)
    for i,(P,Q,k,e) in enumerate(states):
        arrays[f'P_{i}']=P; arrays[f'Q_{i}']=Q
    arrays['orbital_rows']=np.asarray(rows)
    # Preserve radial coefficient arrays and the full native B-spline workspace.
    for name,value in data.items():
        if isinstance(value,np.ndarray): arrays['workspace_'+name]=value
    (HERE/'workspace_scalars.json').write_text(json.dumps({name:value for name,value in data.items()
                                                        if isinstance(value,(int,float,str,bool))},indent=2)+'\n')
    np.savez_compressed(HERE/'radial_inputs.npz',**arrays)
    (HERE/'orbitals.txt').write_text(''.join('%d %d %d %.17g %d\n'%r for r in rows))
    table={}; symmetry_error=0.; cache=ConvolCache(data)
    for kappa in (-1,-2,-3):
        target_ids=[i for i in virtual_ids if states[i][2]==kappa]
        for partner_ids,ket_ids in ((core_ids,virtual_ids),(virtual_ids,core_ids)):
            cp=np.array([states[i][0] for i in ket_ids]).T
            cq=np.array([states[i][1] for i in ket_ids]).T
            for a in target_ids:
                for b in partner_ids:
                    for rank in range(5):
                        bilinear=build_slater_rk_bilinear(*states[a][:2],*states[b][:2],data,rank,cache=cache)
                        values=compute_slater_rk_table(bilinear,cp,cq,cp,cq)
                        for ii,c in enumerate(ket_ids):
                            for jj,d in enumerate(ket_ids):
                                kk=key(rank,a,b,c,d); value=float(values[ii,jj])
                                if kk in table: symmetry_error=max(symmetry_error,abs(table[kk]-value))
                                else: table[kk]=value
    if symmetry_error>1e-12: raise RuntimeError('radial table symmetry gate')
    (HERE/'integrals.txt').write_text(''.join('%d %d %d %d %d %.17g\n'%(*kk,value)
                                             for kk,value in sorted(table.items())))
    def factory(targets,partner,kets,data,cache):
        b=by_array[id(partner[0]),id(partner[1])]
        def evaluator(t):
            a=by_array[id(t[0]),id(t[1])]
            def radial(k,Pc,Qc,Pd,Qd):
                c=by_array[id(Pc),id(Qc)]; d=by_array[id(Pd),id(Qd)]
                return table[key(k,a,b,c,d)]
            return radial
        return [evaluator(t) for t in targets]
    def basis(_,k,**kwargs):
        assert kwargs['potential_mode']=='VN' and kwargs['orthogonal_to_occupied']
        return [],blocks[k]
    requests=[]; measured={}; direct_error=0.
    for k in (-1,-2,-3):
        ids=[i for i in virtual_ids if states[i][2]==k]
        targets=[states[i] for i in ids]; energy=targets[0][3]
        terms=[]
        args=dict(reference_identity=f'Ne:matched:AMBiT:{k}',virtual_kappas=KAPPAS,
                  energy_window_au=50.,n_virt_max=3)
        with patch('atomx.methods.mbpt.build_virtual_basis',side_effect=basis), \
             patch('atomx.methods.mbpt_matrix._radial_table_evaluators',side_effect=factory):
            sigma=build_mbpt2_matrix_self_energy(core,targets,term_recorder=lambda p,r,d:terms.append((p,r,d)),**args)
        with patch('atomx.methods.mbpt.build_virtual_basis',side_effect=basis):
            direct=build_mbpt2_matrix_self_energy(core,targets,**args)
        direct_error=max(direct_error,float(np.max(abs(direct.evaluate(energy)-sigma.evaluate(energy)))))
        matrices={d:sum((r/(energy-p) for p,r,dd in terms if d==dd),np.zeros((3,3)))
                  for d in ('2p1h','2h1p')}
        for ii,a in enumerate(ids):
            for jj,b in enumerate(ids):
                requests.append((a,b,energy)); measured[a,b]=[float(matrices[d][ii,jj]) for d in ('2p1h','2h1p')]+[float(sigma.evaluate(energy)[ii,jj])]
        np.savez_compressed(HERE/f'native_sigma_kappa_{k}.npz',poles=sigma.poles_Ha,residues=sigma.residues_Ha2,
                            energy=energy,eps=sigma.reference_energies_Ha,pp=matrices['2p1h'],hh=matrices['2h1p'])
    (HERE/'requests.txt').write_text(''.join('%d %d %.17g\n'%r for r in requests))
    env=compile_probe()
    with (HERE/'ambit_sigma_output.txt').open('w') as out, (HERE/'ambit_sigma_stderr.txt').open('w') as err:
        subprocess.run([str(HERE/'ambit_sigma_probe'),str(HERE/'orbitals.txt'),str(HERE/'integrals.txt'),str(HERE/'requests.txt')],env=env,stdout=out,stderr=err,check=True,timeout=300)
    warnings=(HERE/'ambit_sigma_stderr.txt').read_text()
    if warnings.strip(): raise RuntimeError('AMBiT reference reported warnings: '+warnings)
    ref=np.loadtxt(HERE/'ambit_sigma_output.txt'); comparisons=[]; max_error=0.
    for a,b,e,pp,hh,total in ref:
        values=measured[int(a),int(b)]; errors=np.array([pp,hh,total])-values
        max_error=max(max_error,float(np.max(abs(errors))))
        comparisons.append(dict(a=int(a),b=int(b),energy_Ha=e,atomx_Ha=values,
                                ambit_Ha=[pp,hh,total],error_Ha=errors.tolist()))
        np.testing.assert_allclose([pp,hh,total],values,atol=1e-11,rtol=1e-10)
    assert max(abs(v[2]) for (a,b),v in measured.items() if a!=b)>1e-7
    assert direct_error<1e-12
    report=dict(pass_=True,reference_source='fe1115aaed97b340e1f4d625d451b165c5bcd7a5',
                orbitals=len(states),integrals=len(table),matrix_entries=len(comparisons),
                components=['2p1h','2h1p','total'],max_error_Ha=max_error,
                native_vs_canonical_table_Ha=direct_error,radial_symmetry_error_Ha=symmetry_error,
                comparisons=comparisons,seconds=time.monotonic()-started,
                max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [HERE/'orbitals.txt',HERE/'integrals.txt',HERE/'requests.txt',HERE/'radial_inputs.npz']})
    (HERE/'matched_sigma_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','input_sha256')}))

if __name__=='__main__': main()
