"""Native closed-core linearized SD DC/DCB experiments and saved equations."""
import argparse,json,os,resource,time
from pathlib import Path
import numpy as np
from atomx.methods.dirac_fock import DFSolver,Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.linearized_sd import compute
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    results=[]
    setups=[('Be',4,[Subshell(1,-1,2)],'1s2',('2s','2p'),(-1,1,-2),30),
            ('Ne',10,[Subshell(1,-1,2),Subshell(2,-1,2),Subshell(2,1,2),Subshell(2,-2,4)],'1s2 2s2 2p6',('3s','3p','3d'),(-1,1,-2,2,-3),40)]
    for name,z,shells,reference,targets,kappas,ns in setups:
        core=DFSolver(Z=z,subshells=shells,ns=ns,ks=7).solve()
        assert core.converged
        orbitals={k:v[:2] for k,v in core.orbitals_by_subshell().items()}
        virtual_energies={}
        for k in kappas:
            _,virtual=build_virtual_basis(core,k,e_max=50,potential_mode='VN',orthogonal_to_occupied=True)
            n=int(targets[0][:-1]);orbitals[n,k]=virtual[0][:2];virtual_energies[str((n,k))]=float(virtual[0][2])
        for tag,breit,frozen in [('DC',False,()),('DCB',True,())]+([('frozen_DC',False,('1s',))] if name=='Ne' else []):
            t=time.monotonic()
            result=compute(reference,targets,orbitals,core._bsp_data,z,ns,reference_identity=f'{name}:point:ns{ns}:ks7:VN:first-complement:{tag}',frozen_orbitals=frozen,include_breit=breit,nuclear_potential=core._nuclear_potential)
            path=output/f'{name}_{tag}_equations.npz'
            np.savez_compressed(path,amplitudes=result.amplitudes,K=result.equation_matrix_Ha,subtraction=result.singles_product_subtraction_Ha)
            record=dict(atom=name,tag=tag,reference_energy_Ha=result.reference_energy_Ha,correlation_energy_Ha=result.correlation_energy_Ha,total_energy_Ha=result.total_energy_Ha,equation_residual_Ha=result.equation_residual_Ha,provenance=dict(result.provenance),seconds=time.monotonic()-t,virtual_energies_Ha=virtual_energies,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,load=os.getloadavg())
            results.append(record)
            (output/'native_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k!='provenance'}),flush=True)
if __name__=='__main__':main()
