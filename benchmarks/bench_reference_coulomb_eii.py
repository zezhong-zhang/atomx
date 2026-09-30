"""Fresh FAC EII panel with the identical bare Coulomb one-electron field."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.potentials.static_field import static_hartree_potential
from atomx.scattering.electron_ionization.dw import dw_eii_cross_section
from benchmarks.bench_reference_structure import run_reference
from benchmarks.fac_runner import parse_ci


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fac',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--z',type=int,nargs='+',default=[2,26,82])
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    report=dict(scope='Hydrogen-like EII; point nucleus, infinite mass, corrections off, bare Coulomb field; partial-wave and energy-share convergence remain separate checks',cases=[],fac_sha256=hashlib.sha256(args.fac.read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/atomx').rglob('*.py')})
    for z in args.z:
        start=time.perf_counter()
        atom=Atom(z,config='1s1',engine='bspline',nuclear='point',corrections='none').solve(tol=1e-10)
        if not atom.converged:raise RuntimeError('bound state not converged')
        scf_seconds=time.perf_counter()-start
        threshold=-atom.orbital_energies['1s1/2']*HARTREE_EV
        factors=np.array([1.5,2.,5.]);excess=(factors-1)*threshold
        folder=out/f'Z{z:03d}';folder.mkdir()
        # Dummy symbol with explicit nuclear charge; all nuclear parameters
        # are set explicitly by SetAtom, independent of an isotope default.
        from atomx.constants import chemical_symbols
        lines=[f"SetAtom('{chemical_symbols[z]}',{z},-1,0)", 'SetSE(0)','SetVP(0)','SetMS(0,0)','SetBreit(0)',
            'SetConfigEnergyMode(-1)',"Closed('')","Config('g','1s1')","Config('ion','')","OptimizeRadial(['g'])",
            "Structure('levels.b',['g','ion'])","MemENTable('levels.b')","PrintTable('levels.b','levels.txt',1)",
            "DumpMeanField('mean-field.txt')","SetCIQkMode('dw')",'SetCIPWOptions(1e-3,36,4,36)',
            "SetOption('ionization:maxk',6)",
            f'SetCIEGrid({excess.tolist()})',f'SetUsrCIEGrid({excess.tolist()})',"CITable('ci.b',['g'],['ion'])","PrintTable('ci.b','ci.txt',1)"]
        (folder/'run.sf').write_text('\n'.join(lines)+'\n')
        fac_seconds=run_reference([str(args.fac.resolve()),'run.sf'],folder,timeout=240)
        refs=parse_ci(folder/'ci.txt')
        if len(refs)!=1:raise RuntimeError('expected one ground-to-bare ionization channel')
        ref=refs[0];fac_values=np.array(ref['sigma_1e20cm2'])*1e-20
        if len(fac_values)!=len(factors):raise RuntimeError('FAC grid size mismatch')
        # CI table rows use %11.4E. Check the requested nodes at that
        # documented print precision, not with a tighter numeric tolerance.
        np.testing.assert_array_equal(ref['E_eV'],[float(f'{e:.4E}') for e in excess])
        table_lines=(folder/'ci.txt').read_text().splitlines()
        threshold_index=next(i for i,line in enumerate(table_lines) if line.startswith('NTEGRID'))
        fac_threshold=float(table_lines[threshold_index+1])
        start=time.perf_counter()
        dw=dw_eii_cross_section(atom.result,factors*threshold,kappa_max=10,n_outgoing_energies=25,
            partial_wave_tail='none',
            continuum_model=ContinuumModel.shooting(n_grid=3000))
        native_seconds=time.perf_counter()-start
        sample=np.geomspace(1e-12,100.,100)
        field=static_hartree_potential(atom.result,screening='mean')
        fac_field=np.loadtxt(folder/'mean-field.txt')
        case=dict(z=z,threshold_eV=threshold,fac_threshold_eV=fac_threshold,factors=factors.tolist(),
            native_partial_wave_tail='none',native_lmax=10,fac_incident_lmax=36,fac_eject_lmax=4,fac_max_rank=3,
            atomx_sigma_cm2=dw.sigma_total_cm2.tolist(),fac_sigma_cm2=fac_values.tolist(),ratios=(dw.sigma_total_cm2/fac_values).tolist(),
            atomx_scf_seconds=scf_seconds,atomx_eii_seconds=native_seconds,fac_process_seconds=fac_seconds,
            atomx_field_max_error=float(np.max(abs(sample*field(sample)+z))),fac_field_max_error=float(np.max(abs(fac_field[:,1]+z))),
            native_provenance=dw.provenance,raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()})
        # The public continuum-model object is represented explicitly by its
        # numerical settings, rather than an opaque repr in the artifact.
        case['native_provenance']['continuum_model']={'method':'shooting','n_grid':3000}
        report['cases'].append(case)
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(z,case['ratios'],native_seconds,fac_seconds,flush=True)


if __name__=='__main__':main()
