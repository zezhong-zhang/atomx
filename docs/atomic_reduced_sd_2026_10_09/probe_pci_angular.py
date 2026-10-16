"""Execute upstream pCI libD S(k,k1,k2), with its declared phase mapping."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from atomx.angular.coefficients import reduced_C_jj
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--library-path',default='');ap.add_argument('--revision',help='Explicit pinned revision for an extracted archive without .git');args=ap.parse_args()
SOURCE=args.source;OUTPUT=args.output_dir
paths=['lib/all-order/include/libD.f','lib/all-order/include/global.par','lib/all-order/allcore-ci.f','lib/all-order/valsd-ci.f']
source_sha={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in paths}
if (SOURCE/'.git').exists():
    revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if args.revision and args.revision!=revision:raise ValueError('Explicit revision differs from checkout')
else:
    pinned=json.loads((HERE/'final_pci/pci_angular_validation.json').read_text())
    if args.revision!=pinned['upstream_revision'] or source_sha!=pinned['source_sha256']:
        raise ValueError('Archive replay requires the exact pinned revision and original source hashes')
    revision=args.revision
OUTPUT.mkdir(parents=True,exist_ok=False)
keys=tuple(k for k in range(-6,7) if k)
rows=[(a,b,k) for a in keys for b in keys for k in range(13)]
exe=args.executable.resolve()
env=dict(os.environ);env['LD_LIBRARY_PATH']=args.library_path+':'+env.get('LD_LIBRARY_PATH','')
out=subprocess.check_output([str(exe)],input=''.join(f'{a} {b} {k}\n' for a,b,k in rows),text=True,env=env)
(OUTPUT/'pci_angular_output.txt').write_text(out)
errors=[];cases=[]
for expected,line in zip(rows,out.splitlines(),strict=True):
    a,b,k,v=line.split(); a,b,k=int(a),int(b),int(k); v=float(v)
    assert (a,b,k)==expected
    phase=(-1)**(abs(a)+abs(b)+k-1)
    mapped=phase*reduced_C_jj(a,b,k)
    errors.append(abs(v-mapped));cases.append(dict(kappa_a=a,kappa_b=b,rank=k,pci_S=v,atomx_mapped=mapped))
r=dict(rows=len(rows),maximum_error=max(errors),gate=1e-12,passed=max(errors)<1e-12,
       relation='pCI S(k,a,b)=(-1)^(|a|+|b|+k-1) atomx reduced_C_jj(a,b,k)',
       scope='Independent live angular convention check only, not pCI radial/SD equation equality',
       upstream_revision=revision,
       source_sha256=source_sha,
       executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),cases=cases)
(OUTPUT/'pci_angular_validation.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
assert r['passed']
