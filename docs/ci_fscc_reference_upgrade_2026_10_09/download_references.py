"""Recreate pinned official source copies and explicit local compiler ports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = (
    ('pCI', 'https://github.com/ud-pci/pCI.git', '4d685d82bfdf27876098a87808dadc5fcf3038d1',
     'pCI_gnu_string_constructor.patch'),
    ('AMBiT', 'https://github.com/drjuls/AMBiT.git', 'fe1115aaed97b340e1f4d625d451b165c5bcd7a5', None),
    ('EXP-T', 'https://github.com/aoleynichenko/EXP-T.git', 'aa8d809ad5818d4aa618dcdadbd7f12257ecbc28',
     'EXP-T_gnu15_openblas_declaration.patch'),
)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--apply-build-ports', action='store_true')
    args=parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    here=Path(__file__).resolve().parent
    records=[]
    for name, url, revision, patch in SOURCES:
        repo=args.destination/name
        if not repo.exists():
            subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True,timeout=300)
            subprocess.run(['git','fetch','--depth','1','origin',revision],cwd=repo,check=True,timeout=300)
            subprocess.run(['git','checkout','--detach',revision],cwd=repo,check=True,timeout=60)
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=repo,text=True).strip()
        if actual!=revision or origin!=url:
            raise ValueError(f'Existing {name} differs from pinned official source; preserve it and use a new destination')
        if args.apply_build_ports and patch:
            path=here/patch
            already=subprocess.run(['git','apply','--reverse','--check',str(path)],cwd=repo,capture_output=True)
            if already.returncode:
                subprocess.run(['git','apply','--check',str(path)],cwd=repo,check=True)
                subprocess.run(['git','apply',str(path)],cwd=repo,check=True)
        records.append(dict(code=name,url=url,revision=actual,
                            tracked_changes=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
                            build_port_sha256=hashlib.sha256((here/patch).read_bytes()).hexdigest()
                            if args.apply_build_ports and patch else None))
    (args.destination/'pinned_download_replay.json').write_text(json.dumps(records,indent=2))
