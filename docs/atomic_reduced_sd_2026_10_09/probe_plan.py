"""Exact scalar planning from a sealed complete native-band identity."""
import argparse,hashlib,json,os,resource,time
from pathlib import Path
from atomx.methods.linearized_sd import plan_scalar_sd
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--band',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
if args.output.exists():raise FileExistsError('choose a fresh output')
s=json.loads(args.band.read_text());i=s['identity'];t=time.monotonic()
plan=plan_scalar_sd(i['occupied_keys'],i['virtual_keys'])
r=plan.resources();r.update(planning_seconds=time.monotonic()-t,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,occupied_radial=len(plan.occupied_keys),virtual_radial=len(plan.virtual_keys),source_band=str(args.band),source_band_sha256=hashlib.sha256(args.band.read_bytes()).hexdigest(),load=os.getloadavg(),channels=[dict(J2=k[0],parity=k[1],occupied_pairs=v[0],virtual_pairs=v[1]) for k,v in plan.pair_channels.items()])
args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='channels'},indent=2))
