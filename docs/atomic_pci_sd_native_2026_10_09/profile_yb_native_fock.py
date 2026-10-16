"""Read-only sampling of the exact full Yb one-body construction stage."""

import faulthandler
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

from atomx.physics.csf.spinor_integrals import build_spinor_integrals
from atomx.potentials.nuclear import make_nuclear_potential

HERE = Path(__file__).resolve().parent
OUT = HERE/"yb_native_profile001"
OUT.mkdir(exist_ok=False)
ROOT = HERE.parents[1]
source = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((ROOT/'src/atomx').rglob('*.py'))}
(OUT/'contract.json').write_text(json.dumps(dict(
    fixed="all 68 core electrons; original Fermi/ns123; all 414 l<=3/<5 Ha virtuals",
    hypothesis="native Fock/validation stage or Fortran rhocore explains long preparation",
    sampling_seconds=15, diagnostic_seconds=90,
    stopping="diagnostic timeout is not a failed SD or physical gate; original workloads remain running",
    source_sha256=source), indent=2)+"\n")
plan = json.loads((HERE/'yb_full_run003/yb_full_plan.json').read_text())
occ, vir = (tuple(map(tuple, plan[k])) for k in ('occupied_keys', 'virtual_keys'))
with np.load(HERE/'yb_full_run003/yb_full_inputs.npz') as data:
    ws = {k.removeprefix('workspace_'):data[k] for k in data if k.startswith('workspace_')}
    for k, a in tuple(ws.items()):
        if a.ndim == 0:
            ws[k] = a.item()
    waves = {a:(data[f'P_{a[0]}_{a[1]}'],data[f'Q_{a[0]}_{a[1]}']) for a in occ+vir}
with (OUT/'stacks.log').open('w') as f:
    faulthandler.dump_traceback_later(15, repeat=True, file=f)
    def timeout(signum, frame):
        raise TimeoutError('predeclared 90-second stage sampling complete; no numerical result claimed')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    started = time.perf_counter()
    try:
        result = build_spinor_integrals(waves, ws, 70, ws['ns'], occupied_keys=occ,
            virtual_keys=vir, nuclear_potential=make_nuclear_potential(70,'fermi'),
            representation='on_demand', max_integral_bytes=32*1024**3)
        (OUT/'result.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
            provenance=dict(result.identity), diagnostics=result.diagnostics()),indent=2)+"\n")
    finally:
        signal.alarm(0)
        faulthandler.cancel_dump_traceback_later()
