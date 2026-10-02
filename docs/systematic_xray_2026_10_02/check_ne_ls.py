"""Supplementary default public Atom Ne anchor, separate from DFSolver sweep."""
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ne_test',root/'tests/test_auger_ne_kll_xatom.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
result = m.ne_khole_result.__wrapped__()
rates = m._channel_rates(result)
rows = [dict(channel=c,atomx_per_s=v,xatom_per_s=m.XATOM_NE_KLL[c][0]*m.AU_RATE_PER_S,
             ratio=v/(m.XATOM_NE_KLL[c][0]*m.AU_RATE_PER_S)) for c,v in rates.items()]
record = dict(mode='ls',shooting_grid=400,bsp_box=float(result.bsp_data['t'][-1]),
              reference='archived XATOM fixture, explicit XATOM kinetic energies',rows=rows,
              total_ratio=sum(rates.values())/(m.XATOM_TOTAL_AU*m.AU_RATE_PER_S))
(Path(__file__).parent/'ne_ls.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
