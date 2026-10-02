"""Supplementary current jj API against the archived five-level FAC panel."""
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fe_test',root/'tests/test_auger_fac_panel.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
reference = json.loads(m.KLL_FIXTURE.read_text())
result = m.fe17_result.__wrapped__()
rows = m.live_rates.__wrapped__(reference,result)
(Path(__file__).parent/'fe_jj.json').write_text(json.dumps({'mode':'jj','reference':'archived FAC fixture; 2p-hole LMM, no fresh FAC resonance run','rows':rows},indent=2)+'\n')
print(m._report(rows))
