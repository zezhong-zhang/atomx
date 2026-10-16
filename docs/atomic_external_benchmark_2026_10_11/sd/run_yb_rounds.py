"""Three independent, unchanged complete Yb rounds; each gets original caps."""
import subprocess
import sys
from pathlib import Path
OWNER=Path(__file__).resolve().parent
for round_number in (0,1,2):
    with (OWNER/f'yb_round{round_number:03d}.log').open('w') as log:
        run=subprocess.run([sys.executable,str(OWNER/'yb_round_probe.py'),str(round_number)],stdout=log,stderr=subprocess.STDOUT,timeout=610)
    print({'round':round_number,'exit_code':run.returncode},flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)
