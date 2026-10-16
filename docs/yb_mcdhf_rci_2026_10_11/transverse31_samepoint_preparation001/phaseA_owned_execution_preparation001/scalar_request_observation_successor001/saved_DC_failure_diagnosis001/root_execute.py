"""Foreground one-attempt guardian; diagnostic child imports no Atomx."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

here = Path(__file__).resolve().parent
started = time.monotonic()
with (here / 'execution.log').open('xb') as log:
    process = subprocess.Popen(
        ['/data/code/atomx/.venv/bin/python', '-I', str(here / 'run.py')],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stopped = False
    try:
        code = process.wait(timeout=75)
    except subprocess.TimeoutExpired:
        stopped = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
result = here / 'actual/result.json'
receipt = dict(status='foreground_closed', exit_code=code, wall_stop=stopped,
               elapsed_uncontrolled_s=time.monotonic() - started, controlled_timing=False,
               result_sha256=hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None,
               guardian_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(here / 'root_execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
raise SystemExit(code)
