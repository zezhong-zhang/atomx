"""Read-only bounded observation of native child OpenMP task counts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

master = int(sys.argv[1])
owner = Path(__file__).resolve().parent
observations = {}
started = time.monotonic()

def descendants(pid):
    children = Path(f'/proc/{pid}/task/{pid}/children')
    try:
        ids = [int(value) for value in children.read_text().split()]
    except FileNotFoundError:
        return []
    result = list(ids)
    for child in ids:
        result.extend(descendants(child))
    return result

while Path(f'/proc/{master}').exists() and time.monotonic()-started<600:
    for pid in descendants(master):
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if command[0].decode().rsplit('/',1)[-1] != 'ambit':
                continue
            cwd = Path(f'/proc/{pid}/cwd').resolve()
            if not cwd.is_relative_to(owner):
                continue
            tasks = len(list(Path(f'/proc/{pid}/task').iterdir()))
        except (FileNotFoundError,ProcessLookupError):
            continue
        row = observations.setdefault(str(pid),dict(pid=pid,directory=str(cwd),samples=0,max_observed_threads=0,
                                               first_utc=datetime.now(timezone.utc).isoformat()))
        row['samples'] += 1
        row['max_observed_threads'] = max(row['max_observed_threads'],tasks)
        row['last_utc'] = datetime.now(timezone.utc).isoformat()
    time.sleep(.02)
(owner/'threads_observed.json').write_text(json.dumps(dict(scope='Read-only /proc native child task count sampled every20ms; only later corrected native runs sampled. Unobserved earlier records stay null.',elapsed_seconds=time.monotonic()-started,
                                                        observations=observations),indent=2)+'\n')
