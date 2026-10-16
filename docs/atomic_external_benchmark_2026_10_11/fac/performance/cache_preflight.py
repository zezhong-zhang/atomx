"""Bounded fresh, per-ion Numba cache diagnostic; preserves failed global epoch."""
import importlib.util
import json
from pathlib import Path

OWNER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fac_performance_runner',OWNER/'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
original_environment = r.environment
cache = OWNER/'fresh_cache002/cache/Z082'
cache.mkdir(parents=True,exist_ok=True)
def scoped_environment(workers):
    env = original_environment(workers)
    env['NUMBA_CACHE_DIR'] = str(cache)
    return env
r.environment = scoped_environment
results = []
with r.timing_lock():
    for repeat in range(2):
        record = r.timed_case('Atomx',1,82,OWNER/'fresh_cache002/preflight'/f'repeat{repeat:03d}'/'Z082')
        results.append(record)
        print(json.dumps(record),flush=True)
        r.dump(OWNER/'fresh_cache002/preflight.json',results)
        if not record['science_ok']:
            break
