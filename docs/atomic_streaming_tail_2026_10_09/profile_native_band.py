"""Bounded owned-process profiling; records partial work, not a solved model."""
import cProfile
import json
from pathlib import Path
import resource
import runpy
import signal
import time

HERE=Path(__file__).resolve().parent
profile=cProfile.Profile()
start=time.monotonic()


def stop(signum, frame):
    profile.disable()
    profile.dump_stats(HERE/'full_band_partial_profile.pstats')
    print(json.dumps(dict(status='declared 20-second partial-profile stop',
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)
    raise SystemExit(0)


signal.signal(signal.SIGALRM,stop)
signal.setitimer(signal.ITIMER_REAL,20)
profile.enable()
runpy.run_path(str(HERE/'probe_full_band.py'),run_name='__main__')
signal.setitimer(signal.ITIMER_REAL,0)
stop(None,None)
