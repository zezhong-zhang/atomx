"""Read-only /proc CPU-time evidence for actual native worker activity."""
import json
import os
from pathlib import Path
import subprocess
import time
import datetime
OWNER=Path(__file__).resolve().parent
TICKS=os.sysconf('SC_CLK_TCK')
seen={}
last_stamp={}
start=time.monotonic()
with (OWNER/'thread_samples.jsonl').open('w') as out:
    while time.monotonic()-start<1500:
        listing=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        processes=[]
        for line in listing:
            fields=line.strip().split(None,1)
            if len(fields)==2 and 'python' in fields[1] and '/sd/yb_round_probe.py ' in fields[1] and 'ps -eo' not in fields[1]:
                processes.append((int(fields[0]),int(fields[1].split()[-1])))
        for pid,round_id in processes:
            folder=OWNER/f'yb_round{round_id:03d}'
            try:
                contract=json.loads((folder/'contract.json').read_text())
                sequence=[8]+contract['orders'][0]
                f=folder/'observations.json'
                observations=json.loads(f.read_text()) if f.exists() else []
                expected=sequence[len(observations)] if len(observations)<len(sequence) else None
                current={}
                for path in Path(f'/proc/{pid}/task').glob('*/stat'):
                    stat=path.read_text().rsplit(')',1)[-1].split()
                    current[int(path.parent.name)]=(int(stat[11])+int(stat[12]))/TICKS
                now=time.monotonic();elapsed=now-last_stamp.get(pid,now)
                deltas={tid:cpu-seen.get(pid,{}).get(tid,cpu) for tid,cpu in current.items()}
                active=[tid for tid,value in deltas.items() if elapsed>0 and value/elapsed>.25]
                row=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=pid,round=round_id,
                         completed_observations=len(observations),next_requested_workers=expected,
                         interval_seconds=elapsed,thread_cpu_seconds=current,delta_thread_cpu_seconds=deltas,
                         active_threads_over_25_percent=active,active_count=len(active))
                out.write(json.dumps(row)+'\n');out.flush();seen[pid]=current;last_stamp[pid]=now
            except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):pass
        if all((OWNER/f'yb_round{r:03d}/result.json').exists() for r in (0,1,2)):break
        time.sleep(2)
