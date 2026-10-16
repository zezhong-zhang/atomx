"""Render normalized CPU scaling from completed, guarded full-work measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OWNER=Path(__file__).resolve().parent
data=json.loads((OWNER/'result.json').read_text())
if data['status']!='complete':raise RuntimeError('three full Yb rounds required before final figure')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for key,label,color in [('finite_whole_action','Na+ complete five-column action','#a25421'),('yb_full_term2','Yb all 414 virtuals: complete term2+response','#2766a8')]:
    rows=[r for r in data[key]['scaling'] if isinstance(r.get('workers',r.get('backend')),int)]
    x=[r.get('workers',r.get('backend')) for r in rows]
    if not data[key]['timing_guard_pass']:
        raise RuntimeError('failed load guard: figure must not show controlled speedup')
    axes[0].plot(x,[r['speedup'] for r in rows],'o-',color=color,label=label)
    axes[1].plot(x,[100*r['efficiency'] for r in rows],'o-',color=color,label=label)
axes[0].plot([1,8],[1,8],'--',color='#aaa',label='Ideal within each workload')
axes[1].axhline(100,linestyle='--',color='#aaa')
axes[0].set_ylabel('Paired median speedup, T1 / Tp')
axes[1].set_ylabel('Parallel efficiency, 100 x speedup / p (%)')
for ax in axes:
    ax.set_xlabel('CPU workers; inner BLAS = 1')
    ax.set_xticks([1,2,4,8]);ax.grid(alpha=.25);ax.set_xlim(.75,8.25)
axes[0].set_ylim(0,8.3);axes[1].set_ylim(0,105)
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('Fresh fixed-work CPU measurements; full signed arrays verified',fontsize=12)
fig.savefig(OWNER/'cpu_scaling.png',dpi=200)
fig.savefig(OWNER/'cpu_scaling.svg')
